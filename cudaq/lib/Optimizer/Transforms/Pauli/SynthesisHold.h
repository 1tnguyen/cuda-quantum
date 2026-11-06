/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/

#pragma once

#include "SynthesisNativeRefinement.h"
#include <array>
#include <functional>

namespace cudaq::pauli::graph::detail {

/// Every callback returns a fully reconstructed, phase-exact unitary. A local
/// source is never an incumbent: the callback must rebuild its graph and frame.
using HoldSynthesis = std::function<Result(unsigned, const std::vector<Gate> &,
                                           const Options &, SearchBudget *)>;

inline Result holdFailure(std::string error, SearchTermination status) {
  Result result;
  result.error = std::move(error);
  result.termination = status;
  result.stats.budgetExhausted = status == SearchTermination::Exhausted;
  return result;
}

/// Reconstruct a generated three-wire four-T gadget without its CZ/Z debt.
/// The T decoder deliberately requires a T reduction; this equal-T identity
/// instead reduces native gates. Scalar interruptions commute and are retained
/// verbatim. Only bounded, phase-exact local proofs authorize replacements.
inline bool refineHoldThreeWireTemplates(unsigned width, Result &result,
                                         const Options &options,
                                         SearchBudget &budget) {
  constexpr size_t maxGates = 262144, window = 32, length = 14;
  const auto &source = result.gates;
  if (!options.algebraicCandidates || source.size() < length ||
      source.size() > maxGates || budget.stopped())
    return true;
  // Prepay the bounded full-word copy and final objective comparison, so an
  // exhausted local proof can still return an already complete alternative.
  if (!budget.take(16 * (source.size() + width + 1) *
                   (uint64_t(width) / 64 + 1)))
    return true;
  std::vector<Gate> word;
  word.reserve(source.size());
  unsigned rewrites = 0;
  size_t at = 0;
  while (at < source.size() && !budget.stopped()) {
    if (!budget.take(window * (uint64_t(width) / 64 + 1)))
      break;
    std::array<const Gate *, length> gates{};
    size_t count = 0, end = at;
    if (source[at].kind == Kind::CX)
      for (; end < source.size() && end - at < window && count < length; ++end)
        if (source[end].kind != Kind::Phase)
          gates[count++] = &source[end];
    bool matched = count == length;
    unsigned a = 0, b = 0, c = 0;
    if (matched) {
      a = gates[0]->b;
      b = gates[0]->a;
      c = gates[2]->b;
      auto one = [&](size_t i, Kind kind, unsigned q) {
        return gates[i]->kind == kind && gates[i]->a == q;
      };
      auto cx = [&](size_t i, unsigned control, unsigned target) {
        return one(i, Kind::CX, control) && gates[i]->b == target;
      };
      auto rz = [&](size_t i, unsigned q, int sign) {
        const auto &g = *gates[i];
        if (g.kind != Kind::Rotate || g.pauli.x ||
            g.pauli.z != QubitSet::single(q))
          return false;
        auto quarters = tcount::detail::quarters(g.angle);
        if (!quarters)
          return false;
        int k = *quarters * (g.pauli.sign ? -1 : 1);
        // A rotation has period 4*pi, not 2*pi: preserve its scalar sign.
        return ((k - sign) % 16) == 0;
      };
      const auto &cz = *gates[1];
      matched = a < width && b < width && c < width && a != b && a != c &&
                b != c && cz.kind == Kind::CZ &&
                ((cz.a == b && cz.b == c) || (cz.a == c && cz.b == b)) &&
                cx(2, b, c) && one(3, Kind::H, b) && cx(4, b, a) &&
                rz(5, a, -1) && cx(6, c, a) && rz(7, b, 1) && cx(8, c, b) &&
                rz(9, a, 1) && cx(10, b, a) && rz(11, b, -1) &&
                one(12, Kind::H, b) && one(13, Kind::Z, b);
    }
    if (!matched) {
      word.push_back(source[at++]);
      continue;
    }
    std::vector<Gate> replacement{
        {Kind::CX, 1, 2}, {Kind::CX, 1, 0}, {Kind::H, 1}};
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
      replacement.push_back({Kind::CX, 2, 1});
      replacement.push_back(Gate::rotation({0, 2}, {-1, 4}));
      replacement.push_back({Kind::CX, 0, 1});
      replacement.push_back(Gate::rotation({0, 2}, {1, 4}));
    }
    replacement.push_back({Kind::H, 1});
    // Compact the actual matched operations, including angle signs, before
    // independent replay. No proof depends on a benchmark name or input seed.
    std::vector<Gate> local;
    auto index = [&](unsigned q) { return q == a ? 0u : q == b ? 1u : 2u; };
    for (const Gate *g : gates) {
      Gate mapped = *g;
      if (mapped.kind == Kind::Rotate)
        mapped.pauli = {0, QubitSet::single(index(mapped.pauli.z.first())),
                        mapped.pauli.sign};
      else {
        mapped.a = index(mapped.a);
        if (mapped.kind == Kind::CX || mapped.kind == Kind::CZ)
          mapped.b = index(mapped.b);
      }
      local.push_back(std::move(mapped));
    }
    if (!budget.take(10000))
      break;
    if (!graph::equivalent(3, local, replacement)) {
      result = holdFailure("hold three-wire template phase replay failed",
                           SearchTermination::InvalidProof);
      return false;
    }
    // Copy all intervening scalar operations exactly; they commute with the
    // proved local unitary, including opaque scalar angles.
    for (size_t i = at; i < end; ++i)
      if (source[i].kind == Kind::Phase)
        word.push_back(source[i]);
    const std::array<unsigned, 3> physical{a, b, c};
    for (auto gate : replacement) {
      if (gate.kind == Kind::Rotate)
        gate.pauli = {0, QubitSet::single(b)};
      else {
        gate.a = physical[gate.a];
        if (gate.kind == Kind::CX)
          gate.b = physical[gate.b];
      }
      word.push_back(std::move(gate));
    }
    ++rewrites;
    at = end;
  }
  word.insert(word.end(), source.begin() + at, source.end());
  if (rewrites &&
      tcount::cost(width, word).t <= tcount::cost(width, source).t &&
      circuitScore(width, word, options) < circuitScore(width, source, options))
    result.gates = std::move(word);
  return true;
}

/// Compose source-bound native certificates only with an already synthesized
/// word. Ranking follows the requested objective, with a separate T ceiling.
inline bool refineHoldNative(unsigned width, Result &result,
                             const Options &options, SearchBudget &budget) {
  if (!options.algebraicCandidates || budget.stopped())
    return true;
  if (!budget.take((result.gates.size() + 1) * (uint64_t(width) / 64 + 1)))
    return true; // The complete, checked word is still usable.
  const auto source = result.gates;
  auto limits = boundedNativeRefinementLimits(options, source.size());
  limits.maxCandidates = 16;
  limits.maxWork = 2000000000;
  limits.maxBackendWork = 200000000;
  auto candidates =
      refineSynthesizedNative(width, source, options, limits, budget);
  if (candidates.status == NativeRefinementStatus::InvalidProof) {
    result = holdFailure(candidates.error, SearchTermination::InvalidProof);
    return false;
  }
  for (const auto &candidate : candidates.alternatives) {
    if (!candidate.matchesSource(width, source)) {
      result = holdFailure("hold refinement source binding failed",
                           SearchTermination::InvalidProof);
      return false;
    }
    if (tcount::cost(width, candidate.gates()).t <=
            tcount::cost(width, result.gates).t &&
        circuitScore(width, candidate.gates(), options) <
            circuitScore(width, result.gates, options))
      result.gates = candidate.gates();
  }
  // Preserve the established native portfolio and its T reductions first.
  // The optional equal-T reconstruction starts from that same generated source;
  // it must not replace an incumbent whose T count was already reduced.
  if (source.size() > 262144 || budget.stopped() ||
      !budget.take((source.size() + 1) * (uint64_t(width) / 64 + 1)))
    return true;
  Result rewritten;
  rewritten.gates = source;
  if (!refineHoldThreeWireTemplates(width, rewritten, options, budget)) {
    result = std::move(rewritten);
    return false;
  }
  if (native_refinement_detail::sameWord(source, rewritten.gates) ||
      tcount::cost(width, rewritten.gates).t >
          tcount::cost(width, result.gates).t)
    return true;
  const auto rewrittenSource = rewritten.gates;
  auto continued =
      refineSynthesizedNative(width, rewrittenSource, options, limits, budget);
  if (continued.status == NativeRefinementStatus::InvalidProof) {
    result = holdFailure(continued.error, SearchTermination::InvalidProof);
    return false;
  }
  for (const auto &candidate : continued.alternatives) {
    if (!candidate.matchesSource(width, rewrittenSource)) {
      result = holdFailure("hold template refinement source binding failed",
                           SearchTermination::InvalidProof);
      return false;
    }
    if (tcount::cost(width, candidate.gates()).t <=
            tcount::cost(width, result.gates).t &&
        circuitScore(width, candidate.gates(), options) <
            circuitScore(width, result.gates, options))
      result.gates = candidate.gates();
  }
  if (tcount::cost(width, rewrittenSource).t <=
          tcount::cost(width, result.gates).t &&
      circuitScore(width, rewrittenSource, options) <
          circuitScore(width, result.gates, options))
    result.gates = rewrittenSource;
  return true;
}

/// Partition before frame absorption. Exact block proofs compose on unchanged
/// physical wires, including entangled inputs and scalar phase. An individual
/// operation wider than the support cap remains intact, never truncated.
inline Result synthesizeHoldBlocks(unsigned width,
                                   const std::vector<Gate> &input,
                                   const Options &options, SearchBudget &budget,
                                   unsigned maxWidth,
                                   const HoldSynthesis &synthesize) {
  constexpr unsigned maxBlockGates = 64;
  Result result;
  QubitSet active;
  std::vector<Gate> block;
  auto flush = [&]() -> bool {
    if (block.empty())
      return true;
    if (!budget.take((block.size() + 1) * (uint64_t(width) / 64 + 1))) {
      result = holdFailure("hold checkpoint mapping work limit",
                           SearchTermination::Exhausted);
      return false;
    }
    std::vector<unsigned> physical;
    for (unsigned q : active.set_bits())
      physical.push_back(q);
    bool identityRotation =
        std::any_of(block.begin(), block.end(),
                    [](const Gate &gate) { return gate.kind == Kind::Rotate; });
    if (physical.empty() && !identityRotation) {
      if (block.size() > options.maxGates - result.gates.size()) {
        result = holdFailure("hold checkpoint scalar gate cap",
                             SearchTermination::Unsupported);
        return false;
      }
      result.gates.insert(result.gates.end(), block.begin(), block.end());
      block.clear();
      return true;
    }
    // Identity rotations still need exact conversion to scalar phases (and
    // rejection of opaque identity angles). Use an otherwise idle real wire
    // for the independently checked normalization, without introducing support.
    if (physical.empty())
      physical.push_back(0);
    auto localIndex = [&](unsigned q) {
      return unsigned(std::lower_bound(physical.begin(), physical.end(), q) -
                      physical.begin());
    };
    auto map = [](Gate gate, const auto &index) {
      if (gate.kind == Kind::Rotate) {
        Pauli p;
        p.sign = gate.pauli.sign;
        for (unsigned q : gate.pauli.x.set_bits())
          p.x.set(index(q));
        for (unsigned q : gate.pauli.z.set_bits())
          p.z.set(index(q));
        gate.pauli = std::move(p);
      } else if (gate.kind != Kind::Phase) {
        gate.a = index(gate.a);
        if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
            gate.kind == Kind::Swap)
          gate.b = index(gate.b);
      }
      return gate;
    };
    std::vector<Gate> local;
    local.reserve(block.size());
    for (const auto &gate : block)
      local.push_back(map(gate, localIndex));
    Options settings = options;
    // Keep the bounded diagnostic log available for whole-route decisions;
    // thousands of local continuations must not evict their summaries.
    settings.diagnostics = nullptr;
    settings.holdCheckpoints = false;
    settings.commutingCandidates = settings.jointPhaseCandidates = 0;
    auto synthesized = synthesize(physical.size(), local, settings, &budget);
    if (!synthesized) {
      result = std::move(synthesized);
      return false;
    }
    if (synthesized.gates.size() > options.maxGates - result.gates.size()) {
      result = holdFailure("hold checkpoint output gate cap",
                           SearchTermination::Unsupported);
      return false;
    }
    if (!budget.take((synthesized.gates.size() + 1) *
                     (uint64_t(width) / 64 + 1))) {
      result = holdFailure("hold checkpoint lifting work limit",
                           SearchTermination::Exhausted);
      return false;
    }
    // The same injective map embeds both source and result; the local unitary
    // proof therefore extends by identity on every other physical wire.
    for (const auto &gate : synthesized.gates)
      result.gates.push_back(
          map(gate, [&](unsigned q) { return physical[q]; }));
    result.stats.rotationsBefore += synthesized.stats.rotationsBefore;
    result.stats.rotationsAfter += synthesized.stats.rotationsAfter;
    result.stats.tqes += synthesized.stats.tqes;
    result.stats.swaps += synthesized.stats.swaps;
    result.stats.candidates += synthesized.stats.candidates;
    result.stats.dependencies += synthesized.stats.dependencies;
    block.clear();
    active = {};
    return true;
  };
  for (const auto &gate : input) {
    QubitSet support;
    if (gate.kind == Kind::Rotate)
      support = gate.pauli.x | gate.pauli.z;
    else if (gate.kind != Kind::Phase) {
      support.set(gate.a);
      if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
          gate.kind == Kind::Swap)
        support.set(gate.b);
    }
    if (!block.empty() && ((active | support).count() > maxWidth ||
                           block.size() == maxBlockGates))
      if (!flush())
        return result;
    active |= support;
    block.push_back(gate);
  }
  if (!flush())
    return result;
  auto cleanupOptions = options;
  cleanupOptions.diagnostics = nullptr;
  refineHoldNative(width, result, cleanupOptions, budget);
  return result;
}

/// A fixed source-only portfolio. Mandatory local reconstruction finishes
/// before optional searches; local slice exhaustion never invalidates it.
inline Result synthesizeHoldPortfolio(unsigned width,
                                      const std::vector<Gate> &input,
                                      const Options &options,
                                      SearchBudget &budget,
                                      const HoldSynthesis &synthesize) {
  auto record = [&](const char *route, const Result &candidate,
                    uint64_t before) {
    if (options.diagnostics)
      options.diagnostics->record(
          {route, candidate ? "verified" : "declined", candidate.error,
           input.size(), candidate.gates.size(), budget.work() - before});
  };
  uint64_t before = budget.work();
  auto best =
      synthesizeHoldBlocks(width, input, options, budget, 3, synthesize);
  record("hold-checkpoint-3", best, before);
  if (!best)
    return best;
  for (bool global : {false, true}) {
    auto time = budget.timeRemainingMs();
    if (budget.stopped() || !budget.remaining() || (time && !*time))
      break;
    SearchBudget optional(budget,
                          std::min<uint64_t>(budget.remaining(),
                                             global ? 1000000000 : 10000000000),
                          global ? 5000 : 30000);
    before = budget.work();
    auto candidate = global ? synthesize(width, input, options, &optional)
                            : synthesizeHoldBlocks(width, input, options,
                                                   optional, 4, synthesize);
    record(global ? "hold-global" : "hold-checkpoint-4", candidate, before);
    if (!candidate && candidate.termination == SearchTermination::InvalidProof)
      return candidate;
    if (candidate &&
        tcount::cost(width, candidate.gates).t <=
            tcount::cost(width, best.gates).t &&
        circuitScore(width, candidate.gates, options) <
            circuitScore(width, best.gates, options))
      best = std::move(candidate);
  }
  return best;
}
} // namespace cudaq::pauli::graph::detail
