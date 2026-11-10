/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/

#pragma once

#include "SynthesisHoldOneQubit.h"
#include "SynthesisNativeRefinement.h"
#include "cudaq/Optimizer/Pauli/Optimizer.h"
#include <array>
#include <functional>
#include <unordered_map>

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

/// Refine completed reconstructions before comparing their boundary choices.
/// This continuation consumes generated words only. Its caller owns a separate
/// optional slice, so extra composition cannot starve the established routes.
inline bool continueHoldNative(unsigned width, Result &candidate,
                               const Options &options, SearchBudget &budget) {
  if (!options.algebraicCandidates || candidate.gates.size() > 262144)
    return true;
  for (unsigned round = 0; round < 4 && !budget.stopped(); ++round) {
    OptimizerOptions settings;
    settings.enableGraphSynthesis = false;
    settings.enableGateCount = true;
    settings.maxGates = std::min(options.maxGates, 262144u);
    settings.tcount.rounds = 2;
    settings.tcount.maxCandidates = 32;
    settings.tcount.maxWork = 200000000;
    // The enclosing reconstruction route already has a deadline. A second,
    // fixed three-second slice can discard every large cleanup before its
    // proof completes, despite ample time remaining in that route.
    SearchBudget slice(budget,
                       std::min<uint64_t>(budget.remaining(), 2000000000));
    auto refined = optimize(width, candidate.gates, settings, &slice);
    if (!refined) {
      candidate = holdFailure(refined.error, SearchTermination::InvalidProof);
      return false;
    }
    auto native = nativeWord(refined.gates, options);
    if (native.size() > options.maxGates ||
        tcount::cost(width, native).t >
            tcount::cost(width, candidate.gates).t ||
        circuitScore(width, native, options) >=
            circuitScore(width, candidate.gates, options))
      break;
    candidate.gates = std::move(native);
  }
  return true;
}

/// Complete a large generated word through the native adapter. Its temporary
/// expansion allowance is separate from the retained-word cap: otherwise the
/// adapter can reject cleanup before it starts. This helper belongs only to a
/// deferred hold alternative, leaving earlier search trajectories untouched.
inline bool completeHoldLargeNative(unsigned width, Result &candidate,
                                    const Options &options,
                                    SearchBudget &budget) {
  if (!options.algebraicCandidates || candidate.gates.size() > 262144 ||
      width > NativeRefinementLimits{}.maxWidth ||
      (!options.nativeCX && !options.nativeCZ))
    return true;
  for (unsigned round = 0; round < 4 && !budget.stopped(); ++round) {
    if (!budget.take((candidate.gates.size() + 1) * (uint64_t(width) / 64 + 1)))
      break;
    // The adapter reserves successive fractions for other proposals. On a
    // large word that can exhaust cleanup before it publishes any reduction.
    // Complete the checked GC backend under this stage's own bounded allowance
    // first, then let the adapter add its T-merge/linear alternatives. The GC
    // contract returns the source on exhaustion, never an unchecked prefix.
    gatecount::Options cleanupOptions;
    SearchBudget cleanupBudget(
        budget, std::min<uint64_t>(budget.remaining(), 2000000000));
    auto cleaned =
        gatecount::optimize(width, candidate.gates, cleanupOptions,
                            cleanupBudget, std::min(options.maxGates, 262144u));
    if (!options.nativeCX || !options.nativeCZ || !options.nativeSwap) {
      uint64_t loweredSize = 0;
      for (const auto &gate : cleaned)
        loweredSize += gate.kind == Kind::Swap && !options.nativeSwap
                           ? (options.nativeCX ? 3 : 9)
                       : (gate.kind == Kind::CX && !options.nativeCX) ||
                               (gate.kind == Kind::CZ && !options.nativeCZ)
                           ? 3
                           : 1;
      if (loweredSize <= std::min(options.maxGates, 262144u) &&
          budget.take(loweredSize + 1))
        cleaned = nativeWord(cleaned, options);
      else
        cleaned = candidate.gates;
    }
    if (!budget.take(6 * (candidate.gates.size() + cleaned.size() + 1) *
                     (uint64_t(width) / 64 + 1)))
      break;
    if (tcount::cost(width, cleaned).t <=
            tcount::cost(width, candidate.gates).t &&
        circuitScore(width, cleaned, options) <
            circuitScore(width, candidate.gates, options))
      candidate.gates = std::move(cleaned);
    const auto source = candidate.gates;
    auto limits = boundedNativeRefinementLimits(options, source.size());
    limits.maxGates = std::min(options.maxGates, 262144u);
    // Source <= 262,144 and the adapter's width <= 4,096 bound this scratch
    // allowance to 2,101,256 operations. Retained results keep the smaller cap.
    limits.maxTemporaryGates = std::max<uint64_t>(
        limits.maxTemporaryGates, 8 * (source.size() + 1) + width);
    limits.maxWork = 2000000000;
    limits.maxBackendWork = 200000000;
    auto refined =
        refineSynthesizedNative(width, source, options, limits, budget);
    if (refined.status == NativeRefinementStatus::InvalidProof) {
      candidate = holdFailure(refined.error, SearchTermination::InvalidProof);
      return false;
    }
    for (const auto &alternative : refined.alternatives) {
      if (!budget.take(6 *
                       (source.size() + alternative.gates().size() +
                        candidate.gates.size() + 1) *
                       (uint64_t(width) / 64 + 1)))
        break;
      if (!alternative.matchesSource(width, source)) {
        candidate = holdFailure("hold large completion source binding failed",
                                SearchTermination::InvalidProof);
        return false;
      }
      if (alternative.gates().size() <= limits.maxGates &&
          tcount::cost(width, alternative.gates()).t <=
              tcount::cost(width, candidate.gates).t &&
          circuitScore(width, alternative.gates(), options) <
              circuitScore(width, candidate.gates, options))
        candidate.gates = alternative.gates();
    }
    if (native_refinement_detail::sameWord(source, candidate.gates))
      break;
  }
  return true;
}

/// Bounded exact-source memoization. Each instance belongs to one fixed option
/// profile; discrepancySteps is the only setting allowed to vary between calls.
struct HoldReconstructionCache {
  struct CachedBlock {
    unsigned width;
    unsigned discrepancy;
    std::vector<Gate> source;
    Result result;
  };
  std::unordered_map<size_t, CachedBlock> cache;
  size_t cachedGates = 0;
  Result reconstruct(unsigned localWidth, const std::vector<Gate> &word,
                     const Options &settings, SearchBudget *work,
                     const HoldSynthesis &synthesize) {
    if (!work->take((word.size() + 1) * (uint64_t(localWidth) / 64 + 1)))
      return holdFailure("hold reconstruction cache budget exhausted",
                         SearchTermination::Exhausted);
    auto hash =
        llvm::hash_combine(localWidth, settings.discrepancySteps, word.size());
    for (const auto &gate : word)
      hash = llvm::hash_combine(
          hash, unsigned(gate.kind), gate.a, gate.b, gate.pauli.x.hash(),
          gate.pauli.z.hash(), gate.pauli.sign, gate.angle.numerator,
          gate.angle.denominator, gate.angle.symbol, gate.angle.negative);
    size_t key = size_t(hash);
    auto found = cache.find(key);
    if (found != cache.end() && found->second.width == localWidth &&
        found->second.discrepancy == settings.discrepancySteps &&
        native_refinement_detail::sameWord(found->second.source, word)) {
      if (!work->take(found->second.result.gates.size() + 1))
        return holdFailure("hold reconstruction cached copy budget exhausted",
                           SearchTermination::Exhausted);
      return found->second.result;
    }
    auto rebuilt = synthesize(localWidth, word, settings, work);
    // Reuse a certificate only for the identical compact source and width.
    // Hash collisions decline caching; they never authorize proof reuse.
    const size_t storage = word.size() + rebuilt.gates.size();
    if (rebuilt && found == cache.end() && cache.size() < 512 &&
        storage <= 65536 - cachedGates && work->remaining() > storage &&
        work->take(storage + 1)) {
      cache.emplace(key, CachedBlock{localWidth, settings.discrepancySteps,
                                     word, rebuilt});
      cachedGates += storage;
    }
    return rebuilt;
  }
};

inline Result synthesizeHoldCachedBlocks(unsigned width,
                                         const std::vector<Gate> &input,
                                         const Options &options,
                                         SearchBudget &budget, unsigned cap,
                                         const HoldSynthesis &synthesize) {
  HoldReconstructionCache cache;
  auto reconstruct = [&](unsigned localWidth, const std::vector<Gate> &word,
                         const Options &settings, SearchBudget *work) {
    return cache.reconstruct(localWidth, word, settings, work, synthesize);
  };
  return synthesizeHoldBlocks(width, input, options, budget, cap, reconstruct);
}

/// Rebuild finer local graphs in both traversal directions. Scalar witnesses
/// commute with every block and do not need to consume its operation quota.
/// In particular, a one-wire cap still rebuilds an individual wider operation
/// on its full support; it never copies an incoming entangler as an incumbent.
inline Result synthesizeHoldFineBlocks(
    unsigned width, const std::vector<Gate> &input, const Options &options,
    SearchBudget &budget, unsigned cap, bool reverse,
    const HoldSynthesis &synthesize, bool continueNative = true) {
  if (!budget.take((input.size() + 1) * (uint64_t(width) / 64 + 1)))
    return holdFailure("hold fine checkpoint input budget exhausted",
                       SearchTermination::Exhausted);
  std::vector<Gate> body, phases;
  for (const auto &gate : input)
    (gate.kind == Kind::Phase ? phases : body).push_back(gate);
  if (reverse)
    body = tcount::detail::adjoint(body);
  auto result =
      synthesizeHoldCachedBlocks(width, body, options, budget, cap, synthesize);
  if (!result)
    return result;
  if (phases.size() > options.maxGates - result.gates.size())
    return holdFailure("hold fine checkpoint scalar gate cap",
                       SearchTermination::Unsupported);
  if (reverse)
    result.gates = tcount::detail::adjoint(result.gates);
  result.gates.insert(result.gates.end(), phases.begin(), phases.end());
  if (continueNative)
    continueHoldNative(width, result, options, budget);
  return result;
}

inline Gate expandHoldGate(Gate gate, const std::vector<unsigned> &wires) {
  if (gate.kind == Kind::Rotate) {
    Pauli p;
    p.sign = gate.pauli.sign;
    for (unsigned q : gate.pauli.x.set_bits())
      p.x.set(wires[q]);
    for (unsigned q : gate.pauli.z.set_bits())
      p.z.set(wires[q]);
    gate.pauli = std::move(p);
  } else if (gate.kind != Kind::Phase) {
    gate.a = wires[gate.a];
    if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
        gate.kind == Kind::Swap)
      gate.b = wires[gate.b];
  }
  return gate;
}

/// Repair a short reconstruction that has not yet retained the incumbent's T
/// savings. Search high-T four-wire regions with a T-first local objective and
/// all cheapest ready nodes. Trim up to three trailing operations: a maximal
/// closed window can hide a cheaper completion at an interior boundary.
/// Local growth is limited to three gates per T saved; the caller still applies
/// its original whole-circuit gate objective and hard incumbent T ceiling.
inline bool repairHoldT(unsigned width, Result &candidate, uint64_t ceiling,
                        const Options &options, SearchBudget &budget,
                        const HoldSynthesis &synthesize) {
  if (!options.algebraicCandidates || candidate.gates.size() > 262144 ||
      candidate.gates.size() > options.maxGates ||
      !budget.take((2 * candidate.gates.size() + 1) *
                   (uint64_t(width) / 64 + 1)))
    return true;
  std::vector<Gate> body, phases;
  for (const auto &gate : candidate.gates)
    (gate.kind == Kind::Phase ? phases : body).push_back(gate);
  auto settings = options;
  settings.holdCheckpoints = false;
  settings.diagnostics = nullptr;
  settings.objective = SynthesisObjective::TCount;
  settings.candidateNodes = 0;
  settings.commutingCandidates = settings.jointPhaseCandidates = 0;
  settings.maxGates -= phases.size();
  HoldReconstructionCache cache;
  auto remainingT = tcount::cost(width, body).t;
  for (size_t at = 0;
       at < body.size() && remainingT > ceiling && !budget.stopped(); ++at) {
    if (!budget.take(33 * (uint64_t(width) / 64 + 1)))
      break;
    QubitSet active;
    size_t end = at;
    for (; end < body.size() && end - at < 32; ++end) {
      auto support = linear_window_detail::support(body[end]);
      if ((active | support).count() > 4)
        break;
      active |= support;
    }
    if (active.count() < 2 || end - at < 7)
      continue;
    std::vector<unsigned> wires;
    for (unsigned q : active.set_bits())
      wires.push_back(q);
    for (unsigned trim = 0;
         trim < 4 && end - at > trim + 6 && !budget.stopped(); ++trim) {
      const size_t stop = end - trim;
      std::vector<Gate> local;
      for (size_t i = at; i < stop; ++i)
        local.push_back(linear_window_detail::compact(body[i], wires));
      const auto before = tcount::cost(wires.size(), local);
      // Native merging already handles the simple low-T cancellations. Spend
      // this extra reconstruction budget on nontrivial phase-polynomial
      // regions.
      if (before.t < 7)
        continue;
      SearchBudget slice(
          budget, std::min<uint64_t>(budget.remaining(), 100000000), 200);
      auto repaired =
          cache.reconstruct(wires.size(), local, settings, &slice, synthesize);
      if (!repaired) {
        if (repaired.termination == SearchTermination::InvalidProof) {
          candidate = std::move(repaired);
          return false;
        }
        continue;
      }
      const auto after = tcount::cost(wires.size(), repaired.gates);
      if (after.t >= before.t ||
          after.gates > before.gates + 3 * (before.t - after.t) ||
          repaired.gates.size() >
              settings.maxGates - (body.size() - (stop - at)))
        continue;
      if (!budget.take((body.size() + repaired.gates.size() + 1) *
                       (uint64_t(width) / 64 + 1)))
        break;
      std::vector<Gate> word(body.begin(), body.begin() + at);
      for (auto gate : repaired.gates)
        word.push_back(expandHoldGate(gate, wires));
      word.insert(word.end(), body.begin() + stop, body.end());
      body = std::move(word);
      remainingT -= before.t - after.t;
      break;
    }
  }
  body.insert(body.end(), phases.begin(), phases.end());
  candidate.gates = std::move(body);
  return true;
}

/// Overlap closed reconstruction boundaries on a generated incumbent. Each
/// compact local proof extends by identity to all other physical wires.
inline bool refineHoldOverlapping(unsigned width, Result &best,
                                  const Options &options, SearchBudget &budget,
                                  const HoldSynthesis &synthesize,
                                  bool wideReverse = false) {
  if (!options.algebraicCandidates || best.gates.size() > 262144)
    return true;
  HoldReconstructionCache cache;
  for (unsigned span :
       wideReverse ? std::array{48u, 96u} : std::array{24u, 48u})
    for (unsigned cap = wideReverse ? 6 : 3; cap <= (wideReverse ? 6u : 4u);
         ++cap)
      for (size_t at = 0; at < best.gates.size() && !budget.stopped(); ++at) {
        if (best.gates[at].kind == Kind::Phase)
          continue;
        QubitSet active;
        size_t end = at;
        for (; end < best.gates.size() && end - at < span; ++end) {
          auto support = linear_window_detail::support(best.gates[end]);
          if ((active | support).count() > cap)
            break;
          active |= support;
        }
        if (end - at < 5 || active.count() < 2)
          continue;
        std::vector<unsigned> wires;
        for (unsigned q : active.set_bits())
          wires.push_back(q);
        std::vector<Gate> local;
        for (size_t i = at; i < end; ++i)
          local.push_back(linear_window_detail::compact(best.gates[i], wires));
        for (unsigned discrepancy : {options.discrepancySteps, 0u}) {
          if (budget.stopped())
            break;
          Options settings = options;
          settings.holdCheckpoints = false;
          settings.diagnostics = nullptr;
          settings.commutingCandidates = settings.jointPhaseCandidates = 0;
          settings.discrepancySteps = discrepancy;
          SearchBudget slice(
              budget, std::min<uint64_t>(budget.remaining(), 100000000), 200);
          auto source = wideReverse ? tcount::detail::adjoint(local) : local;
          auto candidate = cache.reconstruct(wires.size(), source, settings,
                                             &slice, synthesize);
          if (!candidate) {
            if (candidate.termination == SearchTermination::InvalidProof) {
              best = std::move(candidate);
              return false;
            }
            continue;
          }
          if (wideReverse)
            candidate.gates = tcount::detail::adjoint(candidate.gates);
          if (tcount::cost(wires.size(), candidate.gates).gates >=
                  tcount::cost(wires.size(), local).gates ||
              candidate.gates.size() >
                  options.maxGates - (best.gates.size() - (end - at)))
            continue;
          if (!budget.take((best.gates.size() + candidate.gates.size() + 1) *
                           (uint64_t(width) / 64 + 1)))
            break;
          std::vector<Gate> word(best.gates.begin(), best.gates.begin() + at);
          for (auto gate : candidate.gates)
            word.push_back(expandHoldGate(gate, wires));
          word.insert(word.end(), best.gates.begin() + end, best.gates.end());
          if (tcount::cost(width, word).t <=
                  tcount::cost(width, best.gates).t &&
              circuitScore(width, word, options) <
                  circuitScore(width, best.gates, options)) {
            best.gates = std::move(word);
            break;
          }
        }
      }
  return true;
}

/// Give adjacent checkpoints a wider reverse reconstruction without consuming
/// the earlier window search's budget. Scalar witnesses commute with the body;
/// keep them outside the window quota and restore them before final admission.
inline bool refineHoldWideReverse(unsigned width, Result &best,
                                  const Options &options, SearchBudget &budget,
                                  const HoldSynthesis &synthesize) {
  if (!options.algebraicCandidates || best.gates.size() > 262144 ||
      best.gates.size() > options.maxGates ||
      !budget.take((best.gates.size() + 1) * (uint64_t(width) / 64 + 1)))
    return true;
  Result candidate;
  std::vector<Gate> phases;
  for (const auto &gate : best.gates)
    (gate.kind == Kind::Phase ? phases : candidate.gates).push_back(gate);
  auto settings = options;
  settings.maxGates -= phases.size();
  if (!refineHoldOverlapping(width, candidate, settings, budget, synthesize,
                             true)) {
    best = std::move(candidate);
    return false;
  }
  candidate.gates.insert(candidate.gates.end(), phases.begin(), phases.end());
  if (tcount::cost(width, candidate.gates).t <=
          tcount::cost(width, best.gates).t &&
      circuitScore(width, candidate.gates, options) <
          circuitScore(width, best.gates, options))
    best.gates = std::move(candidate.gates);
  return true;
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
      synthesizeHoldCachedBlocks(width, input, options, budget, 3, synthesize);
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
    auto candidate = global
                         ? synthesize(width, input, options, &optional)
                         : synthesizeHoldCachedBlocks(width, input, options,
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
  // Keep the established portfolio's schedule intact. Every new route starts
  // from the lossless source and must reconstruct its complete unitary.
  // Preserve the incumbent's T savings even for the total-gate objective.
  // A shorter source-T candidate is not sufficient when hold already saved T.
  std::vector<Result> pendingTRepair;
  size_t pendingGates = 0;
  if (input.size() <= 262144)
    for (unsigned cap : {1u, 2u})
      for (bool reverse : {false, true}) {
        if (budget.stopped())
          break;
        SearchBudget optional(
            budget, std::min<uint64_t>(budget.remaining(), 20000000000ull),
            12000);
        before = budget.work();
        auto candidate = synthesizeHoldFineBlocks(
            width, input, options, optional, cap, reverse, synthesize);
        if (candidate && options.objective == SynthesisObjective::TotalGates &&
            tcount::cost(width, candidate.gates).t >
                tcount::cost(width, best.gates).t &&
            circuitScore(width, candidate.gates, options) <
                circuitScore(width, best.gates, options) &&
            pendingTRepair.size() < 2 &&
            candidate.gates.size() <= 262144 - pendingGates &&
            optional.remaining() > candidate.gates.size() &&
            optional.take(candidate.gates.size() + 1)) {
          pendingGates += candidate.gates.size();
          pendingTRepair.push_back(candidate);
        }
        record(cap == 1 ? (reverse ? "hold-local-1-reverse" : "hold-local-1")
                        : (reverse ? "hold-local-2-reverse" : "hold-local-2"),
               candidate, before);
        if (!candidate &&
            candidate.termination == SearchTermination::InvalidProof)
          return candidate;
        if (candidate &&
            tcount::cost(width, candidate.gates).t <=
                tcount::cost(width, best.gates).t &&
            circuitScore(width, candidate.gates, options) <
                circuitScore(width, best.gates, options))
          best = std::move(candidate);
      }
  if (!budget.stopped()) {
    SearchBudget optional(
        budget, std::min<uint64_t>(budget.remaining(), 30000000000ull), 40000);
    before = budget.work();
    if (!refineHoldOverlapping(width, best, options, optional, synthesize))
      return best;
    record("hold-overlapping", best, before);
  }
  if (!budget.stopped()) {
    SearchBudget optional(
        budget, std::min<uint64_t>(budget.remaining(), 20000000000ull), 20000);
    before = budget.work();
    if (!refineHoldWideReverse(width, best, options, optional, synthesize))
      return best;
    record("hold-wide-reverse", best, before);
  }
  // Finish the established incumbent's searches before admitting repaired
  // alternatives. Redirecting those searches to a new intermediate word can
  // otherwise lose improvements found along the previous trajectory.
  if (!pendingTRepair.empty() && !budget.stopped()) {
    SearchBudget optional(
        budget, std::min<uint64_t>(budget.remaining(), 20000000000ull), 10000);
    for (auto &candidate : pendingTRepair) {
      if (optional.stopped())
        break;
      if (tcount::cost(width, candidate.gates).gates >
          tcount::cost(width, best.gates).gates)
        continue;
      before = budget.work();
      if (!repairHoldT(width, candidate, tcount::cost(width, best.gates).t,
                       options, optional, synthesize))
        return candidate;
      record("hold-t-repair", candidate, before);
      if (tcount::cost(width, candidate.gates).t <=
              tcount::cost(width, best.gates).t &&
          circuitScore(width, candidate.gates, options) <
              circuitScore(width, best.gates, options))
        best = std::move(candidate);
    }
  }
  for (bool reverse : {false, true}) {
    if (!options.algebraicCandidates || budget.stopped() ||
        best.gates.size() > 262144)
      break;
    SearchBudget optional(
        budget, std::min<uint64_t>(budget.remaining(), 20000000000ull), 12000);
    before = budget.work();
    if (!optional.take((best.gates.size() + 1) * (uint64_t(width) / 64 + 1)))
      break;
    auto source = reverse ? tcount::detail::adjoint(best.gates) : best.gates;
    auto candidate = normalizeHoldOneQubit(width, source, options, optional);
    if (candidate && reverse)
      candidate.gates = tcount::detail::adjoint(candidate.gates);
    if (candidate)
      continueHoldNative(width, candidate, options, optional);
    record(reverse ? "hold-one-wire-reverse-completion"
                   : "hold-one-wire-completion",
           candidate, before);
    if (!candidate && candidate.termination == SearchTermination::InvalidProof)
      return candidate;
    if (candidate &&
        tcount::cost(width, candidate.gates).t <=
            tcount::cost(width, best.gates).t &&
        circuitScore(width, candidate.gates, options) <
            circuitScore(width, best.gates, options))
      best = std::move(candidate);
  }
  // A large reverse reconstruction can be longer before native completion,
  // and must not be ranked at that intermediate point. The fixed optional
  // allowance funds reconstruction and three completed native stages; the
  // enclosing program deadline remains authoritative.
  if (options.algebraicCandidates && input.size() > 32768 &&
      input.size() <= 262144 && !budget.stopped()) {
    SearchBudget optional(
        budget, std::min<uint64_t>(budget.remaining(), 30000000000ull), 90000);
    before = budget.work();
    auto candidate = synthesizeHoldFineBlocks(width, input, options, optional,
                                              1, true, synthesize, false);
    auto finish = [&](Result &word) {
      SearchBudget stage(
          optional, std::min<uint64_t>(optional.remaining(), 10000000000ull),
          30000);
      return completeHoldLargeNative(width, word, options, stage);
    };
    if (candidate && candidate.gates.size() <= 262144) {
      if (!finish(candidate))
        return candidate;
      for (bool reverse : {false, true}) {
        if (optional.stopped() || !optional.take((candidate.gates.size() + 1) *
                                                 (uint64_t(width) / 64 + 1)))
          break;
        auto source = reverse ? tcount::detail::adjoint(candidate.gates)
                              : candidate.gates;
        auto normalized =
            normalizeHoldOneQubit(width, source, options, optional);
        if (normalized && reverse)
          normalized.gates = tcount::detail::adjoint(normalized.gates);
        if (normalized && !finish(normalized))
          return normalized;
        if (!normalized &&
            normalized.termination == SearchTermination::InvalidProof)
          return normalized;
        if (normalized &&
            tcount::cost(width, normalized.gates).t <=
                tcount::cost(width, candidate.gates).t &&
            circuitScore(width, normalized.gates, options) <
                circuitScore(width, candidate.gates, options))
          candidate = std::move(normalized);
      }
    }
    record("hold-large-completion", candidate, before);
    if (!candidate && candidate.termination == SearchTermination::InvalidProof)
      return candidate;
    if (candidate && candidate.gates.size() <= 262144 &&
        tcount::cost(width, candidate.gates).t <=
            tcount::cost(width, best.gates).t &&
        circuitScore(width, candidate.gates, options) <
            circuitScore(width, best.gates, options))
      best = std::move(candidate);
  }
  return best;
}
} // namespace cudaq::pauli::graph::detail
