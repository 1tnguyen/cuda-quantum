/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SynthesisCost.h"
#include "SynthesisLinearWindows.h"
#include "cudaq/Optimizer/Pauli/GateCount.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>
#include <memory>

namespace cudaq::pauli::graph::detail {

struct NativeRefinementLimits {
  unsigned maxCandidates = 16;
  unsigned maxGates = 32768;
  unsigned maxTemporaryGates = 262144;
  unsigned maxWidth = 4096;
  unsigned gateCountRounds = 3;
  bool enableTCount = true;
  bool enableLinearWindows = true;
  unsigned maxLinearSources = 4;
  uint64_t maxLinearWork = 20000000;
  uint64_t maxWork = 100000000;
  uint64_t maxBackendWork = 10000000;
};
enum class NativeRefinementStatus {
  Complete,
  Unchanged,
  Unsupported,
  Exhausted,
  InvalidProof
};
struct NativeRefinementResult;
NativeRefinementResult
refineSynthesizedNative(unsigned, const std::vector<Gate> &, const Options &,
                        const NativeRefinementLimits &, SearchBudget &);

namespace native_refinement_detail {
struct Source {
  unsigned width;
  std::vector<Gate> gates;
};
inline bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}
inline uint64_t add(uint64_t a, uint64_t b) {
  return llvm::SaturatingAdd<uint64_t>(a, b);
}
inline uint64_t multiply(uint64_t a, uint64_t b) {
  return llvm::SaturatingMultiply<uint64_t>(a, b);
}
inline std::optional<SearchBudget> slice(SearchBudget &parent, uint64_t cap) {
  auto remainingTime = parent.timeRemainingMs();
  if (parent.remaining() < 2 || !cap || (remainingTime && *remainingTime < 2))
    return std::nullopt;
  return SearchBudget(parent, std::min(cap, parent.remaining() / 2),
                      remainingTime ? *remainingTime / 2 : 0);
}
} // namespace native_refinement_detail

/// Full-unitary refinement relative to one owned, already synthesized word.
/// The caller retains the incoming measured-synthesis certificate and composes
/// it with this checked-backend contract. This adapter neither reads nor
/// accepts a separate original/preserve-network seed. GC/T witness checks are
/// internal; their APIs do not export invalid-proof diagnostics. A rejected
/// linear-window certificate propagates the explicit InvalidProof status.
class CertifiedNativeRefinement {
public:
  enum class Proof { CheckedNativeUnitaryRewritesAndLowering };
  Proof proof() const { return Proof::CheckedNativeUnitaryRewritesAndLowering; }
  unsigned width() const { return origin->width; }
  const std::vector<Gate> &gates() const { return word; }
  const CircuitCost &cost() const { return evaluated; }
  const std::vector<Gate> &sourceGates() const { return origin->gates; }
  bool matchesSource(unsigned width, const std::vector<Gate> &gates) const {
    return origin->width == width &&
           native_refinement_detail::sameWord(origin->gates, gates);
  }

private:
  friend NativeRefinementResult
  refineSynthesizedNative(unsigned, const std::vector<Gate> &, const Options &,
                          const NativeRefinementLimits &, SearchBudget &);
  CertifiedNativeRefinement(
      std::shared_ptr<const native_refinement_detail::Source> source,
      std::vector<Gate> gates, CircuitCost cost)
      : origin(std::move(source)), word(std::move(gates)), evaluated(cost) {}
  std::shared_ptr<const native_refinement_detail::Source> origin;
  std::vector<Gate> word;
  CircuitCost evaluated;
};

struct NativeRefinementResult {
  std::vector<CertifiedNativeRefinement> alternatives;
  NativeRefinementStatus status = NativeRefinementStatus::Unchanged;
  std::string error;
  uint64_t work = 0, materializedGateBound = 0, decodedGateBound = 0;
  unsigned gateCountCalls = 0, tCountCalls = 0, proposals = 0;
  unsigned sizeRejections = 0, localExhaustions = 0;
  unsigned linearCalls = 0, linearExhaustions = 0;
  uint64_t linearWork = 0, linearWindowsChecked = 0, linearRewrites = 0;
  uint64_t linearMapStates = 0, linearCommutationChecks = 0;
  bool tCountSkippedBySize = false;
  bool decodeSkippedBySize = false;
  tcount::Statistics backend;
};

/// Invoke only in the full-optimization track, on a complete circuit produced
/// and checked by synthesis. A source-to-result full-unitary proof composes
/// with arbitrary-input, ordered terminal measurement semantics. This routine
/// never changes readout associations and never substitutes an unchecked word.
/// Final native ranking and all-five regression guards belong to the caller.
inline NativeRefinementResult
refineSynthesizedNative(unsigned width, const std::vector<Gate> &synthesized,
                        const Options &options,
                        const NativeRefinementLimits &limits,
                        SearchBudget &shared) {
  namespace nr = native_refinement_detail;
  NativeRefinementResult result;
  uint64_t initialWork = shared.work();
  auto unsupported = [&](const char *reason) {
    result.status = NativeRefinementStatus::Unsupported;
    result.error = reason;
    result.work = shared.work() - initialWork;
    return result;
  };
  if (!width || width > limits.maxWidth || !limits.maxCandidates ||
      limits.maxCandidates > 16 || !limits.maxGates ||
      !limits.maxTemporaryGates || synthesized.size() > limits.maxGates ||
      synthesized.size() > limits.maxTemporaryGates ||
      !limits.gateCountRounds || limits.gateCountRounds > 3 ||
      !limits.maxBackendWork || (!options.nativeCX && !options.nativeCZ))
    return unsupported("unsupported native-refinement dimensions or limits");
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost})
    if (!std::isfinite(cost) || cost < 0)
      return unsupported("invalid native-refinement cost profile");
  auto allowance = nr::slice(shared, limits.maxWork);
  if (!allowance) {
    result.status = NativeRefinementStatus::Exhausted;
    return result;
  }
  auto &budget = *allowance;
  auto finish = [&] {
    result.work = budget.work();
    if (budget.stopped() || result.localExhaustions ||
        result.linearExhaustions) {
      result.status = NativeRefinementStatus::Exhausted;
      result.error =
          "native refinement exhausted; completed candidates retained";
    } else if (!result.alternatives.empty())
      result.status = NativeRefinementStatus::Complete;
    return result;
  };
  // Validate/prepay the source before any unbudgeted backend cost or copy.
  if (!budget.take(nr::add(
          nr::multiply(width, width),
          nr::multiply(synthesized.size() + 1, uint64_t(width) / 64 + 1))))
    return finish();
  if (!tcount::detail::valid(width, synthesized))
    return unsupported("invalid synthesized native word");
  for (const auto &gate : synthesized) {
    if ((gate.kind == Kind::Rotate && gate.pauli.weight() != 1) ||
        (gate.kind == Kind::CX && !options.nativeCX) ||
        (gate.kind == Kind::CZ && !options.nativeCZ) ||
        (gate.kind == Kind::Swap && !options.nativeSwap))
      return unsupported(
          "source is not a complete circuit in its native profile");
  }
  auto origin =
      std::make_shared<const nr::Source>(nr::Source{width, synthesized});

  // Backend output consists of named Cliffords/scalars and single-wire
  // rotations. Bound exact entangler lowering before allocating the result.
  auto lower =
      [&](const std::vector<Gate> &word) -> std::optional<std::vector<Gate>> {
    if (!budget.take(nr::multiply(word.size() + 1, uint64_t(width) / 64 + 1)))
      return std::nullopt;
    uint64_t bound = 0;
    for (const auto &g : word) {
      if (g.kind == Kind::Rotate && g.pauli.weight() != 1) {
        ++result.sizeRejections;
        return std::nullopt;
      }
      uint64_t size = 1;
      if ((g.kind == Kind::CX && !options.nativeCX) ||
          (g.kind == Kind::CZ && !options.nativeCZ))
        size = 3;
      if (g.kind == Kind::Swap && !options.nativeSwap)
        size = options.nativeCX ? 3 : 9;
      bound = nr::add(bound, size);
    }
    if (bound > limits.maxGates || bound > limits.maxTemporaryGates) {
      ++result.sizeRejections;
      return std::nullopt;
    }
    if (!budget.take(nr::multiply(bound + 1, 8)))
      return std::nullopt;
    return nativeWord(word, options);
  };
  auto publish = [&](const std::vector<Gate> &word) {
    if (result.alternatives.size() == limits.maxCandidates)
      return;
    auto native = lower(word);
    if (!native)
      return;
    if (!budget.take(nr::multiply(native->size() + synthesized.size() + 1,
                                  result.alternatives.size() + 1)))
      return;
    if (nr::sameWord(*native, synthesized))
      return;
    for (const auto &previous : result.alternatives)
      if (nr::sameWord(previous.gates(), *native))
        return;
    auto cost = evaluateCircuitCost(width, *native, options);
    result.alternatives.push_back(
        CertifiedNativeRefinement(origin, std::move(*native), cost));
  };
  auto cleanup =
      [&](const std::vector<Gate> &word) -> std::optional<std::vector<Gate>> {
    // Internal cleanup methods may briefly add one Pauli per wire or expand
    // one-wire rotations, but never generate a multi-wire rotation. Prepay
    // their mechanical costs/copies, which do not charge the backend budget.
    uint64_t temporary = nr::add(nr::multiply(word.size() + 1, 8), width);
    if (temporary > limits.maxTemporaryGates) {
      ++result.sizeRejections;
      return std::nullopt;
    }
    if (!budget.take(nr::multiply(temporary, 32 * limits.gateCountRounds)))
      return std::nullopt;
    auto local = nr::slice(budget, limits.maxBackendWork);
    if (!local) {
      ++result.localExhaustions;
      return std::nullopt;
    }
    gatecount::Options settings;
    settings.rounds = limits.gateCountRounds;
    ++result.gateCountCalls;
    auto refined =
        gatecount::optimize(width, word, settings, *local, limits.maxGates);
    if (local->stopped())
      ++result.localExhaustions;
    return refined;
  };

  // Run only AFTER every existing GC/T proposal has been generated/cleaned.
  // Do not evict earlier candidates or let a new search spend their allowance.
  // The source and a bounded prefix of the completed old portfolio are the
  // only parents; newly published linear words never recursively seed this
  // stage. All proofs compose back to the same owned synthesized source.
  auto finishWithLinear = [&] {
    const size_t previousCount = result.alternatives.size();
    const size_t count = std::min<size_t>(std::min(limits.maxLinearSources, 4u),
                                          previousCount + 1);
    if (limits.enableLinearWindows && limits.maxLinearWork)
      for (size_t index = 0; index < count; ++index) {
        if (budget.stopped() ||
            result.alternatives.size() == limits.maxCandidates)
          break;
        const auto &parent =
            index ? result.alternatives[index - 1].gates() : synthesized;
        // Copy before publication can reallocate the alternatives vector.
        if (!budget.take(
                nr::multiply(parent.size() + 1, uint64_t(width) / 64 + 1)))
          break;
        auto source = parent;
        LinearWindowLimits settings;
        settings.maxGates = std::min(limits.maxGates, limits.maxTemporaryGates);
        settings.maxWork = limits.maxLinearWork;
        ++result.linearCalls;
        auto linear =
            optimizeLinearWindows(width, source, options, settings, budget);
        result.linearWork += linear.work;
        result.linearWindowsChecked += linear.windowsChecked;
        result.linearRewrites += linear.rewrites;
        result.linearMapStates += linear.mapStates;
        result.linearCommutationChecks += linear.commutationChecks;
        if (linear.status == LinearWindowStatus::Exhausted)
          ++result.linearExhaustions;
        if (linear.status == LinearWindowStatus::InvalidProof) {
          result.alternatives.clear();
          result.status = NativeRefinementStatus::InvalidProof;
          result.error = linear.error;
          result.work = budget.work();
          return result;
        }
        for (const auto &candidate : linear.alternatives) {
          if (!candidate.matchesSource(width, source)) {
            result.alternatives.clear();
            result.status = NativeRefinementStatus::InvalidProof;
            result.error = "linear refinement belongs to a different source";
            result.work = budget.work();
            return result;
          }
          publish(candidate.gates());
        }
      }
    return finish();
  };

  auto cleaned = cleanup(synthesized);
  if (cleaned)
    publish(*cleaned);
  if (!limits.enableTCount || budget.stopped() ||
      result.alternatives.size() == limits.maxCandidates)
    return finishWithLinear();
  const auto &parent = cleaned ? *cleaned : synthesized;
  uint64_t rotations = 0;
  QubitSet active;
  for (const auto &g : parent) {
    if (g.kind == Kind::Phase)
      continue;
    if (g.kind == Kind::Rotate) {
      active |= g.pauli.x | g.pauli.z;
      if (auto k = tcount::detail::quarters(g.angle))
        rotations += *k & 1;
    } else {
      active.set(g.a);
      if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
        active.set(g.b);
    }
  }
  // Every replaced rank<=5 block contains at least two input T rotations.
  // It emits <=31 axes on the existing active wires and one scalar. Retained
  // original Cliffords/scalars are bounded by the whole parent word. Reuse
  // places the same axes in this GENERATED network and obeys the same bound.
  uint64_t decoded = nr::add(
      nr::multiply(parent.size(), 7),
      nr::multiply(
          rotations / 2,
          nr::add(nr::multiply(31, 6 * uint64_t(active.count()) + 1), 1)));
  result.decodedGateBound = decoded;
  result.decodeSkippedBySize = decoded > limits.maxTemporaryGates;
  // Merge only changes/removes existing one-wire rotations and cancels
  // primitive Cliffords. Its stored word cannot grow. Mechanical cost of a
  // one-wire rotation is <=7 gates (Z/S/T plus both Y basis changes), and
  // angles remain exact modulo 16, including the original scalar phase.
  uint64_t storedBound = result.decodeSkippedBySize ? parent.size() : decoded;
  uint64_t mechanicalBound =
      result.decodeSkippedBySize ? nr::multiply(parent.size(), 7) : decoded;
  result.materializedGateBound = mechanicalBound;
  if (mechanicalBound > limits.maxTemporaryGates) {
    result.tCountSkippedBySize = true;
    return finishWithLinear();
  }
  // Six decode/merge strategies, or two merge-only strategies, including
  // comparisons/materialization/copies. One round keeps this source bound.
  if (!budget.take(nr::multiply(mechanicalBound + parent.size() + 1,
                                result.decodeSkippedBySize ? 24 : 64)))
    return finishWithLinear();
  auto local = nr::slice(budget, limits.maxBackendWork);
  if (!local) {
    ++result.localExhaustions;
    return finishWithLinear();
  }
  tcount::Options settings;
  settings.rounds = settings.beamWidth = 1;
  settings.decode = !result.decodeSkippedBySize;
  settings.reuseCliffords = true;
  settings.maxBlockRank = 5;
  settings.maxBlockGates = 256;
  settings.maxCandidates = result.decodeSkippedBySize ? 3 : 7;
  settings.maxWork = limits.maxBackendWork;
  settings.maxTwoQubitGrowth = limits.maxTemporaryGates;
  std::vector<std::vector<Gate>> proposals;
  auto collect = [&](const std::vector<Gate> &word, const tcount::Cost &,
                     uint64_t) {
    ++result.proposals;
    if (word.size() > storedBound || word.size() > limits.maxGates) {
      ++result.sizeRejections;
      return;
    }
    if (proposals.size() >= 6 || nr::sameWord(parent, word))
      return;
    for (const auto &previous : proposals)
      if (nr::sameWord(previous, word))
        return;
    proposals.push_back(word);
  };
  ++result.tCountCalls;
  auto optimized = tcount::optimize(width, parent, settings, &*local, nullptr,
                                    0, 0, collect);
  result.backend = optimized.stats;
  if (local->stopped() || optimized.stats.budgetExhaustions)
    ++result.localExhaustions;
  if (!optimized)
    return unsupported("checked T backend rejected native source/options");
  collect(optimized.gates, optimized.stats.after, 0);
  for (const auto &proposal : proposals) {
    if (budget.stopped() || result.alternatives.size() == limits.maxCandidates)
      break;
    publish(proposal);
    if (auto refined = cleanup(proposal))
      publish(*refined);
  }
  return finishWithLinear();
}
} // namespace cudaq::pauli::graph::detail
