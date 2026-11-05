/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SynthesisMacros.h"
#include "SynthesisRegions.h"
#include <memory>

namespace cudaq::pauli::graph::detail {

struct GeneralLimits {
  /// Pair windows (0,1),(2,3),... and then (1,2),(3,4),... with the
  /// unmatched end windows solved separately. At most two complete plans.
  unsigned maxPlans = 2;
  unsigned maxActiveWires = 32;
  unsigned maxRegionRotations = 128;
  unsigned maxDiscrepancySteps = 2;
  uint64_t maxWork = 1000000000;
  uint64_t maxLocalWork = 50000000;
  unsigned maxCacheEntries = 128;
  uint64_t maxCacheBytes = 16 * 1024 * 1024;
  /// Optional complementary non-monotone provider. Keep a separate invocation
  /// with false to retain the legacy regional plans at the whole-program sink.
  bool useMacros = false;
  MacroLimits macros;
};

struct GeneralCandidates {
  std::vector<RegionalPlan> candidates;
  RegionStatus status = RegionStatus::Complete;
  std::string error;
  uint64_t work = 0;
  unsigned windows = 0, regionsSolved = 0, cacheHits = 0;
  unsigned cacheEntries = 0, completedProposals = 0;
  uint64_t cacheBytes = 0;
  unsigned macroRegions = 0, macroImprovements = 0;
  uint64_t macroPairs = 0;
};

namespace general_detail {
inline bool sameGate(const Gate &a, const Gate &b) {
  return a.kind == b.kind && a.a == b.a && a.b == b.b && a.pauli == b.pauli &&
         a.angle == b.angle;
}

inline bool validNative(unsigned width, const Gate &gate,
                        const Options &options) {
  bool valid = false;
  if (gate.kind == Kind::Rotate)
    valid = !gate.pauli.x && gate.pauli.weight() == 1 &&
            gate.pauli.z.fits(width) &&
            (gate.angle.symbol || gate.angle.denominator > 0);
  else if (gate.kind == Kind::Phase)
    valid = gate.angle.symbol || gate.angle.denominator > 0;
  else
    valid = region_detail::validClifford(width, gate);
  return valid && (gate.kind != Kind::CX || options.nativeCX) &&
         (gate.kind != Kind::CZ || options.nativeCZ) &&
         (gate.kind != Kind::Swap || options.nativeSwap);
}

/// Compact gates have <=32 coordinates, so both APInts stay inline. Arrays
/// have exact lengths: accounting includes Entry and every owned Gate, with
/// no vector capacity or high-wire APInt allocation hidden in the cache.
struct Entry {
  unsigned width;
  size_t inputSize, outputSize;
  std::unique_ptr<Gate[]> input, output;
};

inline uint64_t proofWork(unsigned width, uint64_t size) {
  return llvm::SaturatingAdd<uint64_t>(
      llvm::SaturatingMultiply<uint64_t>(
          3, llvm::SaturatingMultiply<uint64_t>(size, size)),
      llvm::SaturatingMultiply<uint64_t>(size, uint64_t(width) * width + 1));
}

inline unsigned halfTime(const SearchBudget &budget) {
  auto time = budget.timeRemainingMs();
  return time ? *time / 2 : 0;
}
inline bool hasTime(const SearchBudget &budget) {
  auto time = budget.timeRemainingMs();
  return !time || *time >= 2;
}
} // namespace general_detail

/// Compact general-region adapter. Each bounded noncommuting word is solved in
/// full-unitary mode on its exact active coordinates. Its restored Clifford
/// frame makes the exit identity, hence observables and all external quantum
/// wires remain unchanged. Only complete plans escape; the caller still checks
/// the entire measured program. The default local solver retains legacy
/// monotone search. Opt-in macros add bounded non-monotone alternatives with
/// independently checked closed exits; they replace a local legacy body only on
/// strict common native-cost rank.
///
/// The cache is invocation-local: one immutable Options/native profile and
/// one macro mode apply to every entry. Keys include exact ordered gates,
/// signed Paulis, rational/opaque angles and compact width. Every use receives
/// a fresh independently charged full-unitary replay against the current local
/// word. No preserved input network or benchmark reference is accepted.
inline GeneralCandidates
generateGeneralCandidates(const MeasuredProblem &source, const Options &options,
                          const GeneralLimits &limits, SearchBudget &shared) {
  GeneralCandidates result;
  auto fail = [&](RegionStatus status, const char *message) {
    result.status = status;
    result.error = message;
  };
  if (!source.width || !options.maxGates || !options.maxDependencies ||
      source.rotations.size() > options.maxGates || !limits.maxPlans ||
      !limits.maxActiveWires || !limits.maxRegionRotations ||
      source.observables.empty() ||
      source.observables.size() != source.readoutWires.size() ||
      (!options.nativeCX && !options.nativeCZ) ||
      static_cast<unsigned>(options.objective) >
          static_cast<unsigned>(SynthesisObjective::TDepth)) {
    fail(RegionStatus::Unsupported,
         "invalid general-region dimensions or limits");
    return result;
  }
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost, options.actionWeight,
                      options.readoutWeight, options.parallelizationCredit,
                      options.frontierWeight, options.frameWeight})
    if (!std::isfinite(cost) || cost < 0) {
      fail(RegionStatus::Unsupported, "invalid general-region native costs");
      return result;
    }
  // The entire optional provider gets at most half the enclosing work/time.
  // Local exhaustion cannot stop shared or consume its completion reserve.
  if (!general_detail::hasTime(shared) || shared.remaining() < 2) {
    fail(RegionStatus::Exhausted, "no general-region exploration reserve");
    return result;
  }
  SearchBudget budget(shared, std::min(limits.maxWork, shared.remaining() / 2),
                      general_detail::halfTime(shared));
  uint64_t rowWork = uint64_t(source.width) / 64 + 1;
  auto charge = [&](uint64_t count, uint64_t rows = 1) {
    bool accepted =
        budget.take(llvm::SaturatingMultiply<uint64_t>(count, rows));
    result.work = budget.work();
    if (accepted)
      return true;
    fail(RegionStatus::Exhausted, "general-region work budget exhausted");
    return false;
  };
  if (!charge(source.rotations.size() + source.observables.size() + 1, rowWork))
    return result;
  for (const auto &gate : source.rotations)
    if ((gate.kind != Kind::Rotate && gate.kind != Kind::Phase) ||
        (gate.kind == Kind::Rotate &&
         (!gate.pauli.weight() ||
          !(gate.pauli.x | gate.pauli.z).fits(source.width))) ||
        (!gate.angle.symbol && gate.angle.denominator <= 0)) {
      fail(RegionStatus::Unsupported,
           "general regions require valid nonidentity rotations or phases");
      return result;
    }
  pbc::CommutingObservables measurements;
  std::unordered_map<unsigned, size_t> assigned;
  for (size_t i = 0; i < source.observables.size(); ++i) {
    const auto &word = source.observables[i];
    if (!charge(2 * uint64_t(measurements.size()) + 1, rowWork))
      return result;
    if (!word.weight() || !(word.x | word.z).fits(source.width) ||
        source.readoutWires[i] >= source.width || !measurements.insert(word)) {
      fail(RegionStatus::Unsupported, "invalid general-region observables");
      return result;
    }
    auto [at, inserted] = assigned.try_emplace(source.readoutWires[i], i);
    if (!inserted && word != source.observables[at->second]) {
      fail(RegionStatus::Unsupported, "conflicting general-region readouts");
      return result;
    }
  }
  if (measurements.size() != assigned.size()) {
    fail(RegionStatus::Unsupported,
         "general-region readouts require remapping");
    return result;
  }

  std::vector<std::pair<size_t, size_t>> windows;
  pbc::CommutingObservables commuting;
  size_t begin = 0;
  for (size_t i = 0; i < source.rotations.size(); ++i) {
    if (!charge(2 * uint64_t(commuting.size()) + 1, rowWork))
      return result;
    if (source.rotations[i].kind == Kind::Phase)
      continue;
    if (!commuting.insert(source.rotations[i].pauli)) {
      windows.emplace_back(begin, i);
      begin = i;
      commuting = {};
      commuting.insert(source.rotations[i].pauli);
    }
  }
  if (begin < source.rotations.size())
    windows.emplace_back(begin, source.rotations.size());
  result.windows = windows.size();

  // Fixed-size entry array makes the cache's persistent storage bound explicit.
  unsigned entryLimit =
      std::min<uint64_t>(std::min(limits.maxCacheEntries, 4096u),
                         limits.maxCacheBytes / sizeof(general_detail::Entry));
  if (!charge(entryLimit + 1))
    return result;
  std::unique_ptr<general_detail::Entry[]> cache;
  if (entryLimit)
    cache = std::make_unique<general_detail::Entry[]>(entryLimit);
  result.cacheBytes = uint64_t(entryLimit) * sizeof(general_detail::Entry);
  unsigned plans = windows.size() > 1 ? std::min(limits.maxPlans, 2u) : 1u;
  for (unsigned offset = 0; offset < plans; ++offset) {
    if (!charge(source.observables.size() + 1, rowWork))
      return result;
    RegionalPlan plan{{}, source.observables};
    bool unsupported = false;
    for (size_t w = 0; w < windows.size();) {
      size_t next = std::min(w + ((offset && !w) ? 1 : 2), windows.size());
      auto first = windows[w].first, last = windows[next - 1].second;
      w = next;
      if (last - first > limits.maxRegionRotations) {
        unsupported = true;
        break;
      }
      if (!charge(last - first + 1, rowWork))
        return result;
      QubitSet support;
      for (size_t i = first; i < last; ++i)
        if (source.rotations[i].kind == Kind::Rotate)
          support |= source.rotations[i].pauli.x | source.rotations[i].pauli.z;
      if (support.count() > std::min(limits.maxActiveWires, 32u)) {
        unsupported = true;
        break;
      }
      if (!charge(last - first + support.count() + 1, rowWork))
        return result;
      std::vector<unsigned> physical;
      for (unsigned q : support.set_bits())
        physical.push_back(q);
      // A scalar-only region uses a dummy coordinate, on which it acts as I.
      if (physical.empty())
        physical.push_back(0);
      unsigned width = physical.size();
      if (!charge(last - first + 1, width + 1))
        return result;
      std::vector<Gate> local;
      local.reserve(last - first);
      for (size_t i = first; i < last; ++i) {
        const auto &original = source.rotations[i];
        if (original.kind == Kind::Phase) {
          local.push_back(Gate::phase(original.angle));
          continue;
        }
        Pauli compact{{}, {}, original.pauli.sign};
        for (unsigned q = 0; q < width; ++q) {
          compact.x.set(q, original.pauli.x.test(physical[q]));
          compact.z.set(q, original.pauli.z.test(physical[q]));
        }
        local.push_back(Gate::rotation(compact, original.angle));
      }
      size_t available = options.maxGates - plan.prefix.size();
      std::vector<Gate> body;
      bool found = false;
      for (unsigned i = 0; i < result.cacheEntries; ++i) {
        const auto &entry = cache[i];
        if (!charge(local.size() + 1))
          return result;
        if (entry.width != width || entry.inputSize != local.size() ||
            entry.outputSize > available ||
            !std::equal(local.begin(), local.end(), entry.input.get(),
                        general_detail::sameGate))
          continue;
        if (!charge(entry.outputSize + 1))
          return result;
        body.assign(entry.output.get(), entry.output.get() + entry.outputSize);
        found = true;
        ++result.cacheHits;
        break;
      }
      if (!found) {
        if (!available || local.size() > available ||
            !general_detail::hasTime(budget) || budget.remaining() < 2) {
          fail(RegionStatus::Exhausted,
               "no general-region local search reserve");
          return result;
        }
        Options localOptions = options;
        localOptions.release = false;
        localOptions.pruneUnobserved = false;
        localOptions.acceptOnlyImprovement = false;
        localOptions.discrepancySteps =
            std::min(options.discrepancySteps, limits.maxDiscrepancySteps);
        localOptions.maxGates = available;
        SearchBudget search(
            budget, std::min(limits.maxLocalWork, budget.remaining() / 2),
            general_detail::halfTime(budget));
        auto synthesized =
            graph::synthesize(width, local, localOptions, &search);
        result.work = budget.work();
        if (!synthesized) {
          fail(synthesized.stats.budgetExhausted ? RegionStatus::Exhausted
                                                 : RegionStatus::InvalidProof,
               synthesized.error.c_str());
          return result;
        }
        body = std::move(synthesized.gates);
        ++result.regionsSolved;
      }
      // Search and cache lookup are not proof handles. Charge and replay this
      // exact compact input/body before mapping or publishing any cached
      // result.
      uint64_t size = llvm::SaturatingAdd<uint64_t>(local.size(), body.size());
      if (!charge(general_detail::proofWork(width, size)))
        return result;
      for (const auto &gate : body) {
        if (!general_detail::validNative(width, gate, options)) {
          fail(RegionStatus::InvalidProof,
               "general-region solver returned an invalid native body");
          return result;
        }
      }
      if (!graph::equivalent(width, local, body)) {
        fail(RegionStatus::InvalidProof,
             "general-region full-unitary replay failed");
        return result;
      }
      if (body.size() > available) {
        fail(RegionStatus::Exhausted,
             "general-region native gate limit exceeded");
        return result;
      }
      if (!found && limits.useMacros &&
          width <= std::min(limits.macros.maxActiveWires, 8u) &&
          local.size() <= std::min(limits.macros.maxInputGates, 256u) &&
          general_detail::hasTime(budget) && budget.remaining() >= 4) {
        // Legacy is already independently verified. Every optional operation,
        // including closing/scoring/replay, runs in a child so exhaustion can
        // retain that body without stopping the region composition budget.
        auto explore = [&]() -> std::string {
          SearchBudget improvement(budget,
                                   std::min(llvm::SaturatingMultiply<uint64_t>(
                                                limits.macros.maxWork, 2),
                                            budget.remaining() / 2),
                                   general_detail::halfTime(budget));
          if (improvement.remaining() < 4 ||
              !general_detail::hasTime(improvement))
            return {};
          SearchBudget search(
              improvement,
              std::min(limits.macros.maxWork, improvement.remaining() / 2),
              general_detail::halfTime(improvement));
          Options macroOptions = options;
          macroOptions.release = false;
          macroOptions.pruneUnobserved = false;
          macroOptions.acceptOnlyImprovement = false;
          macroOptions.maxGates = available;
          auto alternatives = solveMacroRegion(width, local, macroOptions,
                                               limits.macros, search);
          ++result.macroRegions;
          result.macroPairs += alternatives.expandedPairs;
          for (const auto &proposal : alternatives.candidates) {
            if (proposal.body.size() > available ||
                proposal.exitCliffords.size() >
                    available - proposal.body.size())
              continue;
            if (improvement.remaining() < 2 ||
                !general_detail::hasTime(improvement))
              break;
            SearchBudget proof(improvement, improvement.remaining() / 2,
                               general_detail::halfTime(improvement));
            uint64_t closedSize =
                proposal.body.size() + proposal.exitCliffords.size();
            uint64_t scoreWork = llvm::SaturatingAdd<uint64_t>(
                width, llvm::SaturatingMultiply<uint64_t>(
                           16, closedSize + body.size() + 1));
            if (!proof.take(llvm::SaturatingAdd<uint64_t>(
                    scoreWork, general_detail::proofWork(
                                   width, llvm::SaturatingAdd<uint64_t>(
                                              local.size(), closedSize)))))
              break;
            for (const auto &gate : proposal.body)
              if (!general_detail::validNative(width, gate, options))
                return "macro region returned an invalid native body";
            for (const auto &gate : proposal.exitCliffords)
              if (!region_detail::validClifford(width, gate) ||
                  !general_detail::validNative(width, gate, options))
                return "macro region returned an invalid quantum exit";
            auto closed = proposal.body;
            closed.reserve(closedSize);
            for (auto gate = proposal.exitCliffords.rbegin();
                 gate != proposal.exitCliffords.rend(); ++gate)
              closed.push_back(macro_detail::inverse(*gate));
            // Prefix cost cannot authorize dropping a quantum exit. Both ranks
            // are for restored full-unitary implementations on the same wires.
            if (!(circuitScore(width, closed, options) <
                  circuitScore(width, body, options)))
              continue;
            if (!graph::equivalent(width, local, closed))
              return "macro region full-unitary replay failed";
            body = std::move(closed);
            ++result.macroImprovements;
          }
          return {};
        };
        auto error = explore();
        result.work = budget.work();
        if (!error.empty()) {
          fail(RegionStatus::InvalidProof, error.c_str());
          return result;
        }
      }
      size = llvm::SaturatingAdd<uint64_t>(local.size(), body.size());
      uint64_t bytes = llvm::SaturatingMultiply<uint64_t>(size, sizeof(Gate));
      if (!found && result.cacheEntries < entryLimit &&
          bytes <= limits.maxCacheBytes - result.cacheBytes) {
        if (!charge(size + 1))
          return result;
        auto &entry = cache[result.cacheEntries++];
        entry.width = width;
        entry.inputSize = local.size();
        entry.outputSize = body.size();
        entry.input = std::make_unique<Gate[]>(local.size());
        entry.output = std::make_unique<Gate[]>(body.size());
        std::copy(local.begin(), local.end(), entry.input.get());
        std::copy(body.begin(), body.end(), entry.output.get());
        result.cacheBytes += bytes;
      }
      if (!charge(body.size() + plan.prefix.size() + 1, rowWork))
        return result;
      plan.prefix.reserve(plan.prefix.size() + body.size());
      for (auto gate : body) {
        if (gate.kind == Kind::Rotate) {
          Pauli mapped{{}, {}, gate.pauli.sign};
          for (unsigned q : gate.pauli.x.set_bits())
            mapped.x.set(physical[q]);
          for (unsigned q : gate.pauli.z.set_bits())
            mapped.z.set(physical[q]);
          gate.pauli = std::move(mapped);
        } else if (gate.kind != Kind::Phase) {
          gate.a = physical[gate.a];
          if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
              gate.kind == Kind::Swap)
            gate.b = physical[gate.b];
        }
        plan.prefix.push_back(std::move(gate));
      }
    }
    if (unsupported)
      continue;
    ++result.completedProposals;
    bool duplicate = false;
    for (const auto &previous : result.candidates) {
      if (!charge(plan.prefix.size() + plan.observables.size() + 1, rowWork))
        return result;
      duplicate |= region_detail::samePlan(previous, plan);
    }
    if (!duplicate)
      result.candidates.push_back(std::move(plan));
  }
  if (result.candidates.empty())
    fail(RegionStatus::Unsupported,
         "general-region partition exceeds local limits");
  return result;
}
} // namespace cudaq::pauli::graph::detail
