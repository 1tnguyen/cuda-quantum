/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "SynthesisRegions.h"

namespace cudaq::pauli::graph::detail {

enum class OpenRegionStatus { Complete, Unsupported, Exhausted, InvalidProof };

/// A complete quantum region with body = exitCliffords * source, including
/// scalar phase. The implementation independently checks this exact relation.
/// The exit contains actual native primitive Cliffords; source scalar phases
/// remain in body. No terminal measurement freedom is used inside the region.
struct OpenRegionResult {
  OpenRegionStatus status = OpenRegionStatus::Complete;
  std::vector<Gate> body, exitCliffords;
  graph::Statistics stats;
  std::string error;
  explicit operator bool() const {
    return status == OpenRegionStatus::Complete;
  }
};

/// Implemented with the private graph synthesizer. Input contains normalized
/// rotations and optional scalar phases only. Never prune obligations, impose
/// fictitious measurements, or restore the residual Clifford before returning.
/// A nonzero deviation selects one bounded alternative entangling action.
OpenRegionResult synthesizeOpenRegion(unsigned width,
                                      const std::vector<Gate> &rotations,
                                      const Options &, SearchBudget &,
                                      unsigned deviation = 0);

struct OpenRegionLimits {
  unsigned maxPlans = 4;
  unsigned maxActiveWires = 32, maxRegionRotations = 128;
  /// Zero disables the second sweep; positive values enable deviation one.
  /// At most two pairing offsets times two deviation sweeps are attempted.
  unsigned maxDeviation = 1;
  uint64_t maxWork = 1000000000, maxLocalWork = 50000000;
};

struct OpenRegionCandidates {
  /// Complete native prefixes, still requiring the original fixed readout.
  /// The outer measured checker authorizes whole-program publication.
  std::vector<RegionalPlan> candidates;
  OpenRegionStatus status = OpenRegionStatus::Complete;
  std::string error;
  uint64_t work = 0, transportedExitGates = 0;
  unsigned windows = 0, regionsSolved = 0, completedProposals = 0;
  unsigned plansAttempted = 0;
};

namespace open_detail {
inline bool hasTime(const SearchBudget &budget) {
  auto time = budget.timeRemainingMs();
  return !time || *time >= 2;
}
inline unsigned halfTime(const SearchBudget &budget) {
  auto time = budget.timeRemainingMs();
  return time ? *time / 2 : 0;
}
inline bool validNative(unsigned width, const Gate &gate,
                        const Options &options) {
  bool valid = gate.kind == Kind::Rotate
                   ? !gate.pauli.x && gate.pauli.weight() == 1 &&
                         gate.pauli.z.fits(width) &&
                         (gate.angle.symbol || gate.angle.denominator > 0)
               : gate.kind == Kind::Phase
                   ? gate.angle.symbol || gate.angle.denominator > 0
                   : region_detail::validClifford(width, gate);
  return valid && (gate.kind != Kind::CX || options.nativeCX) &&
         (gate.kind != Kind::CZ || options.nativeCZ) &&
         (gate.kind != Kind::Swap || options.nativeSwap);
}
} // namespace open_detail

/// Compose open quantum regions across adjacent maximal commuting windows.
/// Offset zero pairs (0,1),(2,3),...; offset one keeps the first window alone,
/// then pairs (1,2),(3,4),... . Finish each complete plan before exploring the
/// next offset/deviation. Every selected exit conjugates ALL later rotations
/// and ordered observables; immutable physical readout identities never move.
///
/// This bounded provider keeps no cache or prefix-cost dominance relation.
/// Live memory is one transformed source, one prefix, one compact region and
/// its bounded body/exit, plus at most four completed plans. Copies follow work
/// charges and output caps. A failed later plan never invalidates a completed
/// predecessor; InvalidProof must stop promotion even if predecessors exist.
inline OpenRegionCandidates generateOpenRegionCandidates(
    const MeasuredProblem &source, const Options &options,
    const OpenRegionLimits &limits, SearchBudget &shared) {
  OpenRegionCandidates result;
  auto fail = [&](OpenRegionStatus status, const char *message) {
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
    fail(OpenRegionStatus::Unsupported,
         "invalid open-region dimensions, profile or limits");
    return result;
  }
  for (double value :
       {options.parallelizationCredit, options.frontierWeight,
        options.frameWeight, options.actionWeight, options.readoutWeight,
        options.singleQubitCost, options.tCost, options.cxCost, options.czCost,
        options.swapCost})
    if (!std::isfinite(value) || value < 0) {
      fail(OpenRegionStatus::Unsupported, "invalid open-region cost profile");
      return result;
    }
  if (!open_detail::hasTime(shared) || shared.remaining() < 2) {
    fail(OpenRegionStatus::Exhausted, "no open-region exploration reserve");
    return result;
  }
  SearchBudget budget(shared, std::min(limits.maxWork, shared.remaining() / 2),
                      open_detail::halfTime(shared));
  const uint64_t limbs = uint64_t(source.width) / 64 + 1;
  auto finish = [&]() {
    result.work = budget.work();
    return std::move(result);
  };
  auto charge = [&](uint64_t rows, uint64_t rowWork = 1) {
    if (budget.take(llvm::SaturatingMultiply<uint64_t>(rows, rowWork)))
      return true;
    fail(OpenRegionStatus::Exhausted, "open-region composition work exhausted");
    return false;
  };
  if (!charge(source.rotations.size() + source.observables.size() + 1, limbs))
    return finish();
  for (const auto &gate : source.rotations)
    if ((gate.kind != Kind::Rotate && gate.kind != Kind::Phase) ||
        (gate.kind == Kind::Rotate &&
         (!gate.pauli.weight() || !gate.pauli.x.fits(source.width) ||
          !gate.pauli.z.fits(source.width))) ||
        (!gate.angle.symbol && gate.angle.denominator <= 0)) {
      fail(OpenRegionStatus::Unsupported,
           "open regions require normalized nonidentity rotations or phases");
      return finish();
    }
  pbc::CommutingObservables measurements;
  std::unordered_map<unsigned, size_t> assigned;
  for (size_t i = 0; i < source.observables.size(); ++i) {
    const auto &observable = source.observables[i];
    if (!charge(2 * uint64_t(measurements.size()) + 1, limbs))
      return finish();
    if (!observable.weight() || !observable.x.fits(source.width) ||
        !observable.z.fits(source.width) ||
        source.readoutWires[i] >= source.width ||
        !measurements.insert(observable)) {
      fail(OpenRegionStatus::Unsupported, "invalid open-region observables");
      return finish();
    }
    auto [at, inserted] = assigned.try_emplace(source.readoutWires[i], i);
    if (!inserted && observable != source.observables[at->second]) {
      fail(OpenRegionStatus::Unsupported, "conflicting open-region readouts");
      return finish();
    }
  }
  if (measurements.size() != assigned.size()) {
    fail(OpenRegionStatus::Unsupported,
         "open-region readouts would require classical remapping");
    return finish();
  }
  // Conjugating the whole unconsumed suffix preserves pairwise commutation.
  // These original index intervals therefore remain legal for every plan.
  std::vector<std::pair<size_t, size_t>> windows;
  pbc::CommutingObservables commuting;
  size_t begin = 0;
  for (size_t i = 0; i < source.rotations.size(); ++i) {
    if (!charge(2 * uint64_t(commuting.size()) + 1, limbs))
      return finish();
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
  unsigned offsets = windows.size() > 1 ? 2 : 1;
  unsigned plans =
      std::min(limits.maxPlans, offsets * (limits.maxDeviation ? 2 : 1));
  if (windows.empty())
    plans = 1;
  for (unsigned index = 0; index < plans; ++index) {
    ++result.plansAttempted;
    unsigned offset = index % offsets, deviation = index / offsets;
    if (!charge(source.rotations.size() + source.observables.size() + 1, limbs))
      return finish();
    auto remaining = source.rotations;
    RegionalPlan plan{{}, source.observables};
    bool unsupported = false;
    for (size_t w = 0; w < windows.size();) {
      size_t next = std::min(w + ((offset && !w) ? 1 : 2), windows.size());
      size_t first = windows[w].first, last = windows[next - 1].second;
      w = next;
      if (last - first > limits.maxRegionRotations) {
        unsupported = true;
        break;
      }
      if (!charge(last - first + 1, limbs))
        return finish();
      QubitSet support;
      for (size_t i = first; i < last; ++i)
        if (remaining[i].kind == Kind::Rotate)
          support |= remaining[i].pauli.x | remaining[i].pauli.z;
      if (support.count() > std::min(limits.maxActiveWires, 32u)) {
        unsupported = true;
        break;
      }
      if (!charge(last - first + support.count() + 1, limbs))
        return finish();
      std::vector<unsigned> physical;
      for (unsigned q : support.set_bits())
        physical.push_back(q);
      if (physical.empty())
        physical.push_back(
            0); // Scalar-only word acts identically on this wire.
      unsigned width = physical.size();
      if (!charge(last - first + 1, width + 1))
        return finish();
      std::vector<Gate> local;
      local.reserve(last - first);
      for (size_t i = first; i < last; ++i) {
        const auto &gate = remaining[i];
        if (gate.kind == Kind::Phase) {
          local.push_back(gate);
          continue;
        }
        Pauli compact{{}, {}, gate.pauli.sign};
        for (unsigned q = 0; q < width; ++q) {
          compact.x.set(q, gate.pauli.x.test(physical[q]));
          compact.z.set(q, gate.pauli.z.test(physical[q]));
        }
        local.push_back(Gate::rotation(compact, gate.angle));
      }
      size_t available = options.maxGates - plan.prefix.size();
      if (!available || local.size() > available ||
          !open_detail::hasTime(budget) || budget.remaining() < 2) {
        fail(OpenRegionStatus::Exhausted,
             "no open-region local search reserve");
        return finish();
      }
      Options localOptions = options;
      localOptions.release = false;
      localOptions.pruneUnobserved = false;
      localOptions.acceptOnlyImprovement = false;
      localOptions.maxGates = available;
      SearchBudget search(budget,
                          std::min(limits.maxLocalWork, budget.remaining() / 2),
                          open_detail::halfTime(budget));
      auto implementation =
          synthesizeOpenRegion(width, local, localOptions, search, deviation);
      if (!implementation) {
        if (implementation.status == OpenRegionStatus::Unsupported) {
          unsupported = true;
          break;
        }
        fail(implementation.status, implementation.error.c_str());
        return finish();
      }
      ++result.regionsSolved;
      if (implementation.body.size() > available ||
          implementation.exitCliffords.size() > available) {
        fail(OpenRegionStatus::Exhausted,
             "open-region native gate cap exceeded");
        return finish();
      }
      if (!charge(implementation.body.size() +
                      implementation.exitCliffords.size() + 1,
                  width + 1))
        return finish();
      for (const auto &gate : implementation.body)
        if (!open_detail::validNative(width, gate, options)) {
          fail(OpenRegionStatus::InvalidProof,
               "open-region solver returned an invalid native body");
          return finish();
        }
      for (const auto &gate : implementation.exitCliffords)
        if (!region_detail::validClifford(width, gate) ||
            !open_detail::validNative(width, gate, options)) {
          fail(OpenRegionStatus::InvalidProof,
               "open-region solver returned an invalid quantum exit");
          return finish();
        }
      // The solver checked this exact local source/body/exit. Relabeling its
      // complete active support is injective and also applies to its exit.
      if (!charge(implementation.body.size() +
                      implementation.exitCliffords.size() + plan.prefix.size() +
                      1,
                  limbs))
        return finish();
      auto lift = [&](Gate gate) {
        if (gate.kind == Kind::Rotate) {
          Pauli physicalAxis{{}, {}, gate.pauli.sign};
          for (unsigned q : gate.pauli.x.set_bits())
            physicalAxis.x.set(physical[q]);
          for (unsigned q : gate.pauli.z.set_bits())
            physicalAxis.z.set(physical[q]);
          gate.pauli = std::move(physicalAxis);
        } else if (gate.kind != Kind::Phase) {
          gate.a = physical[gate.a];
          if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
              gate.kind == Kind::Swap)
            gate.b = physical[gate.b];
        }
        return gate;
      };
      plan.prefix.reserve(plan.prefix.size() + implementation.body.size());
      for (const auto &gate : implementation.body)
        plan.prefix.push_back(lift(gate));
      for (const auto &localGate : implementation.exitCliffords) {
        if (!charge(remaining.size() - last + plan.observables.size() + 1,
                    limbs))
          return finish();
        Gate gate = lift(localGate);
        for (size_t i = last; i < remaining.size(); ++i)
          if (remaining[i].kind == Kind::Rotate)
            remaining[i].pauli =
                tcount::detail::conjugate(remaining[i].pauli, gate);
        for (auto &observable : plan.observables)
          observable = tcount::detail::conjugate(observable, gate);
        ++result.transportedExitGates;
      }
    }
    if (unsupported)
      continue;
    ++result.completedProposals;
    bool duplicate = false;
    for (const auto &previous : result.candidates) {
      if (!charge(plan.prefix.size() + plan.observables.size() + 1, limbs))
        return finish();
      duplicate |= region_detail::samePlan(previous, plan);
    }
    if (!duplicate)
      result.candidates.push_back(std::move(plan));
  }
  if (result.candidates.empty())
    fail(OpenRegionStatus::Unsupported,
         "open-region partition exceeds local support or word limits");
  return finish();
}
} // namespace cudaq::pauli::graph::detail
