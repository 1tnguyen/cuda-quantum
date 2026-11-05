/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SynthesisBootstrap.h"
#include "SynthesisCandidates.h"
#include "SynthesisCommuting.h"

namespace cudaq::pauli::graph::detail {

struct RegionLimits {
  /// Bounded whole-program sweeps, capped internally at sixteen. The first
  /// sweep completes before any alternative sweep begins.
  unsigned maxPlans = 4;
  /// Zero retains the existing lockstep sweeps. Positive values first produce
  /// the same greedy plan, then explore a boundary-state beam capped at eight.
  unsigned beamWidth = 0;
  uint64_t maxWork = 100000000;
  /// Further bounded by eight regional candidates, four diagonalizers and
  /// 2^20 DP entries. These are search limits, not semantic restrictions.
  CommutingLimits commuting;
};

enum class RegionStatus { Complete, Unsupported, Exhausted, InvalidProof };

/// All rotation obligations have native implementations in prefix. Observables
/// remain in their original ordered classical identities, expressed after the
/// carried Clifford exit. Physical readout wires remain those of the immutable
/// source problem. A caller must complete readout and check the complete
/// circuit before treating this unverified proposal as a candidate.
struct RegionalPlan {
  std::vector<Gate> prefix;
  std::vector<Pauli> observables;
};

/// Optional ranking of complete proposals, including a caller's signed readout
/// and refinement if desired. Work must be charged to the supplied child;
/// nullopt rejects an unavailable/inadmissible completion, not its equivalence.
/// Only the outer whole-program checker may authorize candidate publication.
using RegionScorer = std::function<std::optional<CostRank>(const RegionalPlan &,
                                                           SearchBudget &)>;

struct RegionCandidates {
  std::vector<RegionalPlan> candidates;
  RegionStatus status = RegionStatus::Complete;
  unsigned regions = 0;
  uint64_t work = 0;
  uint64_t beamExpansions = 0, completedProposals = 0;
  unsigned beamPeakStates = 0;
  std::string error;
};

namespace region_detail {
class Budget {
public:
  Budget(SearchBudget &shared, uint64_t limit, RegionCandidates &result)
      : shared(shared), limit(limit), result(result) {}
  bool take(uint64_t amount = 1) {
    if (result.status == RegionStatus::Exhausted)
      return false;
    if (amount > limit - result.work || !shared.take(amount)) {
      result.status = RegionStatus::Exhausted;
      result.error = "regional composition work budget exhausted";
      return false;
    }
    result.work += amount;
    return true;
  }
  uint64_t remaining() const { return limit - result.work; }
  void accountExternal(uint64_t amount) { result.work += amount; }

private:
  SearchBudget &shared;
  uint64_t limit;
  RegionCandidates &result;
};

/// Account child work exactly once even when a proposal exits early.
struct ChildWork {
  Budget &budget;
  SearchBudget &parent;
  uint64_t before;
  ChildWork(Budget &budget, SearchBudget &parent)
      : budget(budget), parent(parent), before(parent.work()) {}
  ~ChildWork() { budget.accountExternal(parent.work() - before); }
};

inline bool validClifford(unsigned width, const Gate &gate) {
  switch (gate.kind) {
  case Kind::CX:
  case Kind::CZ:
  case Kind::Swap:
    return gate.a < width && gate.b < width && gate.a != gate.b;
  case Kind::H:
  case Kind::S:
  case Kind::Sdg:
  case Kind::X:
  case Kind::Y:
  case Kind::Z:
    return gate.a < width;
  default:
    return false;
  }
}

inline bool samePlan(const RegionalPlan &a, const RegionalPlan &b) {
  if (a.observables != b.observables || a.prefix.size() != b.prefix.size())
    return false;
  for (size_t i = 0; i < a.prefix.size(); ++i) {
    const auto &x = a.prefix[i], &y = b.prefix[i];
    if (x.kind != y.kind || x.a != y.a || x.b != y.b || x.pauli != y.pauli ||
        x.angle != y.angle)
      return false;
  }
  return true;
}
} // namespace region_detail

/// Compose contiguous commuting regions through their full quantum interfaces.
/// A region body G and explicit Clifford exit F satisfy G=F*U. Appending G and
/// conjugating every later rotation and terminal observable by F therefore
/// preserves the original ordered measured problem for arbitrary inputs.
/// No internal region uses terminal measurement freedom or assumes zero inputs.
///
/// The first sweep requests the first implementation of each region. Subsequent
/// sweeps choose successively ranked regional implementations, retaining their
/// distinct exits. This is bounded heuristic exploration, not prefix dominance
/// or an optimality claim. Only complete plans are published; exhaustion or an
/// unsupported later sweep leaves earlier complete plans available.
/// InvalidProof must stop promotion even if earlier plans are present.
inline RegionCandidates generateRegionSweeps(const MeasuredProblem &source,
                                             const Options &options,
                                             const RegionLimits &limits,
                                             SearchBudget &shared) {
  RegionCandidates result;
  region_detail::Budget budget(shared, limits.maxWork, result);
  auto unsupported = [&](const char *message) {
    result.status = RegionStatus::Unsupported;
    result.error = message;
    return result;
  };
  auto exhausted = [&](const char *message) {
    result.status = RegionStatus::Exhausted;
    result.error = message;
    return result;
  };
  auto invalid = [&](const char *message) {
    result.status = RegionStatus::InvalidProof;
    result.error = message;
    return result;
  };
  if (!source.width || !limits.maxPlans || !limits.commuting.maxCandidates ||
      !options.maxGates || source.rotations.size() > options.maxGates ||
      source.observables.empty() ||
      source.observables.size() != source.readoutWires.size() ||
      (!options.nativeCX && !options.nativeCZ))
    return unsupported("invalid regional problem dimensions or limits");
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost})
    if (!std::isfinite(cost) || cost < 0)
      return unsupported("invalid regional native costs");
  uint64_t rowWork = uint64_t(source.width) / 64 + 1;
  auto work = [&](uint64_t count) {
    return llvm::SaturatingMultiply<uint64_t>(count, rowWork);
  };
  if (!budget.take(
          work(source.rotations.size() + source.observables.size() + 1)))
    return result;
  for (const auto &gate : source.rotations)
    if (gate.kind != Kind::Rotate || !gate.pauli.weight() ||
        !(gate.pauli.x | gate.pauli.z).fits(source.width) ||
        (!gate.angle.symbol && gate.angle.denominator <= 0))
      return unsupported("regional input must contain valid Pauli rotations");
  pbc::CommutingObservables measurements;
  std::unordered_map<unsigned, size_t> assigned;
  for (size_t i = 0; i < source.observables.size(); ++i) {
    const auto &observable = source.observables[i];
    if (!observable.weight() ||
        !(observable.x | observable.z).fits(source.width) ||
        source.readoutWires[i] >= source.width)
      return unsupported("regional measurement support or readout is invalid");
    if (!budget.take(work(2 * uint64_t(measurements.size()) + 1)))
      return result;
    if (!measurements.insert(observable))
      return unsupported("regional terminal observables must commute");
    auto [at, inserted] = assigned.try_emplace(source.readoutWires[i], i);
    if (!inserted && observable != source.observables[at->second])
      return unsupported("regional repeated readout identities conflict");
  }
  if (measurements.size() != assigned.size())
    return unsupported("regional readout would require classical remapping");

  // Each new window is maximal in this ordered partition. Subsequent common
  // Clifford conjugations preserve all remaining pairwise commutation, so these
  // immutable index intervals remain legal throughout every sweep.
  std::vector<std::pair<size_t, size_t>> windows;
  pbc::CommutingObservables commuting;
  size_t begin = 0;
  for (size_t i = 0; i < source.rotations.size(); ++i) {
    if (!budget.take(work(2 * uint64_t(commuting.size()) + 1)))
      return result;
    if (!commuting.insert(source.rotations[i].pauli)) {
      windows.push_back({begin, i});
      begin = i;
      commuting = {};
      commuting.insert(source.rotations[i].pauli);
    }
  }
  if (begin < source.rotations.size())
    windows.push_back({begin, source.rotations.size()});
  result.regions = windows.size();

  unsigned plans = std::min(limits.maxPlans, 16u);
  if (windows.empty())
    plans = 1;
  for (unsigned sweep = 0; sweep < plans; ++sweep) {
    if (!budget.take(
            work(source.rotations.size() + source.observables.size() + 1)))
      return result;
    auto remaining = source.rotations;
    RegionalPlan plan{{}, source.observables};
    for (auto [first, last] : windows) {
      size_t availableGates = options.maxGates - plan.prefix.size();
      if (last - first > availableGates)
        return exhausted("regional composition native gate limit exceeded");
      if (!budget.take(work(last - first + 1)))
        return result;
      std::vector<Gate> region(remaining.begin() + first,
                               remaining.begin() + last);
      CommutingLimits local = limits.commuting;
      local.maxCandidates =
          std::min({local.maxCandidates, sweep + 1, unsigned(8)});
      local.maxDiagonalizers = std::min(local.maxDiagonalizers, 4u);
      local.maxExactRank = std::min(local.maxExactRank, 5u);
      local.maxDpEntries = std::min<uint64_t>(local.maxDpEntries, 1048576);
      local.maxWork = std::min(local.maxWork, budget.remaining());
      if (!sweep) {
        bool diagonal = std::all_of(region.begin(), region.end(),
                                    [](const Gate &g) { return !g.pauli.x; });
        local.maxDiagonalizers =
            diagonal ? 0 : std::min(local.maxDiagonalizers, 1u);
      }
      Options regionalOptions = options;
      regionalOptions.maxGates = availableGates;
      auto before = shared.work();
      auto implementations = solveCommutingRegion(
          source.width, region, regionalOptions, local, shared);
      budget.accountExternal(shared.work() - before);
      if (implementations.status == CommutingStatus::InvalidProof)
        return invalid(implementations.error.c_str());
      // Even if a regional search retained a complete local implementation,
      // there is no work reserve here to complete an unfinished whole program.
      if (implementations.status == CommutingStatus::Exhausted)
        return exhausted(implementations.error.c_str());
      if (implementations.status == CommutingStatus::Unsupported ||
          implementations.candidates.empty())
        return unsupported("a commuting region has no complete implementation");
      auto &selected = implementations.candidates[std::min<size_t>(
          sweep, implementations.candidates.size() - 1)];
      if (selected.body.size() > availableGates)
        return exhausted("regional composition native gate limit exceeded");
      if (!budget.take(
              work(selected.body.size() + selected.exitCliffords.size() + 1)))
        return result;
      for (const auto &gate : selected.exitCliffords)
        if (!region_detail::validClifford(source.width, gate))
          return invalid("region exit is not a valid native Clifford frame");
      for (const auto &gate : selected.body) {
        if (gate.kind == Kind::Rotate) {
          if (gate.pauli.x || gate.pauli.weight() != 1 ||
              !gate.pauli.z.fits(source.width) ||
              (!gate.angle.symbol && gate.angle.denominator <= 0))
            return invalid("region body has an invalid native rotation");
        } else if (!region_detail::validClifford(source.width, gate))
          return invalid("region body has an invalid native operation");
        if ((gate.kind == Kind::CX && !options.nativeCX) ||
            (gate.kind == Kind::CZ && !options.nativeCZ) ||
            (gate.kind == Kind::Swap && !options.nativeSwap))
          return invalid(
              "region body does not use the requested native profile");
      }
      plan.prefix.insert(plan.prefix.end(), selected.body.begin(),
                         selected.body.end());
      for (const auto &clifford : selected.exitCliffords) {
        if (!budget.take(
                work(remaining.size() - last + plan.observables.size())))
          return result;
        for (size_t i = last; i < remaining.size(); ++i)
          remaining[i].pauli =
              tcount::detail::conjugate(remaining[i].pauli, clifford);
        for (auto &observable : plan.observables)
          observable = tcount::detail::conjugate(observable, clifford);
      }
    }
    ++result.completedProposals;
    bool duplicate = false;
    for (const auto &previous : result.candidates) {
      if (!budget.take(work(plan.prefix.size() + plan.observables.size() + 1)))
        return result;
      if (region_detail::samePlan(previous, plan)) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate)
      result.candidates.push_back(std::move(plan));
  }
  return result;
}

/// Optional boundary beam. The first greedy proposal remains in output slot
/// zero. Additional complete proposals are ranked before the output cap, while
/// partial states use the cost of a mechanically completed proposal as a search
/// priority. That priority is neither an independently certified upper bound
/// nor a dominance proof between different quantum boundaries.
///
/// Expansions are streamed: at most two beams of eight states plus one child
/// own full words. Every word is bounded by the source/output gate caps and
/// charged before copying. No half-native prefix escapes in candidates.
inline RegionCandidates
generateRegionCandidates(const MeasuredProblem &source, const Options &options,
                         const RegionLimits &limits, SearchBudget &shared,
                         const RegionScorer &scorer = {}) {
  RegionLimits first = limits;
  if (limits.beamWidth)
    first.maxPlans = 1;
  auto result = generateRegionSweeps(source, options, first, shared);
  if (result.status != RegionStatus::Complete || result.candidates.empty() ||
      (!limits.beamWidth && !scorer) || limits.maxPlans <= 1)
    return result;
  const unsigned maxPlans = std::min(limits.maxPlans, 16u);
  SearchBudget slice(
      shared, std::min(limits.maxWork - result.work, shared.remaining()));
  region_detail::Budget budget(slice, limits.maxWork, result);
  const uint64_t rowWork = uint64_t(source.width) / 64 + 1;
  auto work = [&](uint64_t count) {
    return llvm::SaturatingMultiply<uint64_t>(count, rowWork);
  };
  auto stop = [&](const char *message) {
    result.status = RegionStatus::Exhausted;
    result.error = message;
  };
  struct RankedPlan {
    CostRank rank;
    uint64_t ordinal;
    RegionalPlan plan;
  };
  std::vector<RankedPlan> ranked;
  uint64_t ordinal = 0;
  auto publish = [&](RegionalPlan plan, std::optional<CostRank> fallback) {
    for (const auto &old : result.candidates) {
      if (!budget.take(work(plan.prefix.size() + plan.observables.size() + 1)))
        return false;
      if (region_detail::samePlan(old, plan))
        return true;
    }
    auto rank = fallback;
    if (scorer) {
      SearchBudget scoring(slice,
                           std::min(budget.remaining(), slice.remaining()));
      {
        region_detail::ChildWork account(budget, slice);
        rank = scorer(plan, scoring);
      }
      if (scoring.stopped()) {
        stop("regional completed-plan scoring budget exhausted");
        return false;
      }
    }
    if (!rank)
      return true;
    if (std::any_of(rank->begin(), rank->end(),
                    [](double x) { return !std::isfinite(x); })) {
      result.status = RegionStatus::Unsupported;
      result.error = "regional completion score must be finite";
      return false;
    }
    ranked.push_back({*rank, ordinal++, std::move(plan)});
    std::stable_sort(
        ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
          return std::tie(a.rank, a.ordinal) < std::tie(b.rank, b.ordinal);
        });
    if (ranked.size() >= maxPlans)
      ranked.resize(maxPlans - 1);
    // Transactional publication preserves the first complete proposal and any
    // previous shortlist if the final copying charge cannot finish.
    std::vector<RegionalPlan> complete;
    if (!budget.take(work(result.candidates[0].prefix.size() +
                          result.candidates[0].observables.size() + 1)))
      return false;
    complete.push_back(result.candidates[0]);
    for (const auto &candidate : ranked) {
      if (!budget.take(work(candidate.plan.prefix.size() +
                            candidate.plan.observables.size() + 1)))
        return false;
      complete.push_back(candidate.plan);
    }
    result.candidates = std::move(complete);
    return true;
  };
  if (!limits.beamWidth) {
    // A caller may also rank the existing complete sweep set with a callback.
    auto completed = std::move(result.candidates);
    result.candidates.clear();
    result.candidates.push_back(std::move(completed[0]));
    for (size_t i = 1; i < completed.size(); ++i)
      if (!publish(std::move(completed[i]), std::nullopt))
        break;
    return result;
  }
  if (source.rotations.empty())
    return result;

  std::vector<std::pair<size_t, size_t>> windows;
  pbc::CommutingObservables commuting;
  size_t begin = 0;
  for (size_t i = 0; i < source.rotations.size(); ++i) {
    if (!budget.take(work(2 * uint64_t(commuting.size()) + 1)))
      return result;
    if (!commuting.insert(source.rotations[i].pauli)) {
      windows.push_back({begin, i});
      begin = i;
      commuting = {};
      commuting.insert(source.rotations[i].pauli);
    }
  }
  windows.push_back({begin, source.rotations.size()});
  struct State {
    std::vector<Gate> remaining;
    RegionalPlan plan;
    CostRank priority{};
    uint64_t ordinal = 0;
  };
  const unsigned beamWidth = std::min(limits.beamWidth, 8u);
  if (!budget.take(
          work(source.rotations.size() + source.observables.size() + 1)))
    return result;
  std::vector<State> beam{{source.rotations, {{}, source.observables}, {}, 0}};
  uint64_t stateOrdinal = 0;

  auto completionPriority = [&](const State &state,
                                size_t next) -> std::optional<CostRank> {
    SearchBudget scoring(slice,
                         std::min(budget.remaining(), slice.remaining()));
    region_detail::ChildWork account(budget, slice);
    auto charge = [&](uint64_t amount) {
      if (scoring.take(amount))
        return true;
      stop("regional proposal-completion budget exhausted");
      return false;
    };
    if (!charge(work(state.remaining.size() - next + 1)))
      return std::nullopt;
    std::vector<Gate> rest(state.remaining.begin() + next,
                           state.remaining.end());
    Options tailOptions = options;
    tailOptions.maxGates = options.maxGates - state.plan.prefix.size();
    MeasuredProblem residual{source.width, rest, state.plan.observables,
                             source.readoutWires};
    auto bootstrap = buildBootstrapPrefix(residual, tailOptions, &scoring);
    if (!bootstrap) {
      if (scoring.stopped())
        stop("regional bootstrap-completion budget exhausted");
      return std::nullopt;
    }
    // Public fixed readout currently has no budget parameter. Precharge its
    // deterministic tableau-sweep estimate before invoking it, then charge the
    // actual complete native word before copying and scheduling it.
    uint64_t readoutWork = llvm::SaturatingMultiply<uint64_t>(
        state.plan.observables.size() + 1,
        uint64_t(source.width) * source.width + 1);
    if (!charge(readoutWork))
      return std::nullopt;
    auto readout = graph::synthesizeReadout(
        source.width, state.plan.observables, source.readoutWires, options);
    if (!readout) {
      if (!readout.stats.budgetExhausted) {
        result.status = RegionStatus::InvalidProof;
        result.error = readout.error;
      }
      return std::nullopt;
    }
    size_t available = options.maxGates - state.plan.prefix.size();
    if (bootstrap.gates.size() > available ||
        readout.gates.size() > available - bootstrap.gates.size())
      return std::nullopt;
    size_t size = state.plan.prefix.size() + bootstrap.gates.size() +
                  readout.gates.size();
    if (!charge(uint64_t(source.width) + size + 1))
      return std::nullopt;
    auto complete = state.plan.prefix;
    complete.insert(complete.end(), bootstrap.gates.begin(),
                    bootstrap.gates.end());
    complete.insert(complete.end(), readout.gates.begin(), readout.gates.end());
    return circuitScore(source.width, complete, options);
  };

  for (auto [firstRotation, lastRotation] : windows) {
    std::vector<State> next;
    for (const auto &state : beam) {
      size_t available = options.maxGates - state.plan.prefix.size();
      if (lastRotation - firstRotation > available)
        continue;
      if (!budget.take(work(lastRotation - firstRotation + 1)))
        return result;
      std::vector<Gate> region(state.remaining.begin() + firstRotation,
                               state.remaining.begin() + lastRotation);
      CommutingLimits local = limits.commuting;
      local.maxCandidates = std::min(local.maxCandidates, 8u);
      local.maxDiagonalizers = std::min(local.maxDiagonalizers, 4u);
      local.maxExactRank = std::min(local.maxExactRank, 5u);
      local.maxDpEntries = std::min<uint64_t>(local.maxDpEntries, 1048576);
      local.maxWork = std::min(local.maxWork, budget.remaining());
      Options regionalOptions = options;
      regionalOptions.maxGates = available;
      SearchBudget search(slice, std::min(local.maxWork, slice.remaining()));
      CommutingCandidates implementations;
      {
        region_detail::ChildWork account(budget, slice);
        implementations = solveCommutingRegion(source.width, region,
                                               regionalOptions, local, search);
      }
      if (implementations.status == CommutingStatus::InvalidProof) {
        result.status = RegionStatus::InvalidProof;
        result.error = implementations.error;
        return result;
      }
      if (implementations.status == CommutingStatus::Exhausted) {
        stop(implementations.error.c_str());
        return result;
      }
      if (implementations.status == CommutingStatus::Unsupported)
        continue;
      for (const auto &implementation : implementations.candidates) {
        if (implementation.body.size() > available)
          continue;
        if (!budget.take(work(
                state.remaining.size() + state.plan.prefix.size() +
                state.plan.observables.size() + implementation.body.size() +
                implementation.exitCliffords.size() + 1)))
          return result;
        for (const auto &gate : implementation.exitCliffords)
          if (!region_detail::validClifford(source.width, gate)) {
            result.status = RegionStatus::InvalidProof;
            result.error = "beam region has an invalid quantum exit";
            return result;
          }
        for (const auto &gate : implementation.body) {
          bool valid =
              gate.kind == Kind::Rotate
                  ? !gate.pauli.x && gate.pauli.weight() == 1 &&
                        gate.pauli.z.fits(source.width) &&
                        (gate.angle.symbol || gate.angle.denominator > 0)
                  : region_detail::validClifford(source.width, gate);
          valid &= !(gate.kind == Kind::CX && !options.nativeCX) &&
                   !(gate.kind == Kind::CZ && !options.nativeCZ) &&
                   !(gate.kind == Kind::Swap && !options.nativeSwap);
          if (!valid) {
            result.status = RegionStatus::InvalidProof;
            result.error = "beam region has an invalid native body";
            return result;
          }
        }
        State child{state.remaining, state.plan, {}, stateOrdinal++};
        child.plan.prefix.insert(child.plan.prefix.end(),
                                 implementation.body.begin(),
                                 implementation.body.end());
        for (const auto &clifford : implementation.exitCliffords) {
          if (!budget.take(work(child.remaining.size() - lastRotation +
                                child.plan.observables.size())))
            return result;
          for (size_t i = lastRotation; i < child.remaining.size(); ++i)
            child.remaining[i].pauli =
                tcount::detail::conjugate(child.remaining[i].pauli, clifford);
          for (auto &observable : child.plan.observables)
            observable = tcount::detail::conjugate(observable, clifford);
        }
        ++result.beamExpansions;
        result.beamPeakStates = std::max<unsigned>(
            result.beamPeakStates, beam.size() + next.size() + 1);
        auto priority = completionPriority(child, lastRotation);
        if (result.status != RegionStatus::Complete)
          return result;
        if (!priority)
          continue;
        child.priority = *priority;
        if (lastRotation == source.rotations.size()) {
          ++result.completedProposals;
          if (!publish(std::move(child.plan), priority))
            return result;
          continue;
        }
        bool duplicate = false;
        for (const auto &old : next) {
          if (!budget.take(work(child.remaining.size() +
                                child.plan.prefix.size() +
                                child.plan.observables.size() + 1)))
            return result;
          if (commuting_detail::sameWord(old.remaining, child.remaining) &&
              region_detail::samePlan(old.plan, child.plan)) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          next.push_back(std::move(child));
          std::stable_sort(next.begin(), next.end(),
                           [](const State &a, const State &b) {
                             return std::tie(a.priority, a.ordinal) <
                                    std::tie(b.priority, b.ordinal);
                           });
          if (next.size() > beamWidth)
            next.resize(beamWidth);
        }
      }
    }
    if (lastRotation != source.rotations.size() && next.empty()) {
      result.status = RegionStatus::Unsupported;
      result.error = "boundary beam has no completable continuation";
      return result;
    }
    beam = std::move(next);
  }
  return result;
}
} // namespace cudaq::pauli::graph::detail
