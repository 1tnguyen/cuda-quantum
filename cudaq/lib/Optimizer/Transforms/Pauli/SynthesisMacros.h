/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "SynthesisBootstrap.h"
#include "SynthesisCost.h"
#include "TCountInternal.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>

namespace cudaq::pauli::graph::detail {

struct MacroLimits {
  unsigned maxCandidates = 4, beamWidth = 4, maxRounds = 2;
  /// Active support, not physical register width. Hard caps are eight active
  /// wires, 256 input gates, four rounds, eight beam states, sixteen outputs.
  unsigned maxActiveWires = 8, maxInputGates = 128;
  /// Total attempted pairs across all states/rounds, capped at 65536.
  uint64_t maxActionPairs = 1024, maxWork = 1000000;
};

/// Unverified full quantum proposal: body = exitCliffords * original word,
/// including scalar phase. The exit is an actual native Clifford word, not a
/// phase-blind tableau. An outer checker must authorize use of this proposal.
struct MacroImplementation {
  std::vector<Gate> body, exitCliffords;
};

/// Rank the complete body together with a caller's actual boundary completion.
/// Charge all work through the supplied child. Nullopt means unavailable or
/// inadmissible completion. It does not establish inequivalence. Without a
/// scorer the backend restores the inverse exit and scores that complete word.
using MacroScorer = std::function<std::optional<CostRank>(
    const MacroImplementation &, SearchBudget &)>;

enum class MacroStatus { Complete, Unsupported, Exhausted };
struct MacroCandidates {
  /// Slot zero is the mechanical identity-exit baseline, retained on all later
  /// failures. Every other slot is also a complete quantum proposal.
  std::vector<MacroImplementation> candidates;
  MacroStatus status = MacroStatus::Complete;
  uint64_t work = 0, expandedPairs = 0, completedProposals = 0;
  unsigned actions = 0, peakStates = 0;
  std::string error;
};

namespace macro_detail {
inline Gate inverse(Gate gate) {
  if (gate.kind == Kind::S)
    gate.kind = Kind::Sdg;
  else if (gate.kind == Kind::Sdg)
    gate.kind = Kind::S;
  return gate;
}
inline bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}
inline bool sameImplementation(const MacroImplementation &a,
                               const MacroImplementation &b) {
  return sameWord(a.body, b.body) && sameWord(a.exitCliffords, b.exitCliffords);
}
} // namespace macro_detail

/// Bounded nonmonotone Clifford-frame exploration on an ordered region.
/// A state satisfies residual * prefix = exit * source. Carrying C appends C
/// to prefix/exit and conjugates every residual rotation; emitting its first
/// rotation removes that obligation exactly. Only leading one-wire rotations
/// (and explicit phases) are emitted eagerly, so noncommuting order is never
/// approximated. Mechanical completion makes every ranked proposal complete.
///
/// A macro is two actions from {identity, H, S, Sdg, CX, CZ} on active wires,
/// lowered into the configured native basis. No ranking or support-reduction
/// filter occurs between its actions: the first may increase any support.
/// A bounded beam ranks only complete proposals. Different exit frames are
/// never declared dominated by prefix cost. Exact-word dedup is conservative;
/// neither this finite family nor the beam claims global optimality.
inline MacroCandidates
solveMacroRegion(unsigned width, const std::vector<Gate> &rotations,
                 const Options &options, const MacroLimits &limits,
                 SearchBudget &shared, const MacroScorer &scorer = {}) {
  MacroCandidates result;
  SearchBudget budget(shared, std::min(limits.maxWork, shared.remaining()));
  auto finish = [&]() {
    result.work = budget.work();
    return std::move(result);
  };
  auto exhaust = [&](const char *message) {
    result.status = MacroStatus::Exhausted;
    result.error = message;
  };
  auto unsupported = [&](const char *message) {
    result.status = MacroStatus::Unsupported;
    result.error = message;
  };
  auto charge = [&](uint64_t amount) {
    if (budget.take(amount))
      return true;
    exhaust("macro search work budget exhausted");
    return false;
  };
  const uint64_t limbs = uint64_t(width) / 64 + 1;
  auto work = [&](uint64_t rows) {
    return llvm::SaturatingMultiply<uint64_t>(rows, limbs);
  };
  if (!width || width > 4096 || !limits.maxCandidates || !limits.beamWidth ||
      !limits.maxInputGates || !limits.maxActiveWires || !options.maxGates ||
      rotations.size() > std::min(limits.maxInputGates, 256u) ||
      (!options.nativeCX && !options.nativeCZ) ||
      static_cast<unsigned>(options.objective) >
          static_cast<unsigned>(SynthesisObjective::TDepth)) {
    unsupported(
        "unsupported macro dimensions, native profile, objective or limits");
    return finish();
  }
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost})
    if (!std::isfinite(cost) || cost < 0) {
      unsupported("invalid macro native costs");
      return finish();
    }
  if (!charge(work(rotations.size() + 1)))
    return finish();
  QubitSet support;
  for (const auto &gate : rotations) {
    if ((gate.kind != Kind::Rotate && gate.kind != Kind::Phase) ||
        (!gate.angle.symbol && gate.angle.denominator <= 0) ||
        (gate.kind == Kind::Rotate &&
         (!gate.pauli.x.fits(width) || !gate.pauli.z.fits(width)))) {
      unsupported("macro source must contain valid rotations and phases");
      return finish();
    }
    if (gate.kind == Kind::Rotate)
      support |= gate.pauli.x | gate.pauli.z;
  }
  if (support.count() > std::min(limits.maxActiveWires, 8u)) {
    unsupported("macro active support exceeds its explicit search limit");
    return finish();
  }
  const std::vector<Pauli> noObservables;
  const std::vector<unsigned> noReadouts;
  MeasuredProblem source{width, rotations, noObservables, noReadouts};
  auto bootstrap = buildBootstrapPrefix(source, options, &budget);
  if (!bootstrap) {
    if (bootstrap.status == BootstrapStatus::Exhausted)
      exhaust(bootstrap.error.c_str());
    else
      unsupported(bootstrap.error.c_str());
    return finish();
  }
  result.candidates.push_back({std::move(bootstrap.gates), {}});
  result.completedProposals = 1;
  if (limits.maxCandidates == 1 || !limits.maxRounds || !support)
    return finish();

  // At most 109 actions on eight wires, each at most three native gates.
  // This allocation is independent of physical wire indices.
  const unsigned active = support.count();
  const unsigned actionBound = 1 + 3 * active + 3 * active * (active - 1) / 2;
  if (!charge(work(4 * uint64_t(actionBound) + active)))
    return finish();
  std::vector<unsigned> wires;
  for (unsigned q : support.set_bits())
    wires.push_back(q);
  std::vector<std::vector<Gate>> actions(1); // Identity permits single actions.
  auto action = [&](Gate gate) {
    actions.push_back(nativeWord({gate}, options));
  };
  for (unsigned q : wires)
    for (Kind kind : {Kind::H, Kind::S, Kind::Sdg})
      action({kind, q});
  for (size_t i = 0; i < wires.size(); ++i)
    for (size_t j = i + 1; j < wires.size(); ++j) {
      action({Kind::CX, wires[i], wires[j]});
      action({Kind::CX, wires[j], wires[i]});
      action({Kind::CZ, wires[i], wires[j]});
    }
  result.actions = actions.size();
  struct State {
    std::vector<Gate> remaining, prefix, exit;
    size_t next = 0;
    CostRank rank{};
    uint64_t ordinal = 0;
  };
  if (!charge(work(rotations.size() + 1)))
    return finish();
  std::vector<State> beam{{rotations, {}, {}, 0, {}, 0}};
  const unsigned beamWidth = std::min(limits.beamWidth, 8u);
  const unsigned candidateCap = std::min(limits.maxCandidates, 16u);
  const unsigned rounds = std::min(limits.maxRounds, 4u);
  const uint64_t pairCap = std::min<uint64_t>(limits.maxActionPairs, 65536);
  uint64_t ordinal = 0;
  struct Ranked {
    CostRank rank;
    uint64_t ordinal;
    MacroImplementation implementation;
  };
  std::vector<Ranked> ranked;

  auto carry = [&](State &state, const std::vector<Gate> &word) {
    if (word.size() > options.maxGates - state.prefix.size())
      return false;
    if (!charge(work(2 * word.size() +
                     word.size() * (state.remaining.size() - state.next))))
      return false;
    for (const auto &gate : word) {
      state.prefix.push_back(gate);
      state.exit.push_back(gate);
      for (size_t i = state.next; i < state.remaining.size(); ++i)
        if (state.remaining[i].kind == Kind::Rotate)
          state.remaining[i].pauli =
              tcount::detail::conjugate(state.remaining[i].pauli, gate);
    }
    return true;
  };
  auto drain = [&](State &state) {
    while (state.next < state.remaining.size()) {
      const auto &gate = state.remaining[state.next];
      if (gate.kind == Kind::Rotate && gate.pauli.weight() > 1)
        break;
      if (!charge(work(2)))
        return false;
      std::vector<Gate> singleton{gate};
      MeasuredProblem one{width, singleton, noObservables, noReadouts};
      Options tailOptions = options;
      tailOptions.maxGates = options.maxGates - state.prefix.size();
      auto emitted = buildBootstrapPrefix(one, tailOptions, &budget);
      if (!emitted) {
        if (budget.stopped())
          exhaust("macro eager emission work budget exhausted");
        return false;
      }
      if (!charge(work(emitted.gates.size())))
        return false;
      state.prefix.insert(state.prefix.end(), emitted.gates.begin(),
                          emitted.gates.end());
      ++state.next;
    }
    return true;
  };
  auto complete =
      [&](const State &state) -> std::optional<MacroImplementation> {
    if (!charge(work(state.remaining.size() - state.next + 1)))
      return std::nullopt;
    std::vector<Gate> rest(state.remaining.begin() + state.next,
                           state.remaining.end());
    Options tailOptions = options;
    tailOptions.maxGates = options.maxGates - state.prefix.size();
    MeasuredProblem tail{width, rest, noObservables, noReadouts};
    auto completed = buildBootstrapPrefix(tail, tailOptions, &budget);
    if (!completed) {
      if (budget.stopped())
        exhaust("macro proposal completion work budget exhausted");
      return std::nullopt;
    }
    if (!charge(work(state.prefix.size() + state.exit.size() +
                     completed.gates.size() + 1)))
      return std::nullopt;
    MacroImplementation proposal{state.prefix, state.exit};
    proposal.body.insert(proposal.body.end(), completed.gates.begin(),
                         completed.gates.end());
    ++result.completedProposals;
    return proposal;
  };
  auto score =
      [&](const MacroImplementation &candidate) -> std::optional<CostRank> {
    std::optional<CostRank> rank;
    if (scorer) {
      SearchBudget scoring(budget, budget.remaining());
      rank = scorer(candidate, scoring);
      if (scoring.stopped())
        exhaust("macro boundary-completion scoring budget exhausted");
    } else {
      // Completion may exceed the body cap; do not rank an unavailable closed
      // circuit as though its exit were free. A caller with an open boundary
      // can instead supply its own concrete completion scorer.
      if (candidate.exitCliffords.size() >
          options.maxGates - candidate.body.size())
        return std::nullopt;
      uint64_t size = candidate.body.size() + candidate.exitCliffords.size();
      if (!charge(llvm::SaturatingAdd<uint64_t>(
              width, llvm::SaturatingMultiply<uint64_t>(work(size + 1), 16))))
        return std::nullopt;
      auto closed = candidate.body;
      for (auto gate = candidate.exitCliffords.rbegin();
           gate != candidate.exitCliffords.rend(); ++gate)
        closed.push_back(macro_detail::inverse(*gate));
      rank = circuitScore(width, closed, options);
    }
    if (rank && std::any_of(rank->begin(), rank->end(), [](double value) {
          return !std::isfinite(value);
        })) {
      unsupported("macro completion score must be finite");
      return std::nullopt;
    }
    return rank;
  };
  auto publish = [&](MacroImplementation candidate, const CostRank &rank) {
    for (const auto &old : result.candidates) {
      if (!charge(
              work(candidate.body.size() + candidate.exitCliffords.size() + 1)))
        return false;
      if (macro_detail::sameImplementation(old, candidate))
        return true;
    }
    ranked.push_back({rank, ordinal++, std::move(candidate)});
    std::stable_sort(
        ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
          return std::tie(a.rank, a.ordinal) < std::tie(b.rank, b.ordinal);
        });
    if (ranked.size() >= candidateCap)
      ranked.resize(candidateCap - 1);
    // Publish the whole shortlist transactionally. Failed copying charges
    // leave every previously returned complete implementation intact.
    if (!charge(work(result.candidates[0].body.size() + 1)))
      return false;
    std::vector<MacroImplementation> next{result.candidates[0]};
    for (const auto &entry : ranked) {
      if (!charge(work(entry.implementation.body.size() +
                       entry.implementation.exitCliffords.size() + 1)))
        return false;
      next.push_back(entry.implementation);
    }
    result.candidates = std::move(next);
    return true;
  };

  for (unsigned round = 0; round < rounds && !beam.empty(); ++round) {
    std::vector<State> next;
    for (const auto &state : beam) {
      // Cyclic diagonals expose every first action before revisiting it. Tiny
      // action sets are exhaustive within a layer; a cap bounds larger sets.
      for (size_t pair = 1; pair < actions.size() * actions.size(); ++pair) {
        if (result.expandedPairs >= pairCap) {
          exhaust(
              "macro action-pair limit reached; complete proposals retained");
          return finish();
        }
        ++result.expandedPairs;
        size_t a = pair % actions.size();
        size_t b = (pair / actions.size() + a) % actions.size();
        if (!charge(work(state.remaining.size() + state.prefix.size() +
                         state.exit.size() + 1)))
          return finish();
        State child = state;
        child.ordinal = ordinal++;
        result.peakStates = std::max<unsigned>(result.peakStates,
                                               beam.size() + next.size() + 1);
        if (!carry(child, actions[a]) || !drain(child) ||
            !carry(child, actions[b]) || !drain(child)) {
          if (result.status != MacroStatus::Complete)
            return finish();
          continue;
        }
        auto proposal = complete(child);
        if (result.status != MacroStatus::Complete)
          return finish();
        if (!proposal)
          continue;
        auto rank = score(*proposal);
        if (result.status != MacroStatus::Complete)
          return finish();
        if (!rank)
          continue;
        if (!publish(std::move(*proposal), *rank))
          return finish();
        if (child.next == child.remaining.size())
          continue;
        child.rank = *rank;
        bool duplicate = false;
        for (const auto &old : next) {
          if (!charge(work(child.remaining.size() + child.prefix.size() +
                           child.exit.size() + 1)))
            return finish();
          if (old.next == child.next &&
              macro_detail::sameWord(old.remaining, child.remaining) &&
              macro_detail::sameWord(old.prefix, child.prefix) &&
              macro_detail::sameWord(old.exit, child.exit)) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate) {
          next.push_back(std::move(child));
          std::stable_sort(next.begin(), next.end(),
                           [](const auto &a, const auto &b) {
                             return std::tie(a.rank, a.ordinal) <
                                    std::tie(b.rank, b.ordinal);
                           });
          if (next.size() > beamWidth)
            next.resize(beamWidth);
        }
      }
    }
    beam = std::move(next);
  }
  return finish();
}
} // namespace cudaq::pauli::graph::detail
