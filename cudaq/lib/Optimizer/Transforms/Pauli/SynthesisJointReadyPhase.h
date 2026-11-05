/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 * The full license can be found in the LICENSE file at the top level of this  *
 * distribution.                                                               *
 ******************************************************************************/

#pragma once

#include "SynthesisBootstrap.h"
#include "SynthesisOpen.h"
#include <array>
#include <functional>
#include <numeric>

namespace cudaq::pauli::graph::detail {

struct JointPhaseLimits {
  unsigned maxActiveWires = 16, maxRotations = 64, maxRank = 3;
  unsigned maxCandidates = 4, maxBasisTrials = 16, maxDiagonalizers = 4;
  /// One retains the original first-rank layout; larger values additionally
  /// try injective pivots within each generator's natural physical support.
  unsigned maxCoordinateLayouts = 1, maxContextScores = 32;
  uint64_t maxWork = 20000000, maxLiveBytes = 32 * 1024 * 1024;
  uint64_t maxStates = 21504;
};

/// Full quantum relation, INCLUDING scalar phase: body = exitCliffords *
/// source. Both words are in circuit order. Exits contain actual native
/// Cliffords; scalar input phases remain in the body. No measurement or
/// zero-state assumption enters this contract. Publication checks this
/// relation, but the vectors remain mutable proposals: callers must check the
/// complete measured program after carrying an exit through every remaining
/// quantum obligation.
struct JointPhaseImplementation {
  std::vector<Gate> body, exitCliffords;
};

/// Heuristic ranking only, before body-cost truncation. The argument is a
/// complete proposal, not yet a certified source-bound value. Callers charge
/// transport/completion/scoring to the supplied child; nullopt skips a
/// proposal. The solver independently verifies every selected survivor
/// afterwards.
using JointPhaseScorer = std::function<std::optional<CostRank>(
    const JointPhaseImplementation &, SearchBudget &)>;

struct JointPhaseCandidates {
  /// Slot zero is the checked, closed mechanical baseline. maxCandidates==1
  /// deliberately returns only this baseline. Other slots are heuristically
  /// ranked by native body cost with distinct signed exits preferred; this is
  /// NOT dominance of different boundary frames or a complete-program bound.
  std::vector<JointPhaseImplementation> candidates;
  OpenRegionStatus status = OpenRegionStatus::Complete;
  std::string error;
  uint64_t work = 0, statesVisited = 0, completedProposals = 0;
  unsigned rank = 0, basisTrials = 0, diagonalizers = 0;
  unsigned coordinateLayouts = 0, contextScores = 0;
};

namespace joint_phase_detail {
inline uint64_t add(uint64_t a, uint64_t b) {
  return llvm::SaturatingAdd<uint64_t>(a, b);
}
inline uint64_t mul(uint64_t a, uint64_t b) {
  return llvm::SaturatingMultiply<uint64_t>(a, b);
}
inline bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}
inline bool sameAxis(const Pauli &a, const Pauli &b) {
  return a.x == b.x && a.z == b.z;
}
inline unsigned row(unsigned code, unsigned i, unsigned rank) {
  return (code >> (i * rank)) & ((1u << rank) - 1);
}
inline unsigned rowRank(unsigned code, unsigned rank) {
  std::array<unsigned, 3> pivots{};
  unsigned count = 0;
  for (unsigned i = 0; i < rank; ++i) {
    unsigned value = row(code, i, rank);
    for (int bit = int(rank) - 1; bit >= 0; --bit)
      if (value & (1u << bit)) {
        if (pivots[bit])
          value ^= pivots[bit];
        else {
          pivots[bit] = value;
          ++count;
          break;
        }
      }
  }
  return count;
}
inline unsigned resident(unsigned code, unsigned rank, unsigned demanded) {
  unsigned result = 0;
  for (unsigned q = 0; q < rank; ++q)
    result |= 1u << (row(code, q, rank) - 1);
  return result & demanded;
}
struct WalkEntry {
  int32_t parent = -2;
  uint16_t distance = 0;
  uint8_t control = 0, target = 0;
};
struct Diagonalizer {
  std::vector<Gate> gates;
  std::vector<unsigned> wires;
  CostRank cost;
};

/// At most 16^3 injective assignments, without copying any circuit. Missing
/// support is forbidden for the added layouts; Z pivots and less shared wires
/// win ties before physical index. The original layout always stays first.
inline std::vector<std::vector<unsigned>>
naturalLayouts(const std::vector<Pauli> &basis,
               const std::vector<unsigned> &original, unsigned width,
               unsigned cap) {
  struct Layout {
    std::vector<unsigned> wires;
    unsigned basisChanges = 0, interference = 0;
  };
  std::vector<std::vector<unsigned>> result{original};
  if (cap <= 1)
    return result;
  std::vector<Layout> best;
  std::vector<unsigned> wires;
  auto visit = [&](auto &&self, unsigned at, unsigned changes,
                   unsigned interference) -> void {
    if (at == basis.size()) {
      if (wires == original)
        return;
      best.push_back({wires, changes, interference});
      std::stable_sort(
          best.begin(), best.end(), [](const auto &a, const auto &b) {
            return std::tie(a.basisChanges, a.interference, a.wires) <
                   std::tie(b.basisChanges, b.interference, b.wires);
          });
      if (best.size() >= cap)
        best.pop_back();
      return;
    }
    for (unsigned q = 0; q < width; ++q) {
      unsigned axis = basis[at].axis(q);
      if (!axis || std::find(wires.begin(), wires.end(), q) != wires.end())
        continue;
      unsigned shared = 0;
      for (unsigned row = 0; row < basis.size(); ++row)
        shared += row != at && basis[row].axis(q) != 0;
      wires.push_back(q);
      self(self, at + 1,
           changes + (axis == 2   ? 0
                      : axis == 1 ? 1
                                  : 2),
           interference + shared);
      wires.pop_back();
    }
  };
  visit(visit, 0, 0, 0);
  for (auto &layout : best)
    result.push_back(std::move(layout.wires));
  return result;
}
} // namespace joint_phase_detail

/// Jointly visit parity demands using arbitrary CNOT row operations, rather
/// than committing each parity to one fixed star. For rank r<=3 the state is
/// (A in GL(r,2), visited demands), with at most 168*128=21,504 states. Every
/// transition has the same native CX-lowering cost, so BFS finds a shortest
/// parity walk for EACH exit matrix in this fixed diagonalized basis. It does
/// not optimize unrestricted Clifford synthesis, basis choice or final readout.
///
/// Candidate generators include every nonzero element of the source span,
/// ordered by support; this includes low-support differences of an affine
/// packet. Basis trials, diagonalizers, native gates, live storage and work are
/// separately bounded. The baseline is completed first; any later exhaustion
/// preserves all published candidates. InvalidProof clears the entire result.
inline JointPhaseCandidates
solveJointReadyPhaseRegion(unsigned width, const std::vector<Gate> &source,
                           const Options &options,
                           const JointPhaseLimits &limits, SearchBudget &shared,
                           const JointPhaseScorer &scorer = {}) {
  namespace jp = joint_phase_detail;
  JointPhaseCandidates result;
  auto fail = [&](OpenRegionStatus status, const char *message) {
    result.status = status;
    result.error = message;
    if (status == OpenRegionStatus::InvalidProof)
      result.candidates.clear();
  };
  if (!width || width > std::min(limits.maxActiveWires, 16u) ||
      source.size() > std::min(limits.maxRotations, 64u) || !limits.maxRank ||
      !limits.maxCandidates || !options.maxGates ||
      (!options.nativeCX && !options.nativeCZ) ||
      static_cast<unsigned>(options.objective) >
          static_cast<unsigned>(SynthesisObjective::TDepth)) {
    fail(OpenRegionStatus::Unsupported,
         "invalid joint-phase dimensions/profile");
    return result;
  }
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost})
    if (!std::isfinite(cost) || cost < 0) {
      fail(OpenRegionStatus::Unsupported, "invalid joint-phase native cost");
      return result;
    }
  auto time = shared.timeRemainingMs();
  if (shared.remaining() < 2 || !limits.maxWork || (time && *time < 2)) {
    fail(OpenRegionStatus::Exhausted, "no joint-phase resource reserve");
    return result;
  }
  SearchBudget budget(shared, std::min(limits.maxWork, shared.remaining() / 2),
                      time ? *time / 2 : 0);
  auto finish = [&]() {
    result.work = budget.work();
    return std::move(result);
  };
  auto take = [&](uint64_t work) {
    if (budget.take(work))
      return true;
    fail(OpenRegionStatus::Exhausted, "joint-phase work budget exhausted");
    return false;
  };
  const unsigned candidateCap = std::min(limits.maxCandidates, 8u);
  const unsigned basisCap = std::min(limits.maxBasisTrials, 168u);
  const unsigned diagonalizerCap = std::min(limits.maxDiagonalizers, 8u);
  const unsigned layoutCap = std::clamp(limits.maxCoordinateLayouts, 1u, 4u);
  const unsigned contextCap = std::min(limits.maxContextScores, 128u);
  unsigned contextQuota = 0;
  const unsigned diagonalGateCap = 8 * width * width + 8 * width + 16;
  const unsigned gateCap =
      std::min<unsigned>(options.maxGates, source.size() * (8 * width + 3) +
                                               diagonalGateCap + 128);
  // Includes every retained/candidate word, proof target, readout scratch,
  // complete DP table/queue, signed exit rows and span/basis bookkeeping.
  uint64_t gateBytes = sizeof(Gate) + 2 * sizeof(uint64_t);
  uint64_t slots = jp::add(jp::mul(2 * candidateCap + 6, gateCap),
                           jp::mul(diagonalizerCap + 4, diagonalGateCap));
  uint64_t bytes = jp::add(
      jp::mul(slots, gateBytes),
      jp::add(jp::mul(21504, sizeof(jp::WalkEntry) + sizeof(uint32_t)), 65536));
  const uint64_t baselineBytes = jp::add(
      jp::mul(3 * uint64_t(gateCap) + source.size() + 32, gateBytes), 16384);
  if (baselineBytes > limits.maxLiveBytes) {
    fail(OpenRegionStatus::Exhausted,
         "joint-phase baseline storage limit exceeded");
    return finish();
  }
  if (!take(jp::mul(source.size() + 1, source.size() + 16)))
    return finish();
  std::vector<Pauli> span(1);
  QubitSet active;
  for (const auto &gate : source) {
    // Bound even ignored scalar fields before any copy. Scalar gates are
    // subsequently normalized to Gate::phase, preserving only their angle.
    if ((gate.kind != Kind::Rotate && gate.kind != Kind::Phase) ||
        !gate.pauli.x.fits(width) || !gate.pauli.z.fits(width) ||
        (!gate.angle.symbol && gate.angle.denominator <= 0) ||
        (gate.kind == Kind::Rotate && !gate.pauli.weight())) {
      fail(OpenRegionStatus::Unsupported, "invalid joint-phase source gate");
      return finish();
    }
    if (gate.kind == Kind::Phase)
      continue;
    for (const auto &other : span)
      if (gate.pauli.anti(other)) {
        fail(OpenRegionStatus::Unsupported,
             "joint-phase region does not commute");
        return finish();
      }
    active |= gate.pauli.x | gate.pauli.z;
    if (std::none_of(span.begin(), span.end(), [&](const Pauli &p) {
          return jp::sameAxis(p, gate.pauli);
        })) {
      if (result.rank == std::min(limits.maxRank, 3u)) {
        fail(OpenRegionStatus::Unsupported, "joint-phase rank limit exceeded");
        return finish();
      }
      auto oldSize = span.size();
      for (size_t i = 0; i < oldSize; ++i)
        span.push_back({span[i].x ^ gate.pauli.x, span[i].z ^ gate.pauli.z});
      ++result.rank;
    }
  }
  if (!take(source.size() + 1))
    return finish();
  std::vector<Gate> normalized;
  normalized.reserve(source.size());
  for (const auto &gate : source)
    normalized.push_back(gate.kind == Kind::Phase ? Gate::phase(gate.angle)
                                                  : gate);
  Options bounded = options;
  bounded.maxGates = gateCap;
  std::vector<std::vector<Pauli>> exitImages;
  std::vector<CostRank> costs;
  auto publish = [&](JointPhaseImplementation proposal, bool baseline) {
    ++result.completedProposals;
    if (!take(jp::mul(proposal.exitCliffords.size() + 1, 2 * width + 1) +
              16 * (proposal.body.size() + width + 1)))
      return false;
    if (proposal.body.size() > gateCap ||
        proposal.exitCliffords.size() > gateCap)
      return true;
    auto cost = circuitScore(width, proposal.body, options);
    if (!baseline && scorer) {
      if (!contextQuota || result.contextScores == contextCap)
        return true;
      --contextQuota;
      ++result.contextScores;
      auto remainingTime = budget.timeRemainingMs();
      if (budget.remaining() < 2 || (remainingTime && *remainingTime < 2)) {
        fail(OpenRegionStatus::Exhausted,
             "joint-phase context reserve exhausted");
        return false;
      }
      SearchBudget context(budget, budget.remaining() / 2,
                           remainingTime ? *remainingTime / 2 : 0);
      auto ranked = scorer(proposal, context);
      if (context.stopped()) {
        fail(OpenRegionStatus::Exhausted,
             "joint-phase context scoring exhausted");
        return false;
      }
      if (!ranked)
        return true;
      cost = *ranked;
    }
    if (std::any_of(cost.begin(), cost.end(),
                    [](double value) { return !std::isfinite(value); })) {
      if (baseline) {
        fail(OpenRegionStatus::Unsupported,
             "joint-phase baseline cost overflow");
        return false;
      }
      return true;
    }
    std::vector<Pauli> images;
    for (unsigned q = 0; q < width; ++q) {
      images.push_back({QubitSet::single(q), {}});
      images.push_back({{}, QubitSet::single(q)});
    }
    for (const auto &gate : proposal.exitCliffords) {
      if (!region_detail::validClifford(width, gate) ||
          !open_detail::validNative(width, gate, options)) {
        fail(OpenRegionStatus::InvalidProof, "invalid joint-phase exit gate");
        return false;
      }
      for (auto &image : images)
        image = tcount::detail::conjugate(image, gate);
    }
    size_t replace = result.candidates.size();
    if (!baseline) {
      for (size_t i = 0; i < result.candidates.size(); ++i)
        if (jp::sameWord(proposal.body, result.candidates[i].body) &&
            jp::sameWord(proposal.exitCliffords,
                         result.candidates[i].exitCliffords))
          return true;
      // Same signed exits are a diversity heuristic only. Whichever actual
      // word survives still receives its own exact scalar/full-unitary proof.
      for (size_t i = 1; i < exitImages.size(); ++i)
        if (images == exitImages[i]) {
          if (!(cost < costs[i]))
            return true;
          replace = i;
          break;
        }
      if (replace == result.candidates.size() && replace == candidateCap) {
        replace = 1;
        for (size_t i = 2; i < costs.size(); ++i)
          if (costs[replace] < costs[i])
            replace = i;
        if (!(cost < costs[replace]))
          return true;
      }
    }
    uint64_t proofWork =
        jp::add(jp::mul(16, jp::mul(source.size() + 1, source.size() + 1)),
                jp::mul(8, jp::mul(normalized.size() + proposal.body.size() +
                                       proposal.exitCliffords.size() + 1,
                                   uint64_t(width) * width + 1)));
    if (!take(proofWork))
      return false;
    for (const auto &gate : proposal.body)
      if (!open_detail::validNative(width, gate, options)) {
        fail(OpenRegionStatus::InvalidProof, "invalid joint-phase body gate");
        return false;
      }
    auto target = normalized;
    target.insert(target.end(), proposal.exitCliffords.begin(),
                  proposal.exitCliffords.end());
    if (!graph::equivalent(width, target, proposal.body)) {
      fail(OpenRegionStatus::InvalidProof,
           "joint-phase full-unitary replay failed");
      return false;
    }
    if (replace == result.candidates.size()) {
      result.candidates.push_back(std::move(proposal));
      exitImages.push_back(std::move(images));
      costs.push_back(cost);
    } else {
      result.candidates[replace] = std::move(proposal);
      exitImages[replace] = std::move(images);
      costs[replace] = cost;
    }
    return true;
  };
  const std::vector<Pauli> noObservables;
  const std::vector<unsigned> noReadouts;
  auto baseline = buildBootstrapPrefix(
      {width, normalized, noObservables, noReadouts}, bounded, &budget);
  if (!baseline) {
    fail(baseline.status == BootstrapStatus::Exhausted
             ? OpenRegionStatus::Exhausted
             : OpenRegionStatus::Unsupported,
         "joint-phase mechanical baseline unavailable");
    return finish();
  }
  if (!publish({std::move(baseline.gates), {}}, true))
    return finish();
  if (!result.rank || candidateCap == 1 || !basisCap || !diagonalizerCap)
    return finish();
  // Optional storage cannot prevent a smaller, already feasible baseline.
  if (bytes > limits.maxLiveBytes) {
    fail(OpenRegionStatus::Exhausted,
         "joint-phase exploration storage limit exceeded");
    return finish();
  }

  const unsigned rank = result.rank, codes = 1u << (rank * rank);
  const unsigned masks = 1u << ((1u << rank) - 1);
  if (!take(codes * (rank * rank + 1) + 512))
    return finish();
  std::vector<unsigned> group;
  std::array<int, 512> groupIndex;
  groupIndex.fill(-1);
  unsigned identity = 0;
  for (unsigned q = 0; q < rank; ++q)
    identity |= (1u << q) << (q * rank);
  for (unsigned code = 0; code < codes; ++code)
    if (jp::rowRank(code, rank) == rank) {
      groupIndex[code] = group.size();
      group.push_back(code);
    }
  const size_t stateCount = group.size() * masks;
  if (stateCount > std::min<uint64_t>(limits.maxStates, 21504)) {
    fail(OpenRegionStatus::Exhausted, "joint-phase state-table limit exceeded");
    return finish();
  }
  std::vector<unsigned> basisOrders = group;
  auto basisWeight = [&](unsigned code) {
    unsigned weight = 0;
    for (unsigned q = 0; q < rank; ++q)
      weight += span[jp::row(code, q, rank)].weight();
    return weight;
  };
  if (!take(group.size() * 64))
    return finish();
  std::stable_sort(
      basisOrders.begin(), basisOrders.end(), [&](unsigned a, unsigned b) {
        return std::pair{basisWeight(a), a} < std::pair{basisWeight(b), b};
      });
  std::vector<unsigned> wires;
  for (unsigned wire : active.set_bits()) {
    wires.push_back(wire);
    if (wires.size() == rank)
      break;
  }
  std::vector<jp::Diagonalizer> diagonalizers;
  for (unsigned code : basisOrders) {
    if (result.basisTrials == basisCap)
      break;
    if (!take(rank + 1))
      return finish();
    ++result.basisTrials;
    std::vector<Pauli> basis;
    for (unsigned q = 0; q < rank; ++q)
      basis.push_back(span[jp::row(code, q, rank)]);
    if (layoutCap > 1 &&
        !take(64 * jp::mul(width + 1, jp::mul(width + 1, width + 1))))
      return finish();
    auto layouts = jp::naturalLayouts(basis, wires, width, layoutCap);
    for (const auto &coordinates : layouts) {
      ++result.coordinateLayouts;
      auto tails = generateReadoutCandidates(
          width, basis, coordinates, bounded,
          {1, std::min<uint64_t>(1000000, budget.remaining())}, budget);
      if (tails.status == ReadoutStatus::InvalidProof) {
        fail(OpenRegionStatus::InvalidProof,
             "joint-phase diagonalizer proof failed");
        return finish();
      }
      for (const auto &tail : tails.candidates) {
        if (!take(16 * (tail.gates().size() + width + 1)))
          return finish();
        if (tail.gates().size() > diagonalGateCap)
          continue;
        auto cost = circuitScore(width, tail.gates(), options);
        if (std::any_of(cost.begin(), cost.end(),
                        [](double value) { return !std::isfinite(value); }))
          continue;
        bool duplicate = std::any_of(
            diagonalizers.begin(), diagonalizers.end(), [&](const auto &old) {
              return old.wires == coordinates &&
                     jp::sameWord(old.gates, tail.gates());
            });
        if (duplicate)
          continue;
        diagonalizers.push_back({tail.gates(), coordinates, cost});
        std::stable_sort(
            diagonalizers.begin(), diagonalizers.end(),
            [](const auto &a, const auto &b) { return a.cost < b.cost; });
        if (diagonalizers.size() > diagonalizerCap)
          diagonalizers.pop_back();
      }
      if (tails.status == ReadoutStatus::Exhausted && budget.stopped()) {
        fail(OpenRegionStatus::Exhausted,
             "joint-phase diagonalizer budget exhausted");
        return finish();
      }
    }
  }
  for (const auto &diagonalizer : diagonalizers) {
    ++result.diagonalizers;
    const auto &wires = diagonalizer.wires;
    contextQuota = contextCap / diagonalizers.size() +
                   (result.diagonalizers <= contextCap % diagonalizers.size());
    if (!take((diagonalizer.gates.size() + 1) * (source.size() + 1) +
              stateCount * 2))
      return finish();
    std::array<std::vector<Angle>, 8> demands;
    std::vector<Gate> phases;
    unsigned demanded = 0;
    for (const auto &gate : normalized) {
      if (gate.kind == Kind::Phase) {
        phases.push_back(gate);
        continue;
      }
      auto axis = gate.pauli;
      for (const auto &clifford : diagonalizer.gates)
        axis = tcount::detail::conjugate(axis, clifford);
      unsigned mask = 0;
      QubitSet expected;
      for (unsigned q = 0; q < rank; ++q)
        if (axis.z.test(wires[q])) {
          mask |= 1u << q;
          expected |= QubitSet::single(wires[q]);
        }
      if (axis.x || axis.z != expected || !mask) {
        fail(OpenRegionStatus::InvalidProof,
             "joint-phase basis failed to diagonalize span");
        return finish();
      }
      demands[mask].push_back(axis.sign ? gate.angle.negated() : gate.angle);
      demanded |= 1u << (mask - 1);
    }
    std::vector<jp::WalkEntry> table(stateCount);
    std::vector<uint32_t> queue;
    queue.reserve(stateCount);
    unsigned start =
        groupIndex[identity] * masks + jp::resident(identity, rank, demanded);
    table[start].parent = -1;
    queue.push_back(start);
    for (size_t head = 0; head < queue.size(); ++head) {
      if (!take(rank * rank * 4 + 1))
        return finish();
      ++result.statesVisited;
      unsigned at = queue[head], code = group[at / masks], seen = at % masks;
      for (unsigned control = 0; control < rank; ++control)
        for (unsigned target = 0; target < rank; ++target) {
          if (control == target)
            continue;
          unsigned nextCode =
              code ^ (jp::row(code, control, rank) << (target * rank));
          unsigned next = groupIndex[nextCode] * masks +
                          (seen | jp::resident(nextCode, rank, demanded));
          if (table[next].parent != -2)
            continue;
          table[next] = {int32_t(at), uint16_t(table[at].distance + 1),
                         uint8_t(control), uint8_t(target)};
          queue.push_back(next);
        }
    }
    std::vector<unsigned> endpoints;
    if (!take(group.size() * 16))
      return finish();
    for (unsigned i = 0; i < group.size(); ++i)
      if (table[i * masks + demanded].parent != -2)
        endpoints.push_back(i * masks + demanded);
    std::stable_sort(endpoints.begin(), endpoints.end(),
                     [&](unsigned a, unsigned b) {
                       return std::pair{table[a].distance, a} <
                              std::pair{table[b].distance, b};
                     });
    for (unsigned endpoint : endpoints) {
      if (scorer && !contextQuota)
        break;
      unsigned distance = table[endpoint].distance;
      uint64_t gates = diagonalizer.gates.size() + source.size() +
                       uint64_t(distance) * (options.nativeCX ? 1 : 3);
      if (gates > gateCap)
        continue;
      if (!take(8 * (gates + distance + 1)))
        return finish();
      std::vector<unsigned> path;
      for (unsigned at = endpoint; table[at].parent >= 0; at = table[at].parent)
        path.push_back(at);
      std::reverse(path.begin(), path.end());
      JointPhaseImplementation candidate{phases, diagonalizer.gates};
      candidate.body.insert(candidate.body.end(), diagonalizer.gates.begin(),
                            diagonalizer.gates.end());
      unsigned seen = 0;
      auto emit = [&](unsigned code) {
        for (unsigned q = 0; q < rank; ++q) {
          unsigned mask = jp::row(code, q, rank), bit = 1u << (mask - 1);
          if (!(seen & bit)) {
            for (const auto &angle : demands[mask])
              candidate.body.push_back(
                  Gate::rotation({{}, QubitSet::single(wires[q])}, angle));
            seen |= bit;
          }
        }
      };
      emit(identity);
      for (unsigned at : path) {
        auto native = nativeWord(
            {{Kind::CX, wires[table[at].control], wires[table[at].target]}},
            options);
        candidate.body.insert(candidate.body.end(), native.begin(),
                              native.end());
        candidate.exitCliffords.insert(candidate.exitCliffords.end(),
                                       native.begin(), native.end());
        emit(group[at / masks]);
      }
      if ((seen & demanded) != demanded) {
        fail(OpenRegionStatus::InvalidProof,
             "joint-phase parity walk missed a demand");
        return finish();
      }
      if (!publish(std::move(candidate), false))
        return finish();
    }
  }
  // Slot zero never moves. Stable native-cost ordering applies only to the
  // optional complete proposals, leaving actual completion to the caller.
  if (!take(result.candidates.size() * 16))
    return finish();
  std::vector<size_t> order(result.candidates.size());
  std::iota(order.begin(), order.end(), 0);
  if (order.size() > 1)
    std::stable_sort(order.begin() + 1, order.end(),
                     [&](size_t a, size_t b) { return costs[a] < costs[b]; });
  std::vector<JointPhaseImplementation> sorted;
  for (size_t i : order)
    sorted.push_back(std::move(result.candidates[i]));
  result.candidates = std::move(sorted);
  return finish();
}
} // namespace cudaq::pauli::graph::detail
