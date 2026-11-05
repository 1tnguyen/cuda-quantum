/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/
// Source-only composition of exact dependency-ready phase packets.
#pragma once

#include "SynthesisJointReadyPhase.h"
#include "SynthesisCompactTransport.h"
#include "SynthesisDependencies.h"
#include "SynthesisCost.h"
#include <set>

namespace cudaq::pauli::graph::detail {
struct JointPhasePlannerLimits {
  unsigned maxPlans = 4, beamWidth = 2, maxReadyNodes = 32, maxPacketSeeds = 4;
  unsigned maxCompleteScores = 8;
  // One separately bounded score for the complete contextual greedy route;
  // it never consumes any of the eight optional beam completion scores.
  unsigned maxContextCompleteScores = 1;
  bool contextScoring = false;
  // At most 24 solver endpoint callbacks plus four retained-result rescoring
  // calls per packet, and this separate whole-planner cap. Wide readouts must
  // fit the fixed callback work slice; exhaustion never stops the old chain.
  unsigned maxContextCalls = 4096, maxTraceEntries = 128;
  uint64_t maxContextWork = 1000000;
  uint64_t maxWork = 1000000000, maxLocalWork = 20000000;
  uint64_t maxLiveBytes = 128ull << 20, maxExpansions = 4096;
  JointPhaseLimits region;
};
struct JointPhasePacketTrace {
  uint64_t parent = 0, child = 0, exitSignature = 0;
  bool greedy = false, contextualGreedy = false;
  bool contextRanked = false, exactReadout = false;
  std::vector<unsigned> packet, physical;
  unsigned remaining = 0, bodyEntries = 0, exitEntries = 0, physicalSupport = 0;
  double prefixCost = 0, bodyCost = 0, residualCost = 0, readoutCost = 0;
};
struct JointPhaseMemoryPhase {
  unsigned gateCap = 0, stateSlots = 0, peakStates = 0;
  uint64_t retainedBytes = 0, reservedBytes = 0;
  bool admitted = false;
};
struct JointPhasePlannerCandidates {
  std::vector<RegionalPlan> candidates;
  OpenRegionStatus status = OpenRegionStatus::Complete;
  std::string error;
  uint64_t work = 0, expansions = 0, completedProposals = 0;
  uint64_t transportedRows = 0, localExhaustions = 0, localWork = 0;
  uint64_t localStates = 0, localCompleted = 0;
  unsigned packetsSolved = 0, fallbackPackets = 0, packetRotations = 0;
  unsigned peakStates = 0, completeScores = 0, contextCompleteScores = 0;
  unsigned nativeGateCap = 0, finalScoreFallbackPlans = 0;
  JointPhaseMemoryPhase greedyMemory, contextualMemory, beamMemory;
  uint64_t reservedLiveBytes = 0, contextWork = 0;
  uint64_t dependencyWork = 0, packetWork = 0, stateWork = 0;
  uint64_t transportWork = 0, residualScoreWork = 0;
  uint64_t affectedRotations = 0, affectedObservables = 0;
  unsigned exactContextScores = 0, estimatedContextScores = 0;
  unsigned contextCalls = 0, contextMisses = 0, coordinateLayouts = 0;
  unsigned quotaLimitedPackets = 0, minimumEndpointQuota = 24;
  unsigned maximumEndpointQuota = 0;
  std::vector<JointPhasePacketTrace> trace;
};
namespace joint_plan_detail {
// Dynamic distribution, not a promise about future packet ranks: progress may
// change, so recompute after every packet. The hard route cap remains binding.
inline uint64_t packetCallAllowance(uint64_t remainingCalls, unsigned remaining,
                                    unsigned packetSize, unsigned beamWidth,
                                    bool contextualGreedy) {
  if (!remaining || !packetSize)
    return 0;
  uint64_t packets = (uint64_t(remaining) + packetSize - 1) / packetSize;
  // Finish the current contextual route first. An optional later beam uses
  // whatever remains, and has a valid analytic continuation when none remains.
  uint64_t pending =
      contextualGreedy ? packets : std::max(beamWidth, 1u) * packets;
  return remainingCalls / std::max<uint64_t>(pending, 1);
}
inline unsigned endpointQuota(uint64_t remainingCalls, unsigned remaining,
                              unsigned packetSize, unsigned beamWidth,
                              bool contextualGreedy, bool exactReadout,
                              unsigned requested = 24) {
  uint64_t allowance = packetCallAllowance(
      remainingCalls, remaining, packetSize, beamWidth, contextualGreedy);
  if (exactReadout)
    allowance = allowance > 4 ? allowance - 4 : 0;
  return std::min<uint64_t>(std::min(requested, 24u), allowance);
}
struct State {
  std::vector<Gate> rotations;
  std::vector<unsigned char> active;
  RegionalPlan plan;
  std::vector<Gate> exit;
  unsigned remaining = 0;
  std::vector<double> rotationCosts, readoutCosts;
  double residualCost = 0, readoutPotential = 0;
  double prefixCost = 0, priority = 0;
  uint64_t ordinal = 0;
};
// Same analytical native-lowering cost as evaluateNativeCost, without copying
// the rotation or scanning every occupied wire. Pauli sign changes semantics,
// but not this cost model. These functions do not charge work.
inline double rotationCost(const Gate &gate, const Options &options) {
  if (gate.kind == Kind::Phase)
    return 0;
  auto quarter = tcount::detail::quarters(gate.angle);
  unsigned magnitude = quarter ? std::abs(*quarter % 8) : 0;
  double angle = quarter ? ((magnitude & 1) ? options.tCost : 0) +
                               ((magnitude & 2) ? options.singleQubitCost : 0) +
                               ((magnitude & 4) ? options.singleQubitCost : 0)
                         : options.tCost;
  unsigned x = gate.pauli.x.count();
  unsigned y = gate.pauli.x.intersectionCount(gate.pauli.z);
  unsigned weight = x + gate.pauli.z.count() - y;
  return angle + (2 * x + 2 * y) * options.singleQubitCost +
         2 * (weight ? weight - 1 : 0) *
             (options.nativeCX ? options.cxCost
                               : options.czCost + 2 * options.singleQubitCost);
}

// Independent-row fixed-wire decoder estimate. It deliberately ignores sharing
// and interference among rows: it is a heuristic, NEVER an exact readout cost.
inline double readoutPotential(const Pauli &row, unsigned target,
                               const Options &options) {
  unsigned x = row.x.count(), y = row.x.intersectionCount(row.z);
  unsigned weight = x + row.z.count() - y;
  double entangler = options.nativeCX
                         ? options.cxCost
                         : options.czCost + 2 * options.singleQubitCost;
  double swap = options.nativeSwap ? options.swapCost : 3 * entangler;
  return (x + y + unsigned(row.sign)) * options.singleQubitCost +
         (weight ? weight - 1 : 0) * entangler +
         ((!row.x.test(target) && !row.z.test(target)) ? swap : 0);
}
inline bool touches(const Pauli &row, const std::vector<unsigned> &physical) {
  return std::any_of(physical.begin(), physical.end(), [&](unsigned q) {
    return row.x.test(q) || row.z.test(q);
  });
}
struct BoundaryScore {
  CostRank rank{};
  double residualCost = 0, readoutCost = 0;
  bool exactReadout = false;
};

// Heuristic rank of a realizable identity-exit residual completion. The exact
// readout is checked here, but this score is NOT a certificate for the supplied
// prefix; the whole measured candidate still needs the caller's replay.
inline std::optional<BoundaryScore>
boundaryScore(unsigned width, const std::vector<Gate> &residual,
              const std::vector<Pauli> &observables,
              const std::vector<unsigned> &readouts, double prefixCost,
              const Options &options, SearchBudget &work, bool &invalidProof) {
  namespace rp = compact_transport;
  uint64_t limbs = uint64_t(width) / 64 + 1;
  uint64_t readoutWork =
      rp::mul(16, rp::mul(observables.size() + 1, uint64_t(width) * width + 1));
  if (!work.take(
          rp::add(readoutWork, rp::mul(32 * (residual.size() + 1), limbs))))
    return std::nullopt;
  auto tail = graph::synthesizeReadout(width, observables, readouts, options);
  if (!tail) {
    if (tail.error == "terminal observable reconstruction failed")
      invalidProof = true;
    // Structural limits or unsupported readout mappings are optional misses.
    return std::nullopt;
  }
  if (!work.take(rp::mul(tail.gates.size() + 1,
                         rp::mul(observables.size() + 16, limbs))))
    return std::nullopt;
  auto transported = observables;
  for (const auto &gate : tail.gates) {
    if (!region_detail::validClifford(width, gate) ||
        !open_detail::validNative(width, gate, options)) {
      invalidProof = true;
      return std::nullopt;
    }
    for (auto &row : transported)
      row = tcount::detail::conjugate(row, gate);
  }
  for (unsigned i = 0; i < transported.size(); ++i)
    if (transported[i] != Pauli{{}, QubitSet::single(readouts[i])}) {
      invalidProof = true;
      return std::nullopt;
    }
  BoundaryScore score;
  score.residualCost = evaluateNativeCost(residual, options).value;
  score.readoutCost = evaluateNativeCost(tail.gates, options).value;
  score.exactReadout = true;
  score.rank = {prefixCost + score.residualCost + score.readoutCost,
                score.readoutCost, score.residualCost, prefixCost, 0};
  if (std::any_of(score.rank.begin(), score.rank.end(),
                  [](double cost) { return !std::isfinite(cost); }))
    return std::nullopt;
  return score;
}
} // namespace joint_plan_detail

/// Every packet is a subset of the exact current dependency-ready frontier.
/// Its body=C*packet carries C through EVERY other live rotation and ordered
/// observable. No selected circuit, preserved network, or initial-state
/// assumptions enter this route. Source-bound measured replay/readout remains
/// mandatory in the caller. This is a bounded heuristic planner, not an
/// optimality certificate; complete greedy output is pinned independently.
inline JointPhasePlannerCandidates
generateJointPhasePlans(const MeasuredProblem &source, const Options &options,
                        const JointPhasePlannerLimits &limits,
                        SearchBudget &shared, const RegionScorer &scorer = {}) {
  namespace jp = joint_plan_detail;
  namespace rp = compact_transport;
  JointPhasePlannerCandidates result;
  auto fail = [&](OpenRegionStatus status, const char *message) {
    result.status = status;
    result.error = message;
    if (status == OpenRegionStatus::InvalidProof)
      result.candidates.clear();
  };
  if (!source.width || source.width > 4096 || source.rotations.size() > 32768 ||
      source.observables.empty() || source.observables.size() > 4096 ||
      source.observables.size() != source.readoutWires.size() ||
      !limits.maxPlans || !limits.maxReadyNodes || !limits.maxPacketSeeds ||
      !limits.region.maxActiveWires || !limits.region.maxRotations ||
      !limits.region.maxRank || !options.maxGates || !options.maxDependencies ||
      (!options.nativeCX && !options.nativeCZ) ||
      static_cast<unsigned>(options.objective) >
          static_cast<unsigned>(SynthesisObjective::TDepth)) {
    fail(OpenRegionStatus::Unsupported, "invalid joint-phase planner contract");
    return result;
  }
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost})
    if (!std::isfinite(cost) || cost < 0) {
      fail(OpenRegionStatus::Unsupported, "invalid joint-phase native profile");
      return result;
    }
  if (shared.remaining() < 2 || !open_detail::hasTime(shared)) {
    fail(OpenRegionStatus::Exhausted, "no joint-phase planner reserve");
    return result;
  }
  SearchBudget budget(shared, std::min(limits.maxWork, shared.remaining() / 2),
                      open_detail::halfTime(shared));
  auto finish = [&] {
    result.work = budget.work();
    return std::move(result);
  };
  auto charge = [&](uint64_t units) {
    if (budget.take(units))
      return true;
    fail(OpenRegionStatus::Exhausted, "joint-phase planner work exhausted");
    return false;
  };
  const uint64_t limbs = uint64_t(source.width) / 64 + 1;
  unsigned beamWidth = std::min(limits.beamWidth, 4u);
  unsigned planCap = std::min(limits.maxPlans, 4u);
  unsigned supportCap = std::min(limits.region.maxActiveWires, 16u);
  unsigned rankCap = std::min(limits.region.maxRank, 3u);
  unsigned packetCap = std::min(limits.region.maxRotations, 64u);
  unsigned readyCap = std::min(limits.maxReadyNodes, 64u);
  unsigned seeds = std::min(limits.maxPacketSeeds, 8u);
  // Admit sequential and beam phases separately. The original source is
  // borrowed; these reservations cover owned source/index/readout scratch,
  // local solver memory, native vectors and actual retained plan capacities.
  unsigned traceCap = std::min(limits.maxTraceEntries, 256u);
  uint64_t traceBytes =
      uint64_t(traceCap) * (sizeof(JointPhasePacketTrace) + 1024);
  uint64_t rowBytes = sizeof(Gate) + 80 + 16 * limbs;
  uint64_t solverBytes =
      std::min(limits.region.maxLiveBytes, limits.maxLiveBytes / 4);
  uint64_t dependencyBytes =
      rp::add(rp::mul(source.rotations.size(), 256 + 16 * limbs),
              rp::mul(std::min<uint64_t>(options.maxDependencies,
                                         rp::mul(source.rotations.size(),
                                                 source.rotations.size()) /
                                             2),
                      16));
  uint64_t sourceRows = source.rotations.size() + source.observables.size();
  // Readout checks its gate cap after one row. Reserve that row's possible
  // native lowering overshoot and vector reallocation, even for a tiny cap.
  uint64_t readoutGrowth =
      rp::mul(3 * (5 * uint64_t(source.width) + 16), rowBytes);
  uint64_t commonBytes = rp::add(
      readoutGrowth,
      rp::add(
          rp::add(solverBytes, traceBytes),
          rp::add(dependencyBytes,
                  rp::add(65536, rp::mul(4, rp::mul(sourceRows, rowBytes))))));
  JointPhaseMemoryPhase *memoryPhase = nullptr;
  auto admitMemory = [&](JointPhaseMemoryPhase &phase, unsigned states,
                         unsigned futurePlans) {
    phase.stateSlots = states;
    if (!charge(result.candidates.size() + 1))
      return false;
    for (const auto &plan : result.candidates)
      phase.retainedBytes =
          rp::add(phase.retainedBytes,
                  rp::mul(plan.prefix.capacity() + plan.observables.capacity(),
                          rowBytes));
    uint64_t fixed = rp::add(
        commonBytes,
        rp::add(
            phase.retainedBytes,
            rp::add(rp::mul(states, rp::mul(sourceRows, rowBytes)),
                    rp::mul(2 * futurePlans,
                            rp::mul(source.observables.size(), rowBytes)))));
    // Each state owns prefix+actual exit. Three capacities per word cover
    // vector growth/reallocation; three additional gate slots cover readout
    // scratch. Local solver proposal storage is reserved in solverBytes.
    uint64_t perGate =
        rp::mul(rowBytes, 6 * uint64_t(states) + 3 + 2 * futurePlans);
    if (!solverBytes || fixed >= limits.maxLiveBytes ||
        perGate > limits.maxLiveBytes - fixed) {
      fail(OpenRegionStatus::Exhausted, "joint-phase memory phase unavailable");
      return false;
    }
    phase.gateCap = std::min<uint64_t>(options.maxGates,
                                       (limits.maxLiveBytes - fixed) / perGate);
    phase.reservedBytes = fixed + phase.gateCap * perGate;
    phase.admitted = true;
    result.reservedLiveBytes =
        std::max(result.reservedLiveBytes, phase.reservedBytes);
    result.nativeGateCap = phase.gateCap;
    memoryPhase = &phase;
    return true;
  };
  if (!admitMemory(result.greedyMemory, 3, 0))
    return finish();
  if (!charge(rp::mul(source.rotations.size() + source.observables.size() + 1,
                      limbs)))
    return finish();
  result.candidates.reserve(planCap);
  result.trace.reserve(traceCap);
  std::vector<Pauli> axes;
  for (const auto &gate : source.rotations) {
    if (gate.kind != Kind::Rotate || !gate.pauli.x.fits(source.width) ||
        !gate.pauli.z.fits(source.width) || !gate.pauli.weight() ||
        (!gate.angle.symbol && gate.angle.denominator <= 0)) {
      fail(OpenRegionStatus::Unsupported,
           "joint-phase planner needs normalized rotations");
      return finish();
    }
    axes.push_back(gate.pauli);
  }
  pbc::CommutingObservables measurements;
  std::set<unsigned> assigned;
  for (size_t i = 0; i < source.observables.size(); ++i) {
    const auto &row = source.observables[i];
    if (!charge(rp::mul(2 * measurements.size() + 1, limbs)))
      return finish();
    if (!row.x.fits(source.width) || !row.z.fits(source.width) ||
        !row.weight() || source.readoutWires[i] >= source.width ||
        !assigned.insert(source.readoutWires[i]).second ||
        !measurements.insert(row)) {
      fail(OpenRegionStatus::Unsupported,
           "invalid joint-phase ordered readout");
      return finish();
    }
  }
  uint64_t dependencyStart = budget.work();
  auto dependencies =
      DependencyIndex::build(axes, options.maxDependencies, &budget);
  result.dependencyWork += budget.work() - dependencyStart;
  if (!dependencies) {
    fail(dependencies.exhausted() ? OpenRegionStatus::Exhausted
                                  : OpenRegionStatus::InvalidProof,
         "joint-phase dependency construction failed");
    return finish();
  }
  uint64_t ordinal = 0;
  auto initial = [&]() -> std::optional<jp::State> {
    if (!charge(rp::mul(source.rotations.size() + source.observables.size() + 1,
                        limbs)))
      return std::nullopt;
    jp::State state;
    state.rotations = source.rotations;
    state.active.assign(source.rotations.size(), 1);
    state.remaining = source.rotations.size();
    state.plan.observables = source.observables;
    if (!charge(rp::mul(
            16 * (source.rotations.size() + source.observables.size() + 1),
            limbs)))
      return std::nullopt;
    for (const auto &gate : source.rotations) {
      state.rotationCosts.push_back(jp::rotationCost(gate, options));
      state.residualCost += state.rotationCosts.back();
    }
    for (unsigned i = 0; i < source.observables.size(); ++i) {
      state.readoutCosts.push_back(jp::readoutPotential(
          source.observables[i], source.readoutWires[i], options));
      state.readoutPotential += state.readoutCosts.back();
    }
    state.ordinal = ordinal++;
    return state;
  };
  auto packet =
      [&](const jp::State &state) -> std::optional<std::vector<unsigned>> {
    auto cursor = DependencyCursor::create(*dependencies.value, &budget);
    if (!cursor) {
      fail(OpenRegionStatus::Exhausted, "joint-phase cursor exhausted");
      return std::nullopt;
    }
    std::vector<unsigned> ready;
    for (unsigned i = 0; i < state.active.size(); ++i)
      if (!state.active[i]) {
        auto consumed = cursor.value->consume(i, &budget);
        if (!consumed) {
          fail(consumed.exhausted() ? OpenRegionStatus::Exhausted
                                    : OpenRegionStatus::InvalidProof,
               "joint-phase cursor restore failed");
          return std::nullopt;
        }
      }
    for (unsigned i = 0; i < state.active.size(); ++i) {
      auto available = cursor.value->ready(i, &budget);
      if (!available) {
        fail(available.exhausted() ? OpenRegionStatus::Exhausted
                                   : OpenRegionStatus::InvalidProof,
             "joint-phase readiness query failed");
        return std::nullopt;
      }
      if (*available.value)
        ready.push_back(i);
    }
    if (ready.empty()) {
      fail(OpenRegionStatus::InvalidProof,
           "nonempty joint-phase residual has no ready node");
      return std::nullopt;
    }
    // Keep source-order and low-support pools independent. Truncating solely
    // by support can remove partners of a ready rank-three affine quartet.
    if (!charge(rp::mul(ready.size() * 32 +
                            uint64_t(2 * seeds) *
                                (readyCap * (8 * rankCap + 8) +
                                 ready.size() * 8 + state.active.size()) +
                            1,
                        limbs)))
      return std::nullopt;
    std::vector<std::vector<unsigned>> pools;
    pools.emplace_back(ready.begin(),
                       ready.begin() +
                           std::min<size_t>(readyCap, ready.size()));
    if (ready.size() > readyCap) {
      auto weighted = ready;
      std::stable_sort(
          weighted.begin(), weighted.end(), [&](unsigned a, unsigned b) {
            return std::pair{state.rotations[a].pauli.weight(), a} <
                   std::pair{state.rotations[b].pauli.weight(), b};
          });
      weighted.resize(readyCap);
      std::sort(weighted.begin(), weighted.end());
      if (weighted != pools[0])
        pools.push_back(std::move(weighted));
    }
    std::vector<unsigned> best;
    bool packetInvalid = false;
    unsigned bestSupport = source.width + 1;
    for (const auto &pool : pools)
      for (unsigned seed = 0; seed < std::min<size_t>(seeds, pool.size());
           ++seed) {
        pbc::CommutingObservables basis;
        QubitSet support;
        std::vector<unsigned> chosen;
        std::vector<bool> picked(state.active.size());
        std::vector<Pauli> span;
        auto admit = [&](unsigned id, bool closeOnly) {
          const auto &axis = state.rotations[id].pauli;
          auto extended = support | axis.x | axis.z;
          if (extended.count() > supportCap)
            return;
          auto trial = basis;
          if (!trial.insert(axis)) {
            packetInvalid = true;
            fail(OpenRegionStatus::InvalidProof,
                 "dependency-ready packet anticommutes");
            return;
          }
          if (trial.size() > rankCap ||
              (closeOnly && trial.size() != basis.size()))
            return;
          if (trial.size() > basis.size()) {
            unsigned old = span.size();
            span.push_back({axis.x, axis.z});
            for (unsigned i = 0; i < old; ++i)
              span.push_back({span[i].x ^ axis.x, span[i].z ^ axis.z});
          }
          basis = std::move(trial);
          support = std::move(extended);
          chosen.push_back(id);
          picked[id] = true;
        };
        for (unsigned offset = 0;
             offset < pool.size() && chosen.size() < packetCap; ++offset)
          admit(pool[(seed + offset) % pool.size()], false);
        // Once a span is chosen, every ready occurrence in that SAME span is
        // eligible, even outside both bounded seed pools. No rank expansion.
        for (unsigned id : ready) {
          if (chosen.size() == packetCap)
            break;
          const auto &axis = state.rotations[id].pauli;
          if (!picked[id] &&
              std::any_of(span.begin(), span.end(), [&](const Pauli &entry) {
                return entry.x == axis.x && entry.z == axis.z;
              })) {
            // A span element cannot introduce support outside its generators.
            chosen.push_back(id);
            picked[id] = true;
          }
        }
        std::sort(chosen.begin(), chosen.end());
        if (chosen.size() > best.size() ||
            (chosen.size() == best.size() && support.count() < bestSupport)) {
          best = std::move(chosen);
          bestSupport = support.count();
        }
      }
    if (packetInvalid)
      return std::nullopt;
    // A broad single axis still admits exact closed Bootstrap completion.
    if (best.empty())
      best.push_back(ready.front());
    return best;
  };

  bool contextualGreedyLane = false;
  auto expand = [&](const jp::State &parent, bool greedyLane, auto &&accept) {
    if (result.expansions >= limits.maxExpansions) {
      fail(OpenRegionStatus::Exhausted, "joint-phase expansion cap");
      return false;
    }
    uint64_t packetStart = budget.work();
    auto selected = packet(parent);
    result.packetWork += budget.work() - packetStart;
    if (!selected)
      return false;
    if (!charge(rp::mul(selected->size() * (supportCap + 2) + 1, limbs)))
      return false;
    QubitSet support;
    for (unsigned id : *selected)
      support |= parent.rotations[id].pauli.x | parent.rotations[id].pauli.z;
    bool compact = support.count() <= supportCap;
    std::vector<unsigned> physical;
    for (unsigned q : support.set_bits())
      physical.push_back(q);
    std::vector<Gate> local;
    for (unsigned id : *selected) {
      const auto &gate = parent.rotations[id];
      if (!compact)
        local.push_back(gate);
      else {
        Pauli axis{{}, {}, gate.pauli.sign};
        for (unsigned q = 0; q < physical.size(); ++q) {
          axis.x.set(q, gate.pauli.x.test(physical[q]));
          axis.z.set(q, gate.pauli.z.test(physical[q]));
        }
        local.push_back(Gate::rotation(std::move(axis), gate.angle));
      }
    }
    Options settings = options;
    settings.release = settings.pruneUnobserved =
        settings.acceptOnlyImprovement = false;
    if (parent.plan.prefix.size() > result.nativeGateCap ||
        parent.exit.size() > result.nativeGateCap) {
      fail(OpenRegionStatus::Exhausted,
           "joint-phase state exceeds current phase gate cap");
      return false;
    }
    settings.maxGates = result.nativeGateCap - parent.plan.prefix.size();
    if (!settings.maxGates || budget.remaining() < 2 ||
        !open_detail::hasTime(budget)) {
      fail(OpenRegionStatus::Exhausted,
           "joint-phase completion reserve exhausted");
      return false;
    }
    // Choose one common scoring mode for this immutable parent. Exact readout
    // decoding must fit every callback's fixed slice, including replay/copies;
    // otherwise all endpoints use the same signed-row estimate.
    uint64_t readoutBound =
        rp::mul(16, rp::mul(source.observables.size() + 1,
                            uint64_t(source.width) * source.width + 1));
    uint64_t readoutTailBound =
        rp::mul(source.observables.size(), 4 * uint64_t(source.width) + 4);
    uint64_t contextBound = rp::add(
        readoutBound,
        rp::add(rp::mul(readoutTailBound + 1,
                        rp::mul(source.observables.size() + 16, limbs)),
                rp::mul(parent.active.size() + source.observables.size() + 1,
                        rp::mul(16 * supportCap + 64, limbs))));
    bool exactContextMode = contextBound <= limits.maxContextWork;
    unsigned remainingCalls =
        std::min(limits.maxContextCalls, 16384u) -
        std::min(result.contextCalls, std::min(limits.maxContextCalls, 16384u));
    unsigned localContextQuota = jp::endpointQuota(
        remainingCalls, parent.remaining, selected->size(), beamWidth,
        contextualGreedyLane, exactContextMode, limits.region.maxContextScores);
    bool fundExactRescoring =
        exactContextMode &&
        jp::packetCallAllowance(remainingCalls, parent.remaining,
                                selected->size(), beamWidth,
                                contextualGreedyLane) >= 4;
    bool scoreLocal = limits.contextScoring && !greedyLane && localContextQuota;
    if (limits.contextScoring && !greedyLane) {
      result.minimumEndpointQuota =
          std::min(result.minimumEndpointQuota, localContextQuota);
      result.maximumEndpointQuota =
          std::max(result.maximumEndpointQuota, localContextQuota);
      result.quotaLimitedPackets +=
          localContextQuota < std::min(limits.region.maxContextScores, 24u);
    }
    JointPhaseCandidates alternatives;
    bool contextInvalid = false;
    std::optional<jp::BoundaryScore> lastContext;
    auto contextual = [&](const JointPhaseImplementation &implementation,
                          SearchBudget &caller) -> std::optional<CostRank> {
      lastContext.reset();
      if (!limits.contextScoring || greedyLane || !compact ||
          result.contextCalls >= std::min(limits.maxContextCalls, 16384u) ||
          caller.remaining() < 2 || !open_detail::hasTime(caller))
        return std::nullopt;
      SearchBudget work(caller,
                        std::min(limits.maxContextWork, caller.remaining() / 2),
                        open_detail::halfTime(caller));
      ++result.contextCalls;
      auto evaluate = [&]() -> std::optional<jp::BoundaryScore> {
        unsigned width = physical.size();
        if (implementation.body.size() > settings.maxGates ||
            implementation.exitCliffords.size() > result.nativeGateCap)
          return std::nullopt;
        uint64_t scan =
            rp::mul(parent.active.size() + parent.plan.observables.size() + 1,
                    2 * width + 1);
        if (!work.take(rp::add(
                scan, rp::mul(implementation.body.size() +
                                  implementation.exitCliffords.size() + 1,
                              16))))
          return std::nullopt;
        for (const auto &gate : implementation.body)
          if (!open_detail::validNative(width, gate, options)) {
            contextInvalid = true;
            return std::nullopt;
          }
        for (const auto &gate : implementation.exitCliffords)
          if (!region_detail::validClifford(width, gate) ||
              !open_detail::validNative(width, gate, options)) {
            contextInvalid = true;
            return std::nullopt;
          }
        std::vector<unsigned> affected, measured;
        if (!implementation.exitCliffords.empty()) {
          for (unsigned i = 0; i < parent.active.size(); ++i)
            if (parent.active[i] &&
                !std::binary_search(selected->begin(), selected->end(), i) &&
                jp::touches(parent.rotations[i].pauli, physical))
              affected.push_back(i);
          for (unsigned i = 0; i < parent.plan.observables.size(); ++i)
            if (jp::touches(parent.plan.observables[i], physical))
              measured.push_back(i);
        }
        if (!work.take(rp::add(
                rp::mul(3 * width, implementation.exitCliffords.size() + 1),
                rp::mul(affected.size() + measured.size(),
                        rp::mul(8 * width + 17, limbs)))))
          return std::nullopt;
        auto images = rp::exitImages(width, implementation.exitCliffords);
        double residual = parent.residualCost,
               potential = parent.readoutPotential;
        for (unsigned id : *selected)
          residual -= parent.rotationCosts[id];
        for (unsigned id : affected) {
          auto gate = parent.rotations[id];
          gate.pauli = rp::transport(gate.pauli, physical, images);
          residual +=
              jp::rotationCost(gate, options) - parent.rotationCosts[id];
        }
        for (unsigned id : measured) {
          auto row =
              rp::transport(parent.plan.observables[id], physical, images);
          potential +=
              jp::readoutPotential(row, source.readoutWires[id], options) -
              parent.readoutCosts[id];
        }
        double prefix = parent.prefixCost +
                        evaluateNativeCost(implementation.body, options).value;
        if (exactContextMode) {
          uint64_t readoutWork =
              rp::mul(16, rp::mul(source.observables.size() + 1,
                                  uint64_t(source.width) * source.width + 1));
          if (work.remaining() < readoutWork)
            return std::nullopt;
          if (!work.take(rp::mul(parent.plan.observables.size() + 1, limbs)))
            return std::nullopt;
          auto rows = parent.plan.observables;
          for (unsigned id : measured)
            rows[id] = rp::transport(rows[id], physical, images);
          const std::vector<Gate> none;
          auto score = jp::boundaryScore(source.width, none, rows,
                                         source.readoutWires, prefix + residual,
                                         settings, work, contextInvalid);
          if (!score)
            return std::nullopt;
          score->residualCost = residual;
          score->rank = {prefix + residual + score->readoutCost,
                         score->readoutCost, residual, prefix, 0};
          ++result.exactContextScores;
          return score;
        }
        ++result.estimatedContextScores;
        jp::BoundaryScore score;
        score.residualCost = residual;
        score.readoutCost = potential;
        score.rank = {prefix + residual + potential, potential, residual,
                      prefix, 0};
        return score;
      };
      auto scored = evaluate();
      result.contextWork += work.work();
      if (!scored) {
        ++result.contextMisses;
        return std::nullopt;
      }
      lastContext = *scored;
      return scored->rank;
    };
    if (compact) {
      auto localLimits = limits.region;
      localLimits.maxCandidates =
          std::max(2u, std::min(localLimits.maxCandidates, 4u));
      localLimits.maxActiveWires = supportCap;
      localLimits.maxRank = rankCap;
      localLimits.maxLiveBytes = solverBytes;
      // Reproduce and publish the earlier greedy route before adding the
      // contextual/natural-coordinate route. Optional search cannot replace it.
      if (greedyLane)
        localLimits.maxCoordinateLayouts = 1;
      localLimits.maxContextScores = localContextQuota;
      SearchBudget search(budget,
                          std::min(limits.maxLocalWork, budget.remaining() / 2),
                          open_detail::halfTime(budget));
      alternatives = solveJointReadyPhaseRegion(
          physical.size(), local, settings, localLimits, search,
          scoreLocal ? JointPhaseScorer(contextual) : JointPhaseScorer{});
      if (contextInvalid) {
        fail(OpenRegionStatus::InvalidProof,
             "joint-phase context proof failed");
        return false;
      }
      result.coordinateLayouts += alternatives.coordinateLayouts;
      ++result.packetsSolved;
      result.packetRotations += selected->size();
      result.localWork += alternatives.work;
      result.localStates += alternatives.statesVisited;
      result.localCompleted += alternatives.completedProposals;
      if (alternatives.status == OpenRegionStatus::InvalidProof) {
        fail(OpenRegionStatus::InvalidProof, alternatives.error.c_str());
        return false;
      }
      result.localExhaustions +=
          alternatives.status == OpenRegionStatus::Exhausted;
    }
    if (alternatives.candidates.empty()) {
      SearchBudget fallback(budget, budget.remaining() / 2,
                            open_detail::halfTime(budget));
      const std::vector<Pauli> none;
      const std::vector<unsigned> noWires;
      auto baseline = buildBootstrapPrefix(
          {compact ? unsigned(physical.size()) : source.width, local, none,
           noWires},
          settings, &fallback);
      if (!baseline) {
        fail(baseline.status == BootstrapStatus::Exhausted
                 ? OpenRegionStatus::Exhausted
                 : OpenRegionStatus::Unsupported,
             baseline.error.c_str());
        return false;
      }
      alternatives.candidates.push_back({std::move(baseline.gates), {}});
      ++result.fallbackPackets;
    }
    std::vector<std::optional<jp::BoundaryScore>> survivorScores(
        alternatives.candidates.size());
    bool rankWithExact =
        limits.contextScoring && !greedyLane && compact && fundExactRescoring;
    if (rankWithExact) {
      for (unsigned i = 0; i < alternatives.candidates.size(); ++i) {
        contextual(alternatives.candidates[i], budget);
        survivorScores[i] = lastContext;
        if (!lastContext || !lastContext->exactReadout)
          rankWithExact = false;
      }
      if (contextInvalid) {
        fail(OpenRegionStatus::InvalidProof,
             "joint-phase context proof failed");
        return false;
      }
    }
    unsigned implementationIndex = 0;
    for (const auto &implementation : alternatives.candidates) {
      unsigned thisImplementation = implementationIndex++;
      if (result.expansions >= limits.maxExpansions) {
        fail(OpenRegionStatus::Exhausted, "joint-phase expansion cap");
        return false;
      }
      if (implementation.body.size() > settings.maxGates ||
          implementation.exitCliffords.size() >
              result.nativeGateCap - parent.exit.size()) {
        fail(OpenRegionStatus::Exhausted, "joint-phase composed gate cap");
        return false;
      }
      unsigned width = compact ? physical.size() : source.width;
      uint64_t beginWork = budget.work();
      if (!charge(rp::mul(2 * parent.rotations.size() +
                              parent.plan.prefix.size() + parent.exit.size() +
                              2 * source.observables.size() +
                              implementation.body.size() + 1,
                          limbs)))
        return false;
      result.stateWork += budget.work() - beginWork;
      std::vector<unsigned> affected, measured;
      if (!implementation.exitCliffords.empty()) {
        beginWork = budget.work();
        if (!charge(
                rp::mul(parent.active.size() + source.observables.size() + 1,
                        2 * width + 1)))
          return false;
        for (unsigned i = 0; i < parent.active.size(); ++i)
          if (parent.active[i] &&
              !std::binary_search(selected->begin(), selected->end(), i) &&
              jp::touches(parent.rotations[i].pauli, physical))
            affected.push_back(i);
        for (unsigned i = 0; i < parent.plan.observables.size(); ++i)
          if (jp::touches(parent.plan.observables[i], physical))
            measured.push_back(i);
        if (!charge(rp::add(
                rp::mul(3 * width, implementation.exitCliffords.size() + 1),
                rp::mul(affected.size() + measured.size(),
                        rp::mul(8 * width + 1, limbs)))))
          return false;
        result.transportWork += budget.work() - beginWork;
      }
      beginWork = budget.work();
      if (!charge(
              rp::mul(16 * (affected.size() + measured.size() +
                            implementation.body.size() + selected->size() + 1),
                      limbs)))
        return false;
      result.residualScoreWork += budget.work() - beginWork;
      for (const auto &gate : implementation.body)
        if (!open_detail::validNative(width, gate, options)) {
          fail(OpenRegionStatus::InvalidProof,
               "invalid joint-phase native body");
          return false;
        }
      for (const auto &gate : implementation.exitCliffords)
        if (!region_detail::validClifford(width, gate) ||
            !open_detail::validNative(width, gate, options)) {
          fail(OpenRegionStatus::InvalidProof,
               "invalid joint-phase actual exit");
          return false;
        }
      auto child = parent;
      child.ordinal = ordinal++;
      for (unsigned id : *selected) {
        child.active[id] = 0;
        child.residualCost -= child.rotationCosts[id];
      }
      child.remaining -= selected->size();
      for (const auto &gate : implementation.body)
        child.plan.prefix.push_back(compact ? rp::lift(gate, physical) : gate);
      if (!implementation.exitCliffords.empty()) {
        auto images = rp::exitImages(width, implementation.exitCliffords);
        for (unsigned id : affected) {
          child.rotations[id].pauli =
              rp::transport(child.rotations[id].pauli, physical, images);
          double cost = jp::rotationCost(child.rotations[id], options);
          child.residualCost += cost - child.rotationCosts[id];
          child.rotationCosts[id] = cost;
        }
        for (unsigned id : measured) {
          child.plan.observables[id] =
              rp::transport(child.plan.observables[id], physical, images);
          double cost = jp::readoutPotential(child.plan.observables[id],
                                             source.readoutWires[id], options);
          child.readoutPotential += cost - child.readoutCosts[id];
          child.readoutCosts[id] = cost;
        }
        for (const auto &gate : implementation.exitCliffords)
          child.exit.push_back(rp::lift(gate, physical));
        result.transportedRows += affected.size() + measured.size();
        result.affectedRotations += affected.size();
        result.affectedObservables += measured.size();
      }
      child.prefixCost = parent.prefixCost +
                         evaluateNativeCost(implementation.body, options).value;
      double residualCost = child.residualCost;
      child.priority = child.prefixCost + residualCost;
      std::optional<jp::BoundaryScore> context;
      if (limits.contextScoring && !greedyLane && compact) {
        if (rankWithExact)
          context = survivorScores[thisImplementation];
        // A scorer miss is not a failed semantic continuation. If ANY exact
        // score is unavailable, ALL siblings use the same cached heuristic
        // scale; never compare an estimate against a successfully exact rank.
        if (!context) {
          jp::BoundaryScore estimate;
          estimate.residualCost = child.residualCost;
          estimate.readoutCost = child.readoutPotential;
          estimate.rank = {
              child.prefixCost + child.residualCost + child.readoutPotential,
              child.readoutPotential, child.residualCost, child.prefixCost, 0};
          context = estimate;
          ++result.estimatedContextScores;
        }
        child.priority = context->rank[0];
      }
      if (result.trace.size() < traceCap) {
        uint64_t signature = 1469598103934665603ull;
        for (const auto &gate : implementation.exitCliffords)
          for (uint64_t value :
               {uint64_t(gate.kind), uint64_t(gate.a), uint64_t(gate.b)})
            signature = (signature ^ value) * 1099511628211ull;
        result.trace.push_back(
            {parent.ordinal, child.ordinal, signature, greedyLane,
             contextualGreedyLane, bool(context),
             context && context->exactReadout, *selected,
             compact ? physical : std::vector<unsigned>{}, child.remaining,
             unsigned(implementation.body.size()),
             unsigned(implementation.exitCliffords.size()),
             unsigned(physical.size()), child.prefixCost,
             evaluateNativeCost(implementation.body, options).value,
             residualCost, context ? context->readoutCost : 0});
      }
      ++result.expansions;
      if (std::isfinite(child.priority) && !accept(std::move(child)))
        return false;
    }
    return true;
  };
  std::vector<CostRank> ranks;
  std::vector<unsigned char> completeRanks;
  ranks.reserve(planCap);
  completeRanks.reserve(planCap);
  auto publish = [&](jp::State &&state, bool first,
                     bool contextualCompletion = false) {
    if (state.remaining) {
      fail(OpenRegionStatus::InvalidProof, "partial joint-phase publication");
      return false;
    }
    ++result.completedProposals;
    for (const auto &old : result.candidates) {
      if (!charge(rp::mul(old.prefix.size() + state.plan.prefix.size() +
                              source.observables.size() + 1,
                          limbs)))
        return false;
      if (region_detail::samePlan(old, state.plan))
        return true;
    }
    CostRank rank{};
    rank[0] = state.priority;
    bool completeRank = false, stopAfterPublication = false;
    if (!first && scorer) {
      unsigned &scoreCount = contextualCompletion ? result.contextCompleteScores
                                                  : result.completeScores;
      unsigned scoreLimit = contextualCompletion
                                ? std::min(limits.maxContextCompleteScores, 1u)
                                : limits.maxCompleteScores;
      if (scoreCount >= scoreLimit) {
        stopAfterPublication = true;
      } else if (budget.remaining() < 2 || !open_detail::hasTime(budget)) {
        fail(OpenRegionStatus::Exhausted, "joint-phase final scoring reserve");
        stopAfterPublication = true;
      } else {
        SearchBudget scoring(budget, budget.remaining() / 2,
                             open_detail::halfTime(budget));
        ++scoreCount;
        auto scored = scorer(state.plan, scoring);
        if (scored && std::all_of(scored->begin(), scored->end(),
                                  [](double x) { return std::isfinite(x); })) {
          rank = *scored;
          completeRank = true;
        } else {
          if (scoring.stopped())
            fail(OpenRegionStatus::Exhausted,
                 "joint-phase final scoring exhausted");
          stopAfterPublication = true;
        }
      }
    }
    if (std::any_of(rank.begin(), rank.end(),
                    [](double x) { return !std::isfinite(x); }))
      return !stopAfterPublication;
    // Complete scores and analytic fallbacks occupy separate ordering tiers;
    // a heuristic never impersonates a measured-completion cost. Every stored
    // value remains an UNVERIFIED RegionalPlan, including scored proposals.
    // Slot zero stays pinned. Capacity was reserved before any optional work,
    // so publication below moves ownership without a post-score budget charge.
    size_t at = first ? 0 : 1;
    while (!first && at < ranks.size() &&
           !(std::pair{!completeRank, rank} <
             std::pair{!bool(completeRanks[at]), ranks[at]}))
      ++at;
    if (at >= planCap)
      return !stopAfterPublication;
    if (result.candidates.size() == planCap) {
      result.candidates.pop_back();
      ranks.pop_back();
      completeRanks.pop_back();
    }
    result.candidates.insert(result.candidates.begin() + at,
                             std::move(state.plan));
    ranks.insert(ranks.begin() + at, rank);
    completeRanks.insert(completeRanks.begin() + at, completeRank);
    if (!first && scorer && !completeRank)
      ++result.finalScoreFallbackPlans;
    return !stopAfterPublication;
  };

  auto greedy = initial();
  if (!greedy)
    return finish();
  result.peakStates = 1;
  memoryPhase->peakStates = 1;
  while (greedy->remaining) {
    std::optional<jp::State> next;
    bool expanded = expand(*greedy, true, [&](jp::State &&candidate) {
      if (!next || std::tie(candidate.priority, candidate.ordinal) <
                       std::tie(next->priority, next->ordinal))
        next = std::move(candidate);
      result.peakStates = std::max(result.peakStates, 3u);
      memoryPhase->peakStates = std::max(memoryPhase->peakStates, 3u);
      return true;
    });
    if (!expanded) {
      // A later optional sibling may exhaust after an already complete local
      // choice. Do not lose that full program merely because siblings failed.
      if (next && !next->remaining &&
          result.status != OpenRegionStatus::InvalidProof)
        publish(std::move(*next), true);
      return finish();
    }
    if (!next) {
      fail(OpenRegionStatus::Unsupported, "no finite joint-phase continuation");
      return finish();
    }
    greedy = std::move(next);
  }
  if (!publish(std::move(*greedy), true))
    return finish();
  greedy.reset();
  if (planCap == 1 || source.rotations.empty())
    return finish();
  // Finish one contextual whole-chain path before broadening the frontier.
  // This shares the same work/expansion/storage caps and leaves the completed
  // control pinned even if any optional continuation or final score exhausts.
  if (limits.contextScoring) {
    if (!admitMemory(result.contextualMemory, 3, 0))
      return finish();
    auto contextual = initial();
    if (!contextual)
      return finish();
    memoryPhase->peakStates = 1;
    contextualGreedyLane = true;
    while (contextual->remaining) {
      std::optional<jp::State> next;
      bool expanded = expand(*contextual, false, [&](jp::State &&candidate) {
        if (!next || std::tie(candidate.priority, candidate.ordinal) <
                         std::tie(next->priority, next->ordinal))
          next = std::move(candidate);
        result.peakStates = std::max(result.peakStates, 3u);
        memoryPhase->peakStates = std::max(memoryPhase->peakStates, 3u);
        return true;
      });
      if (!expanded) {
        if (next && !next->remaining &&
            result.status != OpenRegionStatus::InvalidProof)
          publish(std::move(*next), false, true);
        return finish();
      }
      if (!next) {
        fail(OpenRegionStatus::Unsupported,
             "no finite contextual joint-phase continuation");
        return finish();
      }
      contextual = std::move(next);
    }
    if (!publish(std::move(*contextual), false, true))
      return finish();
    contextualGreedyLane = false;
  }
  if (!beamWidth)
    return finish();
  // Both sequential optional<State> objects have left scope/reset. Retained
  // plans are counted at their actual capacities; reserve every additional
  // beam result that can coexist with those pinned allocations.
  if (!admitMemory(result.beamMemory, 2 * beamWidth + 3, planCap - 1))
    return finish();
  auto start = initial();
  if (!start)
    return finish();
  memoryPhase->peakStates = 1;
  std::vector<jp::State> beam;
  beam.push_back(std::move(*start));
  start.reset();
  while (!beam.empty()) {
    std::vector<jp::State> next;
    for (const auto &state : beam)
      if (!expand(state, false, [&](jp::State &&candidate) {
            result.peakStates = std::max<unsigned>(
                result.peakStates, beam.size() + next.size() + 1);
            memoryPhase->peakStates = std::max<unsigned>(
                memoryPhase->peakStates, beam.size() + next.size() + 1);
            if (!candidate.remaining)
              return publish(std::move(candidate), false);
            // Fund deduplication and the bounded pairwise boundary-diversity
            // selection, including occurrence masks and every stored exit.
            uint64_t comparisonSize = candidate.active.size() +
                                      candidate.exit.size() +
                                      candidate.plan.prefix.size() +
                                      candidate.plan.observables.size() + 1;
            for (const auto &old : next)
              comparisonSize =
                  rp::add(comparisonSize, old.active.size() + old.exit.size() +
                                              old.plan.prefix.size() +
                                              old.plan.observables.size() + 1);
            if (!charge(rp::mul(4 * (next.size() + 1),
                                rp::mul(comparisonSize, limbs))))
              return false;
            for (const auto &old : next)
              if (old.active == candidate.active &&
                  rp::sameWord(old.exit, candidate.exit) &&
                  region_detail::samePlan(old.plan, candidate.plan))
                return true;
            next.push_back(std::move(candidate));
            std::stable_sort(next.begin(), next.end(),
                             [](const auto &a, const auto &b) {
                               return std::tie(a.priority, a.ordinal) <
                                      std::tie(b.priority, b.ordinal);
                             });
            if (next.size() > beamWidth) {
              // Preserve distinct carried exits/obligation sets before keeping
              // another cheap prefix with the same boundary. This is diversity,
              // not dominance; only full measured completions are ranked
              // finally.
              std::vector<size_t> keep;
              for (size_t i = 0; i < next.size() && keep.size() < beamWidth;
                   ++i) {
                bool same = false;
                for (size_t j : keep)
                  same |= next[i].active == next[j].active &&
                          rp::sameWord(next[i].exit, next[j].exit);
                if (!same)
                  keep.push_back(i);
              }
              for (size_t i = 0; i < next.size() && keep.size() < beamWidth;
                   ++i)
                if (std::find(keep.begin(), keep.end(), i) == keep.end())
                  keep.push_back(i);
              std::vector<jp::State> retained;
              for (size_t i : keep)
                retained.push_back(std::move(next[i]));
              next = std::move(retained);
            }
            return true;
          }))
        return finish();
    beam = std::move(next);
  }
  return finish();
}
} // namespace cudaq::pauli::graph::detail
