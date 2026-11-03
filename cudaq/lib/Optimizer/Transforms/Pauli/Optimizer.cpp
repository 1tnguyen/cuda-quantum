/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Composition and selection of verified Pauli optimization candidates.
///
/// Runs T-count and graph-synthesis strategies with a shared budget and a
/// bounded beam. Temporary entangler growth may be explored, but only
/// candidates meeting the final cost constraints can replace the retained
/// incumbent.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Pauli/Optimizer.h"
#include "TCountInternal.h"
#include <algorithm>
#include <tuple>

using namespace cudaq::pauli;
namespace {
using Cost = tcount::Cost;
using Circuit = std::vector<Gate>;
/// Combined-portfolio circuit awaiting acceptance or further composition.
struct Candidate {
  /// Original or verified circuit; may temporarily exceed growth limits.
  Circuit gates;
  /// Cached mechanical cost for objective ordering and admissibility checks.
  Cost cost;
  uint64_t id = 0;
};
bool same(const Circuit &a, const Circuit &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}
/// Build a lexicographic ranking key: smaller is better, with the selected
/// objective first and the remaining metrics used as ordered tie-breakers.
/// Compare tuples from left to right, stopping at the first unequal metric.
/// Only when all preceding metrics tie does the next metric affect the rank.
/// Equal tuples have equal rank. A later metric cannot compensate for a worse
/// earlier metric; these are not weighted sums.
///
/// For Objective::TCount, the order is T count, two-qubit count, circuit depth,
/// total gate count, then T depth. A circuit with 10 T gates and 100 entanglers
/// ranks ahead of one with 11 T gates and 20 entanglers, regardless of their
/// remaining metrics. If both have 10 T gates, the one with 20 entanglers wins.
/// If their entangler counts also tie, circuit depth decides next.
///
/// Ranking does not imply acceptance: optimize() checks candidate admissibility
/// and growth limits separately. Even a better-ranked circuit can be rejected
/// for exceeding the permitted entangler or depth growth.
auto score(const Cost &c, Objective objective) {
  switch (objective) {
  case Objective::TotalGates:
    return std::tuple{c.gates, c.t, c.depth, c.twoQubit, c.tDepth};
  case Objective::TCount:
    // T count, then entanglers, circuit depth, total gates, and T depth.
    return std::tuple{c.t, c.twoQubit, c.depth, c.gates, c.tDepth};
  case Objective::TDepth:
    // T depth, then T count, entanglers, circuit depth, and total gates.
    return std::tuple{c.tDepth, c.t, c.twoQubit, c.depth, c.gates};
  case Objective::Entanglers:
    // Entanglers, then circuit depth, T count, total gates, and T depth.
    return std::tuple{c.twoQubit, c.depth, c.t, c.gates, c.tDepth};
  case Objective::Depth:
    // Circuit depth, then entanglers, T count, total gates, and T depth.
    return std::tuple{c.depth, c.twoQubit, c.t, c.gates, c.tDepth};
  }
  // Defensive fallback; optimize() rejects invalid objectives before scoring.
  return std::tuple{c.t, c.twoQubit, c.depth, c.gates, c.tDepth};
}
} // namespace

/// Search a bounded beam of circuits by composing T reduction and graph
/// synthesis. Each strategy verifies its own transformation before its result
/// enters the portfolio; retain() applies the shared cost and growth policy.
/// Keep the best admissible circuit independently of the search beam, so failed
/// proposals or exhausted search budgets cannot discard a verified incumbent.
OptimizerResult cudaq::pauli::optimize(unsigned width, const Circuit &input,
                                       const OptimizerOptions &options,
                                       SearchBudget *sharedBudget,
                                       Telemetry *telemetry) {
  OptimizerResult result;
  // Invalid input or settings are API errors, unlike a valid search that finds
  // no improvement. Reject them before installing the unchanged-input fallback.
  if (!tcount::detail::valid(width, input)) {
    result.error =
        "Pauli optimization requires valid gates on a nonzero wire count";
    return result;
  }
  // Composition requires full-unitary equivalence. Measurement-only release
  // semantics cannot certify a circuit that subsequent strategies may reuse.
  if (!options.rounds || !options.beamWidth || options.beamWidth > 64 ||
      !options.maxCandidates || !options.maxWork || !options.maxGates ||
      !options.gatecount.rounds || options.graphSynthesis.release ||
      static_cast<unsigned>(options.objective) >
          static_cast<unsigned>(Objective::TotalGates)) {
    result.error = "Pauli optimization requires positive limits, beam width "
                   "1..64, a valid objective and unitary hold semantics";
    return result;
  }
  // Let each strategy validate its own options using an empty one-wire circuit,
  // including disabled strategies, without searching the actual input circuit.
  auto validT = tcount::optimize(1, {}, options.tcount);
  auto validGraph = graph::synthesize(1, {}, options.graphSynthesis);
  if (!validT || !validGraph) {
    result.error = !validT ? validT.error : validGraph.error;
    return result;
  }
  result.gates = input;
  // An oversized but valid input is a clean skip, not a failed optimization.
  // Cost statistics remain unavailable on this path.
  if (input.size() > options.maxGates) {
    result.stats.skippedSize = true;
    return result;
  }
  result.stats.before = result.stats.after = tcount::cost(width, input);
  // A caller-supplied budget replaces the local outer allowance. All strategy
  // calls and their verification share it; measure only this invocation's work.
  SearchBudget localBudget(options.maxWork, options.timeoutMs);
  SearchBudget &budget = sharedBudget ? *sharedBudget : localBudget;
  uint64_t initialWork = budget.work();
  const uint64_t ceiling =
      result.stats.before.twoQubit + options.maxTwoQubitGrowth;
  // Anchor final growth limits to the original circuit, not each parent, so
  // successive rounds cannot accumulate an allowance repeatedly. T objectives
  // additionally forbid increasing T count, even when optimizing T depth.
  auto admissible = [&](const Cost &c) {
    return c.twoQubit <= ceiling &&
           c.depth <= result.stats.before.depth + options.maxDepthGrowth &&
           ((options.objective != Objective::TCount &&
             options.objective != Objective::TDepth &&
             options.objective != Objective::TotalGates) ||
            c.t <= result.stats.before.t);
  };
  auto better = [&](const Cost &a, const Cost &b) {
    return score(a, options.objective) < score(b, options.objective);
  };
  // Count outer strategy invocations, independently of candidates explored
  // inside a strategy. A failed invocation still consumes its reserved slot.
  auto startCandidate = [&] {
    if (result.stats.candidates >= options.maxCandidates || !budget.take())
      return false;
    ++result.stats.candidates;
    return true;
  };
  // Gate-count transformations preserve the existing Clifford network and
  // exact scalar phase. Their internal choices use total gates first; outer
  // retention still applies the user's objective and growth limits.
  auto runGateCount = [&](const Candidate &parent) -> std::optional<Candidate> {
    if (!options.enableGateCount || !startCandidate())
      return std::nullopt;
    ++result.stats.gateCountCandidates;
    TelemetryScope observation(telemetry, "gate-count", parent.id);
    auto start = budget.work();
    if (telemetry)
      observation.event.before = parent.cost;
    auto gates = gatecount::optimize(width, parent.gates, options.gatecount,
                                     budget, options.maxGates, telemetry,
                                     parent.id, observation.event.id,
                                     &observation.event.selectedChild,
                                     observation.verificationTime(), ceiling);
    auto cost = tcount::cost(width, gates);
    if (telemetry) {
      observation.event.after = cost;
      observation.event.work = budget.work() - start;
    }
    return Candidate{std::move(gates), cost, observation.event.id};
  };
  // Produce a verified T-portfolio proposal. The temporary branch relaxes its
  // entangler allowance for subsequent reconstruction, not for final
  // acceptance.
  auto runT = [&](const Candidate &parent,
                  bool temporary) -> std::optional<Candidate> {
    if (!options.enableTCount || !startCandidate())
      return std::nullopt;
    ++result.stats.tcountCandidates;
    // Link this invocation to its parent candidate. Nested T-strategy events
    // use this scope as their enclosing event; the returned candidate keeps its
    // ID so retention decisions can be reported after the scope ends.
    TelemetryScope observation(
        telemetry, temporary ? "t-portfolio.temporary" : "t-portfolio",
        parent.id);
    uint64_t initialWork = budget.work();
    if (telemetry) {
      observation.event.before = parent.cost;
      observation.event.unresolvedBefore = unsynthesizedRotations(parent.gates);
    }
    auto settings = options.tcount;
    uint64_t bound =
        ceiling + (temporary ? options.intermediateTwoQubitGrowth : 0);
    // The nested API accepts growth relative to its parent. Convert the outer
    // absolute ceiling to that allowance, clamping subtraction at zero and the
    // result to unsigned. A parent already above the bound gets no extra room;
    // retain() still enforces the actual final ceiling.
    settings.maxTwoQubitGrowth = unsigned(
        std::min<uint64_t>(bound - std::min(bound, parent.cost.twoQubit),
                           std::numeric_limits<unsigned>::max()));
    // Assess reconstruction before the inner T-first beam prunes alternatives.
    // For example, 47 T / 80 entanglers can clean up to fewer total gates than
    // 47 T / 72 entanglers. Both must reach the outer objective's cost model.
    std::optional<Candidate> refined;
    auto visit = [&](const Circuit &gates, const Cost &cost, uint64_t id) {
      if (options.objective != Objective::TotalGates ||
          !options.enableGateCount || budget.stopped())
        return;
      if (auto cleaned = runGateCount(Candidate{gates, cost, id}))
        if (admissible(cleaned->cost) &&
            (!refined || better(cleaned->cost, refined->cost)))
          refined = std::move(*cleaned);
    };
    auto candidate =
        tcount::optimize(width, parent.gates, settings, &budget, telemetry,
                         parent.id, observation.event.id, visit);
    if (candidate && refined && better(refined->cost, candidate.stats.after)) {
      candidate.gates = std::move(refined->gates);
      candidate.stats.after = refined->cost;
      candidate.stats.selectedCandidate = refined->id;
      candidate.stats.changed = better(refined->cost, parent.cost);
    }
    if (telemetry) {
      observation.event.work = budget.work() - initialWork;
      observation.event.selectedChild = candidate.stats.selectedCandidate;
      observation.event.counters = {
          {"merges", candidate.stats.merges},
          {"commutation_checks", candidate.stats.commutationChecks},
          {"decoded_blocks", candidate.stats.decodedBlocks},
          {"verified_blocks", candidate.stats.verifiedBlocks},
          {"candidates", candidate.stats.candidates},
          {"rounds", candidate.stats.rounds},
          {"budget_exhaustions", candidate.stats.budgetExhaustions}};
      observation.event.verificationNs = candidate.stats.verificationNs;
    }
    // Strategy failures and oversized proposals are local rejections. Neither
    // replaces the incumbent nor turns an otherwise valid portfolio run into
    // an API error.
    if (!candidate || candidate.gates.size() > options.maxGates) {
      if (telemetry) {
        observation.event.decision = "rejected";
        observation.event.reason =
            candidate ? "candidate size limit" : candidate.error;
      }
      ++result.stats.rejectedCandidates;
      return std::nullopt;
    }
    // All T transformations already replay their own exact witnesses.
    if (telemetry) {
      observation.event.after = candidate.stats.after;
      observation.event.unresolvedAfter =
          unsynthesizedRotations(candidate.gates);
    }
    return Candidate{std::move(candidate.gates), candidate.stats.after,
                     observation.event.id};
  };
  // Reconstruct a parent using its own graph replay proof. This proof checks
  // the reconstruction step; earlier T identities rely on the T strategy's
  // witnesses, giving a chain of verified transformations from the input.
  auto runGraphSynthesis =
      [&](const Candidate &parent) -> std::optional<Candidate> {
    if (!options.enableGraphSynthesis || !startCandidate())
      return std::nullopt;
    ++result.stats.graphSynthesisCandidates;
    TelemetryScope observation(telemetry, "graph-synthesis", parent.id);
    uint64_t initialWork = budget.work();
    if (telemetry) {
      observation.event.before = parent.cost;
      observation.event.unresolvedBefore = unsynthesizedRotations(parent.gates);
    }
    auto settings = options.graphSynthesis;
    settings.maxGates = std::min(settings.maxGates, options.maxGates);
    // Disable only the standalone cost gate, not equivalence verification.
    // A graph proposal may help the selected portfolio objective even when it
    // would lose under the graph synthesizer's standalone ranking.
    settings.acceptOnlyImprovement = false; // The shared objective decides.
    auto candidate = graph::synthesize(width, parent.gates, settings, &budget,
                                       observation.verificationTime());
    if (telemetry) {
      observation.event.work = budget.work() - initialWork;
      observation.event.counters = {
          {"rotations_before", candidate.stats.rotationsBefore},
          {"rotations_after", candidate.stats.rotationsAfter},
          {"dependencies", candidate.stats.dependencies},
          {"entangling_actions", candidate.stats.tqes},
          {"swaps", candidate.stats.swaps},
          {"candidates", candidate.stats.candidates},
          {"budget_exhausted", candidate.stats.budgetExhausted}};
    }
    if (!candidate) {
      if (telemetry) {
        observation.event.decision = "rejected";
        observation.event.reason = candidate.error;
      }
      ++result.stats.rejectedCandidates;
      result.stats.budgetExhaustions += candidate.stats.budgetExhausted;
      return std::nullopt;
    }
    // Rank both strategy families with the same mechanical cost model, rather
    // than comparing graph heuristic scores with T-portfolio gate counts.
    auto cost = tcount::cost(width, candidate.gates);
    if (telemetry) {
      observation.event.after = cost;
      observation.event.unresolvedAfter =
          unsynthesizedRotations(candidate.gates);
    }
    return Candidate{std::move(candidate.gates), cost, observation.event.id};
  };
  // The beam holds alternatives worth expanding; result holds the best accepted
  // circuit. A candidate need not beat that incumbent to remain in the beam.
  std::vector<Candidate> beam{{input, result.stats.before}};
  auto retain = [&](std::vector<Candidate> &next, Candidate candidate) {
    // This is the final acceptance boundary, including for temporary proposals.
    // Verification alone does not authorize exceeding the user's growth limits.
    if (!admissible(candidate.cost)) {
      if (telemetry)
        telemetry->decision(candidate.id, "growth_limit");
      ++result.stats.rejectedCandidates;
      return;
    }
    // Replace the incumbent only on a strict lexicographic improvement. Equal
    // costs keep the previous result and its telemetry winner.
    if (better(candidate.cost, result.stats.after)) {
      result.gates = candidate.gates;
      result.stats.after = candidate.cost;
      result.stats.changed = true;
      if (telemetry)
        telemetry->winner = candidate.id;
    }
    // Deduplicate exact gate sequences, not costs: equally ranked circuits can
    // expose different reductions when another strategy processes them.
    if (std::none_of(next.begin(), next.end(), [&](const Candidate &existing) {
          return same(existing.gates, candidate.gates);
        })) {
      if (telemetry)
        telemetry->decision(candidate.id, "admissible");
      next.push_back(std::move(candidate));
    } else if (telemetry)
      telemetry->decision(candidate.id, "duplicate");
    // Preserve insertion order on cost ties, then keep only beamWidth choices.
    // Pruning limits future exploration; it does not undo an incumbent update.
    std::stable_sort(next.begin(), next.end(),
                     [&](const Candidate &a, const Candidate &b) {
                       return better(a.cost, b.cost);
                     });
    if (next.size() > options.beamWidth) {
      if (telemetry)
        for (size_t i = options.beamWidth; i < next.size(); ++i)
          telemetry->decision(next[i].id, "beam_pruned");
      next.resize(options.beamWidth);
    }
  };
  // First establish the strict T-only incumbent, before spending work on
  // optional reconstruction. The original remains the fallback throughout.
  if (auto candidate = runGateCount(beam.front()))
    retain(beam, std::move(*candidate));
  if (auto candidate = runT(beam.front(), false))
    retain(beam, std::move(*candidate));
  for (unsigned round = 0; round < options.rounds && !budget.stopped();
       ++round) {
    // Expand a fixed snapshot for this round. Preserve existing alternatives in
    // next; ordinary proposals become parents only in the following round.
    auto next = beam;
    for (const auto &parent : beam) {
      if (auto candidate = runGateCount(parent))
        retain(next, std::move(*candidate));
      if (auto candidate = runGraphSynthesis(parent))
        retain(next, std::move(*candidate));
      if (auto candidate = runT(parent, false))
        retain(next, std::move(*candidate));
      if (options.enableGraphSynthesis && options.intermediateTwoQubitGrowth) {
        // Explicit two-stage branch: allow extra parity-network entanglers to
        // expose T savings, then try to remove that overhead by reconstruction.
        // Unlike ordinary proposals, the temporary form is used immediately.
        if (auto temporary = runT(parent, true)) {
          bool needsRescue = !admissible(temporary->cost);
          if (auto reconstructed = runGraphSynthesis(*temporary)) {
            // Rescue counts restored admissibility, not necessarily a win over
            // the incumbent; retain() still ranks and filters the result.
            if (needsRescue && admissible(reconstructed->cost))
              ++result.stats.rescuedCandidates;
            retain(next, std::move(*reconstructed));
          }
          // The intermediate form itself may already satisfy final limits.
          // Otherwise retain() rejects it even if reconstruction failed.
          retain(next, std::move(*temporary));
        }
      }
      if (budget.stopped() || result.stats.candidates >= options.maxCandidates)
        break;
    }
    // Stop when the ordered beam is unchanged, not merely when costs tie.
    // This is a search fixed point under the current strategies and bounds,
    // not a proof that the circuit is globally optimal.
    bool fixed = next.size() == beam.size();
    for (size_t i = 0; fixed && i < next.size(); ++i)
      fixed = same(next[i].gates, beam[i].gates);
    beam = std::move(next);
    if (fixed || result.stats.candidates >= options.maxCandidates)
      break;
  }
  // Budget/candidate exhaustion ends exploration successfully with the best
  // verified admissible result so far (possibly the original input).
  result.stats.work = budget.work() - initialWork;
  result.stats.budgetExhaustions +=
      budget.stopped() || result.stats.candidates >= options.maxCandidates;
  return result;
}
