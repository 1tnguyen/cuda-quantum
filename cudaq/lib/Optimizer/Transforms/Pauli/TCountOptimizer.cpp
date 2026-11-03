/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Bounded search over T-count reduction strategies.
///
/// Explores merging, commuting-block decoding, and Clifford reuse in forward
/// and adjoint traversal orders. A bounded beam retains alternative circuits
/// while T-count and entangler guards protect the original incumbent.
//===----------------------------------------------------------------------===//

#include "TCountInternal.h"
#include <algorithm>
#include <limits>

using namespace cudaq::pauli;
using namespace cudaq::pauli::tcount;
namespace {
/// Deduplicate exact native sequences, not equal costs or equivalent unitaries.
/// Different representations may expose different rewrites in a later round.
bool same(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}
/// One retained T-portfolio circuit and its cached mechanical cost.
struct Candidate {
  /// Native gate sequence produced by a checked strategy or the original input.
  std::vector<Gate> gates;
  /// Mechanical-lowering metrics used for beam ordering and growth checks.
  Cost cost;
  /// Telemetry identity linking this candidate to the strategy that produced
  /// it.
  uint64_t id = 0;
};
} // namespace
/// Explore bounded combinations of checked T-reduction strategies. Preserve
/// the original as the incumbent until a strictly better admissible circuit is
/// found; search exhaustion does not invalidate already verified rewrites.
cudaq::pauli::tcount::Result
cudaq::pauli::tcount::optimize(unsigned width, const std::vector<Gate> &input,
                               const Options &options, SearchBudget *shared,
                               Telemetry *telemetry, uint64_t parent,
                               uint64_t scope, CandidateVisitor visitor) {
  Result result;
  if (!detail::valid(width, input)) {
    result.error =
        "T-count optimization requires valid gates on a nonzero wire count";
    return result;
  }
  // The decoder enumerates subset tables and codewords explicitly. Rank 5
  // bounds those structures independently of the physical circuit width.
  if (!options.rounds || !options.beamWidth || !options.maxWork ||
      !options.maxCandidates || !options.maxBlockGates ||
      !options.maxBlockRank || options.maxBlockRank > 5) {
    result.error =
        "T-count search bounds must be positive; block rank must be 1..5";
    return result;
  }
  result.gates = input;
  result.stats.before = result.stats.after = cost(width, input);
  detail::Budget budget(options, result.stats, shared);
  std::vector<Candidate> beam{{input, result.stats.before, parent}};
  for (unsigned round = 0; round < options.rounds && !budget.stopped();
       ++round) {
    // Expand a fixed parent set for this round. Carry parents forward so an
    // unhelpful strategy cannot evict a better circuit just by being attempted.
    auto next = beam;
    auto consider = [&](std::vector<Gate> gates, TelemetryEvent &event) {
      auto score = cost(width, gates);
      // Growth is measured against the original input, not the current parent,
      // so successive rounds cannot repeatedly spend the same allowance.
      if (score.t > result.stats.before.t ||
          score.twoQubit >
              result.stats.before.twoQubit + options.maxTwoQubitGrowth) {
        if (telemetry)
          event.decision = "growth_limit";
        return;
      }
      for (auto &candidate : next)
        if (same(candidate.gates, gates)) {
          if (telemetry)
            event.decision = "duplicate";
          return;
        }
      if (telemetry)
        event.decision = "admissible";
      // Offer verified alternatives before the T-only ordering discards them.
      // A fewer-entangler winner can lose on total gates after frame cleanup.
      if (visitor)
        visitor(gates, score, event.id);
      next.push_back({std::move(gates), score, event.id});
    };
    for (auto &candidate : beam) {
      // Each strategy starts from the same parent. Keep multiple equal-T
      // forms so a later strategy can expose a reduction missed by its peers.
      for (unsigned strategy = 0; strategy < (options.reuseCliffords ? 6u : 4u);
           ++strategy) {
        // Even/odd variants traverse forward/adjoint. Pairs select merge,
        // decode with reuse, and decode without reuse; disabling reuse leaves
        // just the merge and decode-without-reuse pairs.
        if ((strategy < 2 && !options.merge) ||
            (strategy >= 2 && !options.decode) ||
            ((strategy & 1) && !options.reverse))
          continue;
        if (budget.stopped() ||
            result.stats.candidates == options.maxCandidates)
          break;
        ++result.stats.candidates;
        static constexpr const char *names[] = {
            "merge.forward",          "merge.reverse",
            "decode.reuse.forward",   "decode.reuse.reverse",
            "decode.rebuild.forward", "decode.rebuild.reverse"};
        unsigned variant =
            strategy >= 2 && !options.reuseCliffords ? strategy + 2 : strategy;
        TelemetryScope observation(telemetry, names[variant], candidate.id,
                                   scope);
        auto initial = result.stats;
        if (telemetry) {
          observation.event.before = candidate.cost;
          observation.event.unresolvedBefore =
              unsynthesizedRotations(candidate.gates);
        }
        budget.verificationNs = observation.verificationTime();
        bool reverse = strategy & 1;
        // Optimize U^dagger and adjoint the result back. Reversing gate order
        // alone would not preserve the circuit or expose the intended search.
        auto source =
            reverse ? detail::adjoint(candidate.gates) : candidate.gates;
        auto strategyOptions = options;
        // Share the island's entangler allowance across transformations.
        // Earlier cancellations can fund a later T-reducing reconstruction.
        uint64_t ceiling =
            result.stats.before.twoQubit + options.maxTwoQubitGrowth;
        strategyOptions.maxTwoQubitGrowth =
            unsigned(std::min<uint64_t>(ceiling - candidate.cost.twoQubit,
                                        std::numeric_limits<unsigned>::max()));
        if (strategy >= 4)
          strategyOptions.reuseCliffords = false;
        auto gates =
            strategy < 2
                ? detail::merge(source, width, options, budget, result.stats)
                : detail::decode(source, width, strategyOptions, budget,
                                 result.stats);
        if (reverse)
          gates = detail::adjoint(gates);
        budget.verificationNs = nullptr;
        result.stats.verificationNs += observation.event.verificationNs;
        {
          TelemetryScope cleanup(telemetry, "clifford-cleanup", candidate.id,
                                 observation.event.id);
          observation.event.selectedChild = cleanup.event.id;
          uint64_t work = result.stats.work;
          if (telemetry) {
            cleanup.event.before = cost(width, gates);
            cleanup.event.unresolvedBefore = unsynthesizedRotations(gates);
          }
          gates = detail::cancelCliffords(gates, budget);
          if (telemetry) {
            cleanup.event.after = cost(width, gates);
            cleanup.event.unresolvedAfter = unsynthesizedRotations(gates);
            cleanup.event.work = result.stats.work - work;
          }
        }
        if (telemetry) {
          observation.event.after = cost(width, gates);
          observation.event.unresolvedAfter = unsynthesizedRotations(gates);
          observation.event.work = result.stats.work - initial.work;
          observation.event.counters = {
              {"merges", result.stats.merges - initial.merges},
              {"commutation_checks",
               result.stats.commutationChecks - initial.commutationChecks},
              {"decoded_blocks",
               result.stats.decodedBlocks - initial.decodedBlocks},
              {"verified_blocks",
               result.stats.verifiedBlocks - initial.verifiedBlocks},
              {"budget_exhaustions",
               result.stats.budgetExhaustions - initial.budgetExhaustions}};
          if (budget.stopped())
            observation.event.reason =
                "budget exhausted; only checked rewrites retained";
        }
        consider(std::move(gates), observation.event);
      }
    }
    // Keep stable ties to make bounded search deterministic. Ranking is
    // lexicographic (T, entanglers, depth, total gates), not a weighted sum.
    std::stable_sort(next.begin(), next.end(),
                     [](auto &a, auto &b) { return a.cost < b.cost; });
    if (next.size() > options.beamWidth) {
      if (telemetry)
        for (size_t i = options.beamWidth; i < next.size(); ++i)
          telemetry->decision(next[i].id, "beam_pruned");
      next.resize(options.beamWidth);
    }
    // Stop only when the ordered retained sequences are unchanged. Equal costs
    // alone do not imply a fixed point, and this is not global optimality.
    bool fixed = next.size() == beam.size();
    for (size_t i = 0; fixed && i < next.size(); ++i)
      fixed = same(next[i].gates, beam[i].gates);
    beam = std::move(next);
    ++result.stats.rounds;
    if (beam.front().cost < result.stats.after) {
      result.stats.after = beam.front().cost;
      result.gates = beam.front().gates;
      result.stats.changed = true;
      result.stats.selectedCandidate = beam.front().id;
    }
    if (result.stats.candidates == options.maxCandidates) {
      ++result.stats.budgetExhaustions;
      break;
    }
    if (fixed)
      break;
  }
  return result;
}
