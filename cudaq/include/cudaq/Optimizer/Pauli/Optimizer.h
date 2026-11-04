/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Shared portfolio interface for Pauli circuit optimization.
///
/// Combines T-count reduction and graph synthesis under one objective and
/// search budget. Candidate selection bounds secondary costs and retains a
/// verified incumbent when a strategy cannot produce an admissible improvement.
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Pauli/GateCount.h"
#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include "cudaq/Optimizer/Pauli/TCount.h"
#include <limits>

namespace cudaq::pauli {

/// Primary metric for lexicographic portfolio selection; see objective below.
enum class Objective { TCount, TDepth, Entanglers, Depth, TotalGates };

/// Compose exact T reduction and graph reconstruction under a shared budget.
/// Costs describe mechanical lowering, not routing, gate durations, or magic
/// factories. Arbitrary rotations are not approximately synthesized here, so
/// their eventual T cost depends on downstream synthesis. Defaults are native
/// API defaults; the Quake PBC pass overrides some, notably the growth
/// allowance.
/// All candidates require their strategy's proof, irrespective of these knobs.
struct OptimizerOptions {
  /// Lexicographic selection order (T = T count, E = entanglers, D = depth,
  /// G = total gates, TD = T depth): TCount uses (T,E,D,G,TD), TDepth uses
  /// (TD,T,E,D,G), Entanglers uses (E,D,T,G,TD), Depth uses (D,E,T,G,TD),
  /// and TotalGates uses (G,T,D,E,TD).
  /// TCount/TDepth/TotalGates forbid increasing T above the original; the other
  /// two have no separate T nonincrease guard. Changing objective can trade
  /// lower priority metrics for the primary one, within the growth limits
  /// below. Start with the metric the target actually charges for. This selects
  /// among candidates; it does not automatically retune graph heuristic
  /// weights.
  Objective objective = Objective::TotalGates;

  /// enableTCount runs the merge/commuting-block decoding portfolio. Disable
  /// for a synthesis-only comparison or to save that search/proof cost, at the
  /// risk of missing phase identities and T reductions.
  /// enableGraphSynthesis runs entangler reconstruction and enables the
  /// temporary-growth reconstruction branch. Disable for a cheaper T-only run,
  /// at the risk of rejecting T savings with expensive parity reconstruction.
  /// Keep both true for composition. All three families disabled retains the
  /// input. Graph simplification may still merge rotations when only graph
  /// synthesis is on. Nested settings are validated even when their strategy is
  /// disabled.
  bool enableTCount = true, enableGraphSynthesis = true;

  /// rounds bounds combined strategy-composition rounds (positive), distinct
  /// from tcount.rounds inside each T invocation. More rounds can expose new
  /// reductions after reconstruction; convergence or budgets can stop earlier.
  /// beamWidth retains this many distinct combined candidates (1..64). One is
  /// greedy; larger beams preserve alternatives but use more memory and work.
  /// maxCandidates bounds outer strategy invocations (positive): each T call
  /// or graph call consumes one, independently of tcount.maxCandidates inside
  /// each T call. Raising it permits more exploration, not a larger graph DAG.
  /// Start with defaults; increase these together with maxWork only when the
  /// corresponding limit is reached. Wider search at fixed work is not always
  /// better because it may prevent useful later strategies from running.
  unsigned rounds = 2, beamWidth = 2, maxCandidates = 256;

  /// Positive deterministic search/proof work allowance when no sharedBudget
  /// is supplied. An external sharedBudget replaces this outer allowance;
  /// nested T-call work limits still apply. Units estimate work, not seconds
  /// or bytes. Raise for more quality exploration at higher compilation cost;
  /// lower to bound effort. Inspect exhaustion before increasing other limits.
  uint64_t maxWork = 10000000;

  /// Outer wall-clock deadline in milliseconds; zero disables the local timer.
  /// A supplied sharedBudget replaces this timer. Deadline checks occur at
  /// budget charges, not by preemption, and do not bound the entire compiler.
  /// Lower values limit search opportunity; larger values may improve quality
  /// until another bound dominates. Prefer zero/fixed work for reproducibility.
  unsigned timeoutMs = 0;

  /// Positive bound on input and candidate native gate-vector lengths.
  /// Oversized input is returned unchanged with skippedSize; oversized
  /// candidates are rejected. Also clamps graphSynthesis.maxGates. Increasing
  /// admits larger circuits but raises storage/replay costs; decreasing skips
  /// more work. This does not cap target gate count or total allocated bytes.
  unsigned maxGates = 8192;

  /// Absolute final entangler allowance above the original mechanical cost.
  /// Zero forbids a net increase; larger values can admit lower T/depth at the
  /// cost of extra parity networks. Use zero for a conservative comparison and
  /// raise only for an explicit trade-off. The PBC CLI default is instead
  /// UINT_MAX, a permissive finite allowance, not this native API default.
  unsigned maxTwoQubitGrowth = 0;

  /// Absolute final depth allowance above the original mechanical depth.
  /// Zero forbids a net increase; larger values may admit T/entangler savings
  /// with longer schedules. The default UINT_MAX is a permissive finite
  /// allowance. Tighten for latency-sensitive targets; this is logical gate
  /// depth, not a duration bound or a T-depth/magic-factory constraint.
  unsigned maxDepthGrowth = std::numeric_limits<unsigned>::max();

  /// Additional temporary entanglers above the final entangler ceiling for
  /// a T-reduction-then-reconstruction branch. Zero disables this extra branch;
  /// larger values explore T savings requiring more parity work before graph
  /// reconstruction, at extra search/memory cost. Useful only with both
  /// strategy
  /// families enabled. Start at 256 and inspect rescuedCandidates before
  /// raising
  /// it. Every selected result must still satisfy the final growth limits;
  /// this option never authorizes committing an over-limit intermediate form.
  unsigned intermediateTwoQubitGrowth = 256;

  /// Per-invocation T strategy controls; see tcount::Options for the
  /// phase-polynomial/Reed-Muller decoder and individual tuning guidance.
  /// Local rounds, beam, candidate/work limits and deadline apply inside each
  /// call in addition to the shared budget. maxTwoQubitGrowth is overridden
  /// with the remaining global or temporary allowance for that call. Raising
  /// nested budgets alone cannot bypass an exhausted outer/shared budget.
  tcount::Options tcount;

  /// Enable native total-gate rewrites independently of the other families.
  /// Disabled by default to preserve existing exploration presets. The total-
  /// gate objective also forbids increasing T count over the original input.
  bool enableGateCount = false;

  /// Network-preserving Pauli/phase/Clifford strategy controls.
  gatecount::Options gatecount;

  /// Per-invocation graph controls; see graph::Options for singlet/factor
  /// synthesis, ready-node weighting, and ASAP parallelization credit.
  /// The portfolio clamps maxGates, forces acceptOnlyImprovement=false, and
  /// selects by objective instead. The nested objective follows this outer
  /// objective here; tune parallelizationCredit to bias synthesis toward depth.
  /// release must remain false: this native optimizer preserves the full
  /// unitary. Weights influence candidate quality/runtime, never proof
  /// requirements.
  graph::Options graphSynthesis;
};

/// Combined portfolio accounting, separate from nested strategy statistics.
struct OptimizerStatistics {
  /// Input and retained-output costs; unavailable when skippedSize is true.
  tcount::Cost before, after;
  /// Shared work units charged during this optimizer invocation.
  uint64_t work = 0;
  /// Total outer strategy calls and the T, graph and total-gate subsets.
  unsigned candidates = 0, tcountCandidates = 0, graphSynthesisCandidates = 0,
           gateCountCandidates = 0;
  /// Failed/inadmissible candidates and recorded budget/candidate-limit events.
  unsigned rejectedCandidates = 0, budgetExhaustions = 0;
  /// Inadmissible temporary candidates made admissible by graph reconstruction.
  unsigned rescuedCandidates = 0;
  /// Strict cost improvement, and input skipped for exceeding maxGates.
  bool changed = false, skippedSize = false;
};
/// Best verified admissible circuit, with the input as fallback.
struct OptimizerResult {
  /// Selected native circuit; do not consume when error is nonempty.
  std::vector<Gate> gates;
  /// Combined costs, work, and candidate-selection counters.
  OptimizerStatistics stats;
  /// Empty on success, including a bounded search that retains the input.
  std::string error;
  explicit operator bool() const { return error.empty(); }
};

/// Exact, unitary-preserving Pauli optimization. Strategy-local proof chains
/// certify every candidate; graph synthesis replay checks its own step,
/// not a T-decoding identity it cannot prove. One work/deadline budget covers
/// all strategies and replay. Size limits bound stored candidates and the DAG.
/// The original and the strict T-only result remain incumbent choices.
/// Costs describe exact mechanical lowering, excluding routing and cleanup.
OptimizerResult optimize(unsigned width, const std::vector<Gate> &,
                         const OptimizerOptions & = {},
                         SearchBudget *sharedBudget = nullptr,
                         Telemetry *telemetry = nullptr);
} // namespace cudaq::pauli
