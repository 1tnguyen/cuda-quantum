/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Bounded synthesis of Pauli rotations and Clifford frames.
///
/// Declares synthesis controls, statistics, and restricted equivalence
/// checking. The API distinguishes full-unitary preservation from the terminal-
/// measurement contract used when releasing unobservable output state.
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Pauli/Circuit.h"
#include "cudaq/Optimizer/Pauli/PBC.h"
#include "cudaq/Optimizer/Pauli/SearchBudget.h"
#include <cstdint>
#include <string>
#include <vector>

namespace cudaq::pauli::graph {

// Compatibility names for the standalone synthesis API.
using ::cudaq::pauli::Angle;
using ::cudaq::pauli::Gate;
using ::cudaq::pauli::Kind;
using ::cudaq::pauli::Pauli;

enum class SynthesisObjective { TotalGates, TCount, Entanglers, Depth, TDepth };
enum class JointPhaseAcceptance { AllFive, GatesAndT };

/// Controls for one graph-synthesis invocation, not a global optimum search.
/// The singlet/factor costs, ready-node frontier, and hold/release distinction
/// follow Paykin et al., "PCOAST: A Pauli-based Quantum Circuit Optimization
/// Framework", Fig. 12 and Sec. V-C: https://arxiv.org/abs/2305.10966v2.
/// The exact scoring and ASAP scheduling policy here are implementation
/// choices,
/// not a reproduction of unpublished SDK tie breaking or paper presets.
///
/// Start with the defaults and compare emitted entanglers, depth, total gates,
/// and runtime on the intended workload. Larger search limits or weights do not
/// guarantee better circuits, particularly under a fixed shared work budget.
/// These are native API defaults; pass wrappers may override them. No option
/// disables correctness checking. Check Result before using its gate vector:
/// an exhausted graph search/replay can return an error with a partial
/// candidate.
struct Options {
  SynthesisObjective objective = SynthesisObjective::TotalGates;
  /// Score the actual emitted Clifford action and terminal measurement network.
  double actionWeight = 0.5;
  double readoutWeight = 0.125;
  /// Native entanglers. H, S/Sdg, T/Tdg and Pauli gates form the fixed
  /// one-qubit Clifford+T basis. At least one of CX/CZ must be native.
  bool nativeCX = true, nativeCZ = true, nativeSwap = true;
  double singleQubitCost = 1, tCost = 1, cxCost = 1, czCost = 1, swapCost = 1;
  /// Fixed deterministic tie ordering; zero uses candidate enumeration order.
  uint64_t tieSeed = 4;
  /// Try one alternative at each of the first N entangling decisions, sharing
  /// the invocation's work budget. Zero requests one greedy synthesis.
  unsigned discrepancySteps = 22;
  /// Optional measurement-only deletion, independently replay checked.
  bool pruneUnobserved = false;
  /// Reward candidate gates whose ASAP schedule fits below the current depth.
  /// Zero disables this reward. Increasing it favors parallel placement, which
  /// may lower depth at the expense of entangler count or terminal-frame cost.
  /// It does not directly minimize T depth or T count. Start at zero for a
  /// count-focused run; increase gradually when measuring depth/count
  /// trade-offs.
  /// Must be finite and nonnegative; there is no calibrated universal best
  /// value.
  double parallelizationCredit = 0;

  /// Extra weight for dependency-free nodes in the support-cost delta score.
  /// Zero removes the extra frontier term, not the frontier eligibility rule.
  /// Larger values favor making ready nodes cheap now; smaller values emphasize
  /// the sampled remaining population. Either can change entanglers and depth:
  /// excessive frontier bias may miss sharing with later nodes. Start at one
  /// and vary it only against measured output costs. Must be
  /// finite/nonnegative.
  double frontierWeight = 1;

  /// Weight of the estimated terminal Clifford-frame cost change, when scored.
  /// Increasing it discourages simplifying rotations by leaving an expensive
  /// frame to reconstruct; too much weight may sacrifice shared rotation work.
  /// Zero ignores this scoring penalty, but still preserves the required frame.
  /// Start at 0.1; raise it when final Clifford reconstruction dominates the
  /// entangler/depth cost. This is a heuristic, not a hard output-cost bound.
  /// Must be finite and nonnegative.
  double frameWeight = 0.1;

  /// Number of remaining nodes sampled for each candidate's population score.
  /// Zero scores all remaining nodes. A positive cap reduces scoring work but
  /// can miss distant sharing and worsen entanglers/depth. This is not
  /// recursive
  /// search through this many future gates: the frontier and frame terms are
  /// scored separately. Start with zero on small graphs; use a finite cap on
  /// large graphs if scoring consumes the work budget.
  unsigned lookahead = 0;

  /// Number of minimum-cost ready nodes used to generate candidate actions.
  /// Zero considers every cheapest ready node. This bounds source nodes, not
  /// the number of candidate gates: each node can generate multiple pair/basis
  /// choices. Smaller caps save generation/scoring work but make selection more
  /// sensitive to node order and can miss useful sharing. Start with zero for
  /// quality comparisons; use a small positive cap to bound large-graph search.
  unsigned candidateNodes = 0;

  /// Merge equal Pauli axes across commuting nodes before constructing the DAG.
  /// Usually reduces rotation count and downstream entanglers; exact angle
  /// cancellation can also remove T contributions. The simplification scan is
  /// quadratic in the worst case and is charged to a supplied work budget.
  /// Keep enabled normally; disable to isolate synthesis behavior or when this
  /// preprocessing dominates. Disabling does not disable equivalence checking.
  bool simplify = true;

  /// Carry trailing local Clifford basis changes into the residual graph/frame
  /// instead of immediately emitting their inverses around each entangler.
  /// This can reduce one-qubit gates and depth by sharing later basis changes,
  /// but shifts work into subsequent choices and terminal reconstruction.
  /// Keep true as the initial choice; compare false if frame overhead dominates
  /// or to study symmetric entangling templates. Neither setting is universally
  /// better, and both preserve the chosen semantic contract.
  bool asymmetric = true;

  /// Standalone acceptance guard: return the original if synthesis does not
  /// strictly improve the selected objective tuple. Applies only when every
  /// input rotation is already single-qubit; multi-qubit Pauli input needs a
  /// caller's lowered-cost comparison. Keep true for standalone gate-circuit
  /// optimization.
  /// False permits a verified but more expensive candidate, useful for
  /// portfolio
  /// composition. The combined optimizer sets this false and applies its own
  /// objective/guards. It never turns off semantic replay. This guard applies
  /// to synthesize(); synthesizeMeasured() always returns a verified native
  /// completion because normalized input has no native-network incumbent.
  bool acceptOnlyImprovement = true;

  /// Select terminal joint Z-measurement equivalence instead of full-unitary
  /// preservation. True can remove rotations invisible to all final outcomes,
  /// reducing T count, entanglers, or depth; it may change the quantum state.
  /// The caller must establish that every output is terminally measured in Z
  /// and that no quantum state escapes. Keep false for unitary/controlled uses.
  /// This is the paper's release-vs-hold semantic distinction, not a
  /// performance
  /// hint. The combined unitary optimizer rejects release=true.
  bool release = false;

  /// Maximum native gate-vector length for both input and emitted output.
  /// Must be positive. Oversized input or output fails this synthesis
  /// candidate;
  /// it never truncates gates. Larger values admit more circuits/reconstruction
  /// work but increase storage and replay cost. Size this above expected input
  /// and output lengths and use a shared work budget as well; this is not a
  /// byte
  /// limit or a bound on gates after subsequent target lowering.
  unsigned maxGates = 32768;

  /// Maximum number of logical noncommuting occurrence pairs in the rotation
  /// word. Must be positive. This retains the original DAG edge-cap meaning:
  /// the exact oracle stores axis classes and conflicts instead of all edges,
  /// but never drops constraints to satisfy this limit. Stored memory depends
  /// on distinct axes and their conflicts, reported separately in Statistics.
  /// Reducing lookahead does not reduce dependency-index construction.
  unsigned maxDependencies = 1000000;

  /// Maximum entangling synthesis steps, including terminal-frame synthesis.
  /// A step selects an entangling action; it is not one arbitrary emitted gate
  /// or one scored candidate. Larger values allow longer reconstructions and
  /// more runtime, not a wider search at each step. Zero permits no such steps
  /// (it is not unlimited); exceeding the limit fails the candidate. Retain a
  /// positive bound and use maxGates/shared work to bound the other costs.
  unsigned maxSteps = 1000000;

  /// Optional measured-program providers, run only after the existing search
  /// has produced a verified incumbent. Zero disables the corresponding
  /// provider. All providers share the invocation budget, preserve fixed
  /// physical readouts, and may replace the incumbent only without worsening
  /// any of its five integral metrics. These controls do not affect
  /// synthesize() or permit a preserve-network circuit to seed normalized-PBC
  /// synthesis.
  unsigned representationSteps = 0;
  unsigned readoutCandidates = 0;
  unsigned commutingCandidates = 0;
  /// Optional full-optimization alternatives from the checked merge/decoder
  /// backend. Zero keeps the fixed-word synthesis track separate.
  unsigned algebraicCandidates = 0;
  /// Zero preserves legacy-first scheduling. A positive limit first proves a
  /// mechanical completion from normalized PBC, then bounds legacy search in
  /// a child slice so other providers retain completion/proof resources.
  /// It is a work allowance, not a circuit-size or quality threshold.
  uint64_t legacyWorkLimit = 0;
  /// Optional deadline for the complete pre-joint measured portfolio. Zero
  /// retains the shared invocation deadline. When joint planning is enabled,
  /// a positive value also reserves its aggregate work allowance where the
  /// parent can fund both lanes. This bounds scheduling, not candidate quality.
  unsigned legacyPortfolioTimeoutMs = 0;
  /// Append an independent indexed sign/frame search after all older routes
  /// finish. Its complete continuation uses one greedy rollout without the
  /// repeated readout-cost heuristic, then the enabled regional/algebraic
  /// providers. Zero disables this route; the existing representation search
  /// and its configured step bound remain unchanged.
  unsigned indexedRepresentationSteps = 0;
  /// Append source-only joint-phase planning after the existing verified
  /// portfolio. Zero disables this route; at most four complete plans are
  /// retained per source. Candidate generation never receives the incumbent
  /// native network. AllFive retains the existing five-metric guard;
  /// GatesAndT permits secondary-metric tradeoffs against the same immutable
  /// incoming incumbent. Neither policy relaxes semantic verification.
  unsigned jointPhaseCandidates = 0;
  JointPhaseAcceptance jointPhaseAcceptance = JointPhaseAcceptance::AllFive;
  /// Work per planner source in the first profile, and aggregate work/time
  /// across all joint profiles, completion and proofs. The enclosing budget
  /// remains authoritative. Local exhaustion retains completed candidates.
  uint64_t jointPhasePlannerWork = 3000000000;
  uint64_t jointPhaseMaxWork = 100000000000;
  unsigned jointPhaseTimeoutMs = 120000;
  /// Append an independent wider profile (10B planner work, 16K callbacks and
  /// expansions) after the first profile (4K callbacks and expansions). Each
  /// owns its scoring cache and has at most 60 seconds, within the aggregate
  /// deadline. Both use 128 MiB planner storage and the same final admission
  /// policy. This is a fixed portfolio, not a circuit-specific preset.
  bool jointPhaseWideSearch = false;
};

/// Graph preprocessing, search, and acceptance counters for one synthesis call.
struct Statistics {
  /// Rotation counts before/after preprocessing and logical dependency edges.
  unsigned rotationsBefore = 0, rotationsAfter = 0, dependencies = 0;
  /// Distinct unsigned axes and directed conflict entries in the exact index.
  /// These describe compressed storage, independently of the logical edge cap.
  unsigned dependencyAxisClasses = 0;
  uint64_t dependencyConflictEntries = 0;
  /// Entangling actions, frame SWAPs, generated candidates, and released nodes.
  unsigned tqes = 0, swaps = 0, candidates = 0, released = 0;
  /// False when the standalone cost guard substitutes the original circuit.
  bool accepted = true;
  /// A work/deadline or structural synthesis limit prevented completion.
  bool budgetExhausted = false;
};
/// Synthesis output and diagnostics; failure may leave a partial gate vector.
struct Result {
  /// Usable only if error is empty; accepted alone does not indicate success.
  std::vector<Gate> gates;
  /// Preprocessing, synthesis, and acceptance accounting.
  Statistics stats;
  /// Empty after successful semantic checking; otherwise explains failure.
  std::string error;
  explicit operator bool() const { return error.empty(); }
};

/// Ultra-greedy singlet/factor synthesis (PCOAST, Fig. 12 and Sec. V-C).
/// The candidate is checked by exact signed-Pauli/Clifford replay before
/// return. Release preserves the final joint Z measurement distribution, not a
/// unitary.
Result synthesize(unsigned width, const std::vector<Gate> &, const Options &,
                  SearchBudget *budget = nullptr,
                  uint64_t *verificationNs = nullptr);

/// Joint terminal readout on the specified physical wires. Empty error means
/// success; returns no partial network on invalid/dependent readouts.
Result synthesizeReadout(unsigned width, const std::vector<Pauli> &,
                         const std::vector<unsigned> &readoutWires,
                         const Options &);

/// Reconstruct a normalized measured program. The returned gate network
/// includes terminal readout Cliffords; measurements are Z on readoutWires.
/// Every candidate is replay checked before selection. No full-unitary or
/// classical output-permutation equivalence is assumed.
pbc::Result synthesizeMeasured(unsigned width, const std::vector<Gate> &,
                               const std::vector<Pauli> &,
                               const std::vector<unsigned> &readoutWires,
                               const Options &, SearchBudget * = nullptr);

/// Gate cost in the configured primitive basis; includes exact Clifford+T
/// lowering of one-wire rotations. Does not approximate arbitrary rotations.
double nativeGateCost(const std::vector<Gate> &, const Options &);

/// Independent traversal of the emitted circuit: pull Cliffords through its
/// rotations and compare the partially commuting rotation words and exact
/// Clifford phase. This is a restricted algebraic checker, not a
/// general-purpose equivalence solver. False/unknown results must never be
/// committed.
bool equivalent(unsigned width, const std::vector<Gate> &,
                const std::vector<Gate> &);

} // namespace cudaq::pauli::graph
