/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Exact T-count optimization of native Pauli circuits.
///
/// Declares the bounded merge/decoding portfolio, its cost model, and
/// statistics. Rewrites preserve scalar phase, and candidate selection accounts
/// for the entangling cost of reconstructing optimized rotations.
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Pauli/Circuit.h"
#include "cudaq/Optimizer/Pauli/SearchBudget.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <cstdint>
#include <string>
#include <vector>

namespace cudaq::pauli {
struct Telemetry;
}

namespace cudaq::pauli::tcount {

/// Controls for one T-count portfolio invocation on a native circuit island.
/// Candidates are ordered by (T count, entanglers, depth, total gates), using
/// mechanical lowering costs rather than routing or magic-factory resources.
/// Limits cover this invocation's strategies and verification; a supplied
/// shared
/// budget additionally bounds work across invocations. Work bounds are
/// deterministic, whereas a wall-clock deadline can change the chosen result.
///
/// The bounded decoder uses the phase-polynomial/Reed-Muller correspondence in
/// Amy and Mosca, "T-count optimization and Reed-Muller codes":
/// https://arxiv.org/abs/1601.07363. It searches a small commuting-generator
/// space, not a global optimum over the physical circuit. Merge/reverse/reuse
/// switches and beam limits are implementation choices rather than paper
/// presets. Start with the defaults, then measure quality and budget
/// exhaustion;
/// wider search under a fixed budget can crowd out other useful strategies.
struct Options {
  /// Merge eligible exact rotations with matching axes after Clifford transport
  /// and checks against intervening operations. Can cancel T rotations or turn
  /// pairs into Cliffords without decoding a whole block. Keep enabled
  /// normally;
  /// false saves this strategy's scans but can miss inexpensive T reductions.
  /// mergeWindow limits pair distance; maxWork bounds search and witness
  /// replay.
  bool merge = true;

  /// Also optimize the adjoint circuit and invert the resulting candidate back.
  /// This explores different merge/block/placement boundaries and may improve
  /// T count or Clifford overhead; it does not change the target unitary.
  /// Enabling adds a reverse variant for each enabled strategy, increasing
  /// runtime and budget use. Keep true for quality; try false for tight latency
  /// budgets and compare the reductions lost on representative inputs.
  bool reverse = true;

  /// Search commuting blocks for phase representations with fewer T terms,
  /// including checked Clifford and scalar corrections. Can find reductions
  /// beyond matching-axis merges, but may introduce costly parity gadgets.
  /// Disable for a cheaper merge-only configuration or an ablation. Keep
  /// enabled
  /// when low-rank phase structure offers gains; maxBlockRank/maxBlockGates
  /// bound
  /// block discovery and decoding, and maxTwoQubitGrowth guards reconstruction.
  bool decode = true;

  /// Place decoded rotations in favorable frames of the original Clifford
  /// network instead of always rebuilding them as block-entry gadgets. This
  /// targets entanglers/depth and can make a T reduction satisfy the growth
  /// cap.
  /// When true, the portfolio also explores decoding without reuse; false keeps
  /// only that cheaper-to-search alternative. Placement scans consume work, so
  /// try false for latency studies. Has no strategy effect when decode is
  /// false.
  bool reuseCliffords = true;

  /// Maximum outer rounds within this T-only portfolio; must be positive.
  /// Additional rounds compose reductions exposed by earlier candidates, but
  /// increase runtime until convergence or another limit stops the search.
  /// One round is a useful low-latency baseline; retain the default for quality
  /// and increase only if rounds are exhausted before candidates stabilize.
  unsigned rounds = 4;

  /// Number of distinct circuits retained between T-portfolio rounds; positive.
  /// One is greedy; larger beams retain equal-T alternatives whose Clifford
  /// structure may enable later reductions. More entries increase circuit
  /// storage and strategy work and may exhaust maxWork before useful later
  /// rounds. Start at two and raise together with an intentional work budget.
  unsigned beamWidth = 2;

  /// Maximum independent commuting-generator rank in a decoded block, in 1..5.
  /// This is not the number of physical wires. A rank-r block has 2^r joint
  /// eigenvalue assignments; exact phase checking grows with that space. Rank
  /// five searches 64 codewords in the implemented punctured Reed-Muller coset.
  /// Below rank four there are no nontrivial coset alternatives, though
  /// repeated
  /// coefficients may still combine. Keep five for strongest bounded decoding;
  /// lower it for speed at the cost of missed phase identities/smaller blocks.
  unsigned maxBlockRank = 5;

  /// Maximum input operations scanned into one decoding block; positive.
  /// Includes intervening Clifford/scalar operations, not only T rotations.
  /// Larger blocks can expose distant phase relations but cost more frame and
  /// placement work; rank, angle, and commutation boundaries still end a block.
  /// Smaller blocks are faster but may split a useful identity. Start at 256
  /// and tune after checking whether block size or generator rank is limiting.
  unsigned maxBlockGates = 256;

  /// Maximum gate-index distance between endpoints of a proposed merge.
  /// Zero removes the distance cap, subject to the work budget. A smaller
  /// positive window limits replay scans and latency but misses long-range T
  /// cancellation; increasing it broadens coverage without allowing a merge
  /// across a failed commutation check. Start at zero for quality and impose
  /// a finite window when long scans dominate. Has no effect if merge is false.
  unsigned mergeWindow = 0;

  /// Maximum strategy invocations in this T-only call; must be positive.
  /// Each enabled forward/reverse merge or decode variant consumes a slot,
  /// whether or not it improves the circuit. Larger caps allow more beam/round
  /// exploration but cost time; smaller caps may prevent later variants from
  /// running. This is not a cap on individual decoded blocks or codewords.
  /// Tune with rounds, beamWidth, and the separate maxWork bound.
  unsigned maxCandidates = 256;

  /// Deterministic work-unit allowance for this invocation, including
  /// witnesses.
  /// Must be positive; units estimate operations, not gates, seconds, or bytes.
  /// Larger values permit more search/proof work and may improve T count at
  /// higher compile cost. A supplied shared budget also applies: increasing
  /// this bound cannot bypass it. Start with the default, inspect exhaustion,
  /// and raise only when quality warrants more work; the verified incumbent
  /// remains available when the budget runs out.
  uint64_t maxWork = 10000000;

  /// Optional wall-clock deadline in milliseconds for this T-only call.
  /// Zero disables this local deadline; maxWork and any shared deadline remain.
  /// Smaller values favor latency but may interrupt useful exploration/replay;
  /// larger values give more opportunity until another limit is reached.
  /// Checks occur at budget charges, so this is not a hard preemptive timeout.
  /// Use zero with a fixed work bound for reproducible quality comparisons.
  unsigned timeoutMs = 0;

  /// Extra mechanically lowered two-qubit gates allowed above this call's
  /// input.
  /// Zero requires no net increase; this is an absolute count, not a
  /// percentage.
  /// Raising it can admit lower-T decompositions with more parity computation,
  /// increasing entanglers, total gates, or depth. Keep zero for conservative
  /// use and raise only for an explicit T/Clifford trade-off. The combined
  /// optimizer overrides this per call using its global/intermediate allowance.
  unsigned maxTwoQubitGrowth = 0;
};

/// Mechanical-lowering metrics; excludes routing and magic-state factories.
struct Cost {
  /// T gates, two-qubit gates, gate depth, total gates, and T-layer depth.
  uint64_t t = 0, twoQubit = 0, depth = 0, gates = 0, tDepth = 0;
  bool operator==(const Cost &) const = default;
  bool operator<(const Cost &) const;
};
/// Work counters and selected-circuit metrics for one T portfolio invocation.
struct Statistics {
  /// Input and selected-output costs in the same mechanical cost model.
  Cost before, after;
  /// Charged work units, merge-proof commutation checks, and verified merges.
  uint64_t work = 0, commutationChecks = 0, merges = 0;
  /// Strategy calls, blocks submitted to decoding, and phase-verified blocks.
  unsigned candidates = 0, decodedBlocks = 0, verifiedBlocks = 0;
  /// Completed search rounds and recorded budget-exhaustion events.
  unsigned rounds = 0, budgetExhaustions = 0;
  /// Whether the selected output strictly improves the input cost tuple.
  bool changed = false;
  /// Telemetry ID of the selected strategy; zero denotes the input.
  uint64_t selectedCandidate = 0;
  /// Sum of witness-check times when telemetry is enabled.
  uint64_t verificationNs = 0;
};
/// Verified T-optimization result; check error before using gates.
struct Result {
  /// Selected circuit; retains the input without an admissible improvement.
  std::vector<Gate> gates;
  /// Cost and search accounting for this invocation.
  Statistics stats;
  /// Empty on success; a nonempty message invalidates the output contract.
  std::string error;
  explicit operator bool() const { return error.empty(); }
};

/// Optional observer of each distinct, verified, growth-admissible proposal
/// before T-only beam pruning. A composing optimizer can refine and rank these
/// alternatives for another objective without changing the standalone T search.
/// The gate reference is borrowed for the duration of the call. Any search
/// performed by the observer must charge the enclosing shared budget.
using CandidateVisitor = llvm::function_ref<void(
    const std::vector<Gate> &, const Cost &, uint64_t candidateId)>;

/// Exact, ancilla-free portfolio optimization. Each rewrite has a checked
/// witness, including scalar phase. Unsupported angles remain unchanged and
/// delimit decoding blocks; merging may cross them only when they commute.
/// Candidates compete lexicographically on T count, two-qubit count, depth,
/// and gate count. The original is always retained as a fallback. The current
/// shared Pauli representation uses dynamically sized supports.
Result optimize(unsigned width, const std::vector<Gate> &, const Options & = {},
                SearchBudget *budget = nullptr, Telemetry *telemetry = nullptr,
                uint64_t parent = 0, uint64_t scope = 0,
                CandidateVisitor visitor = {});

/// Cost of mechanical, exact Pauli-gadget lowering. This is not a promise
/// about later target-specific cleanup or physical routing.
Cost cost(unsigned width, const std::vector<Gate> &);

} // namespace cudaq::pauli::tcount
