# Pauli-based optimization in Quake

The implementation represents Pauli-based computation (PBC) as a verified form
of Quake. A shared native library optimizes Pauli rotations and Clifford frames;
MLIR passes validate the input, import it, search for equivalent candidates, and
commit a checked result. This is exploratory compiler work, not a default
CUDA-Q pipeline stage or a claim of globally optimal compilation.

This report describes the CUDA-Q implementation. Benchmark results must be
produced by its compiled passes. Native strategy estimates, emitted Quake counts,
and final target-basis counts are different measurements and must be labeled.

## Implementation and measurement status

The implementation includes T-reduction and graph-synthesis portfolios, a
`total-gates` objective, network-preserving primitive emission, and native
Pauli-frame, phase-placement, and small Clifford-window strategies. The current
validation passes 69 relevant unit tests and all 11 PBC lit tests. Descriptions
and option defaults in this report follow the accompanying source code.

No numerical advantage over CUDA-Q or Qiskit is asserted without a matching
pass run and configuration record. In particular, the available objectives and
lowering path must match the metric being compared. Earlier result tables without
that implementation provenance are not evidence for this pass.

<!-- BEGIN PASS MEASUREMENTS -->
## Measured pass results

Configuration **total-gates-v1**, detailed in Appendix B, uses the compiled
CUDA-Q pass. CUDA-Q and PBC were remeasured after the reproduction fixes below;
the Qiskit references retain the preceding measured run on the same preserved
suite circuits and declared basis. Eight CUDA-Q inputs were imported afresh;
the 118- and 433-qubit adders use their preserved original Quake inputs after fresh imports failed
under the imposed frontend resource cap. Both then run through the same compiled
CUDA-Q baseline and PBC pass. No cases were added.
Measurements are excluded from gate counts and from the circuit depths in this
table. T and two-qubit/depth columns show CUDA-Q baseline → PBC output.

Completed paired measurements: **10/10**. All ten pass independent equivalence
checking up to global phase. The implemented pass beats the proposed CUDA-Q
total on eight cases, ties one, and is one gate higher on QFT-2. It reduces total
gates versus the measured CUDA-Q baseline on nine cases and leaves QFT-2 unchanged.
T count never increases; `rc_adder_6` drops from 63 to 47 and `csla_mux_3` from
64 to 62. Neither circuit depth nor T depth increases on these ten cases.

| Case | CUDA-Q total | Qiskit 2.6.0 total | PBC total | Best proposed CUDA-Q total (reference study) | T count | Two-qubit count | Depth | Pass / independent verification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| adder_n4_transpiled | 24 | 25 | 22 | 22 | 8 → 8 | 10 → 10 | 12 → 11 | committed; equivalent |
| adder_n10_transpiled | 140 | 142 | 106 | 127 | 32 → 32 | 57 → 53 | 113 → 95 | committed; equivalent_up_to_global_phase |
| adder_n28_transpiled | 418 | 424 | 316 | 369 | 96 → 96 | 171 → 159 | 221 → 185 | committed; equivalent_up_to_global_phase |
| adder_n64_transpiled | 974 | 988 | 736 | 857 | 224 → 224 | 399 → 371 | 437 → 365 | committed; equivalent_up_to_global_phase |
| adder_n118_transpiled | 1808 | 1834 | 1366 | 1589 | 416 → 416 | 741 → 689 | 761 → 635 | committed; equivalent_up_to_global_phase |
| adder_n433_transpiled | 6673 | 6769 | 5041 | 5859 | 1536 → 1536 | 2736 → 2544 | 2651 → 2210 | committed; equivalent |
| multiply_n13_transpiled | 105 | 95 | 91 | 93 | 32 → 32 | 40 → 40 | 40 → 39 | committed; equivalent_up_to_global_phase |
| rc_adder_6 | 191 | 185 | 147 | 172 | 63 → 47 | 91 → 76 | 111 → 91 | committed; equivalent |
| csla_mux_3 | 161 | 157 | 156 | 157 | 64 → 62 | 76 → 72 | 69 → 69 | committed; equivalent |
| qft_n002 | 8 | 7 | 8 | 7 | 3 → 3 | 3 → 3 | 6 → 6 | skipped; equivalent |

The proposal column is copied from the supplied Gate-count fixes study; it is
not a run of that patch set. Qiskit here is the retained measured 2.6.0 reference, not the study's
unversioned reference. Its output is translated to the same named Clifford+T
basis before counting. Any final permutation/layout remains part of Qiskit's
output interpretation. The PBC pass preserves the original physical readouts.

<!-- END PASS MEASUREMENTS -->

## Benchmark configuration and evidence

Every benchmark configuration must record the following **before running a
case**, and every result must identify that configuration. “Default,” “tuned,”
or “same parameters” alone is insufficient.

| Record | Required information |
|---|---|
| Compiler identity | Run-artifact manifest identifying the exact source/build, executable hash and build type |
| Input identity | cudaq-bench source manifest, suite, exact case name, corpus lock/hash and compiler-input hash |
| Actual command | Executable, complete argument vector, exact pass pipeline and every explicitly supplied PBC option |
| Effective parameters | All PBC options, including omitted defaults resolved from that build; relevant environment values or explicitly unset state |
| Surrounding compilation | Pass placement, upstream legalization, downstream cleanup, output basis, optimization level and any synthesis tolerance |
| Baselines | CUDA-Q command without PBC; Qiskit version, basis, optimization level, approximation settings and random seed |
| Outcome | Committed improvement, unchanged/no improvement, unsupported input, resource limit or failure; an unchanged output does not prove the search completed |
| Correctness | Equivalence contract, checker and result, with input/output hashes |
| Counts | Counts from the actual final output at the same compilation boundary for all compared tools |

The complete command and expanded options belong in the report or its appendix,
not only in a script. A shared configuration ID can avoid repeating them in every
row, but per-case overrides must be explicit. Raw logs and outputs must retain the
same ID. Record `CUDAQ_PBC_TELEMETRY` even when unset; an explicit `telemetry`
option takes precedence. A timeout or work-limit fallback must remain visible.

For total-gate comparisons, count each gate in the declared output basis once;
exclude measurements, allocation, sinks and scalar phase. Do not substitute the
number of PBC operations or a native cost estimate. Report T count, two-qubit
count and depth alongside total gates to expose trade-offs. Use the same input,
output ordering and measurement contract. The report need not include timing
columns to retain these reproducibility records.

## Reference study

The supplied **“Clifford+T gate-count fixes”** proposal draws on the
**2026-10-29 Clifford+T/Qiskit gap study** and **2026-10-30 Toffoli cancellation
study**, using the study's recorded **CUDA-Q pipeline through `cudaq-opt`**. It discusses ten
CUDA-Q total-gate losses among 195 paired Clifford+T cases. Its Pauli-propagation
numbers are an offline model, swap elision is proposed, and the RSA matcher
change uses a pipeline stand-in. Its figures are external references, not
measurements of this PBC implementation or a rerun of that patch set.

| Case | Reported CUDA-Q total gates | Reported Qiskit total gates | Best reported/proposed CUDA-Q total gates |
|---|---:|---:|---:|
| adder_n4_transpiled | 24 | 23 | 22 |
| adder_n10_transpiled | 141 | 140 | 127 |
| adder_n28_transpiled | 419 | 418 | 369 |
| adder_n64_transpiled | 975 | 974 | 857 |
| adder_n118_transpiled | 1809 | 1808 | 1589 |
| adder_n433_transpiled | 6674 | 6673 | 5859 |
| multiply_n13_transpiled | 105 | 95 | 91 |
| rc_adder_6 | 191 | 185 | 172 |
| csla_mux_3 | 161 | 157 | 157 |
| qft_n002 | 8 | 7 | 7 |

These are **total gates, not two-qubit counts**. The proposal's QFT value assumes
swap elision with appropriate output interpretation. It is not a fixed-output
unitary comparison. The Qiskit version and full command used by the author have
not been established here. Fresh paired measurements must identify their own
versions and commands rather than inherit these labels.

## 1. PBC encoding and native representation

### Operations and exact angles

Existing `quake.exp_pauli` carries an uncontrolled rotation about a static Pauli
word. The word can contain X, Y and Z on arbitrarily many represented wires;
the angle can be exact or an opaque scalar SSA value. Quake uses `exp(+i e P)`;
the native library uses `exp(-i theta P / 2)`, so `theta = -2e`. Adjoint flags and
signed axes must be reflected in that conversion.

`quake.pbc_angle = array<i64: numerator, denominator>` annotates the exponent
`e / pi` exactly. The denominator must be positive and the floating operand must
agree with it. A nearby floating-point value is never promoted to an exact
rational by guesswork. Integer ranges still bound exact arithmetic; unsupported
or overflowing arithmetic cannot authorize an approximate rewrite.

The added `quake.mpp` operation measures a signed Pauli product with projectors
`(I + P)/2` and `(I - P)/2`. This joint measurement preserves coherence inside each
eigenspace; measuring each factor separately would have different semantics.
The first operand also carries the designated readout-wire identity through PBC
rewrites. It can retain an I factor if the observable's support moves elsewhere.
The sign, classical result and register name remain part of the measurement.
Scalar results preserve the source type: `!quake.measure` or
`!cc.measure_handle`, including through PBC conversion and lowering.

### Internal structures and frames

A native circuit is an ordered vector of Clifford, rotation and scalar-phase
operations. Each Pauli has X/Z support bitsets and a sign; overlapping X/Z bits
encode Y. Dynamically sized LLVM integer storage removes the previous
machine-word qubit limit. Circuit size, storage and work budgets still apply.

A Clifford frame records the signed images of Pauli generators. Conjugating a
rotation through a Clifford changes its axis, exposing equal rotations that were
not local neighbors in the gate circuit. Normalization absorbs Clifford action
into later rotation axes and terminal observables. The dependency DAG used for
graph synthesis is one analysis view of these data, not the only representation.

### Supported contract

The passes support single-block scalar `!quake.wire` ownership, scalar wire
arguments and `quake.null_wire` preparations, followed by terminal commuting
measurements and explicit sink closure. No quantum output may escape. The
preserved observable is the **ordered joint measurement distribution for every
input state**, not just the all-zero input and not the discarded post-measurement
quantum state. Native unitary rewrites preserve the stronger unitary contract,
including scalar phases, before this measured-function boundary.

The importer accepts supported H, S/Sdg, T/Tdg, X/Y/Z, CX, CZ, SWAP, Rx/Ry/Rz,
scalar phase, static `exp_pauli`, and scalar X/Y/Z or Pauli-product measurements.
It does not accept arbitrary controlled gates, Toffoli, CY, dynamic Pauli words,
reference/vector ownership, regions, adaptive feedback or quantum returns.
Such operations require prior legalization or a clean optional-pass fallback.
X/Y measurement results must go directly to sinks. Recognized pure scalar
arithmetic and classical results are retained; unknown operations or metadata
are not silently discarded.

## 2. Compilation flow and failure behavior

| Pass | Contract |
|---|---|
| `quake-verify-pbc` | Verify the entire function is in the supported measured PBC form |
| `quake-to-pbc` | Strictly import supported Quake and absorb Cliffords into rotations and terminal observables |
| `quake-optimize-pbc` | Require PBC input, run native strategies, normalize and emit checked PBC |
| `quake-lower-pbc` | Strictly lower rotations and a shared terminal-readout network to primitive gates |
| `quake-try-pbc` | Attempt a complete measured-function optimization transaction from supported primitive or PBC input |

The optional wrapper imports the original native gate network directly; it does
not simply run the strict passes consecutively. Normally it normalizes the native
winner back to PBC before lowering. The `preserve-clifford-network=true`
option instead emits the checked native Clifford/rotation network and prepares
the original terminal observables. This keeps circuit-level cancellations from
being lost through independent parity-gadget reconstruction. It is an optional
wrapper control; strict `quake-optimize-pbc` still emits PBC.

All paths preflight the full function before editing it. Candidate emission uses
a private function, verifies the result, and commits only on success. Unsupported
input, no admissible improvement, representability limits and exhausted budgets
leave the optional wrapper's original function unchanged, without statistics
attributes. Invalid IR, invalid options and internal reconstruction failures
produce pass failure without committing that candidate. Pass failure stops the
pipeline; it is not a synonym for skipping optimization. Transactions apply per
function, not retroactively to earlier passes or other functions.

### Lowering and readout

Rotations use basis changes and CX parity compute/uncompute. Exact Clifford+T
angles become named gates directly; opaque scalar angles remain Rz operations.
Exact rational angles outside that supported set require explicit
`allow-approximate-angles=true` on `quake-lower-pbc` to become floating operands.
That option is rounding at the representation boundary, not approximate
Clifford+T synthesis or an end-to-end error guarantee. The optional wrapper does
not enable it.

Terminal commuting observables use a **shared Clifford readout network**. It maps
each observable to Z on its designated physical readout wire, implements signs
coherently, then measures. All Clifford work precedes the first measurement.
There is no quantum uncompute after readout. Readouts needing unsupported
classical remapping or additional qubits are rejected. This function-level
lowering uses the sink-only contract; it need not preserve an unobservable
post-measurement state.

## 3. Optimization methods

### 3.1 Clifford-aware Pauli rotation merging

Transport rotation axes into a common Clifford frame and index candidate equal
axes. Before combining angles, replay the intervening gates and check the signed
transport and commutation conditions. Equal axes alone are not a proof: an
intervening anticommuting rotation may block a merge. Forward and adjoint searches
can expose different candidates. Exact quarter-turn pairs can cancel or become
Cliffords, reducing T count without approximate synthesis.

This uses the Pauli-rotation viewpoint of Zhang and Chen,
[“Optimizing T gates in Clifford+T circuit as pi/4 rotations around Paulis”](https://arxiv.org/abs/1903.12456).
It is also related to the rotation-merging problem studied by Vandaele, Perdrix
and Vuillot in
[“Optimal number of parametrized rotations and Hadamard gates in parametrized Clifford circuits with non-repeated parameters”](https://arxiv.org/abs/2407.07846).
Our implementation uses a latest-candidate axis index and explicit replay. It
is not a literal implementation of FastTMerge's algorithm or its complexity
bound, and does not inherit a global T-optimality guarantee.

### 3.2 Bounded commuting-block decoding

Express a commuting block in independent signed Pauli generators. Odd phase
coefficients correspond to T rotations. Search the associated small Reed–Muller
coset, reconstruct even Clifford and scalar corrections, then verify the exact
phase function for every assignment of generator eigenvalues. The independent
rank is capped at five, where there are 64 coset codewords; it is not a five-qubit
limit on the physical circuit.

The decoding relation is adapted from Amy and Mosca,
[“T-count optimization and Reed–Muller codes”](https://arxiv.org/abs/1601.07363).
We use exhaustive bounded-rank decoding and exact signed-Pauli reconstruction,
not an unrestricted whole-circuit decoder. It can find identities beyond
pairwise merging, but additional parity computation can outweigh the T saving
under a gate-count or depth objective.

The decoder also tries placing reconstructed rotations in favorable frames of
the existing Clifford network. That reuse applies to its native candidate.
Preserving the candidate network at the final emission boundary matters:
normalization followed by independent gadgets can discard its placement benefit.

### 3.3 Signed Pauli-frame propagation

The total-gate family removes explicit X/Y/Z gates into a pending signed Pauli
frame. H exchanges its X/Z components; CX spreads X from control to target and Z
from target to control. Anticommutation changes a rotation's sign. S/Sdg can
exchange with the corresponding exact scalar correction. At the boundary, emit
the residual Pauli product. Forward and adjoint traversals expose different
cancellations, and local cleanup tests the resulting gate cost.

The candidate cleanup folds named S/Sdg/Z and exact Rz gates using one phase
coefficient, retaining the scalar correction between the two conventions. It
then exposes a Pauli in equal-cost phase decompositions: `S T = Z Tdg` and
`Sdg Tdg = Z T` for named gates. Each side has two gates and one T, but the Z
can propagate through a Clifford network and cancel. Exact rotations retain
their additional scalar phase. Candidate ranking happens after this cleanup;
otherwise an equal-cost representation choice can hide a later improvement.

This implements the propagation idea in the **Gate-count fixes study**, inside
the shared native representation. The implementation retains the scalar phases
needed by native unitary semantics, explores both directions, and composes the
result with other strategies. Expected opportunities are repeated Pauli
corrections separated by Clifford networks. The pending frame can spread and
increase the number of emitted Paulis, so propagation is a candidate rather
than an unconditional rewrite.

### 3.4 Phase placement on existing parity networks

Within supported CX/SWAP and exact Z-phase blocks, track the binary parity held
by each wire. Combine rotations with the same parity and place the result at an
existing occurrence of that parity. Explore both the first and last available
locations. Last placement uses the full lifetime of a parity: immediately
before the next gate changes it, or at the block end. A CX control retains its
parity, so its phase can pass that CX. The implementation records these endpoints
lazily instead of scanning every wire after every entangler. The entangling skeleton remains intact; the selected placement can
expose inverse pairs or shorter adjacent Clifford words.

Replay the output parity map and exact rotation coefficients modulo 16 to check
that relocation preserved the block, including scalar signs. This adapts the
phase-polynomial viewpoint also used in the Amy–Mosca work. It does not implement
a general nonlinear Boolean optimizer, arbitrary parity-network synthesis or
phase folding across a retained Toffoli. Benefits require repeated parities or
placement-sensitive neighboring gates.

### 3.5 Small Clifford-window synthesis

Find contiguous Clifford windows on one or two wires. A breadth-first table of
signed Clifford actions supplies a shortest word in the implemented native gate
basis. Recover and independently check its scalar correction before replacement.
The table has a fixed small width; the full circuit can be much wider. Retain
windows when a checked shorter replacement is unavailable.

A shortest total-gate word can use more entanglers. The enclosing optimizer
passes its absolute entangler ceiling into this strategy. If a cleaned proposal
exceeds that ceiling, retry the same proposal with window replacements that do
not increase the window's entangler count. This retains independent Pauli,
phase-placement and other window improvements instead of rejecting the entire
proposal. Both attempts share the work budget and exact checks. An unrestricted
result that satisfies the ceiling is retained, allowing local increases offset
by cancellations elsewhere. The retry is conservative: it does not enumerate
every alternative word under an entangler constraint.

This generalizes local Clifford/template cleanup from the **Gate-count fixes
study**: equivalent Clifford words can be recognized through their Pauli action
rather than individual Quake rewrite patterns. Shortest-word optimality is only
for that window and declared basis. It does not prove global circuit optimality,
include CY as a one-gate native primitive, or optimize routed hardware cost.

### 3.6 Pauli graph synthesis

Construct a DAG whose dependencies preserve the order of anticommuting rotations.
Merge compatible equal axes, emit available one-wire rotations, and greedily
choose entangling actions that reduce the remaining support/frame cost. Ready-node
weight, terminal-frame weight, candidate sampling and ASAP scheduling credit
control this search. The final Clifford action is reconstructed and checked by
independent replay.

This adapts the rotation/frame subset of Paykin et al.,
[“PCOAST: A Pauli-based Quantum Circuit Optimization Framework,” extended v2](https://arxiv.org/abs/2305.10966v2),
especially Figure 12 and Section V-C. Resource bounds, scalar-phase checking,
ASAP scoring and portfolio acceptance are implementation choices. The pass does
not reproduce the paper's full preparation/measurement graph machinery or claim
its benchmark results. The unified native optimizer uses unitary hold semantics;
there is no release-mode pass option.

### 3.7 Composition and selection

The combined portfolio keeps distinct candidates under a bounded beam, composes
T reduction and graph synthesis, and can temporarily permit extra entanglers
before reconstruction. The total-gate family adds propagation, placement and
Clifford-window candidates while retaining the original as fallback. Shared
budgets cover search and checks; increasing a budget does not prove or guarantee
better output.

The `total-gates` objective ranks total gates first and forbids a baseline
T-count increase. Existing objectives remain `t-count`, `t-depth`, `entanglers`
and `depth`. A native winner is not automatically a pass-level improvement:
final comparison includes the chosen emitted network and readout cost, then
applies growth and representability guards. Total gates measured after later
CUDA-Q cleanup remain a separate benchmark result.

With `objective=total-gates` and `gate-count=true`, each distinct, verified,
growth-admissible T-search proposal is offered to the gate-count family **before
the inner T-only beam prunes it**. The enclosing optimizer ranks the cleaned
candidates by its total-gate objective and keeps the best admissible result.
Standalone T optimization retains its original T-first ordering. This uses a
borrowed candidate visitor; it does not duplicate the merging or decoding
algorithms. Refinement calls consume the same outer candidate/work limits and
emit telemetry linked to the T proposal that produced their input. No budget is
increased implicitly.

The native total-gate portfolio is our composition of these methods. It is not a
claim to reproduce a named paper's full optimizer. There is no AlphaTensor model,
unbounded global search, magic-state factory optimization or automatic hardware
mapping in this implementation.

## 4. Correctness and validation

Runtime validation checks ownership, supported operations and attributes, angle
representability, commuting terminal measurements and readout identity. Strategy
checks cover signed rotation transport, exact phase functions and Clifford replay.
The generated MLIR is verified before commit. MLIR verification alone checks IR
validity; it does not prove quantum equivalence.

The regression suite includes independent small-circuit full-unitary comparisons
with scalar phase, all-joint-outcome effect comparisons including off-diagonal
entries, corrupted-witness rejection, high-qubit-index round trips, unknown
metadata and late unknown-operation fallbacks, exact-angle failures, budget and
size bounds, and telemetry behavior. New gate-count methods require isolated
strategy tests as well as composition and MLIR round-trip tests.

These checks support the implementation's stated contracts; they are not a
machine-checked proof of the entire compiler. Benchmark equivalence evidence
must name the checker and its limits. A successful all-zero simulation alone
cannot establish arbitrary-input equivalence. Do not use toy regression fixtures
as added cudaq-bench performance cases.

## 5. Telemetry

`quake-optimize-pbc` and `quake-try-pbc` can append JSONL records using
`telemetry=<path>` or `CUDAQ_PBC_TELEMETRY=<path>`. Explicit options take precedence;
`telemetry=off` disables reporting and `telemetry=-` writes to stderr. Use separate
files for independent processes. Write failure warns without changing the
optimization outcome.

Records include effective native settings and the textual pass configuration,
committed/skipped/failed status, stage and method timings, candidate costs,
selection ancestry, proof/search counters and observed IR metrics. Detailed
events are capped; summaries continue, and truncation is reported. Method timings
are inclusive and candidate improvements refer to individual transitions;
neither should be added indiscriminately across nested or discarded candidates.

Observed total gates exclude measurements; observed depth includes them. Named
T counts do not predict the future synthesis cost of unresolved rotations.
`unsynthesized_rotations` and `t_count_complete` expose that distinction;
magic-state consumption is unavailable. Native costs, readout-inclusive pass
costs and observed IR counts must retain their separate labels. Telemetry
supplements the benchmark command/configuration record; it does not replace it.

## Appendix A. Complete experimental PBC pass option reference

These are the MLIR pass-option defaults. Native C++ APIs can have different
defaults: the native combined optimizer allows zero final two-qubit growth,
while the MLIR pass uses the permissive finite allowance listed below.

### A.1 Pass scope and invocation

| Pass | Purpose | Options |
|---|---|---|
| `quake-verify-pbc` | Check that the entire function satisfies the supported PBC contract | No pass-specific options |
| `quake-to-pbc` | Convert supported Quake to rotations and terminal Pauli measurements | No pass-specific options |
| `quake-optimize-pbc` | Optimize already-verified PBC form; emit PBC | All 38 shared options in A.2–A.6, including A.3a |
| `quake-try-pbc` | Attempt conversion, optimization, and gate lowering as a per-function transaction | All 38 shared options, plus the three wrapper options in A.7 |
| `quake-lower-pbc` | Lower verified PBC to primitive Quake | The angle-lowering option in A.8 |

Options are space-separated inside the pass argument. For example, this requests
a T-only experiment with explicit final entangler/depth growth guards:

```sh
cudaq-opt input.qke \
  --quake-try-pbc='graph-synthesis=false max-two-qubit-growth=0 max-depth-growth=0 telemetry=pbc.jsonl' \
  -o output.qke
```

This is an illustrative command, not a recorded benchmark invocation.
Use `quake-optimize-pbc` only on supported PBC input; it does not implicitly run
conversion. Both optimization passes are opt-in. T reduction and graph synthesis are enabled by default; the total-gate family
is opt-in. Eligibility and search budgets apply to every enabled family.
No option disables correctness checks. Invalid configuration fails the pass,
including invalid nested controls for a disabled strategy.

### A.2 Strategy selection, objectives, and composition

| Option | Default | Behavior and tuning effect |
|---|---|---|
| `t-count` | `true` | Enable the T-reduction family: merging and commuting-block decoding. Disabling saves its search/proof work; graph preprocessing can still merge rotations. |
| `graph-synthesis` | `true` | Enable checked graph-synthesis candidates and the temporary-growth/reconstruction branch. Disabling leaves the enabled T and total-gate strategies. This controls an internal candidate strategy, not an alternative final MLIR lowering route. Disabling all three strategy families leaves the function unchanged. |
| `objective` | `t-count` | Choose `t-count`, `t-depth`, `entanglers`, `depth`, or `total-gates`. Ranking is lexicographic: a later metric cannot compensate for a worse earlier metric. `total-gates` ranks total gates first and forbids a baseline T increase. Changing the objective does not retune graph scoring weights. |
| `portfolio-rounds` | `2` | Positive maximum number of combined T/graph/total-gate composition rounds. More rounds can expose further opportunities, but consume the shared budget; convergence may stop earlier. Distinct from `rounds` inside each T search. |
| `portfolio-beam-width` | `2` | Retain this many distinct combined candidates between rounds; valid range 1–64. One is greedy; larger values preserve alternatives at greater memory/work cost and may spend the budget before later rounds. |
| `max-two-qubit-growth` | `4294967295` | Absolute final mechanical entangler allowance above the original input, not a percentage. Zero forbids net growth. The default is permissive but finite; a T improvement can otherwise come with many additional entanglers. |
| `max-depth-growth` | `4294967295` | Absolute final mechanical circuit-depth allowance above the original. Zero forbids depth growth. This is neither a separate T-depth bound nor a physical-duration constraint. |
| `intermediate-two-qubit-growth` | `256` | Extra temporary entanglers above the final ceiling for a T-reduction candidate followed by graph reconstruction. Zero disables that extra branch. Requires both strategy families to be useful; final growth guards still apply. Larger allowances add exploration, not permission to commit oversized results. |

The native search and final pass acceptance operate on different representations.
Let T be T count, E entangler count, D circuit depth, G total gates, and TD T depth:

| Objective | Native candidate ranking, first metric first | Final comparison after the chosen lowering |
|---|---|---|
| `t-count` | T, E, D, G, TD | T, E, D, TD |
| `t-depth` | TD, T, E, D, G | TD, T, E, D |
| `entanglers` | E, D, T, G, TD | E, D, T, TD |
| `depth` | D, E, T, G, TD | D, E, T, TD |
| `total-gates` | G, T, D, E, TD | G, T, D, E, TD |

The native T objectives additionally disallow an increase over the original
native T count. The total-gate objective has the same T guard; the entangler/depth objectives do not impose that separate
T guard. Final acceptance requires a strict improvement under the final tuple
and compliance with the growth bounds. Its costs include mechanical measurement
lowering. A better native circuit can therefore be rejected after normalization.
Total gates are the primary native and final metric only for `total-gates`;
for the other objectives their earlier tie-breaker role is unchanged. These are logical gate estimates, not routed hardware or
magic-factory costs; arbitrary rotations' eventual synthesized T costs remain a
downstream concern.

### A.3 T-reduction controls

These controls apply inside each enabled T-portfolio invocation.

| Option | Default | Behavior and tuning effect |
|---|---|---|
| `merge` | `true` | Propose equal-axis exact-rotation merges after Clifford transport; replay authorizes each merge. Usually a cheaper source of T cancellation. Disabling does not disable graph's separate `synthesis-simplify` step. |
| `reverse` | `true` | Also search the adjoint/reversed circuit and invert accepted candidates back. Can expose different merge or placement opportunities, at additional work cost. |
| `decode` | `true` | Explore exact rewrites of bounded commuting phase blocks. Can find T reductions beyond pairwise merging, but reconstruction may add entanglers/depth. Disabling selects a cheaper merge-focused search. |
| `reuse-cliffords` | `true` | For decoded terms, try placement in favorable frames of the existing Clifford network as well as the no-reuse variant. False retains only the no-reuse variant. No effect when decoding is disabled. Reuse can avoid fresh parity gadgets but needs extra scans. |
| `rounds` | `4` | Positive maximum composition rounds inside one T invocation. Independent of outer `portfolio-rounds`. More rounds may exploit newly exposed relations; budgets and convergence can stop the search earlier. |
| `beam-width` | `2` | Positive number of distinct circuits retained between T rounds. One is greedy; wider beams increase storage and candidate work. This is separate from the outer `portfolio-beam-width`. |
| `max-block-rank` | `5` | Independent commuting-generator rank limit, 1–5, not physical qubit count. Rank five permits 64 coset codewords. Smaller ranks reduce work but miss identities; below four there are no nontrivial coset alternatives, although duplicate terms can still combine. |
| `max-block-gates` | `256` | Positive input-operation cap per decoding block, including Clifford/scalar operations. Larger blocks can expose distant relations but increase scans. Rank, angle, and commutation boundaries still limit a block. |
| `merge-window` | `0` | Maximum gate-index distance between merge endpoints; zero imposes no distance cap, while work limits still apply. Smaller positive values shorten replay scans but may miss distant merges. No effect on this strategy when `merge=false`. |

### A.3a Total-gate strategy controls

These options apply to both optimization passes. A strict PBC output can lose
network-placement savings when subsequently lowered; the optional wrapper's
`preserve-clifford-network` option retains that network through emission.

| Option | Default | Behavior and tuning effect |
|---|---|---|
| `gate-count` | `false` | Enable the native total-gate strategy family. Can run independently of T decoding and graph synthesis. With `objective=total-gates`, it also refines verified T proposals before T-only pruning. |
| `gate-count-rounds` | `3` | Positive greedy composition-round limit within each gate-count invocation. |
| `gate-commutation-window` | `128` | Maximum backward scan for checked local cancellations; zero removes the distance bound, not the shared work bound. |
| `pauli-propagation` | `true` | Explore forward and adjoint signed Pauli-frame propagation within the gate-count family. |
| `phase-placement` | `true` | Explore first and last existing parity locations for collected exact phases. |
| `clifford-windows` | `true` | Minimize one-/two-wire Clifford windows using shortest native-basis words with exact phase checks. |

### A.4 Shared search and storage bounds

| Option | Default | Behavior and tuning effect |
|---|---|---|
| `max-candidates` | `256` | Positive strategy-invocation cap. Sets both the combined outer cap and the separate cap inside each T invocation; nested gate-count refinements consume outer slots. It is not one flat count of all decoded blocks or graph gates. Larger values allow more exploration only if work remains. |
| `max-work` | `10000000` | Positive deterministic search/proof work allowance, shared across strategies and final PBC normalization, also supplied as each T call's local ceiling. Units estimate work, not gates or time. Raise only to study budget-limited searches; larger budgets do not guarantee better output. |
| `timeout-ms` | `0` | Shared search deadline in milliseconds; zero disables it. Checked at budget charges, not by preemption. It is not a hard deadline for parsing, every cost calculation, or the full compiler pipeline. Wall-clock cutoffs can change results between runs; use zero with fixed work for repeatable comparisons. |
| `max-gates` | `8192` | Positive bound on native input/candidate gate-vector sizes; also clamps graph output. Oversized inputs skip optimization and oversized candidates are rejected. Not a byte limit or a bound on all final MLIR operations; no gates are truncated. |

A local strategy failure can reject only that candidate. If the shared budget
is exhausted, the MLIR wrapper keeps the original function even if native search
had already found an incumbent improvement. Raising beam widths or search windows
does not bypass the shared budget. Use telemetry to distinguish no improvement
from a search that stopped at a limit.

### A.5 Graph-synthesis controls

These controls affect candidate construction when `graph-synthesis=true`.
Weights must be finite and nonnegative. They influence heuristic search, not
the equivalence contract or final objective ordering.

| Option | Default | Behavior and tuning effect |
|---|---|---|
| `synthesis-parallelization-credit` | `0.0` | Credit actions whose emitted depth fits into current ASAP scheduling slack. Zero disables the reward. Raising it can favor circuit depth at the expense of entanglers/frame cost; it does not directly score T depth. |
| `synthesis-frontier-weight` | `1.0` | Extra weight for cost changes on dependency-free nodes, scaled by population/frontier size. Higher values prioritize immediate progress; zero removes the extra weighting, not dependency constraints. Excessive weighting may miss sharing with later nodes. |
| `synthesis-frame-weight` | `0.1` | Weight the mean terminal-frame factor-cost change. Higher values penalize leaving an expensive Clifford reconstruction; zero ignores that cost contribution, never the frame's semantics. |
| `synthesis-lookahead` | `0` | Cap the remaining-node scoring sample; zero scores all. A positive cap reduces work but may miss distant effects. This is a population sample, not recursive lookahead through future circuits. Frontier/frame scoring remains separate. |
| `synthesis-candidate-nodes` | `0` | Cap the minimum-cost ready nodes used to generate actions; zero uses all eligible nodes. Each node can generate multiple wire-pair/basis candidates. Smaller caps reduce work but introduce more order sensitivity. |
| `synthesis-simplify` | `true` | Merge compatible equal-axis rotations before constructing the dependency DAG. Can reduce rotations and T/entangler work; its potentially quadratic scans consume budget. Disable to isolate synthesis or study preprocessing cost. |
| `synthesis-asymmetric` | `true` | Carry trailing local Clifford basis changes into the residual graph/frame rather than immediately emitting their inverses. Can share one-qubit work but shifts frame-reconstruction cost; compare both settings when studying that trade-off. |
| `max-dependencies` | `1000000` | Positive DAG-edge cap. Dense graphs can hit it even with few candidate nodes. Exceeding it rejects the candidate; necessary dependencies are never omitted. The scoring horizon does not bound DAG construction. |
| `synthesis-max-steps` | `1000000` | Entangling-action limit, including terminal-frame reconstruction. Zero permits no such actions; it does not mean unlimited. Exceeding it rejects the candidate. It counts neither all emitted gates nor full-circuit portfolio rounds. |

The combined optimizer requires unitary hold semantics internally; there is no
release-mode pass option. Native standalone controls such as `preferDepth` and
`acceptOnlyImprovement` are not additional MLIR knobs: the portfolio supplies its
own ranking and acceptance. Graph controls affect candidate construction, not
the final emission route. Strict optimization emits normalized PBC; the optional
wrapper can preserve the native network with `preserve-clifford-network=true`.

### A.6 Telemetry controls

| Option | Default | Behavior and tuning effect |
|---|---|---|
| `telemetry` | Empty string | Empty uses `CUDAQ_PBC_TELEMETRY`, or disables reporting if unset. A file path appends JSONL; `-` writes to stderr; `off` explicitly disables reporting even when the environment enables it. A nonempty option takes precedence. Reporting failures warn without failing optimization. |
| `telemetry-max-events` | `4096` | Maximum detailed events retained per function. Zero keeps summaries only. Aggregate counters continue after the cap; dropped-event and ancestry-completeness fields identify truncation. Lowering this cap bounds reporting memory, not optimization work. |

Telemetry records effective options, strategy timings, candidate costs, and
selection decisions as described in the telemetry section. Instrumentation does not change
ranking, but its overhead can affect a wall-clock deadline. Compare output with
fixed work and time the pass with telemetry disabled as well. Use separate files
for independent processes or experiments to avoid mixing appended records.

### A.7 Additional `quake-try-pbc` controls

These three options are exclusive to the optional round-trip wrapper; its other
controls are inherited unchanged from `quake-optimize-pbc`.

| Option | Default | Behavior and tuning effect |
|---|---|---|
| `preserve-clifford-network` | `false` | Emit the verified native Clifford/rotation network and original terminal observables instead of normalizing it to independent PBC gadgets. Important for retaining total-gate savings. This does not permit quantum outputs or bypass import, representability or output verification. |
| `emit-remarks` | `false` | Explain clean fallbacks such as unsupported input, limits, or no admissible improvement. Does not mutate IR or turn failures into skips. Useful for distinguishing eligibility from search quality. |
| `max-operations` | `32768` | Positive total function-operation cap checked before optimization; also bounds the conservative candidate-emission estimate before cloning. Counts more than native quantum gates. Lower values avoid large transactions; larger values admit more work/storage. Zero is invalid. |

### A.8 `quake-lower-pbc` angle control

| Option | Default | Behavior and tuning effect |
|---|---|---|
| `allow-approximate-angles` | `false` | False rejects exact rational angles outside supported Clifford+T multiples. True permits their conversion to floating-point rotation operands, losing exact rational encoding through rounding. It does not perform Clifford+T approximation, set a synthesis tolerance, or prove an end-to-end error bound. Exact Clifford+T angles still use named gates; opaque SSA angles remain rotations. |

This option belongs only to explicit `quake-lower-pbc`. The optional
`quake-try-pbc` round trip does not enable that conversion: an unrepresentable
exact-angle candidate is left uncommitted. An exact Clifford+T comparison must not silently introduce this approximation.

## Appendix B. Measured configuration: total-gates-v1

The table uses one configuration across the ten reference cases. This is an
explicit total-gate experiment, not the default pass configuration. `total-gates`
selects by emitted-network cost; `gate-count=true` enables the native methods;
`preserve-clifford-network=true` prevents their Clifford-network savings from
being lost in PBC renormalization. The final two-qubit-growth allowance is zero;
T count cannot increase under this objective. Depth growth is permitted and is
reported per case. Graph synthesis is disabled in this configuration. The reproduction fixes use
these same parameters, including `reverse=true`, the work bound and the candidate
limits; they change candidate construction and selection, not the tuning budget.

Every option is explicit below, including controls for the disabled graph family.
The only case-specific option is the telemetry output filename. From the compiler
checkout, with `case_dir` set to the absolute case-artifact directory. The
array contains the 40 case-independent options; the invocation supplies the
41st option, the telemetry filename. Appendix D uses the same array for all ten
cases:

```bash
pbc_options=(
  objective=total-gates
  gate-count=true
  gate-count-rounds=3
  gate-commutation-window=128
  pauli-propagation=true
  phase-placement=true
  clifford-windows=true
  t-count=true
  graph-synthesis=false
  merge=true
  reverse=true
  decode=true
  reuse-cliffords=true
  rounds=2
  beam-width=2
  max-block-rank=5
  max-block-gates=256
  merge-window=0
  max-candidates=32
  max-work=100000000
  timeout-ms=0
  max-two-qubit-growth=0
  max-depth-growth=4294967295
  portfolio-rounds=2
  portfolio-beam-width=2
  max-gates=65536
  intermediate-two-qubit-growth=0
  synthesis-parallelization-credit=0.0
  synthesis-frontier-weight=1.0
  synthesis-frame-weight=0.1
  synthesis-lookahead=0
  synthesis-candidate-nodes=0
  synthesis-simplify=true
  synthesis-asymmetric=true
  max-dependencies=1000000
  synthesis-max-steps=1000000
  telemetry-max-events=4096
  emit-remarks=true
  max-operations=131072
  preserve-clifford-network=true
)
env -u CUDAQ_PBC_TELEMETRY OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  prlimit --as=3221225472 --cpu=120 --stack=67108864 --core=0 -- \
  "$PWD/build/bin/cudaq-opt" "$case_dir/baseline.qke" \
  "--quake-try-pbc=${pbc_options[*]} telemetry=$case_dir/telemetry.jsonl" \
  -o "$case_dir/pbc.qke" --mlir-disable-threading
```

The run artifacts record the expanded argument vector before each invocation,
all source/build identities, the executable hash, input/output hashes and logs.
The compiler build is Debug; one compiler process runs at a time. Resource limits
bound each child process. A limit failure is not a completed optimization.

CUDA-Q first runs the suite's Clifford+T target pipeline with epsilon `0.001` and
seed `42`. To use the scalar-wire pass at this boundary, `regtomem` is omitted;
the older `optimize-1q-clifford-t` name in the suite profile is replaced by the
current `func.func(normalize-1q-clifford-t)` pass. The expanded baseline pipeline
is:

```text
func.func(add-dealloc,combine-quantum-alloc,canonicalize,dqe),convert-to-linear-values,unitary-synthesis,apply-op-specialization,func.func(constant-propagation),decomposition{basis=h,s,t,rz,x,y,z,swap,mx,my,mz,x(1),y(1),z(1)},func.func(quake-simplify{threshold=0.001 rotations-to-clifford-t=true clifford-t-epsilon=0.001},normalize-phase-placement,lower-phase,canonicalize,dqe,phase-folding,canonicalize),clifford-t-synthesis{epsilon=0.001 seed=42 skip-below=0 fail-on-controlled-rotation=true},inline,func.func(quake-simplify),func.func(normalize-1q-clifford-t),func.func(phase-folding,normalize-phase-placement,lower-phase,canonicalize,dqe),symbol-dce
```

Both sides receive the following cleanup at the same final boundary. It is also
run before the comparison to obtain `baseline.qke`; the PBC side inserts the
actual pass between the two cleanup invocations, while the reference does not.

```text
func.func(quake-simplify,phase-folding,canonicalize),func.func(normalize-1q-clifford-t),func.func(phase-folding,normalize-phase-placement,lower-phase,canonicalize,dqe)
```

Qiskit uses the suite's `build_clifford_t_pass_manager(optimization_level=2,
synthesis_error=0.001)`, with `seed_transpiler=0`, then `BasisTranslator` to
`h,s,sdg,t,tdg,x,y,z,cx,cy,cz,swap`. Both compilers consume the same saved suite
QPY input. Eight CUDA-Q inputs were imported afresh with
`cudaq.contrib.from_qiskit`; the two largest adders reuse the original paired
Quake frontend inputs. No optimization output is used as a replacement source.
This measures
the named suite pipeline with the stated adaptations, not every CUDA-Q target or
an unspecified default optimization level.

Independent checking compares the actual baseline and optimized gate circuits
up to global phase; physical readout order is checked separately. MQT QCEC uses
version 3.11.0, one thread and a 20-second checker timeout, with simulation
checking disabled.
For up to 15 qubits, the alternating decision-diagram checker is enabled; wider
cases use the ZX checker. Construction checking is disabled. Inconclusive checks
remain labeled inconclusive. These checks concern equivalence of PBC output to
the synthesized CUDA-Q baseline, not the earlier approximation error against the
source circuit or Qiskit's separate compilation.

## Appendix C. Closing the earlier reproduction gaps

The earlier feasibility study supplied target counts, but its optimizer is not
used to produce the measured PBC column. This comparison records the bridge from
that study into the compiled pass. All ten final circuits are produced by
`quake-try-pbc` with configuration **total-gates-v1** and the common downstream
CUDA-Q cleanup from Appendix B. The original Quake inputs are preserved. The
historical harness column is a reference only; it is not relabeled as an
implementation result.

| Case | Earlier feasibility harness total | Implemented pass before fixes | Implemented pass after fixes |
|---|---:|---:|---:|
| adder_n4_transpiled | 22 | 22 | 22 |
| adder_n10_transpiled | 106 | 106 | 106 |
| adder_n28_transpiled | 316 | 320 | 316 |
| adder_n64_transpiled | 736 | 748 | 736 |
| adder_n118_transpiled | 1366 | 1390 | 1366 |
| adder_n433_transpiled | 5041 | 5135 | 5041 |
| multiply_n13_transpiled | 91 | 93 | 91 |
| rc_adder_6 | 147 | 153 | 147 |
| csla_mux_3 | 156 | 156 | 156 |
| qft_n002 | 8 | 8 | 8 |

**Phase representation and cleanup.** The four larger adders retained extra
Paulis; their T and two-qubit counts already matched the earlier results.
The native cleanup split exact rotations into S/Sdg and odd rotations, hiding
Paulis that the earlier cleanup exposed. Folding mixed named/rotation phases
and choosing the Pauli-exposing decomposition in Section 3.3 recovers the adder
and multiply totals. Isolated compiled-pass checks recover adder-28 from
320 to 316 and multiply-13 from 93 to 91 while leaving rc_adder_6 at 153.
Changing traversal order or adding a one-qubit normalizer alone did not close
those gaps; those experimental changes are not retained.

**Pruning before reconstruction.** In rc_adder_6, a T-search proposal with
47 T gates, 80 entanglers and 161 total gates was discarded in favor of one with
47 T gates, 72 entanglers and 153 total gates. Native gate-count cleanup turns
the former into 147 total gates and 76 entanglers. Section 3.7 makes this cleanup
visible before pruning. Disabling reverse T search isolated the profitable
forward branch during diagnosis; the final configuration keeps `reverse=true`
and evaluates both. This is a total-gate trade-off: relative to the previous PBC
result, six total gates are saved while four entanglers are added. Relative to
the CUDA-Q baseline, entanglers still decrease from 91 to 76. T count stays at
47, circuit depth stays at 91, and T depth improves from 35 to 31.

**Placement correctness.** Last-parity placement previously recorded where a
parity was created rather than the end of its lifetime. The corrected endpoint
rule in Section 3.4 is covered by a three-qubit phase-sensitive regression test.
It is not claimed as the cause of the six total-count gaps above.

**Validation.** All ten final outputs pass independent unitary equivalence
checking up to global phase with unchanged physical readout order. The native
regressions also check scalar-sensitive equality for mixed-phase folding and
Pauli exposure, last-parity placement, and both verified T alternatives before
one-entry beam pruning. Together with the existing tests, 69 relevant unit tests
and 11 PBC lit tests pass. These results establish reproduction on these ten
cases; they are not a claim of global optimality or a full-suite comparison.

## Appendix D. Reproduce the ten cases with the compiled pass

This procedure reruns the **actual `quake-try-pbc` implementation** on the saved
original Quake inputs. No Python code below optimizes a circuit. Python only
checks input identity, counts/exports emitted gates, and invokes the independent
checker. The procedure recreates the CUDA-Q and PBC columns; it does not rerun
the external proposal or the historical feasibility optimizer.

### D.1 Build and identify the inputs

Use the compiler checkout's configured build and Bash. Build with bounded
parallelism, then use a new output directory so earlier evidence remains intact:

```bash
set -euo pipefail
cmake --build build --target cudaq-opt -j 2
pbc_opt="$PWD/build/bin/cudaq-opt"
pbc_inputs="$PWD/build/pbc-pass-total-gates-2026-11-03"
pbc_run=$(mktemp -d "$PWD/build/pbc-rerun.XXXXXX")
export pbc_opt pbc_inputs pbc_run
sha256sum "$pbc_opt" > "$pbc_run/compiler.sha256"
git diff HEAD --binary > "$pbc_run/source.patch"
```

`pbc_inputs` is the recorded run-artifact directory, **not a tracked repository
fixture**. On another machine, copy that input bundle and set `pbc_inputs` to
its absolute path. Each directory listed below must contain `input.mlir`,
`case.json` and `frontend.json`; the last file records the input IR hash. Preserve
`input.qpy` and corpus metadata too when transferring evidence or remeasuring
Qiskit. A clean checkout alone does not contain these benchmark inputs.

Use the original `input.mlir`, never a harness-optimized circuit or a previous
`pbc.qke`/`output.qke`. The two largest adders already have valid original Quake
in this bundle; frontend reimport is unnecessary. Fresh import of a different
corpus revision is a new experiment, even when its case name is unchanged.

Create the exact ten-case selection. The first column is the artifact key, not
a compiler flag; the case metadata retains the full cudaq-bench identifier.
The final two columns are the expected CUDA-Q and PBC totals for this report.

```bash
cat > "$pbc_run/cases.tsv" <<'CASES'
e00d79c85c8feec3 adder_n4_transpiled 24 22
c8d218ddd8ec5c12 adder_n10_transpiled 140 106
3773e2a7db8249a0 adder_n28_transpiled 418 316
e34cda2691d90935 adder_n64_transpiled 974 736
6b7e3d29dacfc84d adder_n118_transpiled 1808 1366
86f55a6e5e38dc4a adder_n433_transpiled 6673 5041
918d9599d170a728 multiply_n13_transpiled 105 91
d4175a1a2989e505 rc_adder_6 191 147
71e5e508c0a01e47 csla_mux_3 161 156
1cc4d49e61d1dfa8 qft_n002 8 8
CASES
python3 - <<'PYINPUTS'
import hashlib, json, os, pathlib
root = pathlib.Path(os.environ['pbc_inputs'])
selection = pathlib.Path(os.environ['pbc_run']) / 'cases.tsv'
for line in selection.read_text().splitlines():
    key, name, _, _ = line.split()
    folder = root / key
    metadata = json.loads((folder / 'case.json').read_text())
    frontend = json.loads((folder / 'frontend.json').read_text())
    actual = hashlib.sha256((folder / 'input.mlir').read_bytes()).hexdigest()
    assert actual == frontend['ir_sha256'], (name, 'input hash mismatch')
    assert metadata['case_id'].endswith(name), (name, metadata['case_id'])
    print(name, metadata['case_id'], actual)
PYINPUTS
```

### D.2 Define the exact configuration and pipelines

In the **same Bash session**, execute the `pbc_options=(...)` array definition
from Appendix B. Do not use the short illustrative command in Appendix A;
it uses defaults and is not the measured total-gate configuration. Keep
`reverse=true`, `graph-synthesis=false`, `objective=total-gates`,
`gate-count=true` and `preserve-clifford-network=true`, as well as all the
recorded numeric limits. The commands below supply telemetry per case.

```bash
test "${#pbc_options[@]}" -eq 40
pbc_target='func.func(add-dealloc,combine-quantum-alloc,canonicalize,dqe),convert-to-linear-values,unitary-synthesis,apply-op-specialization,func.func(constant-propagation),decomposition{basis=h,s,t,rz,x,y,z,swap,mx,my,mz,x(1),y(1),z(1)},func.func(quake-simplify{threshold=0.001 rotations-to-clifford-t=true clifford-t-epsilon=0.001},normalize-phase-placement,lower-phase,canonicalize,dqe,phase-folding,canonicalize),clifford-t-synthesis{epsilon=0.001 seed=42 skip-below=0 fail-on-controlled-rotation=true},inline,func.func(quake-simplify),func.func(normalize-1q-clifford-t),func.func(phase-folding,normalize-phase-placement,lower-phase,canonicalize,dqe),symbol-dce'
pbc_cleanup='func.func(quake-simplify,phase-folding,canonicalize),func.func(normalize-1q-clifford-t),func.func(phase-folding,normalize-phase-placement,lower-phase,canonicalize,dqe)'
printf '%s\n' "$pbc_target" > "$pbc_run/target-pipeline.txt"
printf '%s\n' "$pbc_cleanup" > "$pbc_run/cleanup-pipeline.txt"
printf '%s\n' 'total-gates-v1' > "$pbc_run/configuration.txt"
printf '%s\n' 'CUDAQ_PBC_TELEMETRY=unset' 'OMP_NUM_THREADS=1' \
  'OPENBLAS_NUM_THREADS=1' > "$pbc_run/environment.txt"
```

These strings freeze the measured pipeline, including epsilon `0.001`, seed
`42`, scalar-wire retention and the current one-qubit normalization pass name.
Do not silently replace them with the latest target profile or a default
optimization level when checking reproduction.

### D.3 Prepare counting and independent checking

Use a Python environment with **Qiskit 2.6.0 and MQT QCEC 3.11.0**. If needed,
create one inside the new run directory:

```bash
python3 -m venv "$pbc_run/check-env"
"$pbc_run/check-env/bin/python" -m pip install 'qiskit==2.6.0' 'mqt.qcec==3.11.0'
pbc_python="$pbc_run/check-env/bin/python"
"$pbc_python" -m pip freeze > "$pbc_run/checker-packages.txt"
```

The following reader is deliberately limited to the single-function,
scalar-wire, named Clifford+T output used by these ten cases. It rejects
unsupported quantum operations and requires unchanged physical readout order.
Gate counts and depths exclude measurements, allocations, sinks and scalar
phase. It exports the unitary gate portion for QCEC, which checks equality up
to global phase. This is not a general Quake interpreter or an optimizer.

```bash
cat > "$pbc_run/check-output.py" <<'PYCHECK'
import hashlib, json, pathlib, re, sys
from qiskit import qasm2
from mqt import qcec

def read_ir(path):
    text=path.read_text();assert len(re.findall(r'func.func @',text))==1
    wires={};gates=[];n=0;measurements=[];terminal=False
    for line in text.splitlines():
        if 'func.func @' in line:
            for q in re.findall(r'(%\w+): !quake.wire',line):wires[q]=n;n+=1
        m=re.search(r'(.*?)= quake\.(\w+)(<adj>)?(?:\s|$)',line)
        if not m:
            if re.match(r'\s*(?:%[^=]+=\s*)?"?quake\.', line) and not line.lstrip().startswith('quake.sink '):
                raise ValueError('Unsupported Quake syntax: '+line)
            continue
        lhs,k,adj=m.groups();lhs=lhs.strip();args=re.findall(r'%[\w.]+(?:#\d+)?',line[m.end():].split(' : ')[0]);qs=[wires[q] for q in args if q in wires]
        if k=='null_wire':wires[lhs]=n;n+=1;continue
        if k=='mz':
            terminal=True;assert len(qs)==1
            names=re.findall(r'%[\w.]+',lhs);assert len(names)==2
            wires[names[1]]=qs[0];measurements.append(qs[0]);continue
        if k=='discriminate':continue
        assert not terminal, 'gate after readout'
        assert k in {'h','s','t','x','y','z','swap','phase'},(k,line)
        if k!='phase':
            name=k+('dg' if adj else '')
            if '[' in line[m.end():].split(' : ')[0]:
                assert k in ('x','y','z') and len(qs)==2
                name='c'+k
            gates.append([name,qs])
        base=lhs.split(':')[0]
        for i,q in enumerate(qs):wires[base+('#'+str(i) if len(qs)>1 else '')]=q
    assert n>0
    depth=[0]*n;td=[0]*n
    for k,qs in gates:
        assert qs
        d=1+max(depth[q] for q in qs);t=int(k in ('t','tdg'))+max(td[q] for q in qs)
        for q in qs:depth[q]=d;td[q]=t
    metrics={'total':len(gates),'t':sum(k in ('t','tdg') for k,_ in gates),'twoq':sum(len(q)==2 for _,q in gates),'depth':max(depth),'tdepth':max(td)}
    return {'n':n,'gates':gates,'measurements':measurements,'metrics':metrics}
def qasm(d):return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[%d];\n'%d['n']+''.join(k+' '+','.join('q[%d]'%q for q in qs)+';\n' for k,qs in d['gates'])

folder = pathlib.Path(sys.argv[1])
name, expected_cudaq, expected_pbc = sys.argv[2:]
a = read_ir(folder / 'reference.qke')
b = read_ir(folder / 'output.qke')
assert a['n'] == b['n'] and a['measurements'] == b['measurements'], 'readout changed'
for label, circuit in [('reference', a), ('output', b)]:
    (folder / (label + '-gates.json')).write_text(json.dumps(circuit, indent=2))
    (folder / (label + '.qasm')).write_text(qasm(circuit))
wide = a['n'] > 15
options = dict(run_simulation_checker=False, run_zx_checker=wide,
               run_alternating_checker=not wide, run_construction_checker=False,
               nthreads=1, timeout=20)
record = dict(case=name, configuration='total-gates-v1', cudaq=a['metrics'],
              pbc=b['metrics'], checker='MQT QCEC 3.11.0', checker_options=options,
              contract='unitary up to global phase; identical readout order',
              hashes={f: hashlib.sha256((folder / f).read_bytes()).hexdigest()
                      for f in ['input.mlir', 'reference.qke', 'output.qke']})
(folder / 'result.json').write_text(json.dumps(record, indent=2))
qa = qasm2.loads(qasm(a), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
qb = qasm2.loads(qasm(b), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
verification = qcec.verify(qa, qb, **options).json()
record['verification'] = verification
(folder / 'result.json').write_text(json.dumps(record, indent=2))
status = verification['equivalence']
print(f"{name}: CUDA-Q={a['metrics']['total']} PBC={b['metrics']['total']} {status}",
      flush=True)
assert status in ('equivalent', 'equivalent_up_to_global_phase'), verification
assert a['metrics']['total'] == int(expected_cudaq), 'CUDA-Q count changed'
assert b['metrics']['total'] == int(expected_pbc), 'PBC count changed'
PYCHECK
```

### D.4 Run one case at a time and record each result

The loop starts from original Quake, reconstructs the suite baseline, runs the
compiled PBC pass, and gives both comparison outputs the same final cleanup.
The independent check completes and prints its counts before the next case
starts. Each command and all 41 PBC options are recorded before execution.
`prlimit` bounds memory, CPU time and stack size; a nonzero exit stops the loop.
There is no statevector simulation of the large adders.

```bash
run_pbc_opt() {
  local log_prefix=$1
  shift
  local command=(env -u CUDAQ_PBC_TELEMETRY OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    prlimit --as=3221225472 --cpu=120 --stack=67108864 --core=0 --
    "$pbc_opt" "$@" --mlir-disable-threading)
  printf '%q ' "${command[@]}" > "$log_prefix.command.sh"
  printf '\n' >> "$log_prefix.command.sh"
  "${command[@]}" > "$log_prefix.stdout.log" 2> "$log_prefix.stderr.log"
}
while read -r key name expected_cudaq expected_pbc; do
  case_dir="$pbc_run/$key"
  mkdir "$case_dir"
  cp "$pbc_inputs/$key/input.mlir" "$pbc_inputs/$key/case.json" \
     "$pbc_inputs/$key/frontend.json" "$case_dir/"
  sha256sum "$case_dir/input.mlir" > "$case_dir/input.sha256"
  printf '%s\n' "${pbc_options[@]}" "telemetry=$case_dir/telemetry.jsonl" \
    > "$case_dir/pbc-options.txt"
  run_pbc_opt "$case_dir/synthesis" "$case_dir/input.mlir" \
    "--pass-pipeline=builtin.module($pbc_target)" -o "$case_dir/synthesized.qke"
  run_pbc_opt "$case_dir/baseline" "$case_dir/synthesized.qke" \
    "--pass-pipeline=builtin.module($pbc_cleanup)" -o "$case_dir/baseline.qke"
  run_pbc_opt "$case_dir/pbc" "$case_dir/baseline.qke" \
    "--quake-try-pbc=${pbc_options[*]} telemetry=$case_dir/telemetry.jsonl" \
    -o "$case_dir/pbc.qke"
  run_pbc_opt "$case_dir/cleanup" "$case_dir/pbc.qke" \
    "--pass-pipeline=builtin.module($pbc_cleanup)" -o "$case_dir/output.qke"
  run_pbc_opt "$case_dir/reference" "$case_dir/baseline.qke" \
    "--pass-pipeline=builtin.module($pbc_cleanup)" -o "$case_dir/reference.qke"
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    prlimit --as=2147483648 --cpu=30 --core=0 -- \
    "$pbc_python" "$pbc_run/check-output.py" "$case_dir" \
    "$name" "$expected_cudaq" "$expected_pbc" | tee "$case_dir/check.log"
done < "$pbc_run/cases.tsv"
```

For a first smoke test, use only the adder-4 row in a separate run's `cases.tsv`.
Keep the complete selection for reproducing the table. Outputs are separate from
the input bundle; reusing an existing output directory fails rather than
silently appending telemetry or mixing old and new evidence.

Inspect `telemetry.jsonl` and `pbc.stderr.log` for each case. The expected pass
outcome is `committed` for nine cases and `skipped` for QFT-2 because no
improvement was selected. An unsupported-input or work-limit skip is not a
successful reproduction just because a circuit was emitted. The JSON count
comparison and QCEC result must both succeed; an inconclusive check must not be
reported as equivalence. `result.json` records final target-basis counts, not
native estimates or the pre-cleanup telemetry depth.

QFT-2 remains at eight gates under the fixed physical-output contract. The
seven-gate Qiskit/proposal references allow the permutation interpretation
explained in the measured-results section; this procedure does not erase that
swap. To regenerate the Qiskit column, use the paired `input.qpy`, the suite
pass manager and basis translation specified in Appendix B, and record its
layout separately. The proposal column remains an external reference.

<!-- BEGIN ADDITIONAL FIVE -->
## Appendix E. Five additional FTQC cases

These are five distinct existing cudaq-bench cases outside the original ten,
selected by the largest previously recorded absolute total-gate reductions with
nonincreasing T count. This is a selected stress test, not a full-suite aggregate.
The QFT-160 selection used a saved checkpoint from an interrupted case. The two
Ising rows are separate corpus variants. No benchmark cases were added.

All values below come from the compiled pass with the constrained-window retry
in Section 3.5. CUDA-Q references were regenerated with the same compiler and
checked against the preceding references: their gate sequences and physical
readout order are unchanged. Both comparison paths use the Appendix D target
pipeline and common cleanup, starting from preserved original Quake inputs.
The Qiskit 2.6.0 values are retained paired measurements on the identical saved
QPY inputs, normalized to the main table's basis. The supplied Gate-count fixes
study has no proposed CUDA-Q measurements for these cases, hence N/A.
Measurements are excluded; T, two-qubit and depth entries show CUDA-Q → PBC.

### E.1 Exact configuration differences

Use all 40 common options in Appendix B, including `objective=total-gates`,
`graph-synthesis=false`, `preserve-clifford-network=true`, and
`max-two-qubit-growth=0`. Set `telemetry` separately for each case. Only these
three resource limits differ from the original ten-case configuration:

- `max-gates=262144` for all five, instead of `65536`.
- `max-operations=1048576` for all five, instead of `131072`.
- `max-work=1000000000` for `hwb10` and `basis_trotter_n4_transpiled`;
  `max-work=4000000000` for both Ising variants and QFT-160, instead of `100000000`.

These increases were explicitly authorized. With the original limits, Trotter
completed, three cases exhausted the work budget, and QFT exceeded the native
gate cap. A limit skip retains the baseline; it is not an optimized result.
The enlarged-limit runs keep all optimization methods, numeric search settings,
and growth constraints unchanged. The constraint-aware retry uses the existing
entangler ceiling and introduces no new pass option.

Compiler processes were limited to 3 GiB of address space, one thread, 1200 CPU
seconds and a 1250-second wall timeout, with at most two compilers running
together. These are execution bounds, not pass tuning parameters.

### E.2 Measured pass results

| Case | CUDA-Q total | Qiskit 2.6.0 total | PBC total | Best proposed CUDA-Q total (reference study) | T count | Two-qubit count | Depth | Pass / independent verification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hwb10 | 60169 | 69664 | 54614 | N/A | 16437 → 15891 | 31506 → 30007 | 38981 → 36910 | committed; equivalent_up_to_global_phase |
| ising_n420_transpiled | 62351 | 133768 | 49951 | N/A | 23986 → 23680 | 838 → 838 | 291 → 238 | committed; equivalent |
| qft_n160 | 118237 | 380659 | 96771 | N/A | 46959 → 45715 | 2790 → 2790 | 26141 → 21419 | committed; equivalent_up_to_global_phase |
| basis_trotter_n4_transpiled | 20218 | 35930 | 15002 | N/A | 7268 → 7196 | 582 → 341 | 10668 → 7900 | committed; equivalent_up_to_global_phase |
| ising_n420 | 61644 | 132380 | 49719 | N/A | 23986 → 23698 | 838 → 838 | 292 → 232 | committed; equivalent |

All five reduce total gates and T count. None increases entanglers, circuit
depth or T depth. The Ising cases exposed an implementation gap: a shorter local
Clifford word could exceed the global entangler ceiling, causing the whole
proposal to be rejected. The Section 3.5 retry retains the other valid savings
while respecting that ceiling. The original ten cases were also rerun after
this change: every reported metric is unchanged, and all ten again pass QCEC.

**Independent verification.** MQT QCEC 3.11.0 checks the synthesized CUDA-Q
Clifford+T reference against the final emitted circuit, up to global phase.
Physical readout order is checked separately. Simulation is disabled and one
checker thread is used. The Ising cases use the ZX checker with a 20-second
bound; Trotter uses the alternating decision-diagram checker with that bound.
For hwb10 and QFT-160, the successful check uses the alternating checker's
`lookahead` application scheme with a 180-second bound. Earlier inconclusive
checks are retained as evidence, not counted as proofs. Both final comparison
circuits for those two cases have identical SHA-256 hashes to the successful
lookahead checks, so those proofs also apply to the rerun. These checks establish
equivalence to the synthesized baseline; they do not independently certify the
preceding approximation of arbitrary-angle input gates.

### E.3 Reproduce these five cases

Use the Appendix D procedure with the preserved five-case input bundle
(`build/pbc-additional-five-constrained-2026-11-03` in the recorded workspace),
or a copy of that bundle on another machine. It is run evidence, not a tracked
fixture. Replace D.1's selection with:

```bash
pbc_inputs="$PWD/build/pbc-additional-five-constrained-2026-11-03"
export pbc_inputs
cat > "$pbc_run/cases.tsv" <<'CASES'
1099c3cedc4ab7ef hwb10 60169 54614
d20f90ddf2a43532 ising_n420_transpiled 62351 49951
3acc1ad768ab7fc3 qft_n160 118237 96771
1e037e935cad2342 basis_trotter_n4_transpiled 20218 15002
fc7600e2e68178fa ising_n420 61644 49719
CASES
```

Use `total-gates-v1+limits` as the evidence label in D.2's
`configuration.txt` and D.3's result record. This label is not a compiler preset;
the complete recorded option array defines the configuration.

Keep the same target and cleanup pipelines. Inside D.4's per-case loop, before
recording options or invoking PBC, derive the case-specific array from Appendix
B's original array; do not change the original ten-case configuration:

```bash
case "$name" in
  hwb10|basis_trotter_n4_transpiled) pbc_work_limit=1000000000 ;;
  ising_n420|ising_n420_transpiled|qft_n160) pbc_work_limit=4000000000 ;;
  *) echo "Unexpected case: $name" >&2; exit 1 ;;
esac
pbc_large_options=("${pbc_options[@]}")
for pbc_i in "${!pbc_large_options[@]}"; do
  case "${pbc_large_options[$pbc_i]}" in
    max-work=*) pbc_large_options[$pbc_i]="max-work=$pbc_work_limit" ;;
    max-gates=*) pbc_large_options[$pbc_i]=max-gates=262144 ;;
    max-operations=*) pbc_large_options[$pbc_i]=max-operations=1048576 ;;
  esac
done
printf '%s\n' "${pbc_large_options[@]}" \
  "telemetry=$case_dir/telemetry.jsonl" > "$case_dir/pbc-options.txt"
```

After generating `baseline.qke`, replace D.4's PBC invocation with:

```bash
run_pbc_opt "$case_dir/pbc" "$case_dir/baseline.qke" \
  "--quake-try-pbc=${pbc_large_options[*]} telemetry=$case_dir/telemetry.jsonl" \
  -o "$case_dir/pbc.qke"
```

Keep synthesis, baseline generation, final cleanup and reference generation in
their original order. In D.4's wrapper, raise `prlimit --cpu` from `120` to
`1200` and prefix its command array with `timeout --kill-after=5s 1250s`.
Preserve each complete command and all 41 option values.
The exact measured arguments are also in each case's `pbc-command.json` and
`result.json`, with the effective pass configuration in `telemetry.jsonl`.

For the two lookahead checks, adapt D.3's checker configuration as follows;
keep its gate counting and physical-readout checks unchanged:

```python
from mqt.qcec.pyqcec import ApplicationScheme

if name in ("hwb10", "qft_n160"):
    options.update(run_zx_checker=False, run_alternating_checker=True,
                   alternating_scheme=ApplicationScheme.lookahead, timeout=180)
```

Give those checker processes a 200-second CPU cap instead of D.4's 30-second
cap, and wrap them with `timeout --kill-after=5s 210s`. An inconclusive check or a pass-level limit
skip does not satisfy reproduction. The measured table contains emitted outputs,
never native candidate estimates.

<!-- END ADDITIONAL FIVE -->
