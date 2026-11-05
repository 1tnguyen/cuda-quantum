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
  --quake-try-pbc='objective=t-count graph-synthesis=false max-two-qubit-growth=0 max-depth-growth=0 telemetry=pbc.jsonl' \
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
| `objective` | `total-gates` | Choose `t-count`, `t-depth`, `entanglers`, `depth`, or `total-gates`. Ranking is lexicographic: a later metric cannot compensate for a worse earlier metric. `total-gates` ranks total gates first and forbids a baseline T increase. Changing the objective does not retune graph scoring weights. |
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

<!-- BEGIN FULL CI LIVE RUN -->
## Appendix F. CI FTQC Clifford+T benchmark

This run covers all **216 Clifford+T cases** in cudaq-bench's full CI
profile: 203 corpus cases and 13 generated cases. Every case has a final
outcome; none remain unstarted.
CUDA-Q, Qiskit and the compiled PBC pass use the same input circuits, with one
replication per attempted case. Aggregates cover only the verified pairs below.

### F.1 Configuration

The benchmark source is cudaq-bench commit
`2f4d84d78b4eae13d4e479e9453f877f164a838b`; CUDA-Q is the local Debug build at
`b09ae4ad1d21a086417a8875bdf0648f6ad99c96`. The executable SHA-256 is
`5dc022b783b0068021ac9b5bce1c4c1ed93cb0b50ebbbe6ae73a0b4c0133dcdd`.
The corpus archive is checked against the snapshot's LFS lock.

The suite builds each Qiskit circuit, saves its QPY, imports it using
`cudaq.contrib.from_qiskit`, performs the builder's AOT compilation, and writes
that Quake to disk. CUDA-Q runs `canonicalize`,
`hw-jit-prep-pipeline{allow-early-exit=false}`, `jit-deploy-pipeline`, the
snapshot's complete Clifford+T target pipeline, and
`jit-finalize-pipeline{lower-device-calls=true}`. The target includes one-qubit
fusion and `normalize-1q-clifford-t`. A deterministic preflight matches the
native runtime's gate totals and T depth on exact U3, Toffoli and Trotter inputs.
Stock and PBC share the synthesized prefix through its last `dqe`; PBC inserts
`quake-try-pbc` before the common
`func.func(regtomem),symbol-dce` suffix and standard JIT finalization. The PBC
branch removes compiler-generated output annotations with the existing
`func.func(erase-implicit-output)` pass before `quake-try-pbc`; stock retains
those annotations. This pass removes no quantum gate or explicit measurement.
Equivalence checks retain the fixed physical-qubit and measurement contract.
The per-case synthesis epsilon defaults to 0.001. CUDA-Q uses the target's seed
0; sharing the prefix prevents synthesis
variation between the paired outputs.

PBC uses the 40 options in Appendix B with Appendix E's limits:
`max-gates=262144`, `max-operations=1048576`, and `max-work=4000000000`.
The work limit is `1000000000` for `hwb10` and
`basis_trotter_n4_transpiled`. Graph synthesis remains disabled; total gates are
the objective and two-qubit growth is prohibited. Qiskit 2.6.0 uses the suite's
Clifford+T pass manager at optimization level 2 and seed 0. Its output is
translated to `h,s,sdg,t,tdg,x,y,z,cx,cy,cz,swap` for the comparison; native CI
metrics and final layout are retained separately.
OpenQASM 3 inputs use `qiskit-qasm3-import` 0.6.0, `openqasm3` 1.0.1 and
`antlr4-python3-runtime` 4.13.2, satisfying the suite's `qasm3-import` extra.

Four cases contain classical control flow unsupported by CUDA-Q's importer:
`feynman:inverseqft1`, `feynman:inverseqft2`,
`qasmbench:qasmbench_small_ipea_n2_ipea_n2`, and
`qasmbench:qasmbench_small_ipea_n2_ipea_n2_transpiled`.
An audit of the published CI results confirmed these four skips. For these
cases, CUDA-Q is skipped, PBC is unavailable, and Qiskit is attempted
independently. Qiskit's conditional-circuit metrics use the suite's worst-case
branch resource bounds.

### F.2 Measurements

Counts are taken from emitted final circuits, excluding measurements and scalar
global phase. Triples below are CUDA-Q / Qiskit / PBC. PBC is checked against
stock with identical physical readout order and register names. Identical final
gate sequences and physical readouts prove an unchanged circuit; otherwise
MQT QCEC 3.11.0 checks
unitary equivalence up to global phase. It uses alternating decision diagrams
up to 15 qubits and ZX above that, with a 20-second limit; the Appendix E
lookahead settings give hwb10 and QFT-160 180 seconds. Inconclusive checks are
reported explicitly and excluded from verified-pair aggregates. A pass limit
or unsupported-input fallback is distinguished from an optimized result.
PBC's `committed` outcome denotes internal acceptance; the following verification
result is the independent check. For untranspiled QPE9, saved-output identity
comparison permits gates after a measurement only on other qubits, preserving
physical readout order and register names.

Under the local limits in F.3, CUDA-Q encountered nine frontend and fourteen
stock-compilation timeouts, in addition to the four control-flow skips.
PBC timed out on `feynman:hwb11`. Independent verification remains unresolved
for 45 compiled pairs: 44 checks were inconclusive, and QCEC returned
`probably_not_equivalent` for `feynman:ham15_med`. These pairs are excluded from
verified-pair aggregates.

<!-- BEGIN FULL SUITE RESULTS -->
**Coverage: 216/216 cases processed; 143 CUDA-Q/PBC pairs verified.**

| Compiler | Compilation succeeded | Failed | Skipped / unavailable |
|---|---:|---:|---:|
| CUDA-Q | 189 | 23 | 4 |
| Qiskit 2.6.0 | 216 | 0 | 0 |
| PBC | 188 | 1 | 27 |

Among the 143 verified pairs, PBC reduces total gates on 106 and ties on 37; their combined gate count is 11,505,384 → 11,369,641 (1.18% reduction). This aggregate excludes unverified or unavailable pairs.

| Case | CUDA-Q total | Qiskit total | PBC total | T count (C/Q/P) | Two-qubit count (C/Q/P) | Depth (C/Q/P) | T depth (C/Q/P) | Outcome / verification |
|---|---:|---:|---:|---|---|---|---|---|
| feynman:adder | 122 | 138 | 106 | 32 / 48 / 32 | 57 / 65 / 53 | 103 / 99 / 95 | 32 / 32 / 32 | committed: verified admissible improvement; equivalent |
| feynman:adder_8 | 727 | 875 | 621 | 215 / 361 / 173 | 347 / 409 / 320 | 228 / 246 / 205 | 74 / 90 / 66 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:barenco_tof_10 | 365 | 427 | 342 | 100 / 192 / 100 | 176 / 192 / 168 | 324 / 322 / 281 | 88 / 128 / 87 | committed: verified admissible improvement; equivalent |
| feynman:barenco_tof_3 | 50 | 56 | 46 | 16 / 24 / 16 | 22 / 24 / 21 | 44 / 42 / 41 | 16 / 16 / 16 | committed: verified admissible improvement; equivalent |
| feynman:barenco_tof_4 | 95 | 109 | 88 | 28 / 48 / 28 | 44 / 48 / 42 | 84 / 82 / 77 | 28 / 32 / 27 | committed: verified admissible improvement; equivalent |
| feynman:barenco_tof_5 | 140 | 162 | 130 | 40 / 72 / 40 | 66 / 72 / 63 | 124 / 122 / 111 | 38 / 48 / 37 | committed: verified admissible improvement; equivalent |
| feynman:bigadder | 242 | 274 | 209 | 64 / 96 / 64 | 114 / 130 / 106 | 160 / 152 / 144 | 48 / 48 / 48 | committed: verified admissible improvement; equivalent |
| feynman:csla_mux_3 | 161 | 157 | 156 | 64 / 64 / 62 | 76 / 72 / 72 | 69 / 68 / 68 | 26 / 25 / 26 | committed: verified admissible improvement; equivalent |
| feynman:csum_mux_9 | 392 | 420 | 308 | 140 / 196 / 84 | 168 / 168 / 168 | 61 / 60 / 54 | 23 / 24 / 16 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:gf2_10_mult | 1088 | 1257 | 1088 | 410 / 600 / 410 | 609 / 609 / 609 | 293 / 307 / 292 | 117 / 134 / 118 | committed: verified admissible improvement; equivalent |
| feynman:gf2_128_mult | 165019 | 197499 | 165019 | 65664 / 98304 / 65664 | 98685 / 98685 / 98685 | 4073 / 4326 / 4073 | 1652 / 1905 / 1652 | skipped: no admissible improvement; identical gate sequence |
| feynman:gf2_16_mult | 2703 | 3179 | 2703 | 1040 / 1536 / 1040 | 1581 / 1581 / 1581 | 491 / 520 / 491 | 196 / 225 / 196 | skipped: no admissible improvement; identical gate sequence |
| feynman:gf2_256_mult | — | 788219 | — | — / 393216 / — | — / 393981 / — | — / 8671 / — | — / 3824 / — | failed; stock-prefix: timeout; not checked |
| feynman:gf2_32_mult | 10562 | 12538 | 10562 | 4128 / 6144 / 4128 | 6268 / 6268 / 6268 | 1003 / 1064 / 1003 | 404 / 465 / 404 | skipped: no admissible improvement; identical gate sequence |
| feynman:gf2_4_mult | 194 | 213 | 193 | 68 / 96 / 68 | 99 / 99 / 99 | 102 / 106 / 102 | 40 / 45 / 40 | committed: verified admissible improvement; equivalent |
| feynman:gf2_5_mult | 304 | 332 | 302 | 115 / 155 / 115 | 154 / 154 / 154 | 132 / 138 / 131 | 53 / 59 / 53 | committed: verified admissible improvement; equivalent |
| feynman:gf2_64_mult | 41547 | 49595 | 41547 | 16448 / 24576 / 16448 | 24765 / 24765 / 24765 | 2027 / 2152 / 2027 | 820 / 945 / 820 | skipped: no admissible improvement; identical gate sequence |
| feynman:gf2_6_mult | 414 | 465 | 411 | 150 / 216 / 150 | 221 / 221 / 221 | 166 / 174 / 166 | 66 / 75 / 66 | committed: verified admissible improvement; equivalent |
| feynman:gf2_7_mult | 553 | 627 | 550 | 217 / 301 / 217 | 300 / 300 / 300 | 198 / 208 / 198 | 80 / 90 / 80 | committed: verified admissible improvement; equivalent |
| feynman:gf2_8_mult | 709 | 819 | 705 | 264 / 384 / 264 | 405 / 405 / 405 | 235 / 248 / 235 | 92 / 105 / 92 | committed: verified admissible improvement; equivalent |
| feynman:gf2_9_mult | 889 | 1023 | 887 | 351 / 495 / 351 | 494 / 494 / 494 | 260 / 273 / 258 | 106 / 119 / 104 | committed: verified admissible improvement; equivalent |
| feynman:grover_5 | 668 | 732 | 500 | 178 / 296 / 166 | 248 / 288 / 228 | 533 / 498 / 429 | 164 / 192 / 160 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:ham15_high | 3659 | 4789 | 3410 | 1021 / 2173 / 1019 | 1817 / 2149 / 1722 | 2993 / 3071 / 2659 | 871 / 1184 / 873 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:ham15_low | 375 | 427 | 364 | 97 / 147 / 97 | 220 / 236 / 216 | 270 / 273 / 256 | 77 / 85 / 76 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:ham15_med | 871 | 1156 | 771 | 242 / 504 / 211 | 428 / 534 / 393 | 692 / 758 / 574 | 195 / 283 / 169 | committed: verified admissible improvement; verification: probably_not_equivalent; probably_not_equivalent |
| feynman:hwb10 | 60169 | 69664 | 54688 | 16437 / 26915 / 15891 | 31506 / 34828 / 30007 | 38981 / 38548 / 36910 | 11807 / 12863 / 11783 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:hwb11 | 166051 | 194163 | — | 45638 / 75372 / — | 87705 / 97615 / — | 117176 / 115613 / — | 33842 / 39133 / — | failed; pbc: timeout; not checked |
| feynman:hwb12 | 326567 | 386733 | 326567 | 87493 / 152755 / 87493 | 170007 / 191205 / 170007 | 232046 / 225941 / 232046 | 65173 / 77375 / 65173 | skipped: native gate limit; identical gate sequence |
| feynman:hwb6 | 229 | 249 | 212 | 75 / 97 / 75 | 108 / 116 / 105 | 166 / 163 / 152 | 57 / 59 / 55 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:hwb8 | 12546 | 14077 | 11583 | 3617 / 5437 / 3517 | 6585 / 7073 / 6305 | 8522 / 8231 / 8055 | 2594 / 2662 / 2607 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:inverseqft1 | — | 419 | — | — / 156 / — | — / 0 / — | — / 416 / — | — / 156 / — | unsupported classical control flow; not checked |
| feynman:inverseqft2 | — | 116 | — | — / 40 / — | — / 0 / — | — / 108 / — | — / 39 / — | unsupported classical control flow; not checked |
| feynman:mod5_4 | 57 | 60 | 37 | 16 / 22 / 7 | 28 / 28 / 19 | 50 / 48 / 30 | 14 / 16 / 6 | committed: verified admissible improvement; equivalent |
| feynman:mod_adder_1024 | 3035 | 3859 | 2780 | 1011 / 1739 / 1011 | 1392 / 1702 / 1306 | 2472 / 2539 / 2115 | 746 / 989 / 758 | committed: verified admissible improvement; equivalent |
| feynman:mod_mult_55 | 109 | 117 | 92 | 37 / 45 / 35 | 44 / 48 / 40 | 52 / 51 / 46 | 20 / 19 / 18 | committed: verified admissible improvement; equivalent |
| feynman:mod_red_21 | 231 | 262 | 196 | 73 / 107 / 73 | 93 / 105 / 87 | 176 / 166 / 149 | 55 / 59 / 55 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:qcla_adder_10 | 440 | 481 | 419 | 162 / 212 / 162 | 205 / 215 / 200 | 72 / 75 / 66 | 26 / 29 / 25 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:qcla_com_7 | 335 | 406 | 298 | 95 / 169 / 95 | 152 / 174 / 144 | 77 / 85 / 72 | 26 / 34 / 26 | committed: verified admissible improvement; equivalent |
| feynman:qcla_mod_7 | 742 | 827 | 678 | 249 / 351 / 237 | 348 / 366 / 324 | 211 / 211 / 194 | 74 / 83 / 73 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:qft | 496 | 1021 | 411 | 199 / 299 / 195 | 12 / 12 / 12 | 330 / 670 / 266 | 131 / 189 / 127 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:qft_4 | 171 | 484 | 157 | 67 / 146 / 67 | 46 / 45 / 46 | 146 / 445 / 130 | 52 / 127 / 52 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| feynman:rc_adder_6 | 191 | 185 | 147 | 63 / 67 / 47 | 91 / 83 / 76 | 111 / 103 / 91 | 40 / 39 / 31 | committed: verified admissible improvement; equivalent |
| feynman:tof_10 | 215 | 247 | 183 | 71 / 103 / 71 | 86 / 102 / 78 | 180 / 171 / 157 | 62 / 68 / 62 | committed: verified admissible improvement; equivalent |
| feynman:tof_3 | 40 | 44 | 36 | 15 / 19 / 15 | 16 / 18 / 15 | 33 / 31 / 31 | 13 / 12 / 13 | committed: verified admissible improvement; equivalent |
| feynman:tof_4 | 65 | 73 | 57 | 23 / 31 / 23 | 26 / 30 / 24 | 54 / 51 / 49 | 20 / 20 / 20 | committed: verified admissible improvement; equivalent |
| feynman:tof_5 | 90 | 102 | 78 | 31 / 43 / 31 | 36 / 42 / 33 | 75 / 71 / 67 | 27 / 28 / 27 | committed: verified admissible improvement; equivalent |
| feynman:vbe_adder_3 | 101 | 128 | 97 | 24 / 56 / 24 | 54 / 58 / 52 | 66 / 76 / 61 | 16 / 30 / 16 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_large_adder_n118_adder_n118 | 1574 | 1782 | 1366 | 416 / 624 / 416 | 741 / 845 / 689 | 643 / 639 / 635 | 224 / 224 / 224 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_large_adder_n118_adder_n118_transpiled | 1809 | 1834 | 1366 | 416 / 624 / 416 | 741 / 845 / 689 | 762 / 679 / 635 | 224 / 224 / 224 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_adder_n28_adder_n28 | 364 | 412 | 316 | 96 / 144 / 96 | 171 / 195 / 159 | 193 / 189 / 185 | 64 / 64 / 64 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_large_adder_n28_adder_n28_transpiled | 419 | 424 | 316 | 96 / 144 / 96 | 171 / 195 / 159 | 222 / 199 / 185 | 64 / 64 / 64 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_adder_n433_adder_n433 | 5809 | 6577 | 5041 | 1536 / 2304 / 1536 | 2736 / 3120 / 2544 | 2218 / 2214 / 2210 | 784 / 784 / 784 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_large_adder_n433_adder_n433_transpiled | 6674 | 6769 | 5041 | 1536 / 2304 / 1536 | 2736 / 3120 / 2544 | 2652 / 2359 / 2210 | 784 / 784 / 784 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_large_adder_n64_adder_n64 | 848 | 960 | 736 | 224 / 336 / 224 | 399 / 455 / 371 | 373 / 369 / 365 | 128 / 128 / 128 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_large_adder_n64_adder_n64_transpiled | 975 | 988 | 736 | 224 / 336 / 224 | 399 / 455 / 371 | 438 / 391 / 365 | 128 / 128 / 128 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n34_ising_n34 | 4966 | 10636 | 4018 | 1914 / 2841 / 1906 | 66 / 66 / 66 | 309 / 561 / 255 | 124 / 142 / 124 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n34_ising_n34_transpiled | 4977 | 10515 | 4006 | 1914 / 2841 / 1900 | 66 / 66 / 66 | 317 / 563 / 254 | 124 / 142 / 123 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n42_ising_n42 | 6049 | 13102 | 4926 | 2348 / 3500 / 2328 | 82 / 82 / 82 | 324 / 570 / 256 | 122 / 143 / 122 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n420_ising_n420 | 62106 | 132380 | 49981 | 23986 / 36123 / 23684 | 838 / 838 / 838 | 301 / 585 / 242 | 116 / 150 / 116 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n420_ising_n420_transpiled | 62449 | 133768 | 49901 | 23986 / 36123 / 23684 | 838 / 838 / 838 | 293 / 620 / 239 | 116 / 150 / 116 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n66_ising_n66 | 9606 | 20394 | 7686 | 3712 / 5564 / 3646 | 130 / 130 / 130 | 306 / 557 / 248 | 120 / 146 / 118 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n66_ising_n66_transpiled | 9650 | 20373 | 7651 | 3712 / 5564 / 3644 | 130 / 130 / 129 | 301 / 561 / 246 | 120 / 146 / 120 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n98_ising_n98 | 14421 | 30133 | 11567 | 5565 / 8375 / 5483 | 194 / 194 / 194 | 312 / 574 / 249 | 120 / 146 / 118 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_ising_n98_ising_n98_transpiled | 14510 | 30263 | 11536 | 5563 / 8375 / 5481 | 194 / 194 / 194 | 315 / 570 / 245 | 120 / 146 / 118 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_multiplier_n350_multiplier_n350 | 166584 | 304183 | 161182 | 39202 / 136080 / 39062 | 92754 / 145880 / 92148 | 85210 / 141075 / 80334 | 21622 / 53489 / 21486 | committed: verified admissible improvement; verification: inconclusive; inconclusive |
| qasmbench:qasmbench_large_multiplier_n350_multiplier_n350_transpiled | 176925 | 326475 | 161229 | 39202 / 136080 / 39062 | 92754 / 145880 / 92197 | 94856 / 160187 / 80315 | 21622 / 53489 / 21487 | committed: verified admissible improvement; verification: inconclusive; inconclusive |
| qasmbench:qasmbench_large_multiplier_n400_multiplier_n400 | 217578 | 397635 | 217578 | 51202 / 177920 / 51202 | 121204 / 190720 / 121204 | 111400 / 184438 / 111400 | 28307 / 69929 / 28307 | skipped: search size/work/deadline limit; identical gate sequence |
| qasmbench:qasmbench_large_multiplier_n400_multiplier_n400_transpiled | — | 426711 | — | — / 177920 / — | — / 190720 / — | — / 209467 / — | — / 69929 / — | failed; stock-prefix: timeout; not checked |
| qasmbench:qasmbench_large_multiplier_n45_multiplier_n45 | 2756 | 4796 | 2599 | 650 / 2124 / 634 | 1500 / 2286 / 1428 | 1370 / 2224 / 1252 | 333 / 846 / 319 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_multiplier_n45_multiplier_n45_transpiled | 2985 | 5189 | 2611 | 650 / 2124 / 634 | 1500 / 2286 / 1430 | 1518 / 2502 / 1247 | 333 / 846 / 319 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_multiplier_n75_multiplier_n75 | 7651 | 13619 | 7307 | 1802 / 6060 / 1774 | 4206 / 6510 / 4082 | 3833 / 6304 / 3547 | 942 / 2394 / 916 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_multiplier_n75_multiplier_n75_transpiled | 8215 | 14681 | 7320 | 1802 / 6060 / 1774 | 4206 / 6510 / 4086 | 4257 / 7122 / 3533 | 942 / 2394 / 916 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_large_qft_n160_qft_n160 | 119613 | 380659 | 96773 | 46959 / 105786 / 46029 | 2790 / 5700 / 2790 | 26147 / 65327 / 21420 | 10551 / 17920 / 10392 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_qft_n160_qft_n160_transpiled | 106920 | 380702 | 87577 | 41873 / 105786 / 41247 | 2790 / 5700 / 2790 | 25485 / 65327 / 21369 | 10067 / 17919 / 10059 | committed: verified admissible improvement; verification: no_information; no_information |
| qasmbench:qasmbench_large_qft_n29_qft_n29 | 18253 | 57351 | 15052 | 7266 / 15920 / 7122 | 432 / 722 / 432 | 4479 / 10962 / 3602 | 1774 / 2986 / 1746 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_qft_n29_qft_n29_transpiled | 16994 | 57394 | 13851 | 6634 / 15920 / 6528 | 432 / 722 / 432 | 4343 / 10962 / 3549 | 1683 / 2985 / 1674 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_large_qft_n320_qft_n320 | 239671 | 775539 | 239671 | 95439 / 215546 / 95439 | 5670 / 11780 / 5670 | 52701 / 131727 / 52701 | 21271 / 36160 / 21271 | skipped: search size/work/deadline limit; identical gate sequence |
| qasmbench:qasmbench_large_qft_n320_qft_n320_transpiled | 216825 | 775582 | 216825 | 84913 / 215546 / 84913 | 5670 / 11780 / 5670 | 52650 / 131727 / 52650 | 20307 / 36159 / 20307 | skipped: search size/work/deadline limit; identical gate sequence |
| qasmbench:qasmbench_large_qft_n63_qft_n63 | 44289 | 141263 | 36199 | 17568 / 39244 / 17220 | 1044 / 2014 / 1044 | 10041 / 25072 / 8228 | 4052 / 6862 / 3990 | committed: verified admissible improvement; verification: no_information; no_information |
| qasmbench:qasmbench_large_qft_n63_qft_n63_transpiled | 40124 | 141306 | 32643 | 15780 / 39244 / 15424 | 1044 / 2014 / 1044 | 9561 / 25072 / 8175 | 3859 / 6861 / 3848 | committed: verified admissible improvement; verification: no_information; no_information |
| qasmbench:qasmbench_medium_bigadder_n18_bigadder_n18 | 242 | 274 | 209 | 64 / 96 / 64 | 114 / 130 / 106 | 160 / 152 / 144 | 48 / 48 / 48 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_medium_bigadder_n18_bigadder_n18_transpiled | 278 | 282 | 209 | 64 / 96 / 64 | 114 / 130 / 106 | 171 / 157 / 142 | 48 / 48 / 48 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_medium_ising_n26_ising_n26 | 3785 | 7480 | 3045 | 1454 / 2132 / 1442 | 50 / 50 / 50 | 282 / 532 / 228 | 110 / 145 / 109 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_medium_ising_n26_ising_n26_transpiled | 3814 | 7489 | 3035 | 1454 / 2132 / 1444 | 50 / 50 / 50 | 284 / 531 / 225 | 110 / 145 / 110 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_medium_multiplier_n15_multiplier_n15 | 313 | 476 | 280 | 74 / 204 / 70 | 162 / 222 / 142 | 168 / 232 / 147 | 44 / 90 / 41 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_medium_multiplier_n15_multiplier_n15_transpiled | 351 | 524 | 281 | 74 / 204 / 70 | 162 / 222 / 142 | 186 / 258 / 145 | 44 / 90 / 42 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_medium_multiply_n13_multiply_n13 | 93 | 95 | 93 | 32 / 36 / 32 | 40 / 40 / 40 | 41 / 42 / 41 | 16 / 15 / 16 | skipped: no admissible improvement; identical gate sequence |
| qasmbench:qasmbench_medium_multiply_n13_multiply_n13_transpiled | 105 | 95 | 91 | 32 / 36 / 32 | 40 / 40 / 40 | 40 / 42 / 39 | 15 / 15 / 15 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_medium_qft_n18_qft_n18 | 10017 | 30205 | 8104 | 3933 / 8374 / 3855 | 234 / 306 / 234 | 2607 / 6397 / 2108 | 1037 / 1732 / 1020 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_medium_qft_n18_qft_n18_transpiled | 9385 | 30248 | 7594 | 3675 / 8374 / 3589 | 234 / 306 / 234 | 2452 / 6397 / 2049 | 979 / 1731 / 967 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_adder_n10_adder_n10 | 122 | 138 | 106 | 32 / 48 / 32 | 57 / 65 / 53 | 103 / 99 / 95 | 32 / 32 / 32 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_small_adder_n10_adder_n10_transpiled | 141 | 142 | 106 | 32 / 48 / 32 | 57 / 65 / 53 | 114 / 103 / 95 | 32 / 32 / 32 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_adder_n4_adder_n4 | 23 | 23 | 22 | 8 / 8 / 8 | 10 / 10 / 10 | 11 / 11 / 11 | 2 / 2 / 2 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_small_adder_n4_adder_n4_transpiled | 24 | 25 | 22 | 8 / 8 / 8 | 10 / 10 / 10 | 12 / 13 / 11 | 2 / 2 / 2 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_test_n4 | 800 | 1516 | 616 | 296 / 420 / 288 | 34 / 28 / 27 | 354 / 674 / 265 | 124 / 174 / 122 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_test_n4_transpiled | 885 | 1579 | 654 | 314 / 424 / 310 | 46 / 38 / 30 | 542 / 825 / 396 | 188 / 212 / 187 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_trotter_n4 | 19667 | 47122 | 14410 | 7136 / 12685 / 6886 | 462 / 303 / 304 | 8619 / 20053 / 6292 | 3047 / 5281 / 2988 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_trotter_n4_transpiled | 20299 | 35930 | 14816 | 7190 / 9481 / 7108 | 582 / 351 / 331 | 10704 / 18209 / 7809 | 3766 / 4780 / 3738 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_ipea_n2_ipea_n2 | — | 2331 | — | — / 688 / — | — / 14 / — | — / 2320 / — | — / 686 / — | unsupported classical control flow; not checked |
| qasmbench:qasmbench_small_ipea_n2_ipea_n2_transpiled | — | 2341 | — | — / 688 / — | — / 14 / — | — / 2323 / — | — / 686 / — | unsupported classical control flow; not checked |
| qasmbench:qasmbench_small_ising_n10_ising_n10 | 9302 | 18655 | 7336 | 3644 / 5140 / 3578 | 72 / 90 / 72 | 1385 / 3040 / 1099 | 539 / 835 / 533 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_ising_n10_ising_n10_transpiled | 9314 | 18779 | 7381 | 3642 / 5108 / 3600 | 72 / 90 / 72 | 1390 / 3332 / 1092 | 539 / 926 / 533 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_pea_n5_pea_n5 | 671 | 2988 | 550 | 264 / 831 / 260 | 20 / 26 / 17 | 503 / 2706 / 344 | 197 / 751 / 163 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_small_pea_n5_pea_n5_transpiled | 709 | 2992 | 578 | 272 / 831 / 270 | 20 / 26 / 18 | 527 / 2708 / 361 | 204 / 751 / 170 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_small_qft_n4_qft_n4 | 504 | 1021 | 408 | 199 / 299 / 195 | 12 / 12 / 12 | 338 / 670 / 263 | 131 / 189 / 127 | committed: verified admissible improvement; equivalent |
| qasmbench:qasmbench_small_qft_n4_qft_n4_transpiled | 531 | 756 | 430 | 207 / 226 / 205 | 12 / 12 / 12 | 303 / 620 / 244 | 119 / 185 / 117 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| qasmbench:qasmbench_small_qpe_n9_qpe_n9 | 1208 | 3022 | 1208 | 473 / 863 / 473 | 41 / 43 / 41 | 1192 / 1749 / 1192 | 472 / 506 / 472 | skipped: PBC rotations must precede all measurements and sinks; identical gate sequence |
| qasmbench:qasmbench_small_qpe_n9_qpe_n9_transpiled | 1233 | 3046 | 998 | 467 / 863 / 461 | 41 / 43 / 40 | 855 / 1756 / 694 | 327 / 506 / 321 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| generated_qft:qft_n002 | 8 | 7 | 8 | 3 / 3 / 3 | 3 / 2 / 3 | 6 / 5 / 6 | 3 / 3 / 3 | skipped: no admissible improvement; identical gate sequence |
| generated_qft:qft_n004 | 504 | 1019 | 416 | 199 / 299 / 195 | 14 / 12 / 14 | 338 / 669 / 267 | 131 / 189 / 127 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| generated_qft:qft_n008 | 2379 | 6256 | 1946 | 945 / 1706 / 919 | 60 / 56 / 60 | 930 / 2258 / 758 | 373 / 629 / 365 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| generated_qft:qft_n016 | 8469 | 25279 | 6820 | 3327 / 7002 / 3237 | 206 / 240 / 206 | 2240 / 5567 / 1839 | 903 / 1509 / 888 | committed: verified admissible improvement; equivalent |
| generated_qft:qft_n032 | 20785 | 64755 | 16857 | 8175 / 17978 / 7957 | 502 / 836 / 502 | 4990 / 12207 / 4013 | 1975 / 3328 / 1944 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| generated_qft:qft_n064 | 45339 | 143731 | 36879 | 17871 / 39930 / 17397 | 1094 / 2052 / 1094 | 10212 / 25487 / 8365 | 4119 / 6976 / 4056 | committed: verified admissible improvement; verification: no_information; no_information |
| generated_adder:cuccaro_n002 | 59 | 67 | 51 | 16 / 24 / 16 | 29 / 33 / 27 | 52 / 50 / 48 | 16 / 16 / 16 | committed: verified admissible improvement; equivalent |
| generated_adder:cuccaro_n004 | 117 | 133 | 101 | 32 / 48 / 32 | 57 / 65 / 53 | 102 / 98 / 94 | 32 / 32 / 32 | committed: verified admissible improvement; equivalent |
| generated_adder:cuccaro_n008 | 233 | 265 | 201 | 64 / 96 / 64 | 113 / 129 / 105 | 202 / 194 / 186 | 64 / 64 / 64 | committed: verified admissible improvement; equivalent |
| generated_adder:cuccaro_n016 | 465 | 529 | 401 | 128 / 192 / 128 | 225 / 257 / 209 | 402 / 386 / 370 | 128 / 128 / 128 | committed: verified admissible improvement; equivalent |
| generated_adder:cuccaro_n032 | 929 | 1057 | 801 | 256 / 384 / 256 | 449 / 513 / 417 | 802 / 770 / 738 | 256 / 256 / 256 | committed: verified admissible improvement; equivalent |
| generated_adder:cuccaro_n064 | 1857 | 2113 | 1601 | 512 / 768 / 512 | 897 / 1025 / 833 | 1602 / 1538 / 1474 | 512 / 512 / 512 | committed: verified admissible improvement; equivalent |
| generated_single_rotation:a00_eps1em03 | 0 | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | skipped: no admissible improvement; identical gate sequence |
| ftcircuitbench:adder_adder_10q | 122 | 138 | 106 | 32 / 48 / 32 | 57 / 65 / 53 | 103 / 99 / 95 | 32 / 32 / 32 | committed: verified admissible improvement; equivalent |
| ftcircuitbench:adder_adder_28q | 364 | 412 | 316 | 96 / 144 / 96 | 171 / 195 / 159 | 193 / 189 / 185 | 64 / 64 / 64 | committed: verified admissible improvement; equivalent |
| ftcircuitbench:adder_adder_4q | 23 | 23 | 22 | 8 / 8 / 8 | 10 / 10 / 10 | 11 / 11 / 11 | 2 / 2 / 2 | committed: verified admissible improvement; equivalent |
| ftcircuitbench:adder_adder_64q | 848 | 960 | 736 | 224 / 336 / 224 | 399 / 455 / 371 | 373 / 369 / 365 | 128 / 128 / 128 | committed: verified admissible improvement; equivalent |
| ftcircuitbench:hamiltonians_fermi_hubbard_1d_128q | — | 2535160 | — | — / 648850 / — | — / 59524 / — | — / 742210 / — | — / 195275 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:hamiltonians_fermi_hubbard_1d_18q | 152636 | 358057 | 121693 | 55440 / 91315 / 54602 | 7640 / 8482 / 7172 | 144023 / 274510 / 115264 | 54320 / 70995 / 53542 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_fermi_hubbard_1d_72q | 605975 | 1420904 | 605975 | 221760 / 363390 / 221760 | 29640 / 34204 / 29640 | 496863 / 813050 / 496863 | 187600 / 213515 / 187600 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_128q | — | 4054614 | — | — / 1007308 / — | — / 190924 / — | — / 1475312 / — | — / 374224 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_18q | 272850 | 554679 | 272850 | 95760 / 139484 / 95760 | 18840 / 19802 / 18840 | 231925 / 463692 / 231925 | 85120 / 119124 / 85120 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_72q | — | 2262974 | — | — / 560608 / — | — / 110248 / — | — / 1505532 / — | — / 381804 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_triangular_128q | — | 5552582 | — | — / 1365708 / — | — / 326140 / — | — / 2570632 / — | — / 645864 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_triangular_18q | 390902 | 773194 | 390902 | 136080 / 194784 / 136080 | 29720 / 30562 / 29720 | 347254 / 664512 / 347254 | 126560 / 170964 / 126560 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_triangular_72q | — | 3100445 | — | — / 762208 / — | — / 181892 / — | — / 2366932 / — | — / 596004 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:hamiltonians_heisenberg_1d_100q | — | 1966352 | — | — / 558779 / — | — / 11883 / — | — / 921021 / — | — / 281938 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:hamiltonians_heisenberg_1d_36q | 356435 | 700176 | 356435 | 129000 / 198715 / 129000 | 8600 / 4203 / 8600 | 349065 / 331133 / 349065 | 129000 / 100946 / 129000 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_heisenberg_1d_64q | 614331 | 1254128 | 614331 | 229800 / 356243 / 229800 | 15320 / 7563 / 15320 | 601305 / 589209 / 601305 | 229800 / 180130 / 229800 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_heisenberg_1d_9q | 85081 | 166008 | 65391 | 31800 / 46813 / 31136 | 2120 / 963 / 1762 | 83165 / 82274 / 64133 | 31800 / 24590 / 31129 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_heisenberg_2d_100q | — | 3820963 | — | — / 1096782 / — | — / 23403 / — | — / 406692 / — | — / 124383 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:hamiltonians_heisenberg_2d_36q | 698075 | 1354995 | 698075 | 256200 / 390626 / 256200 | 17080 / 8283 / 17080 | 208785 / 235320 / 208785 | 78600 / 71563 / 78600 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_heisenberg_2d_64q | — | 2430759 | — | — / 698972 / — | — / 14883 / — | — / 321006 / — | — / 97973 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:hamiltonians_heisenberg_2d_9q | 176401 | 330685 | 131867 | 63600 / 95786 / 62616 | 4240 / 1983 / 3563 | 95605 / 107391 / 71774 | 35400 / 32021 / 34848 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_100q | — | 5781095 | — | — / 1665708 / — | — / 35883 / — | — / 2374260 / — | — / 723452 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_36q | — | 2063107 | — | — / 596360 / — | — / 12843 / — | — / 869988 / — | — / 264924 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_64q | — | 3687773 | — | — / 1064074 / — | — / 22923 / — | — / 1529692 / — | — / 466092 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_9q | 263321 | 504599 | 263321 | 96600 / 145823 / 96600 | 6440 / 3123 / 6440 | 199185 / 227052 / 199185 | 75000 / 68532 / 75000 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_ising_1d_100q | 273560 | 817226 | 273560 | 107800 / 209316 / 107800 | 4200 / 7962 / 4200 | 123120 / 543986 / 123120 | 48400 / 140016 / 48400 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_ising_1d_36q | 97240 | 292426 | 79846 | 38808 / 74916 / 38766 | 1512 / 2842 / 1512 | 45402 / 195826 / 38505 | 17808 / 50416 / 17766 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_ising_1d_64q | 175064 | 522026 | 141972 | 68992 / 133716 / 68950 | 2688 / 5082 / 2688 | 79344 / 348146 / 67569 | 31192 / 89616 / 31150 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_ising_1d_9q | 24584 | 71026 | 19944 | 9702 / 18216 / 9660 | 378 / 682 / 378 | 12464 / 48946 / 10478 | 4902 / 12616 / 4860 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_ising_2d_100q | 391200 | 1358452 | 391200 | 155600 / 348632 / 155600 | 8400 / 15924 / 8400 | 39176 / 206706 / 39176 | 15420 / 53216 / 15420 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_ising_2d_36q | 140832 | 485492 | 117032 | 56016 / 124632 / 55850 | 3024 / 5684 / 3024 | 23976 / 119666 / 20293 | 9444 / 30816 / 9385 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_ising_2d_64q | 250920 | 867412 | 250920 | 99584 / 222632 / 99584 | 5376 / 10164 / 5376 | 31698 / 163186 / 31698 | 12432 / 42016 / 12432 | skipped: candidate operation limit; identical gate sequence |
| ftcircuitbench:hamiltonians_ising_2d_9q | 35392 | 117212 | 29243 | 14004 / 30132 / 13934 | 756 / 1364 / 756 | 12658 / 54386 / 10598 | 4962 / 14016 / 4925 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_ising_2d_triangular_100q | 519200 | 1899678 | 519200 | 203400 / 487948 / 203400 | 12600 / 23886 / 12600 | 160678 / 707186 / 160678 | 62740 / 182016 / 62740 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_ising_2d_triangular_36q | 185081 | 678558 | 154181 | 73224 / 174348 / 72914 | 4536 / 8526 / 4536 | 67400 / 293746 / 57038 | 26412 / 75616 / 26244 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_ising_2d_triangular_64q | 330712 | 1212798 | 330712 | 130176 / 311548 / 130176 | 8064 / 15246 / 8064 | 108566 / 478706 / 108566 | 42664 / 123216 / 42664 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_ising_2d_triangular_9q | 46301 | 163398 | 38529 | 18306 / 42048 / 18200 | 1134 / 2046 / 1134 | 23598 / 98178 / 19900 | 9264 / 25286 / 9197 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_128q | 141724 | 297312 | 116152 | 53200 / 80118 / 53056 | 4250 / 4250 / 4250 | 46977 / 111017 / 39039 | 18032 / 29482 / 17990 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_18q | 21210 | 43326 | 17405 | 8020 / 11669 / 7978 | 645 / 645 / 645 | 10791 / 24180 / 9004 | 4146 / 6432 / 4131 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_200q | 221837 | 467786 | 221837 | 83720 / 125948 / 83720 | 6795 / 6795 / 6795 | 108439 / 255954 / 108439 | 41552 / 67912 / 41552 | skipped: candidate operation limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_72q | 79456 | 168317 | 65396 | 29960 / 45188 / 29876 | 2505 / 2505 / 2505 | 33787 / 79614 / 28082 | 12880 / 21000 / 12851 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_128q | 280548 | 714216 | 280548 | 98700 / 184114 / 98700 | 19530 / 19530 / 19530 | 91005 / 228013 / 91005 | 33116 / 59039 / 33116 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_18q | 39236 | 92770 | 32419 | 14270 / 24182 / 14252 | 1915 / 1915 / 1915 | 21457 / 49508 / 17940 | 7966 / 12772 / 7967 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_200q | 467096 | 1142041 | 467096 | 163240 / 297945 / 163240 | 35375 / 35375 / 35375 | 222456 / 553007 / 222456 | 79392 / 142095 / 79392 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_72q | 160466 | 402286 | 132646 | 56440 / 104440 / 56360 | 11110 / 11110 / 11110 | 73346 / 180351 / 62181 | 26292 / 46114 / 26244 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_128q | 442226 | 1088904 | 442226 | 155020 / 277384 / 155020 | 34464 / 34464 / 34464 | 154372 / 370861 / 154372 | 55104 / 93128 / 55104 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_18q | 59559 | 141011 | 49225 | 21620 / 36757 / 21620 | 3115 / 3115 / 3115 | 34177 / 78597 / 28601 | 12648 / 20134 / 12654 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_200q | 721928 | 1753415 | 721928 | 251200 / 451109 / 251200 | 61100 / 61100 / 61100 | 370279 / 894781 / 370279 | 130240 / 224437 / 130240 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_72q | 248388 | 616970 | 248388 | 87120 / 158929 / 87120 | 18775 / 18775 / 18775 | 120330 / 294566 / 120330 | 42732 / 73754 / 42732 | skipped: candidate operation limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_100q | 106132 | 305858 | 85031 | 41580 / 81509 / 40584 | 1485 / 1485 / 1485 | 15391 / 50992 / 12505 | 5992 / 13598 / 5881 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_36q | 37617 | 108226 | 30028 | 14700 / 28837 / 14344 | 525 / 525 / 525 | 6188 / 19824 / 5016 | 2408 / 5278 / 2361 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_64q | 66911 | 194690 | 54109 | 26460 / 51881 / 25824 | 945 / 945 / 945 | 10075 / 33460 / 8292 | 3976 / 8918 / 3901 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_9q | 8494 | 24850 | 6861 | 3360 / 6616 / 3274 | 120 / 120 / 120 | 2264 / 6675 / 1857 | 896 / 1768 / 876 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_100q | 210698 | 644956 | 210698 | 85008 / 171208 / 85008 | 3000 / 3000 / 3000 | 23281 / 65489 / 23281 | 9352 / 17252 / 9352 | skipped: candidate operation limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_36q | 76475 | 230252 | 62502 | 30800 / 61064 / 29692 | 1080 / 1080 / 1080 | 13793 / 38201 / 11338 | 5544 / 10052 / 5340 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_64q | 137526 | 411396 | 110826 | 54544 / 109168 / 52644 | 1920 / 1920 / 1920 | 18989 / 51845 / 15289 | 7448 / 13652 / 7200 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_9q | 19584 | 56816 | 15754 | 7784 / 15038 / 7486 | 270 / 270 / 270 | 6894 / 17927 / 5547 | 2716 / 4692 / 2618 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_100q | 313130 | 968265 | 313130 | 126000 / 256636 / 126000 | 4500 / 4500 / 4500 | 145581 / 422554 / 145581 | 58800 / 110700 / 58800 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_36q | 112927 | 346709 | 91903 | 45360 / 91836 / 43626 | 1620 / 1620 / 1620 | 54168 / 158950 / 44534 | 21840 / 41668 / 21110 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_64q | 202402 | 618339 | 202402 | 80640 / 163848 / 80640 | 2880 / 2880 / 2880 | 94455 / 275152 / 94455 | 38080 / 72104 / 38080 | skipped: candidate operation limit; identical gate sequence |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_9q | 27975 | 86069 | 22962 | 11340 / 22773 / 10896 | 405 / 405 / 405 | 14386 / 43147 / 11987 | 5880 / 11314 / 5684 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_100q | 71000 | 118518 | 58069 | 28000 / 33500 / 27990 | 1000 / 1000 / 1000 | 35360 / 64563 / 29751 | 14140 / 16670 / 14126 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_36q | 25560 | 42678 | 20885 | 10080 / 12060 / 10070 | 360 / 360 / 360 | 12960 / 23603 / 10871 | 5180 / 6110 / 5166 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_64q | 44800 | 75858 | 37154 | 17920 / 21440 / 17910 | 640 / 640 / 640 | 22120 / 41523 / 18818 | 9100 / 10730 / 9090 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_9q | 6300 | 10683 | 5199 | 2520 / 3015 / 2510 | 90 / 90 / 90 | 3420 / 6323 / 2868 | 1400 / 1655 / 1390 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_100q | 104000 | 182518 | 84905 | 42000 / 50000 / 41080 | 2000 / 2000 / 2000 | 11104 / 20769 / 9222 | 4564 / 5384 / 4448 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_36q | 37440 | 65718 | 30615 | 15120 / 18000 / 14800 | 720 / 720 / 720 | 6752 / 12577 / 5602 | 2772 / 3272 / 2700 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_64q | 67840 | 116818 | 54306 | 26880 / 32000 / 26300 | 1280 / 1280 / 1280 | 9180 / 16673 / 7434 | 3668 / 4328 / 3574 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_9q | 9360 | 16443 | 7661 | 3780 / 4500 / 3700 | 180 / 180 / 180 | 3488 / 6433 / 2883 | 1428 / 1688 / 1391 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_100q | 138000 | 246518 | 111326 | 56000 / 66500 / 54080 | 3000 / 3000 / 3000 | 44560 / 83763 / 35879 | 18340 / 21620 / 17495 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_36q | 49680 | 88758 | 40114 | 20160 / 23940 / 19480 | 1080 / 1080 / 1080 | 18720 / 35123 / 15173 | 7700 / 9080 / 7375 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_64q | 88320 | 157778 | 71280 | 35840 / 42560 / 34620 | 1920 / 1920 / 1920 | 30280 / 56883 / 24446 | 12460 / 14690 / 11905 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_9q | 12420 | 22203 | 10058 | 5040 / 5985 / 4870 | 270 / 270 / 270 | 6820 / 12705 / 5534 | 2800 / 3305 / 2680 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hhl_hhl_12q | 131555 | 257513 | 103534 | 51304 / 70263 / 50240 | 2692 / 3084 / 2690 | 102667 / 184988 / 80771 | 40122 / 50357 / 39230 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:hhl_hhl_21q | — | 2487067 | — | — / 682415 / — | — / 29813 / — | — / 1860311 / — | — / 508912 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:hhl_hhl_4q | 926 | 2316 | 734 | 354 / 630 / 352 | 22 / 22 / 18 | 839 / 1722 / 667 | 324 / 460 / 320 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| ftcircuitbench:hhl_hhl_7q | 15395 | 29791 | 12295 | 6059 / 8191 / 5999 | 257 / 317 / 249 | 11321 / 22027 / 9029 | 4458 / 6021 / 4413 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| ftcircuitbench:qft_qft_18q | 10021 | 30205 | 8103 | 3933 / 8374 / 3855 | 234 / 306 / 234 | 2607 / 6397 / 2108 | 1037 / 1732 / 1020 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| ftcircuitbench:qft_qft_29q | 18321 | 57351 | 14972 | 7266 / 15920 / 7122 | 432 / 722 / 432 | 4401 / 10962 / 3604 | 1774 / 2986 / 1746 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| ftcircuitbench:qft_qft_4q | 496 | 1021 | 411 | 199 / 299 / 195 | 12 / 12 / 12 | 330 / 670 / 263 | 131 / 189 / 127 | committed: verified admissible improvement; equivalent |
| ftcircuitbench:qft_qft_63q | 44766 | 141263 | 36231 | 17568 / 39244 / 17102 | 1044 / 2014 / 1044 | 10267 / 25072 / 8228 | 4052 / 6862 / 3990 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:qpe_qpe_h2_0_6_12q | — | 20048891 | — | — / 5512786 / — | — / 232410 / — | — / 15321245 / — | — / 4204099 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:qpe_qpe_h2_0_74_12q | — | 20213908 | — | — / 5553262 / — | — / 232410 / — | — / 15391635 / — | — / 4217711 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:qpe_qpe_h2_1_0_12q | — | 20050898 | — | — / 5523746 / — | — / 232410 / — | — / 15329173 / — | — / 4204505 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:qpe_qpe_h2_1_5_12q | — | 20538805 | — | — / 5612948 / — | — / 232410 / — | — / 15795296 / — | — / 4293008 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:qpe_qpe_hubbard_10q | — | 4188598 | — | — / 1150956 / — | — / 55580 / — | — / 3200591 / — | — / 875817 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:qpe_qpe_hubbard_8q | 529667 | 999931 | 529667 | 205011 / 273889 / 205011 | 11813 / 13262 / 11813 | 427496 / 764444 / 427496 | 165332 / 208602 / 165332 | skipped: native gate limit; identical gate sequence |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_10q | — | 3205952 | — | — / 920421 / — | — / 71814 / — | — / 2779501 / — | — / 795045 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_11q | — | 6214073 | — | — / 1768042 / — | — / 139983 / — | — / 5381576 / — | — / 1521871 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_12q | — | 10979973 | — | — / 3125500 / — | — / 247336 / — | — / 9588277 / — | — / 2710002 / — | failed; stock-prefix: timeout; not checked |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_13q | — | 18449983 | — | — / 5249566 / — | — / 408753 / — | — / 16195524 / — | — / 4578992 / — | failed; frontend: timeout; not checked |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_6q | 80457 | 182235 | 63919 | 31644 / 53256 / 30814 | 1814 / 2002 / 1751 | 78718 / 153395 / 62576 | 31235 / 44969 / 30437 | committed: verified admissible improvement; equivalent_up_to_global_phase |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_7q | 147414 | 307752 | 118339 | 58735 / 90890 / 56729 | 4108 / 5164 / 3830 | 126577 / 258237 / 101462 | 50486 / 76124 / 48792 | committed: verified admissible improvement; verification: no_information; no_information |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_8q | 255293 | 550916 | 255293 | 101712 / 159028 / 101712 | 9502 / 12294 / 9502 | 220910 / 479608 / 220910 | 88163 / 137008 / 88163 | skipped: candidate operation limit; identical gate sequence |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_9q | 678748 | 1325141 | 678748 | 265604 / 375904 / 265604 | 25436 / 30987 / 25436 | 609336 / 1234179 / 609336 | 238259 / 347188 / 238259 | skipped: native gate limit; identical gate sequence |
<!-- END FULL SUITE RESULTS -->

### F.3 Reproduction

The local evidence bundle is `build/pbc-ci-runtime-2026-11-04`. It contains the
216-case inventory, frozen source snapshot and corpus cache, expanded target
pipeline, scripts, package versions, configuration manifest, complete result
matrix and per-case commands, inputs, outputs, telemetry and diagnostics.
The attempted CUDA-Q/PBC cases use the complete runtime pipeline above.
Previously measured Qiskit results are reused only when saved QPY hashes match;
`inherited-qiskit.json` records their provenance. `summary.json` audits coverage,
configuration hashes and the inputs and outputs supporting each result;
`status.json` records completion. The compiler executable and options are
unchanged. `continuation-single.json` and `continuation-importer.json` record
the one-worker continuation; `matrix.partial-145.json` preserves the preceding
results, which the final audit confirms are unchanged. The continuation and
audit commands were:

```bash
python3 build/pbc-ci-runtime-2026-11-04/resume-single.py
python3 build/pbc-ci-runtime-2026-11-04/resume-importer.py
python3 build/pbc-ci-runtime-2026-11-04/audit-single.py
```

The OpenQASM 3 parser packages are installed from the saved `dependency-wheels`
into the isolated `qasm3-deps` directory. `qasm3-dependencies.json` records
their versions, wheel hashes and import smoke check. The importer continuation
prepends that directory to `PYTHONPATH`; the existing Qiskit installation is
unchanged. Initial missing-parser input-load failures are archived per case;
the table contains the final results after all six affected inputs were retried.

The driver sets `PYTHONPATH=build/python:build/pauli-ir/verification-deps`,
`LD_LIBRARY_PATH=build/lib`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and
the corpus-cache path, and unsets `CUDAQ_PBC_TELEMETRY`. The first 145 cases used
at most three workers; the final 71 used **one worker**, with sequential cases
and stages. Each frontend, stock or Qiskit stage has 600 CPU / 605 wall seconds;
PBC has 1200 CPU / 1250 wall seconds.
QCEC has 30 CPU / 35 wall seconds, or 200 CPU / 210 wall seconds for the two
lookahead cases. These are local per-stage limits, not CI timing measurements.

No extra virtual-address-space limit is imposed, matching CI's launcher.
Stack size is 64 MiB and core dumps are disabled.
The driver updates the table and machine-readable matrix as cases finish;
failures do not terminate the remaining selection. `STOP_AFTER_ACTIVE` prevents
further launches and must be removed before an authorized continuation.
The QPE9 accounting correction only parses saved outputs and compares identical
gate sequences; it executes neither compilation nor QCEC. Reproduction needs
the local evidence bundle and a fresh output directory after compiler changes
to preserve the executable identity of these measurements.
<!-- END FULL CI LIVE RUN -->

## Changelog

- **2026-11-05:** Completed all 216 CI FTQC cases in Appendix F, with 143 verified CUDA-Q/PBC pairs and one-worker continuation details.
- **2026-11-04:** Extended Appendix F to 145/216 CI FTQC cases with 117 verified CUDA-Q/PBC pairs and reproduction details; corrected the default objective to `total-gates`.
