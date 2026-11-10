# Pauli-based optimization in Quake

The **PBC synthesis (release)** columns in the main benchmark table use
`semantics=release`: ordered joint terminal measurement probabilities are
preserved for arbitrary inputs, while the final quantum state and full unitary
need not be preserved. The separate **PBC synthesis (hold)** columns now
report the latest fixed-compiler validation of 166 cases, with **152 total-gate
wins, 14 ties and no losses** versus preserved network. The synthesis
passes now default to **hold**, preserving the full unitary and scalar phase.
Historical release configurations must explicitly select `semantics=release`.
Hold results use original pre-normalization circuits and are reported
separately; a release-normalized input cannot recover its original unitary.
See [the hold/release architecture](pbc_synthesis_architecture.md#hold-and-release-synthesis-contracts).

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

Standalone normalized-PBC synthesis has a separate measured-program entry
point, `quake-synthesize-pbc`. Its final complete-circuit portfolio and checked
generated-circuit refinements are documented in
[the synthesis architecture](pbc_synthesis_architecture.md) and
[the final tuning configuration](pbc_synthesis_tuning.md#final-results-for-all-fifteen-cases).
Appendix F adds this path to the full runtime benchmark alongside the
preserve-network optimization, with independent output verification.

No numerical advantage over CUDA-Q or Qiskit is asserted without a matching
pass run and configuration record. In particular, the available objectives and
lowering path must match the metric being compared. Earlier result tables without
that implementation provenance are not evidence for this pass.

<!-- BEGIN PASS MEASUREMENTS -->
## Measured pass results

Benchmark tables call the native, network-preserving techniques **Clifford+T
optimization**, and the circuit reconstruction path **PBC synthesis**. The
former optimizes the existing Clifford+T circuit; the latter synthesizes a new
native circuit from normalized PBC. Compiler pass names and historical evidence
keys such as `pbc` retain their original spelling.

Configuration **total-gates-v1**, detailed in Appendix B, uses the compiled
CUDA-Q pass. CUDA-Q and Clifford+T optimization were remeasured after the reproduction fixes below;
the Qiskit references retain the preceding measured run on the same preserved
suite circuits and declared basis. Eight CUDA-Q inputs were imported afresh;
the 118- and 433-qubit adders use their preserved original Quake inputs after fresh imports failed
under the imposed frontend resource cap. Both then run through the same compiled
CUDA-Q baseline and Clifford+T optimization pass. No cases were added.
Measurements are excluded from gate counts and from the circuit depths in this
table. T and two-qubit/depth columns show CUDA-Q baseline → Clifford+T optimization output.

Completed paired measurements: **10/10**. All ten pass independent equivalence
checking up to global phase. The implemented pass beats the proposed CUDA-Q
total on eight cases, ties one, and is one gate higher on QFT-2. It reduces total
gates versus the measured CUDA-Q baseline on nine cases and leaves QFT-2 unchanged.
T count never increases; `rc_adder_6` drops from 63 to 47 and `csla_mux_3` from
64 to 62. Neither circuit depth nor T depth increases on these ten cases.

| Case | CUDA-Q total | Qiskit 2.6.0 total | Clifford+T optimization total | Best proposed CUDA-Q total (reference study) | T count | Two-qubit count | Depth | Pass / independent verification |
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

Under `semantics=release`, the passes support single-block scalar `!quake.wire` ownership, scalar wire
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

Default `semantics=hold` additionally preserves the full quantum action and
scalar phase, supports scalar SSA quantum returns, and retains the original
terminal measurement instruments, including live X/Y measurement outputs.
Its lossless normalization and network-changing synthesis are described in
[the architecture](pbc_synthesis_architecture.md#hold-and-release-synthesis-contracts).

## 2. Compilation flow and failure behavior

| Pass | Contract |
|---|---|
| `quake-verify-pbc` | Verify the actual operations and recorded hold/release interface |
| `quake-to-pbc` | Retain a lossless rotation word and Clifford/phase frame by default; explicitly release the frame into terminal observables |
| `quake-synthesize-pbc` | Rebuild a native network and complete either its full quantum frame or its release readouts, with the appropriate proof |
| `quake-optimize-pbc` | Run native strategies and Pauli synthesis under the selected boundary contract |
| `quake-lower-pbc` | Lower held rotations/frame with phase and quantum exits intact, or lower released rotations/readouts |
| `quake-try-pbc` | Attempt a complete optimization transaction; hold permits supported scalar quantum returns |

In the historical release workflow, the optional wrapper imports the original native gate network directly; it does
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
its benchmark results. The unified native optimizer uses unitary hold semantics
internally; the pass's `semantics` option independently selects the final
whole-program hold or release boundary.

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
| `quake-to-pbc` | Convert to lossless hold form or explicit released measurement form | `semantics=hold` by default; release-only `assume-measure-all` |
| `quake-synthesize-pbc` | Reconstruct the native network under the selected quantum or measured contract | `semantics=hold` by default, native synthesis and resource controls |
| `quake-optimize-pbc` | Optimize supported PBC under the selected contract | Shared options in A.2–A.6, including A.3a |
| `quake-try-pbc` | Attempt conversion, optimization, and gate lowering as a per-function transaction | Shared options, plus the wrapper options in A.7 |
| `quake-lower-pbc` | Lower verified PBC to primitive Quake | The angle-lowering option in A.8 |

Options are space-separated inside the pass argument. For example, this requests
a T-only experiment with explicit final entangler/depth growth guards:

```sh
cudaq-opt input.qke \
  --quake-try-pbc='semantics=release objective=t-count graph-synthesis=false max-two-qubit-growth=0 max-depth-growth=0 telemetry=pbc.jsonl' \
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
| `semantics` | `hold` | Preserve the complete unitary, scalar phase and scalar SSA quantum outputs. Explicit `release` permits terminal-measurement equivalence and requires no escaping quantum state. This choice is independent of retaining a native Clifford network. Historical result configurations in this report use `release`. |
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

The combined native optimizer requires unitary hold semantics internally;
the pass's `semantics` option selects the whole-program boundary. Native standalone controls such as `preferDepth` and
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
used to produce the measured Clifford+T optimization column. This comparison records the bridge from
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
checker. The procedure recreates the CUDA-Q and Clifford+T optimization columns; it does not rerun
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
The final two columns are the expected CUDA-Q and Clifford+T optimization totals for this report.

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
Measurements are excluded; T, two-qubit and depth entries show CUDA-Q → Clifford+T optimization.

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

| Case | CUDA-Q total | Qiskit 2.6.0 total | Clifford+T optimization total | Best proposed CUDA-Q total (reference study) | T count | Two-qubit count | Depth | Pass / independent verification |
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
profile: 203 corpus cases and 13 generated cases. The original three-compiler
run attempted every case. The Release continuation reuses its completed
measurements, retries missing stages and unresolved Clifford+T optimization
proofs, and adds an independently verified PBC synthesis branch.
CUDA-Q, Qiskit and Clifford+T optimization use the same input circuits, with one
replication per attempted case. Aggregates cover only the verified pairs below.

### F.1 Configuration

The benchmark source is cudaq-bench commit
`2f4d84d78b4eae13d4e479e9453f877f164a838b`; the original CUDA-Q measurements use the local Debug build at
`b09ae4ad1d21a086417a8875bdf0648f6ad99c96`. The executable SHA-256 is
`5dc022b783b0068021ac9b5bce1c4c1ed93cb0b50ebbbe6ae73a0b4c0133dcdd`.
The corpus archive is checked against the snapshot's LFS lock. The separately
identified Release continuation is described in F.4.

The suite builds each Qiskit circuit, saves its QPY, imports it using
`cudaq.contrib.from_qiskit`, performs the builder's AOT compilation, and writes
that Quake to disk. CUDA-Q runs `canonicalize`,
`hw-jit-prep-pipeline{allow-early-exit=false}`, `jit-deploy-pipeline`, the
snapshot's complete Clifford+T target pipeline, and
`jit-finalize-pipeline{lower-device-calls=true}`. The target includes one-qubit
fusion and `normalize-1q-clifford-t`. A deterministic preflight matches the
native runtime's gate totals and T depth on exact U3, Toffoli and Trotter inputs.
Stock and Clifford+T optimization share the synthesized prefix through its last
`dqe`; Clifford+T optimization inserts
`quake-try-pbc` before the common
`func.func(regtomem),symbol-dce` suffix and standard JIT finalization. The optimization
branch removes compiler-generated output annotations with the existing
`func.func(erase-implicit-output)` pass before `quake-try-pbc`; stock retains
those annotations. This pass removes no quantum gate or explicit measurement.
Equivalence checks retain the fixed physical-qubit and measurement contract.
The per-case synthesis epsilon defaults to 0.001. CUDA-Q uses the target's seed
0; sharing the prefix prevents synthesis
variation between the paired outputs.

Clifford+T optimization uses the 40 options in Appendix B with Appendix E's limits:
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
cases, CUDA-Q is skipped, Clifford+T optimization is unavailable, and Qiskit is attempted
independently. Qiskit's conditional-circuit metrics use the suite's worst-case
branch resource bounds.

### F.2 Measurements

Counts are taken from emitted final circuits, excluding measurements and scalar
global phase. The five-way metric groups below are CUDA-Q / Qiskit /
Clifford+T optimization / PBC synthesis (release) / PBC synthesis (hold),
abbreviated **C/Q/O/R/H**. Clifford+T optimization is checked against
stock with identical physical readout order and register names. Identical final
gate sequences and physical readouts prove an unchanged circuit; otherwise
MQT QCEC 3.11.0 checks
unitary equivalence up to global phase. It uses alternating decision diagrams
up to 15 qubits and ZX above that, with a 20-second limit; the Appendix E
lookahead settings give hwb10 and QFT-160 180 seconds. Inconclusive checks are
reported explicitly and excluded from verified-pair aggregates. A pass limit
or unsupported-input fallback is distinguished from an optimized result.
Clifford+T optimization's `committed` outcome denotes internal acceptance; the following verification
result is the independent check. For untranspiled QPE9, saved-output identity
comparison permits gates after a measurement only on other qubits, preserving
physical readout order and register names.

In the original local run under the limits in F.3, CUDA-Q encountered nine frontend and fourteen
stock-compilation timeouts, in addition to the four control-flow skips.
Clifford+T optimization timed out on `feynman:hwb11`. Independent verification was unresolved
for 45 compiled pairs: 44 checks were inconclusive, and QCEC returned
`probably_not_equivalent` for `feynman:ham15_med`. These pairs are excluded from
verified-pair aggregates. These are the historical outcomes; continuation
results and their independent verification status appear in the table.

Both synthesis modes start from the same Clifford+T prefix as the stock branch,
after removing implicit output annotations. Neither consumes the separate
Clifford+T optimization output. The original release continuation described
next precedes the fresh release and hold profiles recorded below.
Earlier release recipes run `quake-to-pbc` followed by
`quake-synthesize-pbc`. Rows marked `native-v1` instead normalize privately
inside `quake-synthesize-pbc`, retaining the original native circuit as a
source-bound incumbent. Expected resource limits can retain that incoming
circuit; its selection is labeled explicitly. Both recipes add terminal Z
measurements on all wires only when no measurements exist, perform checked
native refinement, and use the common finalization suffix before counting.
The preserve-network optimization pass is not run in the synthesis branch.

Release outputs use independent checks with MQT QCEC 3.11.0; latest hold
verification coverage is reported below. The release checker requires the same ordered physical readout wires and register names.
It builds a coherent-copy oracle: execute the circuit, copy each measured Z
value to its own ancilla, then undo the circuit. Oracle equivalence proves the
ordered joint measurement probabilities for arbitrary input states, allowing
the discarded final quantum state to differ. Simulation checking is disabled.
The first attempt uses alternating decision diagrams through 15 input qubits
and ZX above that, one thread and 120 seconds. An unsuccessful ZX attempt is
retained and followed by an independent alternating/lookahead attempt with
180 seconds. Remaining unresolved outputs receive a separate proof-only
continuation with alternating/lookahead, 600 seconds, one thread and a 32 GiB
address-space cap. The same continuation can retry unresolved Clifford+T optimization unitary
checks. It reuses the collected final circuits and metrics, retains all earlier
attempts, and binds its proof to the primary result and final artifact hashes.
Only a completed equivalence proof or identical gate sequence
counts as verified. Emitted metrics may remain visible with an explicit
unverified status; they do not enter verified synthesis aggregates.

<!-- BEGIN FULL SUITE RESULTS -->
**Latest PBC hold: 166 measured cases; 152 wins, 14 ties and 0 losses versus preserved network by total gate count.** No T-for-total-gate exception is needed. Total gates and T count do not regress against the frozen hold admission baseline. Two-qubit count, depth and T depth can trade off and are reported separately.

The hold columns contain only the latest fixed-compiler outputs. The other **50 cases have no measurement in this validation** and their hold cells are blank; they are not counted as current failures. CUDA-Q, Qiskit, preserved-network and release columns retain their latest available reference measurements. Release is reference only and is excluded from the hold win/tie/loss comparison.

**Contracts:** hold preserves the full unitary, scalar phase and physical-wire identities. Release preserves ordered joint terminal measurement probabilities for arbitrary inputs; it need not preserve the full unitary. Preserved network is comparison data only, never a synthesis seed or fallback.

**Verification:** 19 latest outputs have independent passing evidence bound to their output hashes, including one full complex-matrix check with scalar phase. The other unitary checks establish equivalence up to global phase; exact phase is covered by internal source-bound certificates. Both QFT-160 variants have inconclusive short independent checks. Independent verification is not established for the remaining 145 latest outputs. Measurements and independent proof status are reported separately; previous hold verification labels are not carried forward without matching evidence.

Metrics use **C/Q/O/R/H**: CUDA-Q / Qiskit / Clifford+T optimization (preserved network) / PBC synthesis (release) / PBC synthesis (hold). Each mode's five metrics come from one emitted output. Bold marks the smallest reported value for each metric, including ties; it does not imply equivalent semantic contracts across release and hold.

| Compiler / mode | Compilation succeeded | Observed failures | Excluded / unavailable / not run |
|---|---:|---:|---:|
| CUDA-Q | 205 | 7 | 4 |
| Qiskit 2.6.0 | 216 | 0 | 0 |
| Clifford+T optimization | 205 | 0 | 11 |
| PBC synthesis (release) | 197 | 0 | 19 |
| PBC synthesis (hold), latest validation | 166 | 0 | 50 |

Hold compiler SHA-256: `1d413cbe5b52bcd69cf8bdd4f315825816d49c62723d1d8179edf94236e96256`. Release reference compiler SHA-256: `d358d40656d2ba8c7177080c8e54a5203144de2c6d6ee93c48b348d62b2b7486`.

Hold runs locally with frozen source inputs and an 8 GiB per-process address-space limit. Final validation uses at most **two concurrent single-threaded jobs**, reusing completed results from the same compiler. CPU limits are 180 or 300 seconds. The compiler manifest, input/output hashes, all five metrics and proof records are under `build/pbc-hold-research-2026-11-10/`: `candidate-completion-native/compiler-manifest.json`, `completion-native-spots/results.json` and `current-compiler-report.json`. See [the synthesis research report](pbc_hold_research.md) for the implementation findings.

| Case | CUDA-Q total | Qiskit total | Clifford+T optimization total | PBC synthesis (release) total | PBC synthesis (hold) total | T count (C/Q/O/R/H) | Two-qubit count (C/Q/O/R/H) | Depth (C/Q/O/R/H) | T depth (C/Q/O/R/H) | Clifford+T optimization outcome / verification | PBC synthesis (release) outcome / verification | PBC synthesis (hold) outcome / verification |
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
| feynman:adder | 122 | 138 | 106 | 102 | **101** | 32 / 48 / 32 / **31** / **31** | 57 / 65 / 53 / 52 / **50** | 103 / 99 / 95 / 85 / **79** | 32 / 32 / 32 / 27 / **26** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:adder_8 | 727 | 875 | 621 | **489** | 541 | 215 / 361 / **173** / **173** / **173** | 347 / 409 / 320 / **236** / 268 | 228 / 246 / 205 / **156** / 168 | 74 / 90 / 66 / **47** / 55 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:barenco_tof_10 | 365 | 427 | 342 | **268** | 286 | **100** / 192 / **100** / **100** / **100** | 176 / 192 / 168 / **134** / 142 | 324 / 322 / 281 / **207** / 214 | 88 / 128 / 87 / 76 / **71** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:barenco_tof_3 | 50 | 56 | 46 | **37** | 38 | **16** / 24 / **16** / **16** / **16** | 22 / 24 / 21 / **15** / 16 | 44 / 42 / 41 / 35 / **34** | 16 / 16 / 16 / 16 / **15** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:barenco_tof_4 | 95 | 109 | 88 | **69** | 72 | **28** / 48 / **28** / **28** / **28** | 44 / 48 / 42 / **31** / 33 | 84 / 82 / 77 / **59** / 60 | 28 / 32 / 27 / **23** / 25 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:barenco_tof_5 | 140 | 162 | 130 | **102** | 109 | **40** / 72 / **40** / **40** / **40** | 66 / 72 / 63 / **48** / 52 | 124 / 122 / 111 / **84** / 87 | 38 / 48 / 37 / 33 / **31** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:bigadder | 242 | 274 | 209 | 209 | **205** | 64 / 96 / 64 / 64 / **62** | 114 / 130 / 106 / **104** / **104** | 160 / 152 / 144 / 138 / **123** | 48 / 48 / 48 / 45 / **40** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:csla_mux_3 | 161 | 157 | 156 | **146** | 152 | 64 / 64 / **62** / **62** / **62** | 76 / 72 / 72 / **69** / 74 | 69 / 68 / 68 / **57** / 66 | 26 / 25 / 26 / **20** / **20** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| feynman:csum_mux_9 | 392 | 420 | 308 | 289 | **252** | 140 / 196 / **84** / **84** / **84** | 168 / 168 / 168 / 153 / **112** | 61 / 60 / 54 / 71 / **42** | 23 / 24 / **16** / 18 / **16** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| feynman:gf2_10_mult | 1088 | 1257 | 1088 | 1088 | **1087** | **410** / 600 / **410** / **410** / **410** | 609 / 609 / 609 / 609 / **608** | 293 / 307 / 292 / 293 / **223** | 117 / 134 / 118 / 117 / **82** | committed: verified admissible improvement; equivalent | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| feynman:gf2_128_mult | 165019 | 197499 | 165019 | 165019 | **165018** | **65664** / 98304 / **65664** / **65664** / **65664** | 98685 / 98685 / 98685 / 98685 / **98684** | 4073 / 4326 / 4073 / 4073 / **3074** | 1652 / 1905 / 1652 / 1652 / **1146** | skipped: no admissible improvement; identical gate sequence | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| feynman:gf2_16_mult | 2703 | 3179 | 2703 | 2690 | **2683** | **1040** / 1536 / **1040** / **1040** / **1040** | 1581 / 1581 / 1581 / 1568 / **1567** | 491 / 520 / 491 / 489 / **376** | 196 / 225 / 196 / 196 / **138** | skipped: no admissible improvement; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:gf2_256_mult | **657723** | 788219 | **657723** | **657723** | **657723** | **262400** / 393216 / **262400** / **262400** / **262400** | **393981** / **393981** / **393981** / **393981** / **393981** | 8162 / 8671 / 8162 / 8162 / **6138** | 3315 / 3824 / 3315 / 3315 / **2297** | skipped: native gate limit; identical gate sequence | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| feynman:gf2_32_mult | 10562 | 12538 | 10562 | 10562 | **10550** | **4128** / 6144 / **4128** / **4128** / **4128** | 6268 / 6268 / 6268 / 6268 / **6267** | 1003 / 1064 / 1003 / 1003 / **768** | 404 / 465 / 404 / 404 / **282** | skipped: no admissible improvement; identical gate sequence | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| feynman:gf2_4_mult | 194 | 213 | 193 | **184** | **184** | **68** / 96 / **68** / **68** / **68** | 99 / 99 / 99 / 103 / **97** | 102 / 106 / 102 / 97 / **80** | 40 / 45 / 40 / 30 / **25** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:gf2_5_mult | 304 | 332 | 302 | **289** | 294 | **115** / 155 / **115** / **115** / **115** | **154** / **154** / **154** / 159 / 156 | 132 / 138 / 131 / **113** / 126 | 53 / 59 / 53 / **34** / 47 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:gf2_64_mult | 41547 | 49595 | 41547 | 41547 | **41520** | **16448** / 24576 / **16448** / **16448** / **16448** | 24765 / 24765 / 24765 / 24765 / **24764** | 2027 / 2152 / 2027 / 2027 / **1529** | 820 / 945 / 820 / 820 / **570** | skipped: no admissible improvement; identical gate sequence | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| feynman:gf2_6_mult | 414 | 465 | 411 | 403 | **400** | **150** / 216 / **150** / **150** / **150** | 221 / 221 / 221 / 237 / **220** | 166 / 174 / 166 / 168 / **134** | 66 / 75 / 66 / **44** / 50 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:gf2_7_mult | 553 | 627 | 550 | 548 | **547** | **217** / 301 / **217** / **217** / **217** | **300** / **300** / **300** / **300** / 313 | **198** / 208 / **198** / **198** / 209 | 80 / 90 / 80 / 80 / **53** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:gf2_8_mult | 709 | 819 | 705 | 702 | **696** | **264** / 384 / **264** / **264** / **264** | 405 / 405 / 405 / **402** / 410 | **235** / 248 / **235** / **235** / 280 | 92 / 105 / 92 / 92 / **78** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:gf2_9_mult | 889 | 1023 | 887 | 887 | **886** | **351** / 495 / **351** / **351** / **351** | 494 / 494 / 494 / 494 / **493** | 260 / 273 / 258 / 258 / **251** | 106 / 119 / 104 / 104 / **91** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:grover_5 | 668 | 732 | 500 | 512 | **496** | 178 / 296 / **166** / **166** / **166** | 248 / 288 / 228 / 233 / **224** | 533 / 498 / 429 / **380** / 418 | 164 / 192 / 160 / **116** / 154 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:ham15_high | 3659 | 4789 | 3410 | 3653 | **3120** | 1021 / 2173 / 1019 / 1019 / **1016** | 1817 / 2149 / 1722 / 1812 / **1637** | 2993 / 3071 / 2659 / 2982 / **2395** | 871 / 1184 / 873 / 870 / **812** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:ham15_low | 375 | 427 | 364 | 362 | **350** | **97** / 147 / **97** / **97** / **97** | 220 / 236 / 216 / 222 / **204** | 270 / 273 / 256 / **235** / 236 | 77 / 85 / 76 / 73 / **69** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| feynman:ham15_med | 871 | 1156 | 771 | 686 | **654** | 242 / 504 / **211** / 212 / 212 | 428 / 534 / 393 / 355 / **330** | 692 / 758 / 574 / 449 / **413** | 195 / 283 / 169 / 130 / **123** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:hwb10 | 60169 | 69664 | **54688** | 59250 | — | 16437 / 26915 / **15891** / 16437 / — | 31506 / 34828 / **30007** / 30967 / — | 38981 / 38548 / **36910** / 38786 / — | 11807 / 12863 / **11783** / 11803 / — | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | not run in latest validation |
| feynman:hwb11 | 166051 | 194163 | **152364** | 165483 | — | 45638 / 75372 / **44500** / 45638 / — | 87705 / 97615 / **83271** / 87336 / — | 117176 / 115613 / **106212** / 116749 / — | 33842 / 39133 / 33932 / **33840** / — | committed: verified admissible improvement; verification: no_information; no_information | verified; equivalent; fresh-v1 | not run in latest validation |
| feynman:hwb12 | 326567 | 386733 | 326567 | **326190** | — | **87493** / 152755 / **87493** / **87493** / — | 170007 / 191205 / 170007 / **169750** / — | 232046 / **225941** / 232046 / 231794 / — | 65173 / 77375 / 65173 / **65171** / — | skipped: native gate limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| feynman:hwb6 | 229 | 249 | 212 | 207 | **194** | 75 / 97 / 75 / 75 / **72** | 108 / 116 / 105 / 102 / **90** | 166 / 163 / 152 / 142 / **121** | 57 / 59 / 55 / 47 / **41** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:hwb8 | 12546 | 14077 | 11583 | 12231 | **11053** | 3617 / 5437 / 3517 / 3517 / **3509** | 6585 / 7073 / 6305 / 6330 / **5962** | 8522 / 8231 / 8055 / 8334 / **7178** | 2594 / 2662 / 2607 / 2566 / **2167** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:inverseqft1 | — | **419** | — | — | — | — / **156** / — / — / — | — / **0** / — / — / — | — / **416** / — / — / — | — / **156** / — / — / — | unsupported classical control flow; not checked | not rerun: no previous synthesis metrics; prior unsupported control flow | not run in latest validation |
| feynman:inverseqft2 | — | **116** | — | — | — | — / **40** / — / — / — | — / **0** / — / — / — | — / **108** / — / — / — | — / **39** / — / — / — | unsupported classical control flow; not checked | not rerun: no previous synthesis metrics; prior unsupported control flow | not run in latest validation |
| feynman:mod5_4 | 57 | 60 | 37 | **18** | 24 | 16 / 22 / **7** / **7** / 8 | 28 / 28 / 19 / **9** / 14 | 50 / 48 / 30 / **10** / 17 | 14 / 16 / 6 / **3** / 5 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| feynman:mod_adder_1024 | 3035 | 3859 | 2780 | 2945 | **2774** | 1011 / 1739 / 1011 / 1011 / **1004** | 1392 / 1702 / 1306 / 1463 / **1297** | 2472 / 2539 / 2115 / 2141 / **1920** | 746 / 989 / 758 / 701 / **688** | committed: verified admissible improvement; equivalent | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| feynman:mod_mult_55 | 109 | 117 | 92 | **85** | 90 | 37 / 45 / **35** / **35** / **35** | 44 / 48 / 40 / **38** / 40 | 52 / 51 / 46 / **42** / 43 | 20 / 19 / 18 / 16 / **14** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:mod_red_21 | 231 | 262 | 196 | 195 | **187** | **73** / 107 / **73** / **73** / **73** | 93 / 105 / 87 / 85 / **80** | 176 / 166 / 149 / **123** / 125 | 55 / 59 / 55 / **46** / 47 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:qcla_adder_10 | 440 | 481 | 419 | **388** | 406 | **162** / 212 / **162** / **162** / **162** | 205 / 215 / 200 / **169** / 185 | 72 / 75 / 66 / 62 / **58** | 26 / 29 / 25 / 24 / **22** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| feynman:qcla_com_7 | 335 | 406 | 298 | **279** | 288 | **95** / 169 / **95** / **95** / **95** | 152 / 174 / 144 / **136** / **136** | 77 / 85 / 72 / **64** / 71 | 26 / 34 / 26 / **23** / 26 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:qcla_mod_7 | 742 | 827 | 678 | 652 | **648** | 249 / 351 / **237** / **237** / **237** | 348 / 366 / 324 / 307 / **299** | 211 / 211 / 194 / **159** / 180 | 74 / 83 / 73 / **50** / 66 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| feynman:qft | 496 | 1021 | 411 | **405** | 406 | 199 / 299 / **195** / **195** / **195** | 12 / 12 / 12 / **11** / 12 | 330 / 670 / 266 / **265** / **265** | 131 / 189 / **127** / 128 / 128 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:qft_4 | 171 | 484 | 157 | **146** | **146** | **67** / 146 / **67** / **67** / **67** | 46 / 45 / 46 / 35 / **34** | 146 / 445 / 130 / **114** / 117 | **52** / 127 / **52** / **52** / **52** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:rc_adder_6 | 191 | 185 | **147** | 151 | **147** | 63 / 67 / **47** / **47** / **47** | 91 / 83 / **76** / 78 / **76** | 111 / 103 / **91** / 92 / **91** | 40 / 39 / **31** / 33 / **31** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| feynman:tof_10 | 215 | 247 | 183 | **166** | 183 | **71** / 103 / **71** / **71** / **71** | 86 / 102 / 78 / **61** / 78 | 180 / 171 / 157 / **130** / 152 | 62 / 68 / 62 / **53** / 61 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:tof_3 | 40 | 44 | 36 | **33** | 35 | **15** / 19 / **15** / **15** / **15** | 16 / 18 / 15 / **12** / 14 | 33 / 31 / 31 / 29 / **26** | 13 / 12 / 13 / 13 / **11** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| feynman:tof_4 | 65 | 73 | 57 | **52** | 57 | **23** / 31 / **23** / **23** / **23** | 26 / 30 / 24 / **19** / 24 | 54 / 51 / 49 / **40** / 44 | 20 / 20 / 20 / **17** / 19 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:tof_5 | 90 | 102 | 78 | **71** | 78 | **31** / 43 / **31** / **31** / **31** | 36 / 42 / 33 / **26** / 33 | 75 / 71 / 67 / **55** / 62 | 27 / 28 / 27 / **23** / 26 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| feynman:vbe_adder_3 | 101 | 128 | 97 | 86 | **79** | **24** / 56 / **24** / **24** / **24** | 54 / 58 / 52 / 46 / **38** | 66 / 76 / 61 / **46** / 50 | 16 / 30 / 16 / **11** / 15 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_adder_n118_adder_n118 | 1574 | 1782 | 1366 | 1363 | **1314** | 416 / 624 / 416 / 416 / **403** | 741 / 845 / 689 / 672 / **650** | 643 / 639 / **635** / 672 / 654 | 224 / 224 / 224 / 221 / **218** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_adder_n118_adder_n118_transpiled | 1809 | 1834 | **1366** | **1366** | **1366** | **416** / 624 / **416** / **416** / **416** | 741 / 845 / **689** / **689** / **689** | 762 / 679 / **635** / **635** / **635** | **224** / **224** / **224** / **224** / **224** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_adder_n28_adder_n28 | 364 | 412 | 316 | 306 | **304** | 96 / 144 / 96 / 96 / **93** | 171 / 195 / 159 / **146** / 150 | 193 / 189 / 185 / 181 / **174** | 64 / 64 / 64 / 60 / **58** | committed: verified admissible improvement; equivalent | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent unitary verification up to global phase |
| qasmbench:qasmbench_large_adder_n28_adder_n28_transpiled | 419 | 424 | 316 | **306** | 316 | **96** / 144 / **96** / **96** / **96** | 171 / 195 / 159 / **146** / 159 | 222 / 199 / 185 / 181 / **167** | 64 / 64 / 64 / 60 / **55** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent unitary verification up to global phase |
| qasmbench:qasmbench_large_adder_n433_adder_n433 | 5809 | 6577 | 5041 | 5041 | **4849** | 1536 / 2304 / 1536 / 1536 / **1488** | 2736 / 3120 / 2544 / 2544 / **2400** | 2218 / 2214 / **2210** / **2210** / 2334 | 784 / 784 / 784 / 784 / **778** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_adder_n433_adder_n433_transpiled | 6674 | 6769 | **5041** | **5041** | **5041** | **1536** / 2304 / **1536** / **1536** / **1536** | 2736 / 3120 / **2544** / **2544** / **2544** | 2652 / 2359 / **2210** / **2210** / **2210** | **784** / **784** / **784** / **784** / **784** | committed: verified admissible improvement; equivalent | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_adder_n64_adder_n64 | 848 | 960 | 736 | 733 | **708** | 224 / 336 / 224 / 224 / **217** | 399 / 455 / 371 / 360 / **350** | 373 / 369 / **365** / 378 / 366 | 128 / 128 / 128 / 125 / **122** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_adder_n64_adder_n64_transpiled | 975 | 988 | 736 | **733** | 736 | **224** / 336 / **224** / **224** / **224** | 399 / 455 / 371 / **360** / 371 | 438 / 391 / **365** / 378 / **365** | 128 / 128 / 128 / **125** / 128 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent unitary verification up to global phase |
| qasmbench:qasmbench_large_ising_n34_ising_n34 | 4966 | 10636 | 4018 | **3985** | 3991 | 1914 / 2841 / **1906** / **1906** / **1906** | **66** / **66** / **66** / 92 / 67 | 309 / 561 / 255 / **248** / 252 | 124 / 142 / 124 / **122** / 123 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_ising_n34_ising_n34_transpiled | 4977 | 10515 | 4006 | 3985 | **3966** | 1914 / 2841 / **1900** / **1900** / **1900** | **66** / **66** / **66** / 95 / 71 | 317 / 563 / 254 / **250** / 251 | 124 / 142 / 123 / **122** / **122** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_ising_n42_ising_n42 | 6049 | 13102 | 4926 | 4895 | **4881** | 2348 / 3500 / **2328** / **2328** / **2328** | **82** / **82** / **82** / 119 / 84 | 324 / 570 / **256** / 258 / 258 | **122** / 143 / **122** / **122** / 123 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_ising_n420_ising_n420 | 62106 | 132380 | 49981 | 62064 | **49579** | 23986 / 36123 / **23684** / 23986 / **23684** | **838** / **838** / **838** / **838** / 895 | 301 / 585 / 242 / 301 / **241** | **116** / 150 / **116** / **116** / **116** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| qasmbench:qasmbench_large_ising_n420_ising_n420_transpiled | 62449 | 133768 | 49901 | 62449 | **49519** | 23986 / 36123 / **23684** / 23986 / **23684** | **838** / **838** / **838** / **838** / 890 | 293 / 620 / 239 / 293 / **237** | 116 / 150 / 116 / 116 / **115** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent unitary verification up to global phase |
| qasmbench:qasmbench_large_ising_n66_ising_n66 | 9606 | 20394 | 7686 | 7744 | **7631** | 3712 / 5564 / **3646** / **3646** / **3646** | **130** / **130** / **130** / 202 / 132 | 306 / 557 / 248 / 251 / **245** | 120 / 146 / **118** / **118** / **118** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_ising_n66_ising_n66_transpiled | 9650 | 20373 | 7651 | 7697 | **7597** | 3712 / 5564 / **3644** / **3644** / **3644** | 130 / 130 / **129** / 212 / 144 | 301 / 561 / 246 / 247 / **243** | 120 / 146 / 120 / **119** / **119** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_ising_n98_ising_n98 | 14421 | 30133 | 11567 | 11723 | **11491** | 5565 / 8375 / **5483** / **5483** / **5483** | **194** / **194** / **194** / 306 / **194** | 312 / 574 / 249 / 257 / **247** | 120 / 146 / **118** / **118** / **118** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_ising_n98_ising_n98_transpiled | 14510 | 30263 | 11536 | 11708 | **11439** | 5563 / 8375 / **5481** / **5481** / **5481** | **194** / **194** / **194** / 295 / 211 | 315 / 570 / 245 / 254 / **244** | 120 / 146 / **118** / **118** / **118** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_multiplier_n350_multiplier_n350 | 166584 | 304183 | **161182** | 166483 | — | 39202 / 136080 / **39062** / 39202 / — | 92754 / 145880 / **92148** / 92652 / — | 85210 / 141075 / **80334** / 85175 / — | 21622 / 53489 / **21486** / 21622 / — | committed: verified admissible improvement; verification: inconclusive; inconclusive | verified; equivalent; fresh-v1 | not run in latest validation |
| qasmbench:qasmbench_large_multiplier_n350_multiplier_n350_transpiled | 176925 | 326475 | **161229** | 176841 | — | 39202 / 136080 / **39062** / 39202 / — | 92754 / 145880 / **92197** / 92673 / — | 94856 / 160187 / **80315** / 94840 / — | 21622 / 53489 / **21487** / 21622 / — | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | not run in latest validation |
| qasmbench:qasmbench_large_multiplier_n400_multiplier_n400 | 217578 | 397635 | 217578 | **217503** | — | **51202** / 177920 / **51202** / **51202** / — | 121204 / 190720 / 121204 / **121129** / — | 111400 / 184438 / 111400 / **111382** / — | **28307** / 69929 / **28307** / **28307** / — | skipped: search size/work/deadline limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| qasmbench:qasmbench_large_multiplier_n400_multiplier_n400_transpiled | 230996 | 426711 | 230996 | **230939** | — | **51202** / 177920 / **51202** / **51202** / — | 121204 / 190720 / 121204 / **121150** / — | 124021 / 209467 / 124021 / **124014** / — | **28307** / 69929 / **28307** / **28307** / — | skipped: search size/work/deadline limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| qasmbench:qasmbench_large_multiplier_n45_multiplier_n45 | 2756 | 4796 | 2599 | **2273** | 2437 | 650 / 2124 / **634** / **634** / **634** | 1500 / 2286 / 1428 / 1288 / **1275** | 1370 / 2224 / 1252 / 1203 / **1106** | 333 / 846 / 319 / 307 / **302** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_multiplier_n45_multiplier_n45_transpiled | 2985 | 5189 | 2611 | **2337** | 2424 | 650 / 2124 / **634** / **634** / **634** | 1500 / 2286 / 1430 / 1335 / **1286** | 1518 / 2502 / 1247 / 1286 / **1127** | 333 / 846 / 319 / 321 / **310** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_multiplier_n75_multiplier_n75 | 7651 | 13619 | 7307 | **6637** | 6853 | 1802 / 6060 / **1774** / **1774** / **1774** | 4206 / 6510 / 4082 / 3798 / **3651** | 3833 / 6304 / 3547 / 3816 / **3247** | 942 / 2394 / 916 / 963 / **887** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_multiplier_n75_multiplier_n75_transpiled | 8215 | 14681 | 7320 | **6735** | 6825 | 1802 / 6060 / **1774** / **1774** / **1774** | 4206 / 6510 / 4086 / 3869 / **3649** | 4257 / 7122 / 3533 / 3900 / **3320** | 942 / 2394 / 916 / 957 / **901** | committed: verified admissible improvement; equivalent | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_qft_n160_qft_n160 | 119613 | 380659 | 96773 | 119531 | **96632** | 46959 / 105786 / **46029** / 46959 / **46029** | **2790** / 5700 / **2790** / **2790** / **2790** | 26147 / 65327 / **21420** / 26126 / **21420** | 10551 / 17920 / **10392** / 10551 / **10392** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent check inconclusive |
| qasmbench:qasmbench_large_qft_n160_qft_n160_transpiled | 106920 | 380702 | 87577 | 106843 | **87359** | 41873 / 105786 / **41247** / 41873 / **41247** | **2790** / 5700 / **2790** / **2790** / **2790** | 25485 / 65327 / 21369 / 25390 / **21365** | 10067 / 17919 / 10059 / **10031** / 10060 | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent check inconclusive |
| qasmbench:qasmbench_large_qft_n29_qft_n29 | 18253 | 57351 | 15052 | 15091 | **14969** | 7266 / 15920 / **7122** / **7122** / **7122** | **432** / 722 / **432** / 545 / **432** | 4479 / 10962 / **3602** / 5000 / 3629 | 1774 / 2986 / **1746** / 2296 / **1746** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_qft_n29_qft_n29_transpiled | 16994 | 57394 | 13851 | 13937 | **13808** | 6634 / 15920 / **6528** / **6528** / **6528** | **432** / 722 / **432** / 436 / **432** | 4343 / 10962 / 3549 / 3679 / **3528** | 1683 / 2985 / 1674 / 1685 / **1668** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_qft_n320_qft_n320 | 239671 | 775539 | 239671 | 239650 | **206610** | 95439 / 215546 / 95439 / 95439 / **94809** | **5670** / 11780 / **5670** / **5670** / **5670** | 52701 / 131727 / 52701 / 52701 / **45383** | 21271 / 36160 / 21271 / 21271 / **21268** | skipped: search size/work/deadline limit; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_qft_n320_qft_n320_transpiled | 216825 | 775582 | 216825 | 216804 | **187096** | 84913 / 215546 / 84913 / 84913 / **84903** | **5670** / 11780 / **5670** / **5670** / 5672 | 52650 / 131727 / 52650 / 52648 / **44389** | 20307 / 36159 / 20307 / 20307 / **20301** | skipped: search size/work/deadline limit; identical gate sequence | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_qft_n63_qft_n63 | 44289 | 141263 | 36199 | 43994 | **36144** | 17568 / 39244 / **17220** / 17568 / **17220** | **1044** / 2014 / **1044** / **1044** / **1044** | 10041 / 25072 / **8228** / 9979 / **8228** | 4052 / 6862 / **3990** / 4052 / **3990** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_large_qft_n63_qft_n63_transpiled | 40124 | 141306 | 32643 | 32934 | **32556** | 15780 / 39244 / **15424** / **15424** / **15424** | 1044 / 2014 / 1044 / **969** / 1059 | 9561 / 25072 / 8175 / **8054** / 8170 | 3859 / 6861 / 3848 / **3779** / 3848 | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_medium_bigadder_n18_bigadder_n18 | 242 | 274 | 209 | 209 | **205** | 64 / 96 / 64 / 64 / **62** | 114 / 130 / 106 / **104** / **104** | 160 / 152 / 144 / 138 / **123** | 48 / 48 / 48 / 45 / **40** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_medium_bigadder_n18_bigadder_n18_transpiled | 278 | 282 | **209** | **209** | **209** | **64** / 96 / **64** / **64** / **64** | 114 / 130 / 106 / **104** / 106 | 171 / 157 / 142 / **138** / 142 | 48 / 48 / 48 / **45** / 48 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| qasmbench:qasmbench_medium_ising_n26_ising_n26 | 3785 | 7480 | 3045 | 3026 | **3023** | 1454 / 2132 / **1442** / **1442** / **1442** | **50** / **50** / **50** / 69 / 51 | 282 / 532 / 228 / **222** / 229 | 110 / 145 / 109 / **108** / 109 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_medium_ising_n26_ising_n26_transpiled | 3814 | 7489 | 3035 | 3019 | **3011** | 1454 / 2132 / **1444** / **1444** / **1444** | **50** / **50** / **50** / 67 / 55 | 284 / 531 / 225 / **224** / 227 | 110 / 145 / 110 / **109** / 111 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_medium_multiplier_n15_multiplier_n15 | 313 | 476 | 280 | **212** | 237 | 74 / 204 / **70** / **70** / **70** | 162 / 222 / 142 / **113** / 128 | 168 / 232 / 147 / **98** / 99 | 44 / 90 / 41 / 32 / **27** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_medium_multiplier_n15_multiplier_n15_transpiled | 351 | 524 | 281 | **209** | 252 | 74 / 204 / **70** / **70** / **70** | 162 / 222 / 142 / **108** / 134 | 186 / 258 / 145 / **115** / 116 | 44 / 90 / 42 / 35 / **26** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_medium_multiply_n13_multiply_n13 | 93 | 95 | 93 | **73** | 91 | **32** / 36 / **32** / **32** / **32** | 40 / 40 / 40 / **29** / 40 | 41 / 42 / 41 / **19** / 32 | 16 / 15 / 16 / **6** / 13 | skipped: no admissible improvement; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_medium_multiply_n13_multiply_n13_transpiled | 105 | 95 | 91 | **73** | 91 | **32** / 36 / **32** / **32** / **32** | 40 / 40 / 40 / **29** / 40 | 40 / 42 / 39 / **19** / 39 | 15 / 15 / 15 / **6** / 15 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| qasmbench:qasmbench_medium_qft_n18_qft_n18 | 10017 | 30205 | 8104 | 8159 | **8086** | 3933 / 8374 / **3855** / **3855** / **3855** | **234** / 306 / **234** / 303 / **234** | 2607 / 6397 / 2108 / 2573 / **2107** | 1037 / 1732 / **1020** / 1182 / **1020** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_medium_qft_n18_qft_n18_transpiled | 9385 | 30248 | 7594 | 7642 | **7566** | 3675 / 8374 / **3589** / **3589** / **3589** | **234** / 306 / **234** / 258 / **234** | 2452 / 6397 / 2049 / **1972** / 2043 | 979 / 1731 / 967 / **902** / 964 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_adder_n10_adder_n10 | 122 | 138 | 106 | 102 | **101** | 32 / 48 / 32 / **31** / **31** | 57 / 65 / 53 / 52 / **50** | 103 / 99 / 95 / 85 / **79** | 32 / 32 / 32 / 27 / **26** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_adder_n10_adder_n10_transpiled | 141 | 142 | 106 | **102** | 105 | 32 / 48 / 32 / **31** / 32 | 57 / 65 / 53 / **52** / 53 | 114 / 103 / 95 / **85** / 95 | 32 / 32 / 32 / **27** / 32 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| qasmbench:qasmbench_small_adder_n4_adder_n4 | 23 | 23 | 22 | **20** | **20** | 8 / 8 / 8 / **7** / 8 | 10 / 10 / 10 / 10 / **8** | **11** / **11** / **11** / 12 / 14 | **2** / **2** / **2** / 3 / 5 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_adder_n4_adder_n4_transpiled | 24 | 25 | 22 | **20** | **20** | 8 / 8 / 8 / **7** / 8 | 10 / 10 / 10 / 10 / **8** | 12 / 13 / **11** / 12 / 14 | **2** / **2** / **2** / 3 / 5 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_test_n4 | 800 | 1516 | 616 | **606** | 607 | 296 / 420 / **288** / **288** / **288** | 34 / 28 / 27 / **22** / 25 | 354 / 674 / 265 / 271 / **263** | 124 / 174 / **122** / 124 / 124 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_test_n4_transpiled | 885 | 1579 | 654 | 651 | **645** | 314 / 424 / **310** / **310** / **310** | 46 / 38 / 30 / 27 / **25** | 542 / 825 / **396** / 398 / 478 | 188 / 212 / **187** / 188 / 228 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_trotter_n4 | 19667 | 47122 | 14410 | 14249 | **14248** | 7136 / 12685 / **6886** / **6886** / **6886** | 462 / 303 / 304 / 332 / **287** | 8619 / 20053 / 6292 / 6599 / **6238** | 3047 / 5281 / **2988** / 3136 / 2993 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_trotter_n4_transpiled | 20299 | 35930 | 14816 | 14706 | **14688** | 7190 / 9481 / **7108** / **7108** / **7108** | 582 / 351 / 331 / 385 / **320** | 10704 / 18209 / 7809 / **7625** / 7733 | 3766 / 4780 / 3738 / **3660** / 3735 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_ipea_n2_ipea_n2 | — | **2331** | — | — | — | — / **688** / — / — / — | — / **14** / — / — / — | — / **2320** / — / — / — | — / **686** / — / — / — | unsupported classical control flow; not checked | not rerun: no previous synthesis metrics; prior unsupported control flow | not run in latest validation |
| qasmbench:qasmbench_small_ipea_n2_ipea_n2_transpiled | — | **2341** | — | — | — | — / **688** / — / — / — | — / **14** / — / — / — | — / **2323** / — / — / — | — / **686** / — / — / — | unsupported classical control flow; not checked | not rerun: no previous synthesis metrics; prior unsupported control flow | not run in latest validation |
| qasmbench:qasmbench_small_ising_n10_ising_n10 | 9302 | 18655 | 7336 | 7308 | **7286** | 3644 / 5140 / **3578** / **3578** / **3578** | **72** / 90 / **72** / 96 / 75 | 1385 / 3040 / 1099 / 1092 / **1087** | 539 / 835 / 533 / 532 / **531** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_ising_n10_ising_n10_transpiled | 9314 | 18779 | 7381 | 7343 | **7327** | 3642 / 5108 / **3600** / **3600** / **3600** | **72** / 90 / **72** / 99 / 79 | 1390 / 3332 / 1092 / 1089 / **1088** | 539 / 926 / 533 / **532** / 533 | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_pea_n5_pea_n5 | 671 | 2988 | 550 | **544** | 547 | 264 / 831 / **260** / **260** / **260** | 20 / 26 / **17** / 19 / 19 | 503 / 2706 / 344 / 340 / **339** | 197 / 751 / 163 / **162** / **162** | committed: verified admissible improvement; equivalent | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_pea_n5_pea_n5_transpiled | 709 | 2992 | 578 | **569** | **569** | 272 / 831 / **270** / **270** / **270** | 20 / 26 / **18** / 21 / 23 | 527 / 2708 / 361 / **356** / 358 | 204 / 751 / 170 / **169** / **169** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_qft_n4_qft_n4 | 504 | 1021 | 408 | **402** | 404 | 199 / 299 / **195** / **195** / **195** | 12 / 12 / 12 / **11** / 12 | 338 / 670 / **263** / **263** / **263** | 131 / 189 / **127** / 128 / 128 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_qft_n4_qft_n4_transpiled | 531 | 756 | 430 | 429 | **427** | 207 / 226 / **205** / **205** / **205** | **12** / **12** / **12** / 14 / **12** | 303 / 620 / 244 / **201** / 243 | 119 / 185 / 117 / **96** / 118 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| qasmbench:qasmbench_small_qpe_n9_qpe_n9 | **1208** | 3022 | **1208** | — | — | **473** / 863 / **473** / — / — | **41** / 43 / **41** / — / — | **1192** / 1749 / **1192** / — / — | **472** / 506 / **472** / — / — | skipped: PBC rotations must precede all measurements and sinks; identical gate sequence | not rerun: no previous synthesis metrics; prior failed | not run in latest validation |
| qasmbench:qasmbench_small_qpe_n9_qpe_n9_transpiled | 1233 | 3046 | 998 | **975** | 979 | 467 / 863 / 461 / **459** / 461 | 41 / 43 / 40 / **36** / 38 | 855 / 1756 / 694 / 322 / **316** | 327 / 506 / 321 / **149** / **149** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| generated_qft:qft_n002 | 8 | **7** | 8 | **7** | 8 | **3** / **3** / **3** / **3** / **3** | 3 / **2** / 3 / **2** / 3 | 6 / **5** / 6 / **5** / 6 | 3 / 3 / 3 / **2** / 3 | skipped: no admissible improvement; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| generated_qft:qft_n004 | 504 | 1019 | 416 | **411** | 412 | 199 / 299 / **195** / **195** / **195** | 14 / **12** / 14 / 13 / 16 | 338 / 669 / 267 / 270 / **266** | 131 / 189 / **127** / 128 / 128 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| generated_qft:qft_n008 | 2379 | 6256 | 1946 | 1942 | **1920** | 945 / 1706 / **919** / **919** / **919** | 60 / **56** / 60 / 72 / 60 | 930 / 2258 / **758** / 881 / 762 | 373 / 629 / **365** / 415 / 366 | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| generated_qft:qft_n016 | 8469 | 25279 | 6820 | 6863 | **6754** | 3327 / 7002 / **3237** / **3237** / **3237** | **206** / 240 / **206** / 260 / 224 | 2240 / 5567 / **1839** / 2232 / 1851 | 903 / 1509 / **888** / 1031 / **888** | committed: verified admissible improvement; equivalent | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| generated_qft:qft_n032 | 20785 | 64755 | 16857 | 16857 | **16727** | 8175 / 17978 / **7957** / **7957** / **7957** | **502** / 836 / **502** / **502** / **502** | 4990 / 12207 / **4013** / **4013** / 4070 | 1975 / 3328 / **1944** / **1944** / **1944** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| generated_qft:qft_n064 | 45339 | 143731 | 36879 | 45038 | **36569** | 17871 / 39930 / **17397** / 17871 / **17397** | **1094** / 2052 / **1094** / **1094** / **1094** | 10212 / 25487 / **8365** / 10211 / **8365** | 4119 / 6976 / **4056** / 4119 / **4056** | committed: verified admissible improvement; verification: no_information; no_information | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| generated_adder:cuccaro_n002 | 59 | 67 | 51 | **49** | 51 | 16 / 24 / 16 / **15** / 16 | 29 / 33 / 27 / **25** / 27 | 52 / 50 / 48 / **39** / 48 | 16 / 16 / 16 / **13** / 16 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| generated_adder:cuccaro_n004 | 117 | 133 | 101 | **97** | **97** | 32 / 48 / 32 / **31** / **31** | 57 / 65 / 53 / **49** / 50 | 102 / 98 / 94 / 79 / **78** | 32 / 32 / 32 / 27 / **26** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| generated_adder:cuccaro_n008 | 233 | 265 | 201 | **193** | 197 | 64 / 96 / 64 / **63** / **63** | 113 / 129 / 105 / **97** / 104 | 202 / 194 / 186 / **159** / 176 | 64 / 64 / 64 / **55** / 60 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| generated_adder:cuccaro_n016 | 465 | 529 | 401 | 401 | **397** | 128 / 192 / 128 / 128 / **127** | 225 / 257 / 209 / 209 / **208** | 402 / 386 / 370 / 370 / **360** | 128 / 128 / 128 / 128 / **124** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| generated_adder:cuccaro_n032 | 929 | 1057 | 801 | 801 | **797** | 256 / 384 / 256 / 256 / **255** | 449 / 513 / 417 / 417 / **416** | 802 / 770 / 738 / 738 / **728** | 256 / 256 / 256 / 256 / **252** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| generated_adder:cuccaro_n064 | 1857 | 2113 | 1601 | 1601 | **1597** | 512 / 768 / 512 / 512 / **511** | 897 / 1025 / 833 / 833 / **832** | 1602 / 1538 / 1474 / 1474 / **1464** | 512 / 512 / 512 / 512 / **508** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| generated_single_rotation:a00_eps1em03 | **0** | **0** | **0** | **0** | **0** | **0** / **0** / **0** / **0** / **0** | **0** / **0** / **0** / **0** / **0** | **0** / **0** / **0** / **0** / **0** | **0** / **0** / **0** / **0** / **0** | skipped: no admissible improvement; identical gate sequence | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:adder_adder_10q | 122 | 138 | 106 | 102 | **101** | 32 / 48 / 32 / **31** / **31** | 57 / 65 / 53 / 52 / **50** | 103 / 99 / 95 / 85 / **79** | 32 / 32 / 32 / 27 / **26** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent unitary verification up to global phase |
| ftcircuitbench:adder_adder_28q | 364 | 412 | 316 | 306 | **304** | 96 / 144 / 96 / 96 / **93** | 171 / 195 / 159 / **146** / 150 | 193 / 189 / 185 / 181 / **174** | 64 / 64 / 64 / 60 / **58** | committed: verified admissible improvement; equivalent | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:adder_adder_4q | 23 | 23 | 22 | **20** | **20** | 8 / 8 / 8 / **7** / 8 | 10 / 10 / 10 / 10 / **8** | **11** / **11** / **11** / 12 / 14 | **2** / **2** / **2** / 3 / 5 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:adder_adder_64q | 848 | 960 | 736 | 733 | **694** | 224 / 336 / 224 / 224 / **217** | 399 / 455 / 371 / 360 / **350** | 373 / 369 / **365** / 378 / 368 | 128 / 128 / 128 / 125 / **122** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_fermi_hubbard_1d_128q | 1072958 | 2535160 | 1072958 | **1070875** | — | **394240** / 648850 / **394240** / **394240** / — | 50920 / 59524 / 50920 / **50109** / — | 459503 / 742210 / 459503 / **458759** / — | **173600** / 195275 / **173600** / **173600** / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_fermi_hubbard_1d_18q | 152636 | 358057 | 121693 | 151928 | **119964** | 55440 / 91315 / **54602** / 55440 / **54602** | 7640 / 8482 / 7172 / 7188 / **7032** | 144023 / 274510 / 115264 / 143511 / **114140** | 54320 / 70995 / 53542 / 54320 / **53322** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_fermi_hubbard_1d_72q | **605975** | 1420904 | **605975** | — | — | **221760** / 363390 / **221760** / — / — | **29640** / 34204 / **29640** / — / — | **496863** / 813050 / **496863** / — / — | **187600** / 213515 / **187600** / — / — | skipped: native gate limit; identical gate sequence | not rerun: no previous synthesis metrics; prior stopped | not run in latest validation |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_128q | **1994738** | 4054614 | **1994738** | — | — | **680960** / 1007308 / **680960** / — / — | **182440** / 190924 / **182440** / — / — | **824962** / 1475312 / **824962** / — / — | **292320** / 374224 / **292320** / — / — | skipped: function operation limit; identical gate sequence | not rerun: no previous synthesis metrics; prior stopped | not run in latest validation |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_18q | 272850 | 554679 | 272850 | **272138** | — | **95760** / 139484 / **95760** / **95760** / — | 18840 / 19802 / 18840 / **18282** / — | 231925 / 463692 / 231925 / **231656** / — | **85120** / 119124 / **85120** / **85120** / — | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_72q | **1127903** | 2262974 | **1127903** | — | — | **383040** / 560608 / **383040** / — / — | **105404** / 110248 / **105404** / — / — | **830661** / 1505532 / **830661** / — / — | **294000** / 381804 / **294000** / — / — | skipped: function operation limit; identical gate sequence | not rerun: no previous synthesis metrics; prior stopped | not run in latest validation |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_triangular_128q | — | **5552582** | — | — | — | — / **1365708** / — / — / — | — / **326140** / — / — / — | — / **2570632** / — / — / — | — / **645864** / — / — / — | unavailable: Release stock compilation did not complete; stock-prefix: timeout; not checked | not rerun: no previous synthesis metrics; prior failed | not run in latest validation |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_triangular_18q | 390902 | 773194 | 390902 | **390398** | — | **136080** / 194784 / **136080** / **136080** / — | 29720 / 30562 / 29720 / **29334** / — | 347254 / 664512 / 347254 / **347073** / — | **126560** / 170964 / **126560** / **126560** / — | skipped: native gate limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_fermi_hubbard_2d_triangular_72q | **1634014** | 3100445 | **1634014** | — | — | **544320** / 762208 / **544320** / — / — | **177088** / 181892 / **177088** / — / — | **1329436** / 2366932 / **1329436** / — / — | **463120** / 596004 / **463120** / — / — | skipped: function operation limit; identical gate sequence | not rerun: no previous synthesis metrics; prior stopped | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_1d_100q | 960723 | 1966352 | 960723 | **960647** | — | **359400** / 558779 / **359400** / **359400** / — | 23960 / **11883** / 23960 / 23922 / — | 940425 / **921021** / 940425 / 940349 / — | 359400 / **281938** / 359400 / 359400 / — | skipped: native gate limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_1d_36q | 356435 | 700176 | 356435 | **355968** | — | **129000** / 198715 / **129000** / **129000** / — | 8600 / **4203** / 8600 / 8562 / — | 349065 / **331133** / 349065 / 348711 / — | 129000 / **100946** / 129000 / 129000 / — | skipped: native gate limit; identical gate sequence | verified; equivalent_up_to_global_phase; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_1d_64q | 614331 | 1254128 | 614331 | **614255** | — | **229800** / 356243 / **229800** / **229800** / — | 15320 / **7563** / 15320 / 15282 / — | 601305 / **589209** / 601305 / 601229 / — | 229800 / **180130** / 229800 / 229800 / — | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_1d_9q | 85081 | 166008 | 65391 | 83813 | **64957** | 31800 / 46813 / **31136** / 31800 / **31136** | 2120 / **963** / 1762 / 2077 / 1896 | 83165 / 82274 / **64133** / 82279 / 64192 | 31800 / **24590** / 31129 / 31800 / 31134 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_heisenberg_2d_100q | 1915643 | 3820963 | 1915643 | **1915567** | — | **714600** / 1096782 / **714600** / **714600** / — | 47640 / **23403** / 47640 / 47602 / — | 355745 / 406692 / 355745 / **355669** / — | 136200 / **124383** / 136200 / 136200 / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_2d_36q | 698075 | 1354995 | 698075 | **698036** | — | **256200** / 390626 / **256200** / **256200** / — | 17080 / **8283** / 17080 / 17042 / — | 208785 / 235320 / 208785 / **208746** / — | 78600 / **71563** / 78600 / 78600 / — | skipped: native gate limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_2d_64q | 1223411 | 2430759 | 1223411 | **1223335** | — | **456600** / 698972 / **456600** / **456600** / — | 30440 / **14883** / 30440 / 30402 / — | 280505 / 321006 / 280505 / **280429** / — | 107400 / **97973** / 107400 / 107400 / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_2d_9q | 176401 | 330685 | 131867 | 175648 | **130979** | 63600 / 95786 / **62616** / 63600 / **62616** | 4240 / **1983** / 3563 / 4199 / 3827 | 95605 / 107391 / 71774 / 95282 / **71756** | 35400 / **32021** / 34848 / 35400 / 34846 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_100q | 2892723 | 5781095 | 2892723 | **2892647** | — | **1079400** / 1665708 / **1079400** / **1079400** / — | 71960 / **35883** / 71960 / 71922 / — | 1971165 / 2374260 / 1971165 / **1971089** / — | 755400 / **723452** / 755400 / 755400 / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_36q | 1040395 | 2063107 | 1040395 | **1040318** | — | **388200** / 596360 / **388200** / **388200** / — | 25880 / **12843** / 25880 / 25842 / — | 731285 / 869988 / 731285 / **731208** / — | 280200 / **264924** / 280200 / 280200 / — | skipped: native gate limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_64q | 1850811 | 3687773 | 1850811 | **1850735** | — | **690600** / 1064074 / **690600** / **690600** / — | 46040 / **22923** / 46040 / 46002 / — | 1276125 / 1529692 / 1276125 / **1276049** / — | 489000 / **466092** / 489000 / 489000 / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_9q | 263321 | 504599 | 263321 | 262891 | **212598** | 96600 / 145823 / 96600 / 96600 / **96522** | 6440 / **3123** / 6440 / 6375 / 6682 | 199185 / 227052 / 199185 / 198884 / **159752** | 75000 / **68532** / 75000 / 75000 / 75003 | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_1d_100q | 273560 | 817226 | 273560 | 273452 | **229875** | 107800 / 209316 / 107800 / 107800 / **107798** | **4200** / 7962 / **4200** / **4200** / 4313 | 123120 / 543986 / 123120 / 123120 / **105047** | 48400 / 140016 / 48400 / 48400 / **46474** | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_1d_36q | 97240 | 292426 | 79846 | 97240 | **79841** | 38808 / 74916 / **38766** / 38808 / **38766** | **1512** / 2842 / **1512** / **1512** / **1512** | 45402 / 195826 / 38505 / 45402 / **38488** | 17808 / 50416 / **17766** / 17808 / **17766** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_1d_64q | 175064 | 522026 | 141972 | 174904 | **141960** | 68992 / 133716 / **68950** / 68992 / **68950** | **2688** / 5082 / **2688** / **2688** / **2688** | 79344 / 348146 / 67569 / 79342 / **66111** | 31192 / 89616 / 31150 / 31192 / **31089** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_1d_9q | 24584 | 71026 | 19944 | 20077 | **19941** | 9702 / 18216 / **9660** / **9660** / **9660** | **378** / 682 / **378** / 391 / **378** | 12464 / 48946 / 10478 / **10264** / 10460 | 4902 / 12616 / 4860 / **4841** / 4860 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_2d_100q | 391200 | 1358452 | 391200 | **391132** | — | **155600** / 348632 / **155600** / **155600** / — | **8400** / 15924 / **8400** / **8400** / — | **39176** / 206706 / **39176** / **39176** / — | **15420** / 53216 / **15420** / **15420** / — | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_ising_2d_36q | 140832 | 485492 | 117032 | 140832 | **116500** | 56016 / 124632 / **55850** / 56016 / **55850** | **3024** / 5684 / **3024** / **3024** / 3078 | 23976 / 119666 / 20293 / 23976 / **19996** | 9444 / 30816 / **9385** / 9444 / 9396 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_2d_64q | 250920 | 867412 | 250920 | 250816 | **214050** | 99584 / 222632 / 99584 / 99584 / **99544** | **5376** / 10164 / **5376** / **5376** / 5377 | **31698** / 163186 / **31698** / **31698** / 54584 | **12432** / 42016 / **12432** / **12432** / 24103 | skipped: candidate operation limit; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_2d_9q | 35392 | 117212 | 29243 | 29193 | **29055** | 14004 / 30132 / **13934** / **13934** / **13934** | **756** / 1364 / **756** / 958 / **756** | 12658 / 54386 / 10598 / 10806 / **10396** | 4962 / 14016 / **4925** / 5083 / 4939 | committed: verified admissible improvement; equivalent | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_2d_triangular_100q | 519200 | 1899678 | 519200 | **519166** | — | **203400** / 487948 / **203400** / **203400** / — | **12600** / 23886 / **12600** / **12600** / — | 160678 / 707186 / 160678 / **160662** / — | **62740** / 182016 / **62740** / **62740** / — | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_ising_2d_triangular_36q | 185081 | 678558 | 154181 | 185081 | **153503** | 73224 / 174348 / **72914** / 73224 / **72914** | **4536** / 8526 / **4536** / **4536** / **4536** | 67400 / 293746 / 57038 / 67400 / **55009** | 26412 / 75616 / 26244 / 26412 / **25948** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_ising_2d_triangular_64q | 330712 | 1212798 | 330712 | **330636** | — | **130176** / 311548 / **130176** / **130176** / — | **8064** / 15246 / **8064** / **8064** / — | **108566** / 478706 / **108566** / **108566** / — | **42664** / 123216 / **42664** / **42664** / — | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_ising_2d_triangular_9q | 46301 | 163398 | 38529 | 38575 | **38211** | 18306 / 42048 / **18200** / 18306 / **18200** | **1134** / 2046 / **1134** / 1230 / **1134** | 23598 / 98178 / 19900 / **17090** / 19323 | 9264 / 25286 / 9197 / **7911** / 9056 | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_128q | 141724 | 297312 | 116152 | 141724 | **113313** | 53200 / 80118 / **53056** / 53200 / **53056** | **4250** / **4250** / **4250** / **4250** / **4250** | 46977 / 111017 / 39039 / 46977 / **37961** | 18032 / 29482 / 17990 / 18032 / **17955** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_18q | 21210 | 43326 | 17405 | 20985 | **17102** | 8020 / 11669 / **7978** / **7978** / **7978** | 645 / 645 / 645 / **641** / 709 | 10791 / 24180 / 9004 / 10716 / **8954** | 4146 / 6432 / **4131** / 4135 / 4195 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_200q | 221837 | 467786 | 221837 | 221833 | **186211** | 83720 / 125948 / 83720 / 83720 / **83476** | 6795 / 6795 / 6795 / **6791** / 7905 | 108439 / 255954 / 108439 / 108439 / **91292** | **41552** / 67912 / **41552** / **41552** / 41889 | skipped: candidate operation limit; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_72q | 79456 | 168317 | 65396 | 79452 | **63992** | 29960 / 45188 / **29876** / 29960 / **29876** | 2505 / 2505 / 2505 / **2501** / 2502 | 33787 / 79614 / 28082 / 33787 / **27353** | 12880 / 21000 / 12851 / 12880 / **12826** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_128q | **280548** | 714216 | **280548** | **280548** | — | **98700** / 184114 / **98700** / **98700** / — | **19530** / **19530** / **19530** / **19530** / — | **91005** / 228013 / **91005** / **91005** / — | **33116** / 59039 / **33116** / **33116** / — | skipped: native gate limit; identical gate sequence | verified; identical gate sequence; fresh-v1; incoming retained | not run in latest validation |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_18q | 39236 | 92770 | 32419 | 38651 | **31900** | 14270 / 24182 / **14252** / 14270 / **14252** | 1915 / 1915 / 1915 / **1905** / 1912 | 21457 / 49508 / 17940 / 21260 / **17393** | 7966 / 12772 / 7967 / 7966 / **7923** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_200q | **467096** | 1142041 | **467096** | — | — | **163240** / 297945 / **163240** / — / — | **35375** / **35375** / **35375** / — / — | **222456** / 553007 / **222456** / — / — | **79392** / 142095 / **79392** / — / — | skipped: native gate limit; identical gate sequence | not rerun: no previous synthesis metrics; prior stopped | not run in latest validation |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_72q | 160466 | 402286 | 132646 | 160451 | **131451** | 56440 / 104440 / **56360** / 56440 / **56360** | 11110 / 11110 / 11110 / **11095** / 11197 | 73346 / 180351 / 62181 / 73336 / **61502** | 26292 / 46114 / 26244 / 26292 / **26233** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_128q | **442226** | 1088904 | **442226** | — | — | **155020** / 277384 / **155020** / — / — | **34464** / **34464** / **34464** / — / — | **154372** / 370861 / **154372** / — / — | **55104** / 93128 / **55104** / — / — | skipped: native gate limit; identical gate sequence | not rerun: no previous synthesis metrics; prior stopped | not run in latest validation |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_18q | 59559 | 141011 | 49225 | 59078 | **48354** | **21620** / 36757 / **21620** / **21620** / **21620** | 3115 / 3115 / 3115 / **3105** / 3129 | 34177 / 78597 / 28601 / 34001 / **27828** | 12648 / 20134 / 12654 / 12648 / **12603** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_200q | **721928** | 1753415 | **721928** | — | — | **251200** / 451109 / **251200** / — / — | **61100** / **61100** / **61100** / — / — | **370279** / 894781 / **370279** / — / — | **130240** / 224437 / **130240** / — / — | skipped: native gate limit; identical gate sequence | not rerun: no previous synthesis metrics; prior stopped | not run in latest validation |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_72q | 248388 | 616970 | 248388 | **248365** | — | **87120** / 158929 / **87120** / **87120** / — | 18775 / 18775 / 18775 / **18752** / — | 120330 / 294566 / 120330 / **120301** / — | **42732** / 73754 / **42732** / **42732** / — | skipped: candidate operation limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_100q | 106132 | 305858 | 85031 | 106132 | **83653** | 41580 / 81509 / **40584** / 41580 / **40584** | **1485** / **1485** / **1485** / **1485** / **1485** | 15391 / 50992 / 12505 / 15391 / **12390** | 5992 / 13598 / **5881** / 5992 / **5881** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_36q | 37617 | 108226 | 30028 | 30614 | **29453** | 14700 / 28837 / **14344** / **14344** / **14344** | **525** / **525** / **525** / 865 / 560 | 6188 / 19824 / 5016 / 5342 / **4963** | 2408 / 5278 / **2361** / 2427 / **2361** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_64q | 66911 | 194690 | 54109 | 66911 | **53011** | 26460 / 51881 / **25824** / 26460 / **25824** | **945** / **945** / **945** / **945** / 1008 | 10075 / 33460 / 8292 / 10075 / **8147** | 3976 / 8918 / **3901** / 3976 / **3901** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_9q | 8494 | 24850 | 6861 | 6818 | **6728** | 3360 / 6616 / **3274** / **3274** / **3274** | **120** / **120** / **120** / 205 / 128 | 2264 / 6675 / 1857 / 1899 / **1831** | 896 / 1768 / **876** / 881 / **876** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_100q | 210698 | 644956 | 210698 | 210552 | **179794** | 85008 / 171208 / 85008 / 85008 / **83378** | **3000** / **3000** / **3000** / **3000** / 3985 | 23281 / 65489 / 23281 / 23209 / **22686** | **9352** / 17252 / **9352** / **9352** / 10413 | skipped: candidate operation limit; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_36q | 76475 | 230252 | 62502 | 76475 | **61242** | 30800 / 61064 / **29692** / 30800 / **29692** | **1080** / **1080** / **1080** / **1080** / 1226 | 13793 / 38201 / 11338 / 13793 / **11113** | 5544 / 10052 / **5340** / 5544 / **5340** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_64q | 137526 | 411396 | 110826 | 137304 | **108769** | 54544 / 109168 / **52644** / 54544 / **52644** | **1920** / **1920** / **1920** / **1920** / 2138 | 18989 / 51845 / 15289 / 18908 / **14980** | 7448 / 13652 / **7200** / 7447 / **7200** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_9q | 19584 | 56816 | 15754 | 16184 | **15444** | 7784 / 15038 / **7486** / 7490 / **7486** | **270** / **270** / **270** / 557 / 289 | 6894 / 17927 / 5547 / 6071 / **5445** | 2716 / 4692 / **2618** / 2747 / **2618** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_100q | 313130 | 968265 | 313130 | **313031** | — | **126000** / 256636 / **126000** / **126000** / — | **4500** / **4500** / **4500** / **4500** / — | 145581 / 422554 / 145581 / **145563** / — | **58800** / 110700 / **58800** / **58800** / — | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_36q | 112927 | 346709 | 91903 | 112927 | **90133** | 45360 / 91836 / **43626** / 45360 / **43626** | **1620** / **1620** / **1620** / **1620** / 1759 | 54168 / 158950 / 44534 / 54168 / **43780** | 21840 / 41668 / **21110** / 21840 / **21110** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; identical gate sequence; fresh-v1; incoming retained | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_64q | 202402 | 618339 | 202402 | 202256 | **170672** | 80640 / 163848 / 80640 / 80640 / **79058** | **2880** / **2880** / **2880** / **2880** / 3970 | 94455 / 275152 / 94455 / **94397** / 95431 | **38080** / 72104 / **38080** / **38080** / 43854 | skipped: candidate operation limit; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_9q | 27975 | 86069 | 22962 | 23452 | **22512** | 11340 / 22773 / **10896** / **10896** / **10896** | **405** / **405** / **405** / 814 / 436 | 14386 / 43147 / 11987 / 13064 / **11779** | 5880 / 11314 / **5684** / 6001 / **5684** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_100q | 71000 | 118518 | 58069 | **58006** | 58012 | 28000 / 33500 / **27990** / 28000 / **27990** | **1000** / **1000** / **1000** / 1002 / **1000** | 35360 / 64563 / 29751 / 28803 / **28762** | 14140 / 16670 / 14126 / 13646 / **13636** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no completed proof; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_36q | 25560 | 42678 | 20885 | **20696** | 20860 | 10080 / 12060 / **10070** / **10070** / **10070** | **360** / **360** / **360** / 362 / **360** | 12960 / 23603 / 10871 / **10368** / 10522 | 5180 / 6110 / 5166 / **4996** / **4996** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_64q | 44800 | 75858 | 37154 | 44378 | **37112** | 17920 / 21440 / **17910** / 17920 / **17910** | **640** / **640** / **640** / **640** / **640** | 22120 / 41523 / 18818 / 21696 / **18501** | 9100 / 10730 / 9090 / 9100 / **8776** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_9q | 6300 | 10683 | 5199 | **5167** | 5176 | 2520 / 3015 / **2510** / **2510** / **2510** | **90** / **90** / **90** / 102 / **90** | 3420 / 6323 / 2868 / **2822** / 2856 | 1400 / 1655 / 1390 / **1357** / 1381 | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_100q | 104000 | 182518 | 84905 | 103809 | **84772** | 42000 / 50000 / **41080** / 42000 / **41080** | **2000** / **2000** / **2000** / **2000** / **2000** | 11104 / 20769 / **9222** / 11068 / 9223 | 4564 / 5384 / 4448 / 4564 / **4441** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_36q | 37440 | 65718 | 30615 | 36240 | **30545** | 15120 / 18000 / **14800** / **14800** / **14800** | **720** / **720** / **720** / **720** / **720** | 6752 / 12577 / 5602 / 6528 / **5582** | 2772 / 3272 / 2700 / 2700 / **2692** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_64q | 67840 | 116818 | 54306 | 67590 | **54266** | 26880 / 32000 / **26300** / 26880 / **26300** | **1280** / **1280** / **1280** / **1280** / **1280** | 9180 / 16673 / 7434 / 9178 / **7392** | 3668 / 4328 / 3574 / 3668 / **3560** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_9q | 9360 | 16443 | 7661 | 7700 | **7639** | 3780 / 4500 / **3700** / **3700** / **3700** | **180** / **180** / **180** / 219 / **180** | 3488 / 6433 / 2883 / 2963 / **2878** | 1428 / 1688 / 1391 / 1401 / **1390** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_100q | 138000 | 246518 | 111326 | 137861 | **111285** | 56000 / 66500 / **54080** / 56000 / **54080** | **3000** / **3000** / **3000** / **3000** / **3000** | 44560 / 83763 / **35879** / 44486 / 36095 | 18340 / 21620 / 17495 / 18340 / **17491** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_36q | 49680 | 88758 | 40114 | 49108 | **40081** | 20160 / 23940 / **19480** / 20160 / **19480** | **1080** / **1080** / **1080** / **1080** / **1080** | 18720 / 35123 / **15173** / 18383 / 15196 | 7700 / 9080 / 7375 / 7700 / **7371** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_64q | 88320 | 157778 | 71280 | 88114 | **71224** | 35840 / 42560 / **34620** / 35840 / **34620** | **1920** / **1920** / **1920** / **1920** / **1920** | 30280 / 56883 / 24446 / 30158 / **24430** | 12460 / 14690 / **11905** / 12460 / **11905** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_9q | 12420 | 22203 | 10058 | 10268 | **10020** | 5040 / 5985 / **4870** / **4870** / **4870** | **270** / **270** / **270** / 391 / **270** | 6820 / 12705 / 5534 / **4766** / 5523 | 2800 / 3305 / 2680 / **2233** / 2680 | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hhl_hhl_12q | 131555 | 257513 | 103534 | 130978 | **102789** | 51304 / 70263 / **50240** / 51304 / **50240** | 2692 / 3084 / 2690 / 2668 / **2563** | 102667 / 184988 / 80771 / 102247 / **80110** | 40122 / 50357 / 39230 / 40122 / **39150** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hhl_hhl_21q | 1275633 | 2487067 | 1275633 | **1273553** | — | **497107** / 682415 / **497107** / **497107** / — | 26117 / 29813 / 26117 / **26085** / — | 1012668 / 1860311 / 1012668 / **1011070** / — | **393092** / 508912 / **393092** / **393092** / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:hhl_hhl_4q | 926 | 2316 | 734 | **721** | 723 | 354 / 630 / **352** / **352** / **352** | 22 / 22 / 18 / 17 / **16** | 839 / 1722 / 667 / 593 / **590** | 324 / 460 / 320 / 288 / **287** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:hhl_hhl_7q | 15395 | 29791 | 12295 | 12305 | **12227** | 6059 / 8191 / **5999** / **5999** / **5999** | 257 / 317 / **249** / 312 / 254 | 11321 / 22027 / 9029 / 10020 / **8916** | 4458 / 6021 / 4413 / 4880 / **4382** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:qft_qft_18q | 10021 | 30205 | 8103 | 8118 | **8085** | 3933 / 8374 / **3855** / **3855** / **3855** | **234** / 306 / **234** / 279 / **234** | 2607 / 6397 / **2108** / 2294 / **2108** | 1037 / 1732 / **1020** / 1050 / **1020** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent_up_to_global_phase; fresh-v1 | measured; independent unitary verification up to global phase |
| ftcircuitbench:qft_qft_29q | 18321 | 57351 | 14972 | 15219 | **14943** | 7266 / 15920 / **7122** / **7122** / **7122** | **432** / 722 / **432** / 556 / **432** | 4401 / 10962 / **3604** / 4478 / **3604** | 1774 / 2986 / **1746** / 2017 / **1746** | committed: verified admissible improvement; equivalent_up_to_global_phase | unverified; no_information; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:qft_qft_4q | 496 | 1021 | 411 | 408 | **407** | 199 / 299 / **195** / **195** / **195** | 12 / 12 / 12 / **11** / 13 | 330 / 670 / **263** / 267 / 264 | 131 / 189 / **127** / 128 / 128 | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:qft_qft_63q | 44766 | 141263 | 36231 | 44527 | **36002** | 17568 / 39244 / **17102** / 17568 / **17102** | **1044** / 2014 / **1044** / **1044** / **1044** | 10267 / 25072 / **8228** / 10259 / 8277 | 4052 / 6862 / **3990** / 4052 / **3990** | committed: verified admissible improvement; equivalent | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:qpe_qpe_h2_0_6_12q | — | **20048891** | — | — | — | — / **5512786** / — / — / — | — / **232410** / — / — / — | — / **15321245** / — / — / — | — / **4204099** / — / — / — | unavailable: Release stock compilation did not complete; stock-prefix: timeout; not checked | not rerun: no previous synthesis metrics; prior failed | not run in latest validation |
| ftcircuitbench:qpe_qpe_h2_0_74_12q | — | **20213908** | — | — | — | — / **5553262** / — / — / — | — / **232410** / — / — / — | — / **15391635** / — / — / — | — / **4217711** / — / — / — | unavailable: Release stock compilation did not complete; stock-prefix: timeout; not checked | not rerun: no previous synthesis metrics; prior failed | not run in latest validation |
| ftcircuitbench:qpe_qpe_h2_1_0_12q | — | **20050898** | — | — | — | — / **5523746** / — / — / — | — / **232410** / — / — / — | — / **15329173** / — / — / — | — / **4204505** / — / — / — | unavailable: Release stock compilation did not complete; stock-prefix: timeout; not checked | not rerun: no previous synthesis metrics; prior failed | not run in latest validation |
| ftcircuitbench:qpe_qpe_h2_1_5_12q | — | **20538805** | — | — | — | — / **5612948** / — / — / — | — / **232410** / — / — / — | — / **15795296** / — / — / — | — / **4293008** / — / — / — | unavailable: Release stock compilation did not complete; stock-prefix: timeout; not checked | not rerun: no previous synthesis metrics; prior failed | not run in latest validation |
| ftcircuitbench:qpe_qpe_hubbard_10q | 2222706 | 4188598 | 2222706 | **2220658** | — | **859207** / 1150956 / **859207** / **859207** / — | **49559** / 55580 / **49559** / **49559** / — | 1795350 / 3200591 / 1795350 / **1793798** / — | **693815** / 875817 / **693815** / **693815** / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:qpe_qpe_hubbard_8q | 529667 | 999931 | 529667 | **529507** | — | **205011** / 273889 / **205011** / **205011** / — | 11813 / 13262 / 11813 / **11807** / — | 427496 / 764444 / 427496 / **427284** / — | 165332 / 208602 / 165332 / **165298** / — | skipped: native gate limit; identical gate sequence | verified; equivalent; fresh-v1 | not run in latest validation |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_10q | 1550034 | 3205952 | 1550034 | **1547986** | — | **612042** / 920421 / **612042** / **612042** / — | **58358** / 71814 / **58358** / **58358** / — | 1371630 / 2779501 / 1371630 / **1369846** / — | **540678** / 795045 / **540678** / **540678** / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_11q | 2843663 | 6214073 | 2843663 | **2841615** | — | **1125726** / 1768042 / **1125726** / **1125726** / — | 113104 / 139983 / 113104 / **113094** / — | 2543455 / 5381576 / 2543455 / **2541685** / — | **1007365** / 1521871 / **1007365** / **1007365** / — | skipped: function operation limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_12q | — | **10979973** | — | — | — | — / **3125500** / — / — / — | — / **247336** / — / — / — | — / **9588277** / — / — / — | — / **2710002** / — / — / — | unavailable: Release stock compilation did not complete; stock-prefix: timeout; not checked | not rerun: no previous synthesis metrics; prior failed | not run in latest validation |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_13q | — | **18449983** | — | — | — | — / **5249566** / — / — / — | — / **408753** / — / — / — | — / **16195524** / — / — / — | — / **4578992** / — / — / — | unavailable: Release stock compilation did not complete; stock-prefix: timeout; not checked | not rerun: no previous synthesis metrics; prior failed | not run in latest validation |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_6q | 80457 | 182235 | 63919 | 79593 | **63522** | 31644 / 53256 / **30814** / 31644 / **30814** | 1814 / 2002 / 1751 / 1765 / **1691** | 78718 / 153395 / 62576 / 77879 / **62252** | 31235 / 44969 / 30437 / 31235 / **30436** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent full-matrix verification including phase |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_7q | 147414 | 307752 | 118339 | 146847 | **117435** | 58735 / 90890 / **56729** / 58735 / **56729** | 4108 / 5164 / **3830** / 4097 / 3859 | 126577 / 258237 / 101462 / 126045 / **100716** | 50486 / 76124 / 48792 / 50482 / **48756** | committed: verified admissible improvement; equivalent_up_to_global_phase | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_8q | 255293 | 550916 | 255293 | 255010 | **209940** | 101712 / 159028 / 101712 / 101712 / **98550** | 9502 / 12294 / 9502 / 9498 / **9099** | 220910 / 479608 / 220910 / 220671 / **179159** | 88163 / 137008 / 88163 / 88162 / **84636** | skipped: candidate operation limit; identical gate sequence | verified; equivalent; fresh-v1 | measured; independent verification not established for this output |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_9q | 678748 | 1325141 | 678748 | **678618** | — | **265604** / 375904 / **265604** / **265604** / — | 25436 / 30987 / 25436 / **25434** / — | 609336 / 1234179 / 609336 / **609208** / — | 238259 / 347188 / 238259 / **238258** / — | skipped: native gate limit; identical gate sequence | unverified; no_information; fresh-v1 | not run in latest validation |
<!-- END FULL SUITE RESULTS -->

### F.2a Release continuation outcomes

This records the earlier reconciliation run. The current F.2 table also
includes the later capacity continuation and post-refactor updates.

The two-machine continuation finished all queued work: **212 synthesis
attempts, 45 retries of unresolved historical Clifford+T optimization proofs, and 30 additional
proof-only continuations**. The four unsupported classical-control cases remain
explicit skips. No cases remain queued or running.

All **593 previously collected CUDA-Q, Qiskit and Clifford+T optimization metric sets** are
unchanged. Release compilation completed 16 of the 23 previously timed-out
stock cases, and Clifford+T optimization completed its previously timed-out
`hwb11` case. Stock and Clifford+T optimization now each have 205 compiled
outputs. Of those optimization outputs, **202 are
independently verified**, compared with 143 before this continuation. The
alternating/lookahead retries resolved 43 of the 45 earlier inconclusive or
probable-mismatch checks, including the historical `ham15_med` result, using
its unchanged saved output. The remaining Clifford+T optimization checks are `feynman:hwb11`,
`qasmbench:qasmbench_large_multiplier_n350_multiplier_n350`, and
`generated_qft:qft_n064`; their emitted counts remain explicitly unverified.

PBC synthesis emitted **125 native circuits**. The independent checker
ran on every emitted output; **108 passed** and **17 remain unverified** after
the recorded proof attempts. Of the successful checks, 107 are QCEC equivalence
results and one is an identical gate-sequence/readout check. The 17 unresolved
outputs comprise 13 `no_information` results and four attempts without a
completed proof result. Their measured metrics are visible but excluded from
verified aggregates. Synthesis never used a preserve-network output as a seed
or fallback.

The other 87 synthesis attempts produced no completed output: 40 exceeded the
64-million logical dependency cap; 20 exhausted the measured synthesis budget
or found no T-admissible completion; 15 reported invalid input or the gate cap;
one rejected rotations after measurements/sinks; four reached the per-process
address-space allowance; and seven could not complete the prerequisite stock
prefix within its one-hour CPU cap. The original nine frontend timeouts were
resumed from their saved imported raw Quake, so the seven remaining stock
failures are recorded at the current stock-prefix stage. These are bounded-run
outcomes, not zero-gate circuits or estimates of an unavailable synthesis result.

The final audit re-counted all five metrics in all 125 final synthesis outputs,
checked the T-nonincreasing guard, matched every accepted proof to its final
output hash and ordered physical readouts, and confirmed the 593 retained
historical metric sets. The matrix, worker identities, commands, proof attempts,
resource records and raw logs are under `build/pbc-remote-2026-11-06/`;
`merged/summary.json` and `audit.json` give the coverage and consistency checks.

### F.3 Reproduction

The local evidence bundle is `build/pbc-ci-runtime-2026-11-04`. It contains the
216-case inventory, frozen source snapshot and corpus cache, expanded target
pipeline, scripts, package versions, configuration manifest, complete result
matrix and per-case commands, inputs, outputs, telemetry and diagnostics.
The attempted CUDA-Q/Clifford+T optimization cases use the complete runtime pipeline above.
Previously measured Qiskit results are reused only when saved QPY hashes match;
`inherited-qiskit.json` records their provenance. `summary.json` audits coverage,
configuration hashes and the inputs and outputs supporting each result;
`status.json` records completion. The compiler executable and options are
unchanged in that original run. `continuation-single.json` and `continuation-importer.json` record
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
Clifford+T optimization has 1200 CPU / 1250 wall seconds.
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

### F.4 Release continuation and PBC synthesis reproduction

The benchmark helpers are maintained outside the source checkout in
`~/pbc-benchmark-tools/`. Set `pbc_tools` to that directory locally, or to the
external copy on a benchmark host. The continuation inputs and original result
rows are frozen by `python3 "$pbc_tools/run.py" prepare`. The bundle is
`build/pbc-remote-2026-11-06/evidence`; its manifest records each reused QPY,
compiler input, final reference and outstanding Clifford+T optimization output
with SHA-256 hashes. Successful stock, Qiskit and Clifford+T optimization measurements are retained without recompiling
them. A changed compiler receives a new worker manifest and result directory.
The fifteen-case tuning study has a different input boundary; its counts are
not substituted for measurements of these full-runtime inputs.

The compiler source is commit
`247a051f0d471e65d793b26dd5cd271cedca28a0` on `pbc-opt`. Both remote hosts build
it in Release mode. LLVM is the repository's pinned submodule commit
`def143a6c624dc9b991ebfdfec5c36a7084171eb`, with
`tpls/customizations/llvm/mlir-ssi.diff` applied. Unpatched distribution MLIR
is insufficient. The compiler executable, relevant source files, LLVM commit,
patch, CMake cache and runner hashes are recorded per worker. Each worker
snapshots its executable before launching cases. Package versions are pinned to
Qiskit 2.6.0 and MQT QCEC 3.11.0 in an isolated environment.
The two ARM64 Release executables have the same SHA-256:
`c085bfdffd33044d3be02635c3880086f9f7bcdd7b52684b600f54467303a0cb`.
The native preflights reproduced all five metrics for Adder-10
`(102,31,52,85,27)` and QFT-2 `(7,3,2,5,2)` and passed independent QCEC.

The machines are `thiennguyen@10.117.10.135` (80 ARM64 cores) and
`thiennguyen@10.117.6.255` (128 ARM64 cores), each with approximately 498 GiB
physical RAM. Work lives under
`/localhome/thiennguyen/pbc-benchmark-20261106`, on local disks rather than the
full shared home filesystem. The build script caps compilation at 76 and 96
jobs respectively, with two simultaneous LLVM link jobs.

For Ubuntu, install CMake, Ninja, Clang/LLD 22 and `libclang-22-dev`, the development headers for zlib,
libedit, libffi, ncurses, GMP, MPFR and MPC, GNU `time`, and Python venv/pip support. Then:

```bash
pbc_root=/localhome/thiennguyen/pbc-benchmark-20261106
pbc_tools="$pbc_root/tools"
# Transfer ~/pbc-benchmark-tools/ to "$pbc_tools" on the benchmark host.
git clone --depth 1 --branch pbc-opt \
  https://github.com/1tnguyen/cuda-quantum.git cuda-quantum
cd cuda-quantum
git fetch --depth 1 origin 247a051f0d471e65d793b26dd5cd271cedca28a0
git checkout --detach 247a051f0d471e65d793b26dd5cd271cedca28a0
bash "$pbc_tools/build_release.sh" "$PWD" "$pbc_root"
```

On the machine holding the original evidence, prepare one immutable input
bundle, then transfer that bundle and the external helper directory to both hosts:

```bash
pbc_tools="$HOME/pbc-benchmark-tools"
python3 "$pbc_tools/run.py" prepare \
  --configuration "$pbc_tools/synthesis-reconciliation-v1.json" \
  --source build/pbc-ci-runtime-2026-11-04 \
  --output build/pbc-remote-2026-11-06/evidence --shards 2
```

The queue contains 212 synthesis attempts and 45 proof-only retries; four
unsupported control-flow cases remain explicit skips. Twenty-three synthesis
tasks first resume the missing stock compilation stage, and the timed-out optimization
case retries that branch separately. Missing frontend compilation resumes from
the saved imported raw Quake using
`builtin.module(aot-prep-pipeline)`, the same pipeline called by the builder's
`compile()`. Missing stock prefixes use the exact expanded target pipeline
stored in the evidence manifest, including each case's epsilon. Existing
successful prefix inputs are reused byte for byte. The Clifford+T optimization retry retains
Appendix F.1's options and per-case work overrides.

The expanded stock prefix, before the shared native-gate boundary, is:

```bash
pbc_prefix='canonicalize,hw-jit-prep-pipeline{allow-early-exit=false},jit-deploy-pipeline,func.func(add-dealloc,combine-quantum-alloc,canonicalize,dqe),convert-to-linear-values,unitary-synthesis,apply-op-specialization,func.func(constant-propagation),decomposition{basis=h,s,t,rx,ry,rz,x,y,z,swap,mx,my,mz,x(1),y(1),z(1)},func.func(fuse-1q-chains-for-clifford-t{epsilon=0.001},canonicalize,quake-simplify{threshold=0.001 rotations-to-clifford-t=true clifford-t-epsilon=0.001},normalize-phase-placement,lower-phase,canonicalize,dqe,phase-folding,canonicalize),decomposition{enable-patterns=RxToRz,RyToRz},clifford-t-synthesis{epsilon=0.001 seed=0 skip-below=0 fail-on-controlled-rotation=true},inline,func.func(quake-simplify,normalize-1q-clifford-t,phase-folding,normalize-phase-placement,lower-phase,canonicalize,dqe)'
pbc_root=/localhome/thiennguyen/pbc-benchmark-20261106
pbc_tools="$pbc_root/tools"
pbc_opt="$pbc_root/release-pinned/bin/cudaq-opt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1
unset CUDAQ_PBC_TELEMETRY
"$pbc_opt" input.mlir "--pass-pipeline=builtin.module($pbc_prefix)" \
  --mlir-disable-threading -o baseline.qke
"$pbc_opt" baseline.qke \
  --pass-pipeline='builtin.module(func.func(regtomem),symbol-dce,jit-finalize-pipeline{lower-device-calls=true})' \
  --mlir-disable-threading -o stock.qke
"$pbc_opt" baseline.qke \
  --pass-pipeline='builtin.module(func.func(erase-implicit-output))' \
  --mlir-disable-threading -o pbc-input.qke
```

For a case with a different recorded epsilon, substitute its value for every
`epsilon=0.001` and `threshold=0.001` in this prefix. A saved raw frontend input
first runs `--pass-pipeline='builtin.module(aot-prep-pipeline)'`. These stages
are skipped when their successful frozen outputs already exist.

The **uniform final synthesis configuration**, stored in
`$pbc_tools/synthesis-reconciliation-v1.json`, expands to this complete invocation.
These are the settings used for the measured table; later capacity studies use
separately identified configurations rather than changing these measurements.

```bash
pbc_synthesis_options=(
  objective=total-gates
  synthesis-native-entanglers=cx,cz,swap
  synthesis-swap-cost=1
  synthesis-simplify=false
  synthesis-prune-unobserved=false
  synthesis-discrepancy-steps=22
  max-work=1000000000000
  timeout-ms=300000
  max-gates=262144
  max-dependencies=64000000
  synthesis-legacy-max-work=100000000000
  synthesis-legacy-portfolio-timeout-ms=180000
  synthesis-representation-steps=32
  synthesis-readout-candidates=8
  synthesis-commuting-candidates=8
  synthesis-algebraic-candidates=4
  synthesis-indexed-representation-steps=64
  synthesis-joint-phase-candidates=4
  synthesis-joint-phase-acceptance=gates-and-t
  synthesis-joint-phase-planner-work=3000000000
  synthesis-joint-phase-max-work=100000000000
  synthesis-joint-phase-timeout-ms=120000
  synthesis-joint-phase-wide-search=true
)
"$pbc_opt" pbc-input.qke --convert-to-linear-values \
  --mlir-disable-threading -o linear.qke
"$pbc_opt" linear.qke --quake-to-pbc='semantics=release assume-measure-all=true' \
  --quake-verify-pbc --mlir-disable-threading -o normalized.qke
prlimit --as=8589934592 --cpu=360 --stack=67108864 --core=0 -- \
  "$pbc_opt" normalized.qke \
  "--quake-synthesize-pbc=${pbc_synthesis_options[*]}" \
  --mlir-disable-threading -o synthesized.qke
"$pbc_opt" synthesized.qke \
  --pass-pipeline='builtin.module(func.func(regtomem),symbol-dce,jit-finalize-pipeline{lower-device-calls=true})' \
  --mlir-disable-threading -o output.qke
"$pbc_root/venv/bin/python" "$pbc_tools/verify.py" stock.qke output.qke \
  --contract measured --checker auto --timeout 120 --output verification.json
```

The runner imposes a separate 375-second synthesis process timeout, records the
complete argument vector and environment, and retains synthesis work/exhaustion
attributes. Existing search receives up to 180 seconds and the two independent
joint profiles share up to 120 seconds under the 300-second parent deadline.
All cases use the same options. Finite deadlines can affect which checked
candidate completes; the configuration does not promise identical quality
across different CPUs or system loads. Native CX, CZ and SWAP each count once.
The worker manifest resolves every omitted public synthesis option from that
build's `Passes.td` and records the complete effective parameter map before
launching a case. The option reference is in `pbc_synthesis_tuning.md`.

Run shard 0 on `.135` and shard 1 on `.255`:

```bash
pbc_root=/localhome/thiennguyen/pbc-benchmark-20261106
pbc_tools="$pbc_root/tools"
"$pbc_root/venv/bin/python" "$pbc_tools/run.py" run \
  --source "$pbc_root/cuda-quantum" \
  --binary "$pbc_root/release-pinned/bin/cudaq-opt" \
  --evidence "$pbc_root/evidence" --output "$pbc_root/results" \
  --shard 0 --reserve-gib 48 --memory-budget-gib 440
```

The deterministic two-shard allocation balances estimated work, starts the
longest synthesis/preparation cases first, and schedules proof-only retries
after synthesis launches. Each machine can run up to 55 eight-GiB jobs, limited
by its available RAM and CPU affinity. Missing stock stages reserve 32 GiB per
active case and have 3600 CPU / 3615 wall seconds per stage; ordinary synthesis
and initial verification reserve 8 GiB. Proof-only continuations reserve 32 GiB
and have 630 CPU / 645 wall seconds around QCEC's 600-second limit.
The scheduler reserves at least 48 GiB for the
OS, controller and unexpected usage, checks current available memory before
each launch, and limits total reservations to 440 GiB. Each child has an
address-space cap, a CPU cap, a wall timeout, a 64 MiB stack and disabled core
dumps. All runtime/verification thread counts are one, MLIR threading is
disabled and `CUDAQ_PBC_TELEMETRY` is unset. A timed-out process group is
terminated before its reservation is released. Case failures are recorded
without stopping independent queued cases.

Once the primary queue has launched all jobs (`status.json` has no pending
cases), start the proof-only continuation on each host:

```bash
"$pbc_root/venv/bin/python" "$pbc_tools/retry_proofs.py" \
  --evidence "$pbc_root/evidence" --results "$pbc_root/results"
```

It selects only completed outputs whose first proof attempts were unresolved.
It never reruns synthesis, stock compilation or counting. Admission subtracts
the remaining primary reservations and all active continuation reservations
from the same 440 GiB envelope, also checking live available memory. After the
primary queue has no pending jobs its reservations can only decrease. At most
thirteen 32 GiB proof jobs run concurrently, subject to that shared allowance.
`verification-continuation-status.json` records this additional queue and its
reservations. Each attempt keeps separate commands, proof output, CPU/RSS data,
verifier hash and primary-result hash; the merge validates those bindings before
changing a verification status. Drain both services before restarting the primary
queue so that this admission condition remains true.

`status.json` exposes the active queue and reserved memory. Create
`results/STOP_AFTER_ACTIVE` to drain a worker; remove it before resuming. The
same command resumes without repeating completed tasks and rejects changed
compiler/input identity. Interrupted tasks are archived before retry. New
execution/proof limits require a separate recorded run, rather than silently
turning a failed attempt into a verified result.

Copy both result directories back to `host-135` and `host-255` beneath the local
continuation directory, including final circuits and proof files. Merge without
overwriting historical measured metrics:

```bash
pbc_tools="$HOME/pbc-benchmark-tools"
python3 "$pbc_tools/run.py" merge \
  --evidence build/pbc-remote-2026-11-06/evidence \
  --results build/pbc-remote-2026-11-06/host-135 \
            build/pbc-remote-2026-11-06/host-255 \
  --output build/pbc-remote-2026-11-06/merged \
  --report docs/pbc-optimization-report.md
```

The merged matrix records worker/compiler identity for newly completed cells;
retained measurements keep their original identity. It includes all five
native metrics, proof hashes, full commands, environments, elapsed time,
per-process peak RSS and explicit failures. Inconclusive proof retries retain
the previous unresolved status. Structural-limit failures remain failures;
there is no preserve-network fallback counted as a synthesis success.
<!-- END FULL CI LIVE RUN -->

### F.5 Suite-capacity investigation (2026-11-06)

The capacity investigation uses a separate configuration,
`$HOME/pbc-benchmark-tools/synthesis.json`
(`pbc-synthesis-suite-capacity-v3`). The table above retains the measured F.4
values; new entries identify the capacity recipe. Capacity diagnostics retain their own
compiler, configuration, commands, resources and proof files beneath
`build/pbc-synthesis-scale-2026-11-06/` and the remote
`/localhome/thiennguyen/pbc-benchmark-20261106/scale-20261106/` directories.

The rounded limits cover the saved suite's largest normalized input
(1,125,726 rotations) and largest mechanical bootstrap prefix (134,857,624
gates before terminal readout):

| Option | Capacity configuration |
|---|---:|
| `max-gates` | 200,000,000 |
| `max-dependencies` | 1,000,000,000,000 |
| `synthesis-max-steps` | 200,000,000 |
| `max-work` | 100,000,000,000,000,000 |
| `synthesis-legacy-max-work` | 1,000,000,000,000,000 |

All other search and native-gate settings remain those listed in F.4.
Dependency limits and reported logical dependency counts use 64-bit storage.
The enlarged work allowances fund the conservative whole-word replay charge;
the existing search deadlines still apply. Mandatory bootstrap, proof,
emission and compiler lowering also have an external process limit. Optional
joint completion retains its separately bounded storage and gate allowance.

The Release diagnostic build uses the same pinned LLVM installation as F.4,
Clang 22, and the source patch recorded with each compiler. Its final small-case
checks passed 329 native tests. Independent QCEC checks passed for Adder and
QFT-2; their synthesis metrics `(gates, T, two-qubit, depth, T-depth)` were
`(102, 31, 52, 85, 27)` and `(7, 3, 2, 5, 2)` respectively. Adder retains the
F.4 result after separating optional completion storage from whole-program
capacity.

All 12 existing PBC lit tests also passed against that Release compiler. Its
SHA-256 is
`2a8d2c0e70ad6e9424ef1fd2ac2686ef462631b76499acf840dac87d9f5f910b`;
`source-final.patch` records the code changes relative to the recorded branch
base.

For a fresh source checkout at
`247a051f0d471e65d793b26dd5cd271cedca28a0`, apply `source-final.patch` and
reuse F.4's pinned LLVM installation. The capacity build configuration is:

```bash
pbc_root=/localhome/thiennguyen/pbc-benchmark-20261106
pbc_scale="$pbc_root/scale-20261106"
cmake --fresh -G Ninja -S "$pbc_scale/source" -B "$pbc_scale/release" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-22 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-22 \
  -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_DIR="$pbc_root/llvm/lib/cmake/llvm" \
  -DMLIR_DIR="$pbc_root/llvm/lib/cmake/mlir" \
  -DCLANG_INCLUDE_DIRS=/usr/lib/llvm-22/include \
  -DLLVM_EXTERNAL_LIT="$pbc_scale/source/tpls/llvm/llvm/utils/lit/lit.py" \
  -DGIT_SUBMODULE=OFF -DCUDAQ_ENABLE_PROJECTS=cudaq \
  -DCUDAQ_ENABLE_PYTHON_BINDINGS=OFF -DCUDAQ_BUILD_TESTS=OFF \
  -DCUDAQ_DISABLE_CPP_FRONTEND=ON -DCMAKE_DISABLE_FIND_PACKAGE_Clang=ON \
  -DCUDAQ_ENABLE_ALL_BACKEND=OFF -DCUDAQ_ENABLE_REST=OFF \
  -DCUDAQ_ENABLE_ENSMALLEN=OFF -DCUDAQ_ENABLE_NLOPT=OFF \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF -DCUDAQ_SKIP_MPI=ON
cmake --build "$pbc_scale/release" --target cudaq-opt -j 32
```

The pinned installation includes the branch's MLIR SSI customization. System
MLIR is not substituted for it. `final-build-manifest.json`, the CMake cache,
compile commands and source patch retain the full build provenance.

The largest **successful CUDA-Q CI output** in the saved November 4 scheduled
run is `qasmbench:qasmbench_large_qft_n320_qft_n320`: 4,944,141 output gates,
1,975,647 T gates, and 578.28 seconds elapsed under the 600-second compile
limit. CI pipeline `78852944`, job `525761227`, records it as passed and
supported. The branch's existing stock result for that same source case has
239,671 gates; these compiler configurations have separate provenance. The
capacity run reuses that saved branch input and stock output.

The 20,538,805-gate Qiskit case,
`ftcircuitbench:qpe_qpe_h2_1_5_12q`, is also selected by CI's full profile,
with an empty skip list. Its CUDA-Q CI attempt failed with a compile timeout
after 606.03 seconds. The branch's stock-prefix retry sampled
`RotationCombine<RzOp>` commutation search before PBC normalization. The next
capacity target is therefore the successful QFT-320 CI case; the superseded
QPE profile is retained with an explicit stop record.

QFT-320 completed synthesis and final lowering in **803.89 seconds**, with
**20.18 GiB** peak resident memory. Its final circuit contains 16,044,539 gates;
the independent verifier counted the following native metrics before starting
its equivalence check:

| Circuit | Gates | T | Two-qubit | Depth | T-depth |
|---|---:|---:|---:|---:|---:|
| Saved stock CUDA-Q | **239,671** | **95,439** | **5,670** | **52,701** | **21,271** |
| Capacity-v3 PBC synthesis | 16,044,539 | **95,439** | 15,301,148 | 15,194,144 | 95,336 |

The independent QFT-320 attempts did not establish equivalence. Earlier
automatic and lookahead attempts were stopped to release benchmark resources;
the final native-QASM attempt returned `no_information` at its 600-second
checker limit. The circuit
remains **unverified**, with output SHA-256
`a612ec2593d551780d4211b217e2ef698d447c3629184aa908447cc983c1e7cd`.
These counts establish a quality
gap even though the capacity failure is resolved: the synthesized circuit has
about 67 times as many gates as the saved stock circuit. The normalized-source
entry point retains its own synthesized incumbents, rather than the incoming
stock network. Keeping the stock circuit as a verified incumbent before PBC
normalization would address this baseline-quality gap without invoking the
separate Clifford+T optimization pass. That architectural change requires its
own implementation and validation.

The optional joint route has further global admission limits. Both its planner
and completion service reject sources above 32,768 rotations, which excludes
this 95,439-rotation case before regional planning. Raising that cap alone
would still leave the planner's memory preflight: it reserves
`16 * min(max-dependencies, rotations * rotations / 2)` bytes for dependencies,
plus row storage. With this case and the capacity configuration, that term and
its row allowance total approximately **67.9 GiB**, against the configured
**128 MiB** planner allowance. The preflight uses a worst-case logical-edge
bound rather than the compressed index's actual stored conflicts. These are
remaining joint-route scaling gaps; successful outer synthesis currently
comes from the existing routes for this case.

The earlier QSVT-11 probe completed synthesis in 308.23 seconds and final
lowering in 1056.38 seconds. Its independent alternating checker returned
`no_information` at the 600-second checker limit, so it remains **unverified**.
The Fermi–Hubbard-128 probe emitted its internally checked candidate in 861.84
seconds, but subsequent final lowering reached the 3600-CPU-second process
limit. Samples identify repeated MLIR verification, greedy rewriting and
serialization as further costs of large materialized native words. Its
unfinished final circuit is not submitted for independent verification.

For a capacity diagnostic, reuse the successful normalized input and combine
synthesis and final lowering in one compiler invocation. This avoids writing
and reparsing the potentially very large intermediate wire circuit:

```bash
pbc_root=/localhome/thiennguyen/pbc-benchmark-20261106
pbc_scale="$pbc_root/scale-20261106"
pbc_opt="$pbc_scale/tools/cudaq-opt-final"
pbc_tools="$HOME/pbc-benchmark-tools"
mapfile -t pbc_synthesis_options < <(python3 - "$pbc_tools/synthesis.json" <<'PY'
import json
import sys
for key, value in json.load(open(sys.argv[1]))['options'].items():
    print(key + '=' + str(value))
PY
)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 QISKIT_NUM_PROCS=1
unset CUDAQ_PBC_TELEMETRY
timeout --kill-after=15s 3660s \
  prlimit --as=343597383680 --cpu=3600 --stack=67108864 --core=0 -- \
  "$pbc_opt" normalized.qke \
  "--pass-pipeline=builtin.module(func.func(quake-synthesize-pbc{semantics=release ${pbc_synthesis_options[*]}},regtomem),symbol-dce,jit-finalize-pipeline{lower-device-calls=true})" \
  --mlir-disable-threading --mlir-print-op-on-diagnostic=false -o final.qke
timeout --kill-after=15s 3660s \
  prlimit --as=343597383680 --cpu=3600 --stack=67108864 --core=0 -- \
  "$pbc_root/venv/bin/python" "$pbc_tools/verify.py" stock.qke final.qke \
  --contract measured --checker auto --timeout 600 --output verification.json
```

A proof-only lookahead attempt reuses the same stock and final circuit files,
changes `--checker auto` to `--checker lookahead`, and writes a separate
`verification-lookahead.json`. It keeps the same measured contract, 600-second
checker limit and outer process limits. Record both input hashes before
launching it. The parallel QFT-320 attempt reserves 320 GiB on the second host;
its frozen verifier, command, resources and proof are stored separately under
`qft320-proof-lookahead/`. Neither proof attempt reruns compilation.

Additional proof-only diagnostics use
`$pbc_tools/pbc-scale-verify-progress.py --native-qasm` with the same arguments.
This writes the exact `U; coherent-copy; U^-1` oracle as OpenQASM and gives the
files directly to MQT, avoiding Qiskit circuit conversion inside its loader.
Three measurement-contract tests passed for both input routes. Positive and
negative fixtures covering every supported gate gave the same checker outcomes,
and the native oracle's full small-circuit matrix matched the original Qiskit
construction. `--no-preprocessing` additionally disables QCEC's optional
operation reordering, SWAP reconstruction, one-qubit fusion and permutation
elision; that configuration also passed the three contract tests. Every attempt
records its own verifier hash, input format, oracle hashes, options and result.
These are verifier diagnostics; the synthesized circuit and its five metrics
are identical across the attempts.

The Fermi–Hubbard and QSVT probes reserve 320 GiB for their active compiler or
verifier. QFT-320 has a 30,593,319-gate mechanical prefix and uses a 120 GiB
compiler and verification slot, which fits alongside the existing 320 GiB
verification slot within the 440 GiB host budget. Admission also checks live available memory and retains at least
48 GiB for the host. Command records override the generic JSON's F.4 process
defaults with 3600 CPU / 3660 wall seconds and the recorded 120 or 320 GiB
address-space limit. Use this dedicated queue for the
capacity investigation; F.4's many-worker queue still reserves eight GiB per
ordinary synthesis task. A completed compiler output receives independent
verification before it can be labeled verified.

### F.6 Parallel continuation and capacity regression fix (2026-11-06)

The continuation retains all **626** previously collected CUDA-Q, Qiskit and
Clifford+T optimization metric sets and all **125** previous synthesis metric
sets. It collects missing synthesis entries from saved normalized inputs and
stock references. The synthesis pipeline does not run the network-preserving
optimization pass. Results and commands are recorded under
`build/pbc-capacity-continuation-2026-11-06/` and each host's
`/localhome/thiennguyen/pbc-benchmark-20261106/capacity-continuation-20261106/`.
Each entry records its own compiler, configuration and artifact hashes;
`capacity-v3` alone is not a compiler identity.

Spot checks found two quality regressions after raising the global gate cap.
Cap-only ablations reproduced both; raising only work or dependency limits did
not. Native refinement used the global ceiling as temporary search capacity,
admitting additional work before linear cleanup. Clifford-window admission
also estimated local memory from that ceiling and could reject small circuits
under its unchanged local memory bound.

The fix derives native, Four-wire, Clifford-window and algebraic continuation
limits from the actual native source size, retaining the established
262,144-gate local allowance for smaller sources and respecting a smaller
explicit global cap. Larger sources can still be admitted up to the global
200,000,000-gate capacity. The separate joint-completion storage bound remains
unchanged. These are resource controls; they do not add a preserve-network
optimization stage.

Both affected cases restore **all five** previous metrics:

| Case | Gates | T | Two-qubit | Depth | T-depth |
|---|---:|---:|---:|---:|---:|
| `feynman:gf2_32_mult` | 47,673 | 4,128 | 43,448 | 34,932 | 1,813 |
| `ftcircuitbench:hamiltonians_ising_1d_36q` | 176,045 | 38,808 | 76,246 | 147,896 | 32,760 |

The independent verification tool returned `identical gate sequence` for each
fixed capacity output against its reconciliation-cap output. This establishes
the regression comparison; it does not upgrade an unresolved stock-equivalence
result in the table. The fixed build passed **331 native tests** and all **12
PBC lit tests**. The two real-circuit checks and proofs are archived in
`regression-fixed-v2/` beneath the local continuation evidence directory.

The fixed Release compiler SHA-256 is
`6785d49faf1caffec2a2babcfaca0e5fee9a11fa336e374230a87916683fdca1`.
Its complete source patch relative to branch base
`247a051f0d471e65d793b26dd5cd271cedca28a0` is
`pbc-regression-fix-source-v2.patch`, SHA-256
`b29f52ee23d8b351796674158b34c6a96520be2d58087a1d33b696e5b607364c`.
Apply that patch to a fresh base checkout and use F.5's Release build command.
Earlier capacity results retain their earlier compiler identity; active jobs
were allowed to finish with their frozen executable.

Helpers remain outside the source tree in `$HOME/pbc-benchmark-tools/`.
The frozen queue controller and compiler are installed under each remote
continuation root:

```bash
pbc_root=/localhome/thiennguyen/pbc-benchmark-20261106
pbc_run="$pbc_root/capacity-continuation-20261106"
"$pbc_root/venv/bin/python" \
  "$pbc_run/tools/pbc-capacity-suite-6785d49faf1c.py" run --root "$pbc_run"
```

The controller reuses its recorded plan and completed results. It schedules
larger missing synthesis cases first, reserves memory for active jobs, and
admits work only within a **440 GiB combined reservation per host**, with at
least **48 GiB** of live available memory retained for the host. Reservations
range from 8 to 400 GiB according to circuit size. Compiler CPU limits are
3600 seconds for ordinary capacity jobs and 7200 for jobs reserving at least
128 GiB, with 60 additional wall seconds. Resumed final lowering of the
already synthesized Fermi–Hubbard-128 candidate has a recorded 14,400-second
CPU allowance. Each subprocess is single threaded and independently limited.

Synthesis and final lowering share one invocation using F.5's pipeline and
`--verify-each=false` to avoid repeated whole-module verification between
passes. Internal synthesis certificates remain enabled. The final emitted
artifact must then pass a separate compiler parser/verifier invocation:

```bash
"$pbc_run/compiler/cudaq-opt-6785d49faf1c" output.qke \
  '--pass-pipeline=builtin.module()' --emit-bytecode \
  --mlir-disable-threading -o /dev/null
"$pbc_root/venv/bin/python" "$pbc_run/tools/count.py" output.qke \
  --output metrics.json
"$pbc_root/venv/bin/python" "$pbc_run/tools/verify.py" stock.qke output.qke \
  --contract measured --checker auto --timeout 120 \
  --native-qasm --no-preprocessing --output verification.json
```

The controller supplies the recorded address-space and process limits to these
commands. Independent proof jobs have a 600-second CPU / 630-second wall
process allowance in addition to the 120-second checker limit. Metrics use the
same native gate, T, two-qubit, depth and T-depth definitions as F.4. The proof
uses the same arbitrary-input, ordered terminal-measurement contract through
the exact coherent-readout oracle. A timeout or `no_information` remains
explicitly unverified and is excluded from verified aggregates.

### F.7 Final snapshot after the requested stop (2026-11-06)

This is the frozen stop-time snapshot. The current F.2 table reports the
fresh 197-case synthesis rerun; the original snapshot and its hashes remain
preserved.

The run was stopped at the user's request at **05:29 UTC on November 6**.
Both remote queue controllers, active compiler and proof processes, and the
local automatic collector are stopped. Completed artifacts and commands remain
preserved on the benchmark machines; the final metadata snapshot is in
`build/pbc-capacity-continuation-2026-11-06/`.

At the stop, the table contained **197 synthesis metric sets**, including **72 new cases**
and all **125** previously collected synthesis results. All **626** previously
collected CUDA-Q, Qiskit and Clifford+T optimization metric sets are unchanged.
Of the synthesis outputs, **108 are independently verified** and **89 remain
unverified**. Five active proof jobs were interrupted by the requested stop;
their completed circuit metrics are retained with that explicit proof status.

Seven synthesis cases are marked **stopped**: one active compilation was
interrupted and six queued cases were not started. These are separate from the
eight recorded failures: seven upstream stock-prefix timeouts and one PBC
measurement-order validation failure. Four cases have unsupported control
flow. These outcomes account for all 216 cases without treating an interrupted
job as a completed measurement or a synthesis failure.

Each host's `user-stop.json` records the terminated process trees and affected
tasks. The local `user-stop-summary.json` records the final counts, matrix hash,
preservation checks and both stop records. The procedure and earlier queue
snapshots above remain reproduction records for the collected data.

## Appendix G. Hold-contract synthesis: fifteen key cases

Default `semantics=hold` reconstructs the original full unitary and scalar
phase with a newly synthesized network. Appendix F now reports the fresh
release and hold runs separately. This historical table uses the separate fifteen-case
frozen-input study, not the full-runtime corpus in Appendix F. The original ten
cases appear first. Hold uses the fixed `hold-v1-120s` configuration; release
retains the previous reconciliation configuration for these same inputs.

| Circuit | Release gates / T | Hold gates / T | Hold status |
|---|---:|---:|---|
| adder4 | 20 / 7 | 20 / 8 | verified |
| adder10 | 102 / 31 | 145 / 32 | verified |
| adder28 | 306 / 96 | 434 / 96 | verified |
| adder64 | 709 / 224 | 1031 / 224 | verified |
| adder118 | 1361 / 416 | 2216 / 416 | verified |
| adder433 | 6344 / 1536 | — (120 s limit) | 120 s limit |
| multiply13 | 73 / 32 | 91 / 32 | verified |
| rc_adder_6 | 151 / 47 | 163 / 47 | verified |
| csla_mux_3 | 146 / 62 | 177 / 62 | verified |
| qft_n002 | 7 / 3 | 8 / 3 | verified |
| mod5_4 | 18 / 7 | 29 / 8 | verified |
| basis_trotter_n4_transpiled | 14702 / 7108 | 17964 / 7190 | verified |
| basis_trotter_n4 | 14233 / 6886 | 17769 / 7136 | verified |
| basis_test_n4_transpiled | 651 / 310 | 650 / 310 | verified |
| grover_5 | 512 / 166 | 493 / 166 | verified |

All fourteen emitted hold networks passed independent unitary verification;
`adder433` produced no complete candidate before its 120-second limit. The
full metrics, exact-phase checks, fixed options, compiler hash and reproduction
commands are in [the hold synthesis results](pbc_synthesis_tuning.md#hold-contract-measurements-2026-11-06).
The pre/post-change release checks on adder4, QFT-2 and mod5_4 have identical
gate sequences, readout identities and five-metric vectors, and pass independent
ordered-measurement verification. The updated implementation passes 366 native
tests and all 14 PBC lit tests.

## Appendix H. Post-refactor release quality fix (2026-11-06)

Initial focused validation on **10.117.10.135**, after the hold-contract
refactor, checked seven native-input cases. Those spot checks preceded the
fresh 197-case rerun now shown in [the main F.2 table](#f2-measurements).
The current table uses only that fresh batch for synthesis; CUDA-Q, Qiskit and
Clifford+T optimization retain their collected measurements. The spot-check
results and frozen pre-fix matrix remain preserved for provenance.

The fix retains an explicitly imported native source across private release
normalization, verifies its binding, and uses it as an immutable incumbent.
`native-acceptance=all-metrics` requires gates, T, two-qubit gates, depth and
T-depth to stay within that source's costs, plus strict rank improvement.
Expected resource limits retain a checked complete candidate; proof mismatches
remain errors. The preserve-network optimization pass is not run. Shared
parity reconstruction, compressed dependency storage checks and bounded
Clifford cleanup address the diagnosed reconstruction and admission costs.
The global source cap replaces the hidden 32,768-rotation cutoffs; local
scratch and deadline limits remain effective.

### Verification and table selection

All seven native-input all-metrics outputs pass independent arbitrary-input,
ordered joint measurement checks. Adder uses the alternating checker; gf2_32_mult
and ising_1d_36q retain identical gate sequences and readouts. The four other
outputs initially returned `no_information` with ZX at 120 seconds, then passed
the same verifier with `--checker lookahead`, with simulation checking and
preprocessing still disabled. Both attempts and unchanged output hashes are
retained. These initial checks validate the implementation; the current
main table obtains its synthesis measurements from the fresh batch.

The initial Ising-100 G/T check reproduced the previous result: **58,006 gates,
28,000 T gates, 1,002 entanglers, depth 28,803 and T-depth 13,646**. Its gate word
and ordered readouts are identical to the previously independently verified
output. The current verifier confirms that identity, and the saved proof chain
binds both outputs and the stock reference by hash. The fresh strict ZX attempt
is inconclusive; the redundant strict lookahead retry was stopped after this
proof chain was established. The all-metrics alternative has 70,682 gates and
1,000 entanglers. The fresh rerun uniformly uses G/T acceptance. Adder reproduces every previous metric through both the
native-input and normalized-input entry points.

Normalized-input checks isolate reconstruction from the native quality floor.
Ising-64 emits 247,773 gates (previously 264,625); triangular Ising-36 emits
640,784 (previously 761,830). Their full metrics and proof outcomes are retained
in the evidence. These are still much larger than stock: sharing adjacent
gadgets alone does not recover a lost global Clifford network. Normalized input
cannot infer the original native incumbent. The native-input recipe below is
required to enforce the stock-quality floor.
Both normalized-only large-circuit checks remain independently unverified
after `no_information` at the saved checker limit.

Route diagnostics establish that the 38,808-rotation Ising source now enters
joint planning with 21,301,444 logical dependencies represented by 2,316 stored
class conflicts. Triangular Ising-36 also reaches planning, with 47,906,560
logical dependencies and 536,266 stored conflicts. Neither route completes a
better joint plan within its local allowance. Heisenberg-64 still exhausts
dependency construction under the planner's bounded resources and returns its
checked incumbent. These outcomes are distinct from the removed global cap.
The large-buffer cleanup path is additionally exercised by a bounded native
unit test; the native benchmark incumbents here fit the existing owned path.

### Reproduction and evidence

Release compiler SHA-256:
`d358d40656d2ba8c7177080c8e54a5203144de2c6d6ee93c48b348d62b2b7486`.
The build uses branch head `247a051f0d471e65d793b26dd5cd271cedca28a0`,
the saved uncommitted refactor overlay and the implementation overlay
SHA-256 `8648846064d24cea8b61ff93428faf906f0b4b9c7708b32f973d3d1a5adf6f63`.
It reuses the pinned LLVM installation and Release build procedure in F.5.
**375 native tests and all 15 PBC lit tests pass**, including the existing
hold tests and new arbitrary-input measurement-effect checks. Small regression
controls retain the previous Adder and G/T Ising-100 metric vectors.

Use the saved **pre-normalization** scalar-wire `pbc-input.qke`, which includes
the common stock prefix, as input. Do not first run `quake-to-pbc` in this mode:

```sh
OPTIONS='semantics=release native-input=true assume-measure-all=true
  native-acceptance=all-metrics objective=total-gates
  max-gates=200000000 max-dependencies=1000000000000
  max-work=100000000000000000 timeout-ms=300000
  synthesis-dependency-bytes=536870912 synthesis-max-steps=200000000
  synthesis-native-entanglers=cx,cz,swap synthesis-swap-cost=1
  synthesis-simplify=false synthesis-prune-unobserved=false
  synthesis-discrepancy-steps=22 synthesis-representation-steps=32
  synthesis-readout-candidates=8 synthesis-commuting-candidates=8
  synthesis-algebraic-candidates=4 synthesis-indexed-representation-steps=64
  synthesis-legacy-max-work=1000000000000000
  synthesis-legacy-portfolio-timeout-ms=180000
  synthesis-joint-phase-candidates=4
  synthesis-joint-phase-acceptance=gates-and-t
  synthesis-joint-phase-planner-work=3000000000
  synthesis-joint-phase-max-work=100000000000
  synthesis-joint-phase-timeout-ms=120000
  synthesis-joint-phase-wide-search=true'

cudaq-opt pbc-input.qke \
  --pass-pipeline="builtin.module(func.func(quake-synthesize-pbc{$OPTIONS},regtomem),symbol-dce,jit-finalize-pipeline{lower-device-calls=true})" \
  --verify-each=false --mlir-disable-threading \
  --mlir-print-op-on-diagnostic=false -o output.qke
cudaq-opt output.qke --pass-pipeline='builtin.module()' \
  --emit-bytecode --mlir-disable-threading -o /dev/null
python count.py output.qke --output metrics.json
python verify.py stock.qke output.qke --contract measured --checker auto \
  --timeout 120 --native-qasm --no-preprocessing --output verification.json
# If inconclusive, retain the first attempt and retry the unchanged output:
python verify.py stock.qke output.qke --contract measured --checker lookahead \
  --timeout 120 --native-qasm --no-preprocessing --output verification-lookahead.json
```

`assume-measure-all=true` adds terminal Z readouts only when the input has none,
matching the historical benchmark assumption. Explicit ordered readouts and
names remain intact. Native all-metrics admission also constrains joint
selection to all five metrics, regardless of the historical joint-policy
option above. Set `native-acceptance=gates-and-t` explicitly to permit that
tradeoff. A normalized-only reproduction instead uses
`quake-to-pbc='semantics=release assume-measure-all=true'`, verifies PBC, and
omits `native-input`, `assume-measure-all` and `native-acceptance` from synthesis.

Scripts remain outside the source tree. Remote artifacts are under
`/localhome/thiennguyen/pbc-benchmark-20261106/fix-post-refactor-20261106/`:
the immutable compiler is in `compiler/<sha256>/cudaq-opt`, and cases are in
`spots/d358d40656d2/<case-key>/<mode>/`. Local commands, hashes, test logs,
resource records and all independent checker attempts are preserved in
`build/pbc-post-refactor-fix-2026-11-06/`. Its `table-update.json` binds the
seven selected results to the frozen matrix and records the recomputed
aggregates. Workers use one math thread,
32 GiB address-space caps, at most eight concurrent jobs, 900/930-second
compiler CPU/wall caps and 600/630-second proof CPU/wall caps. Checker timeouts
and `no_information` are retained as unverified unless a separate saved check
proves that exact output. The stopped full-suite matrix hash remains
`2a09dc276228c8b495da37cc44aa8f689463ba228e2c250276f7ee3fae31be8f`.

## Changelog

- **2026-11-07:** Added the phase-exact three-wire equal-T hold rewrite and
  restarted all 197 previously measured synthesis cases on 10.117.10.135.
  The main table tracks only the current hold run; pending cells replace the
  previous hold values. CUDA-Q, Qiskit, Clifford+T and release data are retained.

- **2026-11-07:** Retrieved both completed 197-case synthesis batches from
  10.117.10.135 and updated every metric and verification outcome in the main
  table. Release has 197 outputs (162 verified); hold has 166 outputs (146
  verified) and 31 compilation failures. Preserved all prior comparison data
  and recomputed bold minima, including ties.

- **2026-11-06:** Added the integrated hold checkpoint implementation and fifteen-case fixed-profile measurements, including full metric tradeoffs and release regression checks.

- **2026-11-06:** Updated the main CI benchmark table with seven verified post-refactor synthesis results, recomputed aggregates and bold minima, and consolidated the separate spot-check table into recipe/proof notes.

- **2026-11-06:** Added the post-refactor release native-incumbent floor, shared parity reconstruction, compressed admission and bounded cleanup, with focused validation and a separate reproduction recipe.

- **2026-11-06:** Implemented default unitary/phase-preserving hold synthesis and added the fifteen-key-case comparison plus unchanged release spot checks.

- **2026-11-06:** Stopped the run at the user's request and froze 197 synthesis measurements, including 72 new cases, with verification and stopped-job statuses preserved.
- **2026-11-06:** Added missing synthesis measurements, documented the parallel capacity queue, and fixed gate-cap coupling that regressed two spot-checked synthesis outputs.
- **2026-11-06:** Documented rounded full-suite synthesis capacities, 64-bit dependency accounting, streamed cost and replay scaling, optional completion storage isolation, and the largest successful CI target.
- **2026-11-06:** Added the Release continuation, PBC synthesis metric columns, fixed final synthesis recipe, memory-bounded two-machine queue, and mandatory independent proof procedure. Labeled the native network-preserving path Clifford+T optimization.
- **2026-11-05:** Completed all 216 CI FTQC cases in Appendix F, with 143 verified CUDA-Q/Clifford+T optimization pairs and one-worker continuation details.
- **2026-11-04:** Extended Appendix F to 145/216 CI FTQC cases with 117 verified CUDA-Q/Clifford+T optimization pairs and reproduction details; corrected the default objective to `total-gates`.
