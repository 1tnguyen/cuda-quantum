# PBC synthesis tuning

This study tests whether normalized Pauli-based computation can be synthesized
competitively for a gate-based Clifford+T target. The production implementation
is in `cudaq/lib/Optimizer/Transforms/Pauli/GraphSynthesis.cpp`; its public
interface is `cudaq/Optimizer/Pauli/GraphSynthesis.h`. The new implementation
replaces the previous graph synthesizer.

The [final results for all fifteen cases](#final-results-for-all-fifteen-cases)
use one frozen compiler and one configuration, with independent verification
of every output. They include all five synthesis metrics and gate/T comparisons
with stock CUDA-Q and preserve-network PBC.

## Contract and scope

The existing measurements below describe **release** synthesis. Their
reproduction requires `semantics=release` on conversion and synthesis now that
the pass default is **hold**. Hold retains and reconstructs the full terminal
Clifford/phase obligation; its results are reported separately from these
historical release results and use the original pre-normalization inputs.

The subsequent release quality fix adds an explicit `native-input=true` mode
that binds an incoming native incumbent across private normalization and
enforces an immutable five-metric floor by default. Shared parity
reconstruction and compressed dependency admission also apply to normalized
sources. The historical measurements below retain their original recipes.
[Appendix H of the optimization report](pbc-optimization-report.md#appendix-h-post-refactor-release-quality-fix-2026-11-06)
records the new recipe, independently verified spot checks and remaining
normalized-only quality gaps. This mode requires `semantics=release`; the
default hold implementation and the isolated hold tuning results retain their
separate scope.

Release normalization absorbs the Clifford network into rotation axes and signed terminal
observables. Synthesis reconstructs a native circuit that preserves **ordered
joint terminal measurement probabilities for arbitrary input states**. It need
not preserve discarded output states or the full unitary. Physical readout wires,
classical result identities and register names remain fixed; no output-bit
permutation is silently assumed. The separate native `graph::synthesize` entry
point retains full-unitary semantics unless its caller explicitly selects release.

For a benchmark with no measurements, this study explicitly assumes terminal Z
measurements on every wire. `quake-to-pbc` exposes this assumption through
`assume-measure-all=true`; it is off by default and only adds measurements when
none exist. Added measurements are excluded from the reported gate counts.

The isolated experiment retains the synthesizer's small native Clifford action
dictionary. It disables graph preprocessing merges, unobserved-rotation deletion,
T-count decoding, the general circuit-cleanup portfolio, Pauli propagation, phase
placement and Clifford-window optimization. The dictionary uses the existing
small Clifford optimizer to construct actions. The measured synthesizer also
performs the signed Clifford-frame refinement described below as part of native
emission; this does not invoke the network-preserving portfolio.

## Cost model and emission

Support reduction alone does not price the native implementation. A candidate
can shorten a Pauli word while introducing enough basis changes or readout work
to increase the final circuit. The revised greedy score is

```
score(action) = mean_support_change
              + action_weight * native_cost(emitted_action)
              + readout_weight * (readout_cost_after - readout_cost_before)
              + optional_frame_penalty - optional_scheduling_credit
```

The defaults are `action_weight=0.5`, `readout_weight=0.125`, frontier weight 1,
and scheduling credit 0. At frontier weight 1, the support term equally weights
the mean over remaining nodes and the mean over ready nodes. Candidates must
still reduce the cheapest ready-node support cost. In measured synthesis, the
readout term costs the shared terminal reconstruction; full-unitary synthesis
instead uses the retained-frame penalty (default weight 0.1).

The emission profile is the target gate contract. The study profile matches
CUDA-Q and [synthesis off](pbc-optimization-report.md#appendix-b-measured-configuration-total-gates-v1):
named one-qubit Clifford+T gates, CX, CZ and SWAP are
primitive operations, each counted once. Decomposition occurs only when the
configured native basis excludes a gate, and scoring charges that decomposition.

Native action costs include their actual local Clifford gates. The dictionary
includes optimized symmetric Pauli-axis entanglers alongside asymmetric actions.
When a ready rotation has an X or Y axis, emission carries its basis prefix into
all remaining rotations and observables, emits a Z rotation, and retains the
changed frame. It does not immediately emit the inverse basis change. This
keeps the scoring model aligned with the emitted circuit.

Final candidates are independently replay checked and ranked by the requested
objective. The default is `(weighted gates, T, depth, two-qubit gates, T-depth)`,
with a separate T-nonincreasing guard. Unit primitive weights make the primary
metric total gates, as in this study. This is a heuristic search, not a proof of
optimal synthesis or a guarantee to recover the circuit that was normalized.
The default search uses deterministic tie seed 4, one greedy rollout, then one
alternative decision at up to the first 22 decision positions. All rollouts
share the work/deadline budget; an unfinished rollout cannot replace a verified
incumbent.

These choices build on the support/frontier, native-action, asymmetric-emission
and hold/release ideas in [PCOAST](https://arxiv.org/abs/2305.10966v2), Section V
and Figure 12. They are implementation choices for the gate-count target, not a
reproduction of the paper's unpublished presets. Moving Cliffords into terminal
MPP observables is useful only if their eventual hardware cost is accounted for:
this target reconstructs primitive readout gates rather than pricing MPP as free.

## Production interface and reproduction

The production MLIR entry point is **`quake-synthesize-pbc`**. It requires
normalized PBC and emits native gates plus terminal Z measurements. It does not
invoke the T decoder or circuit-cleanup portfolio. `quake-lower-pbc` remains the
mechanical lowering pass for comparison, not a second graph-synthesis backend.
`quake-try-pbc` also uses the new measured synthesizer when graph synthesis is
enabled and `preserve-clifford-network=false`. Its portfolio still applies the
outer objective and growth guards; the isolated study uses
`quake-synthesize-pbc` directly to measure the synthesis result.
If measured synthesis fails, including at a structural search limit,
`quake-try-pbc` leaves the original function unchanged. The strict
`quake-synthesize-pbc` pass reports failure when no verified completion is
available. Invalid input or configuration remains an error.

Set `CUDAQ_BUILD_DIR` to your configured Release build directory, then compile
the repository target:

```sh
cmake --build "$CUDAQ_BUILD_DIR" --target cudaq-opt -j 1
```

Use a Release build for timing. The executable below is built from repository
source through CMake.
Start with a scalar reference-form CUDA-Q baseline in `input.qke`.
[Appendix D of the optimization report](pbc-optimization-report.md#appendix-d-reproduce-the-ten-cases-with-the-compiled-pass)
describes preparation of the benchmark inputs.

```sh
CUDAQ_OPT="$CUDAQ_BUILD_DIR/bin/cudaq-opt"
"$CUDAQ_OPT" input.qke --convert-to-linear-values \
  --mlir-disable-threading -o prepared.qke
"$CUDAQ_OPT" prepared.qke \
  --quake-to-pbc='semantics=release assume-measure-all=true' --quake-verify-pbc \
  --mlir-disable-threading -o normalized.qke
prlimit --as=8589934592 --cpu=3600 --core=0 -- \
  "$CUDAQ_OPT" normalized.qke \
  --quake-synthesize-pbc='semantics=release objective=total-gates synthesis-native-entanglers=cx,cz,swap synthesis-swap-cost=1 synthesis-simplify=false synthesis-prune-unobserved=false synthesis-discrepancy-steps=22 max-work=1000000000000000 timeout-ms=0' \
  --mlir-disable-threading -o synthesized.qke
"$CUDAQ_OPT" synthesized.qke \
  --pass-pipeline='builtin.module(func.func(regtomem))' \
  --mlir-disable-threading -o synthesized-ref.qke
```

Set `synthesis-discrepancy-steps=0` for one greedy rollout. Count emitted native
gates in `synthesized-ref.qke`, excluding allocation, deallocation, global phase
and measurement. Record total gates, T/Tdg, two-qubit gates, ASAP depth, T-depth,
compiler time, compiler/source hashes, exact command and search limits. The
function attributes `pbc.synthesis_work` and `pbc.synthesis_budget_exhausted`
distinguish complete search from selection of a verified incumbent after a limit.
Use at most three concurrent jobs and an 8 GiB address-space limit per compiler.

| Public option | Default | Meaning |
|---|---:|---|
| `objective` | `total-gates` | Also accepts `t-count`, `t-depth`, `entanglers`, `depth` |
| `synthesis-native-entanglers` | `cx,cz,swap` | Allowed native two-qubit gates; CX or CZ is required |
| `synthesis-clifford-cost`, `synthesis-t-cost` | 1 each | One-qubit Clifford and T/Tdg costs |
| `synthesis-cx-cost`, `synthesis-cz-cost`, `synthesis-swap-cost` | 1 each | Native two-qubit costs |
| `synthesis-action-weight`, `synthesis-readout-weight` | 0.5, 0.125 | Local score coefficients |
| `synthesis-frontier-weight`, `synthesis-frame-weight` | 1, 0.1 | Ready-node and retained-frame weights |
| `synthesis-parallelization-credit` | 0 | Scheduling credit |
| `synthesis-lookahead`, `synthesis-candidate-nodes` | 0, 0 | Population/source-node caps; zero considers all |
| `synthesis-tie-seed`, `synthesis-discrepancy-steps` | 4, 22 | Deterministic ordering and bounded alternative search |
| `synthesis-simplify`, `synthesis-prune-unobserved` | true, false | Optional preprocessing and measurement-only deletion |
| `synthesis-asymmetric` | true | Include asymmetric entangling actions |
| `max-work`, `timeout-ms` | 100000000, 0 | Shared work and optional deadline |
| `max-gates`, `max-dependencies`, `synthesis-max-steps` | 32768, 1000000, 1000000 | Structural limits |

The one-qubit basis is fixed to H, S/Sdg, T/Tdg and Pauli gates. This interface
configures the two-qubit basis and primitive costs; it is not an arbitrary gate
set compiler. For a CX-only target, use `synthesis-native-entanglers=cx`: CZ and
SWAP expand to that basis, and the scorer charges the expanded actions. The
study uses native SWAP at unit cost, exactly as the eight-gate CUDA-Q and
synthesis-off QFT baselines do. Excluding SWAP makes its three-CX implementation
visible to scoring. Weight changes affect search and ranking,
not verification or the physical outcome contract.

The corresponding C++ API is:

```cpp
#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include <stdexcept>

cudaq::pauli::graph::Options profile;
profile.simplify = false; // isolate normalization and synthesis
cudaq::pauli::SearchBudget work(1000000000000000ULL);
auto result = cudaq::pauli::graph::synthesizeMeasured(
    width, normalized.rotations, normalized.measurements,
    physicalReadoutWires, profile, &work);
if (!result)
  throw std::runtime_error(result.error);
```

## Ten-case comparison

**Synthesis off** means the `total-gates-v1` configuration in
[Appendix B of the optimization report](pbc-optimization-report.md#appendix-b-measured-configuration-total-gates-v1),
with `graph-synthesis=false`, `gate-count=true`, and
`preserve-clifford-network=true`. Its numbers are the **PBC total** column in that
report's [Measured pass results](pbc-optimization-report.md#measured-pass-results);
[Appendix D](pbc-optimization-report.md#appendix-d-reproduce-the-ten-cases-with-the-compiled-pass)
provides reproduction instructions. It includes network-preserving optimizations,
whereas the synthesis column isolates normalization and synthesis. All columns
count native SWAP as one gate. Production results are collected with the public
pipeline above, with independent verification and recorded commands.

| Case | CUDA-Q | Qiskit | Synthesis off | Initial synthesis |
|---|---:|---:|---:|---:|
| adder_n4_transpiled | 24 | 25 | 22 | 23 |
| adder_n10_transpiled | 140 | 142 | 106 | 119 |
| adder_n28_transpiled | 418 | 424 | 316 | 392 |
| adder_n64_transpiled | 974 | 988 | 736 | 970 |
| adder_n118_transpiled | 1808 | 1834 | 1366 | 1885 |
| adder_n433_transpiled | 6673 | 6769 | 5041 | Interrupted |
| multiply_n13_transpiled | 105 | 95 | 91 | 73 |
| rc_adder_6 | 191 | 185 | 147 | 177 |
| csla_mux_3 | 161 | 157 | 156 | 148 |
| qft_n002 | 8 | 7 | 8 | 7 |

Qiskit's seven-gate QFT-2 reference includes final output interpretation `[1,0]`;
restoring the original physical readout order with a SWAP gives eight gates.
The synthesis result below retains a physical SWAP and the original readout order.

### Production output metrics

Nine cases completed and passed independent QCEC verification of the ordered
joint measurement distribution. The 433-qubit case was manually stopped after
approximately ten minutes; the configured one-hour CPU cap was not reached.
This was a user interruption, not a resource timeout. It produced no completed circuit, so no output metrics
are reported. Measurements are excluded from gates and depth.

| Case | Gates | T | Two-qubit | Depth | T-depth | Search rollouts (maximum) |
|---|---:|---:|---:|---:|---:|---:|
| adder_n4_transpiled | 23 | 8 | 10 | 14 | 4 | 23 |
| adder_n10_transpiled | 119 | 32 | 62 | 80 | 22 | 23 |
| adder_n28_transpiled | 392 | 96 | 200 | 171 | 44 | 23 |
| adder_n64_transpiled | 970 | 224 | 504 | 307 | 81 | 5 |
| adder_n118_transpiled | 1885 | 416 | 994 | 540 | 150 | 1 |
| adder_n433_transpiled | — | — | — | — | — | 1 |
| multiply_n13_transpiled | 73 | 32 | 29 | 19 | 6 | 23 |
| rc_adder_6 | 177 | 63 | 88 | 119 | 42 | 23 |
| csla_mux_3 | 148 | 64 | 70 | 60 | 20 | 23 |
| qft_n002 | 7 | 3 | 2 | 6 | 2 | 23 |

The search allows up to 23 rollouts on the small cases, five on adder_n64,
and one on adder_n118 and adder_n433. An alternative beyond the greedy run's
last decision is not attempted. These are bounded-search results, not an
exhaustive optimum or a uniform search-effort comparison. `rc_adder_6`,
`csla_mux_3` and `qft_n002` use the explicit measure-all convention.

### QFT-2: seven gates with the physical SWAP retained

The following sequences are in execution order, with `CX(c,t)` denoting control
and target. Both retain the final physical SWAP. The seven-gate sequence is
proved exactly below and checked against the compiler output.

| Step | CUDA-Q: 8 gates | Native-SWAP synthesis: 7 gates |
|---|---|---|
| 1 | H(0) | T(1) |
| 2 | T(1) | H(0) |
| 3 | CX(1,0) | T(0) |
| 4 | T†(0) | CX(0,1) |
| 5 | CX(1,0) | T†(1) |
| 6 | T(0) | H(1) |
| 7 | H(1) | SWAP(1,0) |
| 8 | SWAP(0,1) | — |

The CUDA-Q and synthesis-off circuits are the same eight-gate sequence for this
case. The seven-gate result belongs to **normalization plus synthesis**, not to
the synthesis-off configuration:

| Configuration | H | T/T† | CX | SWAP | Total |
|---|---:|---:|---:|---:|---:|
| CUDA-Q baseline | 2 | 3 | 2 | 1 | **8** |
| Synthesis off (`total-gates-v1`) | 2 | 3 | 2 | 1 | **8** |
| Normalization + synthesis | 2 | 3 | 1 | 1 | **7** |

The saving is one CX. SWAP is retained and counts as one gate in all three rows.

Use column state vectors in the ordered basis
`(|00>, |01>, |10>, |11>)`, where the bit labels are `|q1 q0>` and
`T = diag(1, exp(i*pi/4))`. Exact multiplication gives:

```text
                  [ 1   1   1   1 ]
U_CUDAQ = U_off =  [ 1   1  -1  -1 ] / 2
                  [ 1  -1   i  -i ]
                  [ 1  -1  -i   i ]

                  [ 1   1   1   1 ]
U_synthesis =     [ 1   1  -1  -1 ] / 2
                  [ 1  -1   i  -i ]
                  [-1   1   i  -i ]

U_synthesis = D * U_CUDAQ
D = diag(1, 1, 1, -1) = CZ
```

Only the output amplitude for `|11>` changes sign. There is no row permutation.
For every input density matrix `rho` and the same physical output label `b`,
let `P_b = |b><b|`. Since `D† * P_b * D = P_b`,

```text
Pr_synthesis(b) = Tr(P_b * U_synthesis * rho * U_synthesis†)
                = Tr(D† * P_b * D * U_CUDAQ * rho * U_CUDAQ†)
                = Tr(P_b * U_CUDAQ * rho * U_CUDAQ†)
                = Pr_CUDAQ(b).
```

Thus seven gates suffice for the declared terminal-measurement contract without
SWAP elision or bit reordering. The full unitaries differ, even up to a global
phase; this seven-gate circuit does not preserve an unmeasured or controlled
QFT. Adding CZ after it recovers `U_CUDAQ` exactly and costs an eighth gate.
The exact-arithmetic check uses rational coefficients in `Q(exp(i*pi/4))`,
confirms `U_CUDAQ† * U_CUDAQ = I`, `U_synthesis† * U_synthesis = I`,
and the displayed diagonal relation.

## Runtime performance: initial assessment

Synthesis runtime is a separate optimization target. Gate-count improvements do
not establish practical compilation cost, especially on the large adders. The
study records the synthesis invocation separately from normalization, structural
IR conversion and external QCEC verification; compiler rebuilding and QASM
frontend conversion are excluded. Use a Release build and record both elapsed
time and search limits. Comparing an unoptimized compiler to an optimized one,
or one rollout to a multi-rollout search, would misattribute overhead.

<!-- BEGIN SYNTHESIS TIMINGS -->
Both columns below use the same production Release executable. Synthesis off
uses the other report's exact `total-gates-v1` configuration; its ten emitted
metric tuples match the reference results. Times are complete compiler
invocations, including their internal verification. Synthesis normalization and
external QCEC checking are measured separately and excluded here. These are
single runs under the bounded concurrent workload, not a statistical latency
benchmark; small-case times include process startup and timer granularity.

| Case | Synthesis off (s) | Synthesis (s) | Normalization (s) |
|---|---:|---:|---:|
| adder_n4_transpiled | 0.066 | 0.066 | 0.009 |
| adder_n10_transpiled | 0.067 | 0.267 | 0.017 |
| adder_n28_transpiled | 0.116 | 4.252 | 0.017 |
| adder_n64_transpiled | 0.167 | 12.623 | 0.033 |
| adder_n118_transpiled | 0.320 | 64.342 | 0.033 |
| adder_n433_transpiled | 2.652 | Interrupted (~10 min) | 0.166 |
| multiply_n13_transpiled | 0.065 | 0.267 | 0.017 |
| rc_adder_6 | 0.117 | 0.370 | 0.017 |
| csla_mux_3 | 0.066 | 0.470 | 0.019 |
| qft_n002 | 0.066 | 0.065 | 0.017 |

For adder_n118, one greedy synthesis takes 64.342 s versus 0.320 s for synthesis
off. Thus neither the alternative-rollout count nor an unoptimized build alone
explains the large-case cost. The synthesis result also has more gates (1885
versus 1808 CUDA-Q, 1834 Qiskit and 1366 synthesis off), making this a clear
quality-and-runtime follow-up case. No corrective retuning is folded into this
comparison.
<!-- END SYNTHESIS TIMINGS -->

Code inspection identifies the following likely causes; these are hypotheses
for profiling, not measured percentages of runtime:

- **Dependency construction:** the rotation graph checks pairs of rotations for
  noncommutation, giving quadratic pair enumeration before greedy synthesis.
- **Candidate enumeration and scoring:** each entangling decision examines wire
  pairs and local Pauli bases from the cheapest ready nodes. With the default
  uncapped source-node and population settings, every candidate is scored
  against the remaining graph and frontier. This repeats after each decision.
- **Terminal readout costing:** the quality-oriented score reconstructs a shared
  readout network for each candidate. With n wires and m measured observables,
  reconstruction can emit O(nm) gates, each updating m Pauli words: O(nm²)
  conjugation updates, plus support-storage work. When all wires are measured,
  this can be roughly cubic in width **per scored candidate**. Normalization can
  make those words broad even when the original gate network is sparse.
- **Alternative rollouts:** up to 22 alternative decision positions repeat
  graph construction, greedy scoring and proof. A single greedy rollout must
  be timed separately to distinguish intrinsic reconstruction cost from search
  multiplicity.
- **Verification and representation overhead:** independent replay, dynamic
  support copies, and the native action dictionary's initial construction add
  cost. External QCEC time must be kept separate from compiler search time.

Synthesis off retains an existing native network and applies bounded rewrites;
it avoids rebuilding an entangling network from broad normalized Pauli words
and repeatedly estimating its terminal reconstruction. This difference explains
why a smaller emitted circuit can still be much more expensive to compile.

The follow-up should first measure candidate counts, support sizes, time in
readout scoring, dependency construction, replay and dictionary initialization
on the same Release executable. Compare one greedy rollout with the configured
alternative search, using identical inputs, native gates and measurement order.
Only then assess caching, incremental readout costs, candidate caps or search
reuse, reporting their effect on both circuit quality and runtime. No runtime
optimization study is claimed here.

## Validation and remaining tuning

The port passes **76 C++ tests and 12 PBC MLIR/FileCheck tests**, including
phase-sensitive unitary replay, joint measurement effects, dynamic supports,
native versus decomposed SWAP, configured CX/CZ emission, explicit measure-all,
invalid profiles, retaining a verified result at the work limit, and unchanged
optional-pass output at synthesis step, dependency and gate limits. These tests
exercise the production synthesis implementation used for the measurements.

Every accepted synthesis candidate receives signed-Pauli/frame replay. Independent
small-statevector tests compare joint outcome effects, including off-diagonal
entries, and check the CX/CZ/SWAP profiles. Benchmark validation additionally
compares coherent-copy oracles with QCEC. A checker timeout is inconclusive and
must not be reported as equivalence.

The improvements address cost/emission misalignment; they do not establish that
normalization plus resynthesis always beats network preservation. Adder quality
and large-graph runtime remain open tuning targets. Shared readout rescoring,
ready-node candidate enumeration, dependency construction and repeated rollouts
must be profiled before choosing further search approximations. Native MPP or
lattice-surgery costs would require a distinct emission profile rather than
reusing these primitive-gate counts.

## Signed Clifford-frame synthesis

The measured synthesizer now refines each completed graph rollout using an exact
search over the 24 single-qubit Clifford frames. It chooses T or Tdg and carries
the compensating S/Sdg into the residual frame, using `T = S Tdg`. Fixing the
rotation sign can force unnecessary basis changes even when the entangler network
is already good. The search uses backpointers and regions of at most 512 gates.
It changes neither the T count nor the entangler sequence within a rollout.

Internal regions preserve the full unitary up to global phase. After the last
entangler, a measured wire need only preserve its signed Z outcome effect; an
unmeasured wire has no output-state constraint. Physical readout identities stay
fixed. This uses the same terminal-observation freedom that benefits multiply_n13,
csla_mux_3 and QFT-2. It does not delete non-Clifford rotations or invoke the
network-preserving optimization portfolio.

The graph reconstruction is replay-checked before refinement. Each changed local
region is independently checked by exact SO(3) channel arithmetic over
`(a + b sqrt(2)) / 2^T`, using arbitrary-width integers. Final regions check the
required observable rows. The refinement and proof share the synthesis budget;
an unfinished refinement leaves the already verified candidate available. Rollout
ranking uses the refined emitted cost, with a guard against worsening any of the
five reported metrics relative to the unrefined winner among completed rollouts.

This approach is motivated by [Clifford+T normal forms](https://arxiv.org/abs/1312.6584).
The dynamic program minimizes its fixed-rotation-order representation family;
it is not a globally optimal Clifford+T synthesis algorithm. Joint optimization
of rotation and output parities, as in [PhasePoly](https://arxiv.org/abs/2506.20624),
remains a direction for reducing the entangler overhead in the larger adders.

### Fourteen-case comparison

Every run starts from the original scalar CUDA-Q circuit, then normalizes and
calls `quake-synthesize-pbc`. The synthesis-off, preserve-network output is only
a reference. The nine completed original cases are supplemented by the five
largest percentage gate-count gains among verified pairs in the frozen full-suite
snapshot. Adder_n433 is excluded because the original study did not complete it.

All fourteen outputs passed independent QCEC checks of ordered joint measurement
probabilities for arbitrary inputs. There are no regressions in gates, T count,
two-qubit count, depth or T-depth relative to the corresponding previous synthesis
results. The graph and PBC pass test suites also pass all 45 tests, including
exhaustive single-Clifford contexts, random partial-measurement effects, high wire
indices, opaque-angle boundaries, region limits and exhausted budgets.

| Case | CUDA-Q | Qiskit | Preserve network | Before refinement | With refinement |
|---|---:|---:|---:|---:|---:|
| adder_n4 | 24 | 25 | 22 | 23 | 23 |
| adder_n10 | 140 | 142 | 106 | 119 | 119 |
| adder_n28 | 418 | 424 | 316 | 392 | 392 |
| adder_n64 | 974 | 988 | 736 | 970 | 970 |
| adder_n118 | 1808 | 1834 | 1366 | 1885 | 1883 |
| multiply_n13 | 105 | 95 | 91 | 73 | 73 |
| rc_adder_6 | 191 | 185 | 147 | 177 | 177 |
| csla_mux_3 | 161 | 157 | 156 | 148 | 148 |
| qft_n002 | 8 | 7 | 8 | 7 | 7 |
| mod5_4 | 57 | 60 | 37 | 38 | 38 |
| basis_trotter_n4_transpiled | 20299 | 35930 | 14816 | 17990 | 15026 |
| basis_trotter_n4 | 19667 | 47122 | 14410 | 17759 | 14776 |
| basis_test_n4_transpiled | 885 | 1579 | 654 | 790 | 669 |
| grover_5 | 668 | 732 | 500 | 561 | 561 |

The two large Trotter cases improve by 16.5% and 16.8% over their previous
synthesis results and finish approximately 1.4% and 2.5% above preserve-network gate counts.
The smaller basis_test case improves by 15.3%. Adder_n118 remains a gate-count
regression against both CUDA-Q and Qiskit: 1,883 versus 1,808 and 1,834. Its 994
entanglers, versus 741 for CUDA-Q and 689 for preserve-network PBC, identify a
remaining reconstruction problem beyond local Clifford-frame selection.

| Case | Gates | T | Two-qubit | Depth | T-depth | Synthesis seconds |
|---|---:|---:|---:|---:|---:|---:|
| adder_n4 | 23 | 8 | 10 | 14 | 4 | 0.033 |
| adder_n10 | 119 | 32 | 62 | 80 | 22 | 0.215 |
| adder_n28 | 392 | 96 | 200 | 171 | 44 | 3.125 |
| adder_n64 | 970 | 224 | 504 | 307 | 81 | 9.799 |
| adder_n118 | 1883 | 416 | 994 | 540 | 150 | 47.833 |
| multiply_n13 | 73 | 32 | 29 | 19 | 6 | 0.165 |
| rc_adder_6 | 177 | 63 | 88 | 119 | 42 | 0.316 |
| csla_mux_3 | 148 | 64 | 70 | 60 | 20 | 0.366 |
| qft_n002 | 7 | 3 | 2 | 6 | 2 | 0.033 |
| mod5_4 | 38 | 16 | 19 | 24 | 8 | 0.065 |
| basis_trotter_n4_transpiled | 15026 | 7190 | 384 | 7799 | 3685 | 45.164 † |
| basis_trotter_n4 | 14776 | 7136 | 321 | 6750 | 3214 | 45.122 † |
| basis_test_n4_transpiled | 669 | 314 | 27 | 461 | 216 | 0.215 |
| grover_5 | 561 | 178 | 246 | 427 | 136 | 0.917 |

Lower total gates do not imply a win on every metric against the external
baselines. For example, the two large Trotter outputs use 384 and 321 entanglers
versus Qiskit’s 351 and 303; basis_test has T-depth 216 versus CUDA-Q’s 188.

### Reproduction and limits

Artifacts are in `build/pbc-signed-frame-synthesis-2026-11-05/`: `run.py`,
`manifest.json`, `results.json`, frozen inputs, the source snapshot, exact build commands,
and each case’s input, normalized IR, synthesized output, command log and QCEC
record. `cudaq-opt` is a separate production executable linked from the existing
CMake Release build with the changed synthesis translation unit compiled using
the same `-O3` Release flags. The pre-existing compilers remain unchanged.

The run uses one worker, an 8 GiB address-space limit and a 240-second compiler
CPU cap. Native CX, CZ and SWAP cost one each; preprocessing simplification and
unobserved-rotation deletion remain disabled. The original nine retain their
previous discrepancy budgets: 22 for the small cases, 4 for adder_n64 and 0 for
adder_n118, with no internal deadline and `max-work=1000000000000000`. The new
five use 22, `max-gates=262144`, `max-dependencies=64000000`,
`max-work=1000000000000` and `timeout-ms=45000`. Entries marked † exhausted the
search deadline and returned verified incumbents. These are bounded-search
results, not exhaustive optima. Times are single-run synthesis wall times.

The report’s original production commands also reproduce the new behavior with
a compiler built from these sources; no additional pass or cleanup command is
needed. The benchmark driver rejects an existing case directory to preserve
evidence. Reproduce the fourteen cases in a fresh output directory with:

```sh
PBC_SYNTHESIS_OUTPUT=/tmp/pbc-synthesis-repeat \
  python3 build/pbc-signed-frame-synthesis-2026-11-05/run.py
```

## Synthesis architecture review and proposed design

The proposed replacement is a **bounded search over equivalent Pauli
representations and compatible region implementations**. The optimization unit
is a region with a declared quantum boundary, not an individual support-reducing
entangler. Each solver returns verified alternatives with explicit boundary
frames and native costs. A common planner composes those alternatives and keeps
a verified incumbent. The design is supported by the experiments below. The
initial contract and cost adapters have been validated; the complete replacement
pipeline has not yet met the preserve-network benchmark target.

### Evidence and architectural requirements

The adder diagnosis in `build/pbc-adder-structure-2026-11-05/` establishes three
separate limitations. First, Clifford placement changes the reconstruction
problem even at identical T count: one Pauli-angle/frame choice per four-bit
carry block accounts for a substantial part of the gap. Second, the current
minimum-ready-support rule excludes some useful multi-step paths. Third,
averaging support improvement over the entire population changes local action
rankings when unrelated circuits are added. Four independent Adder-10 copies
produce 521 gates, versus 480 from four independent greedy runs. Dividing the
weights by circuit size does not resolve the problem: it worsens Adder-118 to
1,963 gates.

The arithmetic intuition is compute/uncompute cancellation. A carry's quantum
information must remain available across adjacent blocks, but there is freedom
in the Clifford frame used to represent it. Choosing that frame poorly spreads
later Pauli supports and reconstructs work that could cancel. The correct
boundary retains both X and Z information, including signs; treating the carry
as a classical bit or an initialized ancilla would violate arbitrary-input
semantics. A narrow source interaction graph suggests decomposition, but does
not prove a narrow semantic separator after normalization.

There are also distinct non-adder workloads. The following inventory was
computed from all fifteen original inputs, including Adder-433. Counts are for
the normalized original rotation word, before new simplification or pruning.

| Workload | Rotations | Distinct unsigned axes | Ordered noncommuting pairs | Implication |
|---|---:|---:|---:|---|
| Adder-118 | 416 | 416 | 2,064 | Sparse arithmetic dependencies; frame and boundary quality dominate |
| Adder-433 | 1,536 | 1,536 | 7,664 | Scaling test for decomposition and completion latency |
| multiply_n13 | 32 | 32 | 0 | A commuting-region backend is applicable |
| mod5_4 | 16 | 12 | 0 | Commuting phase identities and measurement completion |
| basis_trotter_n4_transpiled | 7,190 | 178 | 11,730,830 | Long word over a small Pauli alphabet; avoid explicit pairwise DAG storage |
| basis_trotter_n4 | 7,136 | 183 | 12,336,902 | Same representation opportunity |
| basis_test_n4_transpiled | 314 | 24 | 17,703 | Small-width exact/local search and repeated frames |
| grover_5 | 178 | 83 | 5,196 | Mixed phase identities and reconstruction; support alone is unreliable |

Some reference improvements include T reduction that the isolated synthesis
experiment deliberately disabled. For example, preserve-network PBC has 47 T
gates versus 63 in the original rc_adder_6, 7 versus 16 in mod5_4, and 166 versus
178 in grover_5. The architecture therefore needs both a fixed-word
reconstruction track and a full optimization track. It must not attribute a
phase-decoding gain to a routing change, or expect fixed-T synthesis to reproduce
all the reference's transformations.

### Mathematical problem and semantic boundaries

For a normalized word `W = R(P_m,theta_m) ... R(P_1,theta_1)` and ordered signed
commuting observables `M_j`, define

```
Pi_b(M) = product_j (I + (-1)^b_j M_j) / 2
E_b     = W† Pi_b(M) W.
```

The measured synthesis problem is to find a native circuit V minimizing the
declared native cost, subject to

```
V† Pi_b(Z_readout) V = E_b          for every ordered outcome b.
```

This is operator equality, so it covers arbitrary inputs, including inputs
entangled with an external system. The physical wire and classical identity of
each outcome are fixed. An internal region must instead preserve its full
quantum transformation on all live outputs unless a separately proved weaker
contract applies. No implicit output permutation, outcome XOR, zero-input
assumption, or ancilla allocation is allowed.

At a terminal boundary, many circuits have the same measurement effects: a
left factor commuting with every output projector is unobservable. The
optimizer should search within this equivalence class rather than restore an
unnecessary full Clifford. QFT-2's seven-gate implementation is the existing
example. At an internal boundary that freedom generally disappears. This
distinction preserves the reason synthesis can already beat network
preservation on multiply_n13, csla_mux_3 and QFT-2.

Use `R(P,theta) = exp(-i theta P/2)`. The exact identity

```
R(P,-pi/4) = R(P,-pi/2) R(P,+pi/4)
```

makes the sign of a T rotation a representation choice with a compensating
Clifford, not an immutable routing requirement. Transporting the compensation
changes later axes and terminal observables. It preserves semantics but can
change dependencies between the transformed suffix and the earlier word;
dependency caches must be rebuilt or invalidated accordingly. Common Clifford
conjugation of the entire residual word, in contrast, preserves its
noncommutation relation.

Formally, let a search state contain the residual rotation obligations, their
ordering constraints, signed boundary frame, terminal obligations, deferred
native tail and any schedule state needed by the objective. Legal transitions
are certified rewrites or native implementations. For the selected finite
transition family, the cost-to-go obeys

```
J(s) = min_a [ native_cost(a) + J(next(s,a)) ].
```

This is a shortest-path/optimal-control formulation on an implicit state graph.
The [Pauli frame graph](https://arxiv.org/abs/2103.08602) supplies a related
view of rotations encountered along a Clifford-frame walk. Here exact
noncommuting order, sign choices and the measured terminal condition must also
be part of the state. A gate need not reduce support immediately to belong to
a useful path. Search budgets, state deduplication and an incumbent replace the
current minimum-support rule as termination mechanisms; this does not imply
enumerating the full Clifford group.

For a region with logical unitary U, a candidate native implementation G has
the boundary relation

```
G F_in = F_out U,
```

with scalar-phase handling declared by its semantic mode. Adjacent candidates
compose when their boundary descriptions agree. A boundary description includes
the signed images of live X/Z generators, deferred local Clifford frames,
physical wire mapping, pending rotation obligations and fixed output identities.
It is not just a support set or a gate count. Quantum wires can be omitted only
with a proof that their remaining obligations are discharged.

For a chain of regions with finite candidate boundary states, let
`K_r(a,b)` be the cost of a verified implementation from boundary state a to b.
Then

```
D_(r+1)(b) = min_a [ D_r(a) + K_r(a,b) ].
```

This is min-plus dynamic programming, equivalently shortest path through a
layered graph. It is exact for the enumerated states and candidates, not for all
quantum circuits. Tree-shaped decompositions use separator messages; general
regions use bounded search or larger overlapping repairs. This follows the
structural lesson of [bucket elimination](https://ics.uci.edu/~csp/r48b.pdf):
complexity depends on the size of the interacting boundary variables. A b-wire
Clifford boundary can still have exponentially many states in b squared. Small
graph separators alone do not make unrestricted frame enumeration practical.

Native counts are additive only for explicitly emitted gates. Deferred basis
changes and possible cross-boundary cancellation must be represented in the
boundary state, or conservatively charged and later re-evaluated. Depth and
T-depth require per-wire arrival profiles; they cannot be added as scalars.
Dominance pruning is valid only between candidates with the same semantic
boundary and compatible future cost state. A cheaper prefix with a different
frame cannot dominate a more expensive prefix merely by gate count.

### Modules and independent implementation contracts

The public MLIR passes remain adapters. Search modules use plain C++ values and
immutable problem descriptions. They do not mutate MLIR, perform hidden target
lowering, read benchmark references, or own a second global objective.

| Module | Input and output contract | Optimization responsibility and independent acceptance test |
|---|---|---|
| Semantic core and proof checker | Signed symplectic Paulis, exact angles, full-unitary or ordered-measurement contracts; check a proposed rewrite or boundary relation | Correctness only. Test sign-sensitive channels, composition, partial measurement, external entanglement, high wire indices, and rejected outcome remapping. Optimizers cannot self-authorize equivalence. |
| Problem index and dependency oracle | Immutable rotation IDs, axis classes and exact ordering; return ready nodes and affected support incidences | Reduce graph construction, memory and update costs without changing legal schedules. Differential-check against all earlier anticommuting rotations. |
| Representation explorer | A region and boundary obligations; return a bounded set of equivalent words, compensating frames and rewrite certificates | Explore Pauli sign/Clifford placement, exact merges and separately enabled decoding. Report structural estimates, never claim they are native costs. Always retain the identity transformation. |
| Region and separator planner | Rotation/support incidence, observables and live quantum obligations; return exact independent components or regions with explicit interfaces | Bound local solver size and preserve sharing. Test independent copies, wire renaming, and adversarial cross-boundary phases. Arithmetic blocks are discovered structurally, never by case name. |
| Commuting-region solver | A proved commuting rotation set, entry frame and output obligations; return native implementations and exit frames | Jointly optimize phase/parity construction and boundary restoration. Compare small regions against exhaustive or exact solutions; charge basis changes and all native gates. |
| General region solver | A bounded noncommuting region, legal actions and boundary contracts; return verified candidates with actual costs | Beam/best-first search with transposition reuse, including bounded non-monotone macros. Match the legacy solver as an available candidate; demonstrate useful alternatives beyond cheapest-ready progress. |
| Clifford and measurement completion | Signed tableau or ordered commuting observables, fixed readout wires and target profile; return feasible completions and optional certified bounds | Optimize the exact required residual transformation. Offer elimination orders and small exact methods; never silently use classical XOR/permutation. Benchmark separately from prefix synthesis. |
| Native emission and local refinement | Candidate operations plus deferred boundary tail and target profile; return native gates, five metrics and local proof | Centralize gate costs, basis lowering and the existing 24-frame DP. Count native SWAP consistently; test decomposed profiles. A local improvement may not alter the declared region interface. |
| Global planner and budget manager | Candidate frontiers, boundary relations, objective, memory/work/deadline limits and verified incumbent | Select compatible implementations, allocate bounded search, reserve completion/proof work and commit only verified admissible results. Preserve quality as more optional candidates are added when budgets do not evict the incumbent. |
| Experiment and telemetry layer | Frozen original inputs, module configuration, compiler hash and seeds; return reproducible metrics and proof status | Attribute time and gains by ablation, report failures/timeouts, and enforce regression rules. It must not affect search decisions. |

The first implementation should share these value types:

```cpp
struct RegionProblem;       // immutable word, contract, target, obligation IDs
struct BoundaryState;      // signed live frame, deferred tail, outcome mapping
struct VerifiedCandidate;  // native body, entry/exit states, cost, proof handle
struct CostVector;         // gates, T, two-qubit, depth, T-depth
struct CostEstimate {      // deliberately distinct meanings
  Optional<CertifiedLowerBound> lower;
  Optional<VerifiedCompletion> upper;
  HeuristicPriority priority;
};

// Conceptual interface; this is not a declaration added to the production API.
CandidateFrontier solve(const RegionProblem &, const BoundaryState &,
                        const ExitContract &, BudgetSlice &, CandidateSink &);
```

Candidate publication is transactional: incomplete output is never a candidate.
`Unsupported`, `ExhaustedWithIncumbent`, `ExhaustedWithoutCompletion`, and
`InvalidProof` are different outcomes. A solver timeout is not a proof of
infeasibility. The candidate sink validates the proof and measures native output
using the common target profile before accepting it.

A cache key includes the semantic contract, exact angles/signs, entry/exit
frames, wire mapping, target profile and algorithm version. Use separate keys
for semantic equivalence and the chosen representation: equal semantics do not
imply equal heuristic costs or equal remaining search state. Reuse across wire
renaming requires a certified relabeling of both gates and output identities.

### Algorithms to adopt, adapt or defer

**Joint parity and frame optimization.**
[PhasePoly](https://arxiv.org/html/2506.20624v1) explicitly combines phase and
output parity matrices, searches multiple completions, and represents sharing
across blocks. Adapt that joint objective to commuting regions. General X/Y/Z
regions require signed symplectic data and exact noncommuting dependencies;
the phase-polynomial algorithm is not a drop-in backend for them. Its
memory-bounded search is useful precedent, but the paper's Hamming-weight and
Gaussian-elimination estimates do not supply an admissible lower bound for our
general measured problem.

**Lazy frame materialization.**
[Lazy synthesis](https://arxiv.org/html/2012.09663v4) separates an efficiently
updated representation from partial native extraction. Adopt the separation:
carry a frame until an operation or boundary needs it, and expose its eventual
cost to the planner. The current asymmetric emission and local signed-frame DP
already implement limited forms of this idea. Extend their scope through the
region interface rather than add another unrelated cleanup pass.

**Sparse elimination and locality.** Treat Clifford reconstruction like a
sparse elimination problem: a cheap pivot can introduce expensive fill into
future rows. Minimum-degree/order proposals, separator-aware pivots and shared
parity reuse are candidate generators, not correctness rules. This provides a
useful explanation for carry support spreading and for output-frame costs
increasing during rotation synthesis. Joint extraction must still evaluate the
actual native circuit.

**Large-neighborhood search.** Once a feasible synthesis exists, freeze its
outside boundary and replace a region, then enlarge or overlap regions if local
progress stops. This adapts the repair approach used in
[constraint-programming/local-search methods](https://research.google/pubs/solving-vehicle-routing-problems-using-constraint-programming-and-metaheuristics/).
It can preserve good synthesized structure while allowing multi-gate changes
that greedy single-step progress forbids. The incumbent here is generated from
the original normalized input, not imported from preserve-network output.

**Exact local solvers.** Use SAT, bounded enumeration or meet-in-the-middle for
small Clifford/frame regions, both to improve candidates and to measure the
heuristics' gap. The corrected version of
[SAT Clifford synthesis](https://arxiv.org/abs/2208.11713v2) explicitly withdraws
the earlier large-instance scalability claim and reports practical experiments
around six qubits. Do not budget an exact whole-adder solve. A recent
[Clifford synthesis study](https://arxiv.org/abs/2503.14660v3) likewise separates
small optimal, intermediate A*, and large greedy methods. An exact solver must
use our native gate weights and boundary contract; CNOT-optimal does not mean
total-gate-optimal.

**Equality saturation and ZX as optional representation providers.**
[Equality saturation](https://arxiv.org/abs/2004.03082) offers a way to retain
equivalent expressions without committing to rewrite order. Initially use a
bounded candidate graph with proof edges; a global e-graph is unnecessary until
the rewrite language and boundary-aware extraction costs are established.
[ZX simplification](https://arxiv.org/abs/1902.03178) can expose identities across
obstructing Clifford structure, but extraction has its own correctness and cost
obligations. Either can later implement the representation-provider contract.
Neither should become a second unchecked compiler hidden inside the planner.

**Objective-specific methods.** Parity-network synthesis has proven hard
restricted cases ([Amy, Azimzadeh and Mosca](https://arxiv.org/abs/1712.01859));
global optimality is not a reasonable default runtime promise. Conversely,
[matroid partitioning](https://arxiv.org/abs/1303.2042) provides useful structure
for T-depth optimization in its applicable circuit class. Keep that as an
objective-specific scheduling backend. It does not solve the current total-gate
and residual-Clifford objective by itself.

### Cost, search and runtime design

The exact objective remains `(G,T,D,E,TD)` for total gates, with unit primitive
weights in this study and T nonincrease relative to the original input.
Alternative target weights belong to the emission profile. Search priorities
may be approximate, but acceptance uses actual emitted costs. The five-metric
regression constraint against the current verified synthesis is an additional
study gate, not a claim that lexicographic optimization inherently preserves
every metric.

Three cost concepts must remain distinct:

- A **feasible completion** is an upper bound for its exact residual problem.
  The current readout algorithm supplies one such completion, not the minimum
  readout cost.
- A **certified lower bound** may prune a branch against an incumbent. Zero is
  a valid weak bound; stronger bounds must state their proof and applicable
  contract. Summing support costs double-counts shared entanglers. Remaining
  T count is not an unconditional lower bound if merging, decoding or terminal
  deletion is still allowed.
- A **heuristic estimate** orders work and shortlists actions. It never proves
  infeasibility or justifies correctness-sensitive pruning. Beam truncation is
  explicitly a bounded-search approximation, not an optimality certificate.

Use exact local cost plus residual estimates in common native-cost units; avoid
global population averages mixed with fixed action penalties. Independent
components must be separable, so adding an unrelated component does not alter
another's local ranking. Equal effort per component or cached repeated regions
is required when testing this property; a fixed total deadline alone does not
promise equal search effort.

Full readout reconstruction should happen for completed states or a small
shortlist, not every primitive candidate. Cache by the exact signed residual
contract and profile. Changes to a Clifford action touch only some support rows,
so use wire-to-axis incidence and rollback/persistent updates. A local physical
action can nevertheless affect many broad rows; locality is a measured property,
not an assumed constant-time guarantee. Keep exact cost deltas separate from
approximate completion estimates.

Avoid materializing the dense occurrence DAG when a small axis alphabet is
available. For an unconsumed rotation at original index i, readiness is exactly

```
for every axis A anticommuting with P_i:
    earliest_unconsumed_occurrence(A) >= i.
```

Store occurrence lists and an axis-conflict relation. With m occurrences, K
distinct axes and A stored directed conflicts, this representation needs
`O(m+K+A)` entries rather than one edge for every noncommuting occurrence pair.
Conflict discovery can still require `O(K^2)` Pauli comparisons. Dense bitsets
or sparse adjacency should be chosen from measured density. Common residual
Clifford changes retain the original axis-class dependency index; algebraic
rewrites invalidate the affected relation. Efficient ready-frontier notification
is a separate implementation task; the representation alone is not an
end-to-end speedup claim.

Search should reuse prefixes and transpositions rather than restart every
alternative rollout. Start with a fast verified completion generated from the
normalized input, then improve it with region solvers and repairs. Reserve work
for completion and proof before launching optional exploration. Keep successful
proofs with the incumbent so budget exhaustion cannot make it unusable. A
mechanical normalized-PBC completion may be expensive in gates; its purpose is
to give bounded search a valid result before quality work, not to satisfy the
preserve-network target by itself.

Instrument exclusive time for normalization, dependency indexing, representation
search, candidate generation, heuristic scoring, actual completion, local
refinement and proof, together with peak memory, cache hits and states explored.
Existing telemetry has inclusive nested totals; those cannot simply be summed
as a pass-time breakdown. Expose time to first verified completion and the
best-cost-versus-work curve. The deterministic work budget is for reproducible
search; wall time and memory are independent operational limits.

### Experimental checks of the design

The architecture study uses frozen original inputs and the existing Release
production compiler. No production synthesis defaults were changed. Artifacts
are in `build/pbc-synthesis-architecture-2026-11-05/`, including `study.py`,
`run_probe.py`, `dependency_probe.py`, input hashes, command logs, generated IR
and independent verification records.

The automatic representation probe considers every T-axis sign flip and
transports its exact compensating Clifford. It accepts a move only when both
the rotation-support sum and observable-support sum do not increase and at least
one decreases; it ranks accepted moves by observable reduction, then rotation
reduction. This is a deliberately simple surrogate with no case-name tests or
reference signs. All changed words are checked against the original by exact
positive-gauge normalization. It is a candidate generator, not a gate-count
guarantee.

On Adder-10/28/64/118 it independently discovers 1/3/7/13 changes and reaches
the same support sums as the reference-guided diagnosis. Adder-118 selection took
0.212 seconds in the first Python probe; subsequent native synthesis took 24.546
seconds in its single Release invocation. These are separate single-run times,
not a statistical speedup measurement. Independent QCEC confirms the resulting
1,691-gate circuit preserves the original ordered joint measurements.

The controlled comparison below uses **one greedy rollout** for both control
and frame probe. The current-synthesis column is the earlier verified portfolio
with its documented, different rollout budgets. All fourteen control/probe
native outputs used in the comparison have independent QCEC verification;
unchanged frame probes serve as their own controls. Adder controls reuse the
previous instrumented production runs with the same synthesis algorithm.

| Case | Preserve network | Current synthesis | Greedy control | Automatic frame probe |
|---|---:|---:|---:|---:|
| adder_n4 | 22 | 23 | 23 | 23 |
| adder_n10 | 106 | 119 | 120 | 120 |
| adder_n28 | 316 | 392 | 405 | 371 |
| adder_n64 | 736 | 970 | 970 | 893 |
| adder_n118 | 1366 | 1883 | 1883 | 1691 |
| adder_n433 | 5041 | Interrupted | — | — |
| multiply_n13 | 91 | 73 | 73 | 73 |
| rc_adder_6 | 147 | 177 | 177 | 174 |
| csla_mux_3 | 156 | 148 | 149 | 149 |
| qft_n002 | 8 | 7 | 7 | 7 |
| mod5_4 | 37 | 38 | 39 | 36 |
| basis_trotter_n4_transpiled | 14816 | 15026 | 15028 | 15046 |
| basis_trotter_n4 | 14410 | 14776 | 14824 | 14797 |
| basis_test_n4_transpiled | 654 | 669 | 672 | 669 |
| grover_5 | 500 | 561 | 567 | 595 |

The negative results are important. Reduced support worsens total gates on the
transpiled Trotter and Grover cases. Adder-28's depth/T-depth become 217/65,
versus 171/44 in the current portfolio; Adder-64's T-depth becomes 92 versus 81.
Thus unconditional frame preprocessing is rejected. Retaining verified
alternatives would admit the Adder-118 and mod5_4 improvements and the
basis_test depth improvement without worsening the current five-metric tuples;
it would retain the current candidates elsewhere. This is offline candidate
selection, not an integrated production result. mod5_4's 36 gates beat the
37-gate reference, but its 16 T gates remain above that reference's 7.

The dependency-oracle prototype checks the exact readiness condition against a
direct earlier-rotation scan on sampled consumed sets for **all fifteen cases**.
Every check passes. For the transpiled Trotter case, the representation stores
7,190 occurrence entries and 15,864 directed axis-conflict entries, versus
11,730,830 explicit occurrence dependencies. For the other large Trotter it
stores 7,136 and 16,648 versus 12,336,902. These are representation-entry counts,
not measured C++ memory savings or compilation speedups. Adder-433 received
structural/oracle checks only; its previously interrupted native synthesis was not rerun in
this architecture study.

### Modular implementation validation

The implementation separates dependency indexing, representation changes,
regional synthesis, fixed-wire readout, native refinement and checked candidate
selection. The proof boundary composes source-bound algebraic certificates,
measured-word replay and local-channel proofs. Internal regions retain their
full quantum exit and scalar phase; complete candidates retain ordered joint
measurement effects for arbitrary inputs. No preserve-network circuit enters
synthesis.

The current Release validation passes **350 unit tests**, including exact
representation-search differentials, four-wire quantum-map checks and the
independent-route regression fixture (`reconciliation-unit-validation/`). The initial
contract adapter reproduced fourteen completed control outputs byte-for-byte;
the dependency-index integration retained them after excluding only the changed
work counter. These controls are frozen in `contracts/` and `modules/` under
`build/pbc-synthesis-redesign-2026-11-05/`. A single-run dependency-index ablation
reduced whole-process peak RSS from 100,624 to 45,840 KiB and from 103,600 to
44,804 KiB on the two Trotter inputs. These are resource observations under
identical fixed-work greedy settings, not repeated speedup measurements.

<!-- reconciliation-results:start -->
### Final results for all fifteen cases

The historical release validation uses one frozen Release compiler and identical pass options
for all fifteen cases. The added hold column uses the separately recorded
`hold-v1-120s` configuration below, on the same original inputs. It compares both total gates and T count with the
preserve-network reference and with the independently verified results
recorded before reconciliation.
**Release: all 15 cases are independently verified; 12 meet both preserve targets,
and all 15 meet the preserve T-count target.**

| Circuit | Stock CUDA-Q gates / T | Preserve gates / T | Release gates / T | Hold gates / T | Release gate gap to preserve |
|---|---:|---:|---:|---:|---:|
| adder4 | 24 / 8 | 22 / 8 | 20 / 7 | 20 / 8 | -2 |
| adder10 | 140 / 32 | 106 / 32 | 102 / 31 | 145 / 32 | -4 |
| adder28 | 418 / 96 | 316 / 96 | 306 / 96 | 434 / 96 | -10 |
| adder64 | 974 / 224 | 736 / 224 | 709 / 224 | 1031 / 224 | -27 |
| adder118 | 1808 / 416 | 1366 / 416 | 1361 / 416 | 2216 / 416 | -5 |
| adder433 | 6673 / 1536 | 5041 / 1536 | 6344 / 1536 | — (120 s limit) | +1303 |
| multiply13 | 105 / 32 | 91 / 32 | 73 / 32 | 91 / 32 | -18 |
| rc_adder_6 | 191 / 63 | 147 / 47 | 151 / 47 | 163 / 47 | +4 |
| csla_mux_3 | 161 / 64 | 156 / 62 | 146 / 62 | 177 / 62 | -10 |
| qft_n002 | 8 / 3 | 8 / 3 | 7 / 3 | 8 / 3 | -1 |
| mod5_4 | 57 / 16 | 37 / 7 | 18 / 7 | 29 / 8 | -19 |
| basis_trotter_n4_transpiled | 20299 / 7190 | 14816 / 7108 | 14702 / 7108 | 17964 / 7190 | -114 |
| basis_trotter_n4 | 19667 / 7136 | 14410 / 6886 | 14233 / 6886 | 17769 / 7136 | -177 |
| basis_test_n4_transpiled | 885 / 314 | 654 / 310 | 651 / 310 | 650 / 310 | -3 |
| grover_5 | 668 / 178 | 500 / 166 | 512 / 166 | 493 / 166 | +12 |

The hold column has 14 independently verified outputs and one synthesis limit;
see [the hold evidence](#hold-contract-measurements-2026-11-06). Positive release
gate gaps remain above the preserve-network target. The stock
and preserve columns are the saved paired references for these exact inputs.

| Circuit | Gates | T | Two-qubit | Depth | T-depth |
|---|---:|---:|---:|---:|---:|
| adder4 | 20 | 7 | 10 | 12 | 3 |
| adder10 | 102 | 31 | 52 | 85 | 27 |
| adder28 | 306 | 96 | 146 | 181 | 60 |
| adder64 | 709 | 224 | 336 | 378 | 125 |
| adder118 | 1361 | 416 | 655 | 670 | 220 |
| adder433 | 6344 | 1536 | 3467 | 2595 | 850 |
| multiply13 | 73 | 32 | 29 | 19 | 6 |
| rc_adder_6 | 151 | 47 | 78 | 92 | 33 |
| csla_mux_3 | 146 | 62 | 69 | 57 | 20 |
| qft_n002 | 7 | 3 | 2 | 5 | 2 |
| mod5_4 | 18 | 7 | 9 | 10 | 3 |
| basis_trotter_n4_transpiled | 14702 | 7108 | 383 | 7619 | 3660 |
| basis_trotter_n4 | 14233 | 6886 | 321 | 6485 | 3098 |
| basis_test_n4_transpiled | 651 | 310 | 27 | 398 | 188 |
| grover_5 | 512 | 166 | 233 | 380 | 116 |

All fifteen outputs retain or improve both primary metrics of the
previously verified best-results snapshot.

Compiler, source/input/output hashes, exact commands and independent proofs:
`reconciliation-full/`. The fixed configuration is recorded in
`reconciliation-config.json`; the prior targets are in `reconciliation-targets.json`.
<!-- reconciliation-results:end -->

An earlier `basis_trotter_n4_transpiled` comparison rejected 14,880 gates, 7,114 T, 384
two-qubit gates, depth 7,734 and T-depth 3,660 against a historical
14,944/7,190/383/7,614/3,618 result. Repeating the frozen earlier compiler
with identical source/input hashes and options also passes QCEC but gives
14,704/7,108/384/7,623/3,660 (`general-repeat-new1/`). This establishes
deadline-sensitive candidate completion rather than a regression
attributable solely to the new modules. Native cleanup now completes
before optional algebra exploration; later candidates retain the existing
five-metric guard against that completed incumbent. It addresses late
cleanup starvation without claiming to recover every historical metric.

A subsequent integrated Native→Four completion fixes the `csla_mux_3` T-count
gap. The intermediate native rewrite has 147 gates, 62 T and T-depth 21, so the
earlier guard discarded it. Completing its four-wire rewrite first yields
**146 gates, 62 T, 69 two-qubit gates, depth 57 and T-depth 20**, improving all
five metrics against the 148/64/70/60/20 result. The actual compiler output
passes QCEC (`native-four-recheck/`) and all 264 core tests pass. This appended
route retains the earlier complete result and composes source-bound native
and linear certificates before final selection. The current table validates this route together with commuting-Clifford
completion and the joint profiles under one frozen compiler.

Independent refinement routes retain immutable inputs and compare completed
circuits. Chaining primitive-frame refinement into generic refinement lost the
parent of an Adder-64 improvement; retaining both routes fixes that composition
error. The compact eight-gate regression fixture and full trace are in
`diagnostics/frame-route-fixture/` and `diagnostics/adder64-whole-trace/`.
Adder-433 also requires retaining a completed primitive-Pauli candidate across
provider changes (`diagnostics/signed-frame-regions/`). Both corrections are
covered by the integrated tests and current equivalence checks.

The initial fixed-word and algebraic runs both verified all fifteen cases and
met five preserve gate targets (`portfolio-fixed/`, `portfolio-algebra/`).
Algebraic reconstruction alone improved mod5_4 from 35 to 23 gates; subsequent
refinement of the generated native circuit supplies the further improvements
above. Local refinement probes on Adder-10, rc_adder_6 and Grover found no
preserve-target candidate hidden by the final metric guards. The remaining
quality gap requires better generated candidates.

Four-wire linear refinement uses a finite, profile-bound dictionary. The
indexed representation route finds all 48 useful Adder-433 sign changes in
32.13 million charged work units. Its completion bounds repeated support
scoring to 64 rows and four generating frontier nodes, while readiness and
proof checks still cover every source rotation and ordered observable. The
route is selected by `synthesis-indexed-representation-steps=64` and retains
the earlier `synthesis-representation-steps=32` route independently.

The isolated ready-axis solver and boundary planner pass 29 tests, including
full quantum effects, scalar phases, external references, fixed readout
identities and bounded-search retention. The solver explores ready rotations,
pivots and eight native macro families; the planner transports actual Clifford
exits through later rotations and ordered observables. These modules remain
outside the benchmark compiler because completed quality probes do not improve
its results. The original bounded beam produces 149, 478, 215 and 688 gates on
Adder-10, Adder-28, rc_adder_6 and Grover. Expanded macro proposals produce
149, 501, 267 and 804 gates, respectively; their corresponding T counts are
32, 96, 47 and 178. Both probes use internal composed proofs, not independent
benchmark acceptance (`drafts/ready-axis/quality-runs/20261105T054453.766426Z/`
and `20261105T061409.520523Z/`).

A separate commuting-Clifford window probe starts from the verified synthesized
circuits, preserving exact scalar phase and source binding. With subsequent
native refinement, it reduces Adder-28/64/118/433 and Grover from
374/911/1646/7869/514 to **364/897/1642/7845/512 gates**, with no increase in
any of the five metrics. Adder-10 and rc_adder_6 remain unchanged. These are
internally certified diagnostic results; the module is now integrated, and all fifteen actual compiler outputs pass
independent QCEC as recorded in the current table. The fifteen local tests and the seven-case
comparison are recorded under `drafts/clifford-windows/`.

Before integration, the joint-phase solver was validated against the finite
rank-three parity-demand state space and exact Clifford exits. Its contextual planner and analytical
cost adapters pass **40 tests**. Natural-support wire layouts and scoring
against the remaining computation produce four candidates meeting both
preserve targets and a further Adder-433 improvement. Independent ordered-measurement QCEC verifies each frozen
artifact (`joint-draft-qcec/`).

| Circuit | Gates | T | Two-qubit | Depth | T-depth | Preserve gates | Preserve T |
|---|---:|---:|---:|---:|---:|---:|---:|
| adder10 | 102 | 31 | 52 | 85 | 27 | 106 | 32 |
| adder28 | 306 | 96 | 146 | 181 | 60 | 316 | 96 |
| adder64 | 709 | 224 | 336 | 378 | 125 | 736 | 224 |
| adder118 | 1361 | 416 | 655 | 670 | 220 | 1366 | 416 |
| adder433 | 6546 | 1536 | 3669 | 2602 | 850 | 5041 | 1536 |

Depth and T-depth tradeoffs require explicit gates-and-T admission rather
than the existing five-metric production guard. These are the earlier independently verified draft measurements. The table
above contains separately verified outputs from the integrated compiler. The small outputs are unchanged from their earlier
QCEC-verified artifacts; exact program and input identity is recorded in
`drafts/joint-ready-phase/quality-runs/20261105T072956.538696Z/`.

Scaling requires complete ready-span retention, cached row costs and bounded
readout estimates where exact scoring exceeds its callback allowance.
Phase-specific memory reservation retains the 128 MiB planner limit while
allowing a completed greedy route before beam allocation. Completed plans
survive optional scoring exhaustion as unverified proposals; caller-side
measured proof remains mandatory. With three billion work units per planner
route, the larger probe completes Adder-64/118, rc_adder_6 and Grover
(`20261105T073116.507554Z/`). The latter two produce 169/47 and 669/166 gates/T,
respectively, and are rejected against the integrated results. Adder-433's wider profile uses ten billion planner work units and 16,384
callback/expansion limits, yielding **6,546 gates/1,536 T**, independently
verified by QCEC (`drafts/joint-wide/quality-runs/20261105T073634.776536Z/`).
Increasing planner storage from 128 to 256 MiB permits more beam completion
but leaves the best program unchanged (`drafts/joint-wide-memory/quality-runs/20261105T074343.283048Z/`).
The draft closes four additional preserve-target gaps; Adder-433 remains
1,505 gates above its target, rc_adder_6 four gates, and Grover twelve gates.
Those measurements used separate profiles. The reconciled compiler now retains
or improves every earlier primary result with one fixed configuration across
all fifteen cases. It further reduces Adder-433 to **6,344 gates/1,536 T**,
leaving a 1,303-gate gap to preserve-network. The remaining gaps are four gates
for rc_adder_6 and twelve for Grover; all fifteen meet the preserve T target.

The reconciled implementation uses a fixed complete-circuit portfolio: the
existing synthesis pipeline followed by two independent joint-phase profiles.
Both profiles run the original and configured indexed source alternatives;
all cases use the same profile order and limits. Each profile owns its scoring
cache. `CompletionSession::score` freezes the first checked heuristic rank;
`complete` can retry optional refinement without changing that planner feedback.
Every checked raw, sink, native and four-wire result reaches immutable-anchor
admission before local continuation selection. Joint-phase admission explicitly
allows secondary-metric tradeoffs while guarding both gates and T. Earlier
five-metric guards remain unchanged. This is a portfolio reconciliation, not a
single planner that interchanges all solvers region by region; the implementation
map and remaining architectural boundaries are documented in
[pbc_synthesis_architecture.md](pbc_synthesis_architecture.md).

All reconciliation cases use **one worker and an 8 GiB address-space limit**.
The shared search deadline is **300 seconds**: the existing portfolio receives
at most 180 seconds, followed by at most 120 seconds for the two joint profiles
(60 seconds each). The existing child reserves the joint work allowance when
the parent can fund both. This is a larger runtime allowance than the preceding
180-second comparisons and is not a like-for-like speed claim. The discrepancy
limit is now **22 for every input**; earlier case-specific limits are not reused.
The existing algorithm's internal indexed continuation retains its documented
greedy settings uniformly across inputs.

The common configuration retains `max-work=1000000000000`,
`synthesis-legacy-max-work=100000000000`, `max-gates=262144`,
`max-dependencies=64000000`, 32 representation steps, eight readout candidates,
eight regional candidates, four algebraic candidates and 64 indexed steps.
It adds four joint-phase plans, `synthesis-joint-phase-acceptance=gates-and-t`,
three billion planner work units for the first profile, the wider ten-billion
profile, and a combined 100-billion joint work allowance. Each planner has
128 MiB storage; the completion service separately bounds owned and temporary
storage within 2 GiB. The wider 256 MiB planner probe did not improve the retained
result, so the fixed configuration uses the smaller tested bound.

The subsequent full-suite capacity configuration is documented in
[the optimization report, F.5](pbc-optimization-report.md#f5-suite-capacity-investigation-2026-11-06).
It raises the outer structural and replay-work limits while retaining these
search strategies and deadlines. Optional joint completion keeps its own
262,144-gate and 2 GiB storage bounds, and native refinement keeps its bounded
temporary allowance. Local native, Four-wire, Clifford and algebraic limits
grow with the actual native source size rather than the global reconstruction
ceiling. The [capacity regression fix](pbc-optimization-report.md#f6-parallel-continuation-and-capacity-regression-fix-2026-11-06)
records the two exact-output spot checks and the compiler used for subsequent
collection. The earlier reconciliation measurements retain the
configuration recorded with them.

The exact options are in `reconciliation-config.json`; the recorded invocation
is `reconciliation-command.json`, both under
`build/pbc-synthesis-redesign-2026-11-05/`. To reproduce a frozen run, use its
`compiler/cudaq-opt`, pass `--source-run` with that run directory, select a fresh
`--output`, and supply the configuration's `option_string`. The runner limits
compiler processes to 360 CPU seconds and 375 wall seconds and independently
checks outputs with QCEC using one thread and a 120-second proof limit.
`--comparison gates-and-t --primary-targets reconciliation-targets.json` stops
on an execution/proof failure or failure to retain either previously verified
primary metric. These target values constrain acceptance only; neither target
circuits nor preserve-network circuits enter synthesis.

`benchmark/run.py` records compiler/source/input/output hashes, commands,
environments, timings, peak RSS and ordered-output QCEC proofs. Its default
`all-metrics` comparison remains strict; explicit `gates-and-t` reports
secondary-only tradeoffs as `verified_metric_tradeoff`. Thirteen policy tests
cover stopping decisions, preserve targets and verified-baseline overlays.
The historical uniform-compiler results remain in `clifford-full/`, with their
then-recorded per-case discrepancy settings. The current table draws only from the completed `reconciliation-full/` run.
All fifteen actual compiler outputs pass independent QCEC, with zero regressions
against the earlier best gate/T counts. Measured synthesis times range from
0.12 to 181.15 seconds; whole-case peak RSS is 188,508 KiB. These are single-run
resource observations, not repeated speedup measurements.

### Implementation sequence and research acceptance gates

1. **Contracts and adapters.** Introduce the shared problem, boundary, candidate
   and proof interfaces. Wrap the existing synthesizer and readout emitter
   unchanged. Require identical baseline results under identical work settings;
   extract cost/emission code without changing policy. This provides a stable
   integration point for independently implemented modules.
2. **Exact dependency and incremental state.** Replace pairwise occurrence-DAG
   storage where beneficial, and add exclusive counters. Differential-check
   ready sets and emitted circuits; measure the Trotter memory/time effect and
   do not accept quality loss from a representation-only change.
3. **Automatic Pauli-frame alternatives.** Port the proved sign/frame rewrite
   and the bounded candidate provider. Keep the unmodified representation.
   Reproduce the reference-free Adder-118 and mod5_4 results, and reject the
   measured Grover/Trotter regressions through common candidate selection.
4. **Joint regional solvers and completion.** Implement the commuting backend
   and interchangeable signed readout completion first. Add a general region
   search supporting multi-step actions. Use QFT-2 and multiply_n13 to ensure
   terminal freedom is retained; use exact small regions to test solver gaps.
5. **Boundary planning and repair.** Start with chain DP over a bounded set of
   frame states, then overlapping-region repair. Verify discovered separators,
   including live carry X/Z obligations. The critical adder test is whether
   improvements transfer from Adder-10/28 to 64/118/433 without new weights or
   case-specific rules. Record boundary-state growth and fallback frequency.
6. **Algebraic optimization integration.** Adapt existing merge/decoder
   identities to return proof-carrying representation alternatives. Evaluate
   fixed-word synthesis and full optimization separately; select decoded words
   by reconstructed native cost, not T count alone.

Each iteration freezes a hypothesis, changes one module, runs its local proof
tests, and compares the same fifteen-case manifest with the previous candidate
set. Report all five native metrics, time to first completion, total synthesis
and proof time, memory, search work, and exact termination status. Use one worker,
an 8 GiB process limit and an optimized Release binary. Include short and longer
fixed-work/latency budgets to distinguish a better search from more search.
The 433-qubit case remains a required completion/scaling test, not an exclusion.

The target is `G_new <= G_preserve` on every case, while retaining the existing
verified five-metric incumbent whenever a proposed candidate regresses it.
Meeting both constraints simultaneously is an empirical goal, not yet proved
feasible. A trade-off candidate can be reported as a Pareto alternative but
cannot be silently accepted under the no-regression policy. Any verification
failure stops promotion; timeout or unknown verification is reported explicitly.

Use Adder-10/28 for arithmetic development and 64/118/433 for size transfer.
Use the known synthesis wins to test whether terminal-measurement advantages
survive the refactor. Add wire-renamed and independent-copy variants, plus
synthetic frame-equivalent inputs, to detect dependence on numbering, circuit
replication or representation. These fifteen already studied cases are a
development/regression suite, not an unseen generalization benchmark; freeze
additional untouched cases before claiming general superiority.

The standalone synthesis-quality track starts from original normalized PBC.
Preserve-network output remains an external target, never a seed or fallback
that can be counted as a synthesis success. Original-source provenance may
optionally propose region boundaries, but a metadata-free normalized-input
ablation must remain available. The existing production portfolio may retain
its separately reported preserve-network fallback without changing this study's
definition of success.

### Reuse and ownership boundaries

Retain `Circuit.h/.cpp` for Pauli/angle values; `PBC.h/.cpp` for normalization
and observable validation; `SearchBudget.h` and `Telemetry.h` for shared
accounting; and the public `GraphSynthesis.h` entry points as compatibility
adapters. Extract the current private frame, conjugation, target-cost and
readout logic into the corresponding modules rather than duplicate it.
`SignedCliffordFrames.h` becomes the local-emission solver, not the global frame
selector. Its exact channel checker remains a useful independent proof path.

Reuse the algebra and witnesses in `TCountMerge.cpp`, `TCountDecode.cpp` and
`TCountVerification.cpp` through adapters; their existing native-circuit and
scalar-phase assumptions must be translated explicitly. `GateCount.cpp`
contains useful local Clifford dictionaries and phase-placement ideas, but
feeding its preserve-network output into standalone synthesis would invalidate
the stated comparison. `Optimizer.cpp` already supplies portfolio and incumbent
concepts; migrate toward one common selection/budget layer instead of adding a
second nested policy system.

For independent implementation, freeze the value/proof/cost interfaces first.
Then dependency indexing, representation exploration, commuting synthesis,
general synthesis, completion, and benchmarking can be separate ownership
areas with disjoint implementation files and shared contract fixtures. Boundary
planning consumes their candidate interfaces. Changes to semantics or the target
profile require a coordinated interface revision; local heuristic experiments
do not. This lets a module improve its subproblem without changing the meaning
of another module's result or weakening global verification.

## Hold contract measurements (2026-11-06)

The `hold-v1-120s` run uses the fifteen original pre-normalization inputs from
`build/pbc-synthesis-redesign-2026-11-05/benchmark/`. It rebuilds native circuits
under the full-unitary contract, including scalar phase and wire identities.
The release figures in the comparison above retain their historical frozen
configuration; the hold run has its own fixed configuration and source hashes.
It does not use a preserve-network output as a synthesis seed or fallback.

**Fourteen of fifteen hold circuits were emitted and independently verified.**
`adder433` exhausted the 120-second synthesis allowance before a complete
candidate existed. `adder118` retained a previously verified incumbent when its
allowance expired. These outcomes are not replaced by original-circuit metrics.

| Circuit | Gates | T | Two-qubit | Depth | T-depth | Hold verification |
|---|---:|---:|---:|---:|---:|---|
| adder4 | 20 | 8 | 8 | 14 | 5 | verified |
| adder10 | 145 | 32 | 74 | 77 | 16 | verified |
| adder28 | 434 | 96 | 215 | 144 | 29 | verified |
| adder64 | 1031 | 224 | 495 | 282 | 62 | verified |
| adder118 | 2216 | 416 | 958 | 555 | 134 | verified |
| adder433 | — | — | — | — | — | 120 s synthesis limit; no complete output |
| multiply13 | 91 | 32 | 39 | 28 | 7 | verified |
| rc_adder_6 | 163 | 47 | 79 | 103 | 34 | verified |
| csla_mux_3 | 177 | 62 | 83 | 67 | 18 | verified |
| qft_n002 | 8 | 3 | 3 | 7 | 2 | verified |
| mod5_4 | 29 | 8 | 17 | 20 | 5 | verified |
| basis_trotter_n4_transpiled | 17964 | 7190 | 368 | 9042 | 3590 | verified |
| basis_trotter_n4 | 17769 | 7136 | 327 | 7801 | 3091 | verified |
| basis_test_n4_transpiled | 650 | 310 | 28 | 397 | 188 | verified |
| grover_5 | 493 | 166 | 228 | 356 | 104 | verified |

All emitted gate portions passed MQT QCEC 3.11.0 unitary equivalence checks,
with simulation checking disabled. The compiler's phase-sensitive source-bound
proofs preserve the stronger exact scalar contract. Independent dense matrices
including actual scalar phase were additionally compared for all six inputs
of at most eight qubits; the largest entrywise error was `2.31e-13` at the
unchanged `1e-10` tolerance. The proof-only retries for the two long Trotter
circuits corrected naive floating-point accumulation of thousands of scalar
angles to `math.fsum`. Both original attempts are retained, and all twenty
saved hold/release outputs have unchanged hashes across those retries.

The release regression spot checks ran the pre-change and post-change compilers
on `adder4`, `qft_n002`, and `mod5_4`, with the same release options and prepared
inputs. All six outputs passed the independent ordered joint-measurement
coherent-copy check. Each pair has identical native gate words, physical
readouts and all five metrics:

| Release spot check | Gates | T | Two-qubit | Depth | T-depth |
|---|---:|---:|---:|---:|---:|
| adder4, before and after | 20 | 7 | 10 | 12 | 3 |
| qft_n002, before and after | 7 | 3 | 2 | 5 | 2 |
| mod5_4, before and after | 18 | 7 | 9 | 10 | 3 |

The build passed **366 native tests and all 14 PBC lit tests**, including the
existing release tests with explicit `semantics=release`.

### Hold reproduction and evidence

Compiler SHA-256: `290bb75d1644a5252f240bf3aa26ab50b50f67bfa36e86fdf34de83139d8ae3e`.
Configuration: one worker, 8 GiB address-space cap, native CX/CZ/SWAP with unit
costs, total-gates objective, `max-gates=262144`,
`max-dependencies=64000000`, `max-work=1000000000000`,
`timeout-ms=120000`, 22 discrepancy decisions, 8 commuting candidates,
4 algebraic candidates, 4 joint-phase plans, 3B first-profile planner work,
100B aggregate joint work, a 120-second joint allowance and the wider profile.
Preprocessing simplification and unobserved pruning are disabled. Readout and
observable representation providers are release-only; the spot checks enable
8 readout candidates, 32 representation steps and 64 indexed steps.

Only compiler-generated `quake.evince` implicit-output annotations are removed
before scalar-wire conversion, as in the existing benchmark workflow. Original
circuits and explicit measurements are retained and hashed. Release inserts
terminal Z measurements only for sources without measurements; hold adds none.
Counts come from the emitted native circuit and exclude scalar phases,
measurements, allocations and sinks.

The tracked runner is [benchmark_contracts.py](../utils/pbc/benchmark_contracts.py).
The complete options, commands, input/output hashes, independent proofs and
retry history are in
`build/pbc-hold-2026-11-06/benchmark-final/results.json`.
`compiler-manifest.json`, `tests.json`, and `lit.log` in the parent directory
identify the build and checks; `runner.executed.py` retains the initial runner.

```sh
PYTHONPATH=/tmp/pbc-hold-python python3 utils/pbc/benchmark_contracts.py \
  --compiler build/pbc-hold-2026-11-06/cudaq-opt-hold \
  --before-compiler build/pbc-hold-2026-11-06/cudaq-opt-before \
  --snapshot build/pbc-synthesis-redesign-2026-11-05/benchmark \
  --output build/pbc-hold-2026-11-06/benchmark-repeat
```

The isolated Python directory contains Qiskit 2.6.0 and MQT QCEC 3.11.0.
`--cases adder4 adder10 ...` selects a smaller subset; `--verify-only` retries
unresolved proofs on saved outputs without compilation or resynthesis.

## Isolated hold-only architecture tuning (2026-11-06)

This separate study targets the original ten circuits with study-proposal data:
adder4, adder10, adder28, adder64, adder118, adder433, multiply13,
rc_adder_6, csla_mux_3 and qft_n002. Its target is a strictly smaller native gate
count than both recorded stock CUDA-Q and Qiskit counts, while preserving the
full unitary, scalar phase and output interface. T count, entanglers, depth and
T-depth remain visible. Release or state-preparation equivalence is ineligible.

All tuning code, compiler variants, immutable source snapshots, generated
circuits and proof artifacts live under `/tmp/pbc-hold-tuning-20261106-ob5kvhcm`. Only this tuning document is
updated in the source tree, which is undergoing an independent refactor.
The initial 46 production source files are reconstructed from the earlier
compiler's source patch and checked against its frozen source manifest.

The investigation starts with architectural ablations: feasible completion and
budget scheduling before global discrepancy search; quantum-frame cost in
partial-plan ranking; and gate/T admission of phase-exact refinements instead
of rejecting useful gate reductions on secondary metrics. Reconstructed PBC
words may be refined, but a preserved original/native optimizer output is not
used as a substitute synthesis result. Every reported improvement requires an
actual emitted circuit and independent unitary verification.

The saved Qiskit QFT-2 result records a swapped final layout. Its seven counted
gates omit the physical output swap that the existing fixed-wire hold result
contains. The mapping and feasibility of a strict improvement target are
audited below before treating this reference as directly comparable.

### Exact QFT-2 bound and comparison audit

The QFT-2 target is already optimal at **8 gates** for the fixed-wire unitary
and declared Clifford+T/CX/CZ/SWAP basis. An exact projective
meet-in-the-middle search enumerated every circuit of at most seven gates,
using radius-three and radius-four sets of 1,952 and 10,072 distinct projective
unitaries. The search even allowed native CY in both directions and arbitrary
free global phase. No shorter implementation exists in that enlarged basis.
The radius search independently recovered minima 1 for H and CZ, 5 for
controlled-S, and 7 for QFT without the final swap. Undefined-behavior
sanitization passed the same exact-integer enumeration.

The saved Qiskit seven-gate circuit differs from the fixed-wire source by a
maximum matrix-entry error of approximately 1; applying its recorded swapped
final layout gives error 0. The historical study also allows swap elision.
Thus strictly beating *all* ten recorded stock/Qiskit totals is infeasible under
the present fixed-wire hold contract: stock QFT-2 is already at the proven
minimum. This is recorded as a lower-bound limitation, not a tuned win or
permission to weaken hold semantics. The final profile below achieves all
nine remaining strict targets.

The exact-search source, sanitized executable/output and Qiskit layout audit
are saved under the isolated workspace as `tools/qft_lower_bound.cpp`,
`qft-lower-bound-ubsan.json`, `qft-search-sanity.log` and `qft-reference/`.

### Architectural ablations and local frame checkpoints

The first controlled ablations expose two different problems. Removing the
five-metric admission restriction admits a few real gate reductions, but does
not explain the large gap. Rescaling the frame penalty or mechanically
reconstructing a feasible PBC prefix does not close it either. A bounded
frontier (one cheapest ready node, a sixteen-node scoring horizon) with 22
discrepancy decisions gives verified totals 134 for adder10, 152 for CSLA and
180 for rc_adder_6. However, the same profile gives 453 for adder28 and 1,109
for adder64. Local action tuning alone therefore fails to scale.

The larger architectural loss happens before those decisions: whole-circuit
Clifford absorption spreads a local arithmetic circuit into wide Pauli axes
and leaves one global frame to reconstruct. Hold must repay that frame cost;
release measurements can hide much of it. The isolated checkpoint variant
retains the input's local frame boundaries until partitioning, then fully
normalizes and resynthesizes each block's Pauli graph **and** its Clifford
frame. Boundaries carry the complete unitary on the same wires. No original
native network is an incumbent or fallback. Certified native cleanup applies
only to the resulting synthesized word, including across block boundaries.

The initial deterministic partition caps each block at three active wires and
64 imported operations, using the same bounded frontier and discrepancy
settings for every case. A larger-support individual operation remains intact.
Each block is compacted onto its active wires, checked independently with
signed/scalar-phase replay, and mapped back to its original wire identities.
Source-bound cleanup certificates compose with those block proofs. Preliminary
independent whole-circuit checks give 130 gates for adder10, 377 for adder28 and
156 for CSLA. These are genuine PBC resynthesis results; the preserved-network
Clifford+T optimizer is reference data only.

The runner's checkpoint profile deliberately receives scalar-wire source IR
before whole-circuit `quake-to-pbc` absorption. Applying the partition after
global absorption cannot recover the lost locality. A production integration
therefore needs explicit local frame boundaries in the PBC representation or
partitioning before conversion, rather than just another scalar heuristic in
the existing global planner.

The refinement boundary also matters. With native refinement disabled, the
three-wire checkpoint profile produces 165/496/161 gates for
adder10/adder28/CSLA; enabling refinement of the generated words gives
130/377/156. The local PBC/frame synthesis and generated-word cleanup are
complementary. The original circuit never enters that cleanup as a seed.

The final fixed portfolio completes the three-wire route first, then tries a
four-wire route with a separate 10-billion-work/30-second slice, then a global
graph/frame route with a 1-billion-work/5-second slice. Optional route
exhaustion retains the already verified checkpoint result. Selection minimizes
the requested total-gate score with nonincreasing T count; secondary metrics
are reported, rather than silently turned into hard admission constraints.
All routes use a sixteen-node scoring horizon, one cheapest frontier seed,
22 discrepancy decisions, exact simplification, four algebraic candidates and
native CX/CZ/SWAP with unit costs. Local Clifford completion considers the
actual native H/S/Sdg/X/Y/Z basis. The final profile does not enable the
experimental regional beam or quantum-context joint planner: neither closed
the gap in its controlled three-case check.

The remaining design shortfall is ranking before native cleanup: a better raw
completion can refine worse than a discarded alternative. The discrepancy-free
global profile reaches 91 on multiply13, whereas the fixed portfolio reaches
93. A production completion interface should rank source-bound, refined full
unitaries, including their frame debt, and retain several distinct boundary
states. It should combine local frame checkpoints with bounded open-boundary
planning to reduce the seven remaining gaps to the native optimizer. Those are
future integration targets, not improvements claimed by this experiment.

### Verified ten-case results: fixed checkpoint portfolio

One fixed profile, with no case IDs or reference counts in synthesis, produces strictly fewer total gates than both recorded stock and Qiskit references on **9 of 10 cases**. All ten outputs pass independent unitary checks. QFT-2 ties the optimal fixed-wire stock result; its seven-gate Qiskit/study values use swap elision as audited above.

| Circuit | Stock | Qiskit | Study proposal | Native Clifford+T | Earlier hold | Tuned hold | Compile wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adder4 | 24 | 25 | 22 | 22 | 20 | **20** | 0.30 |
| adder10 | 140 | 142 | 127 | 106 | 145 | **128** | 0.89 |
| adder28 | 418 | 424 | 369 | 316 | 434 | **370** | 2.83 |
| adder64 | 974 | 988 | 857 | 736 | 1031 | **828** | 6.68 |
| adder118 | 1808 | 1834 | 1589 | 1366 | 2216 | **1506** | 13.03 |
| adder433 | 6673 | 6769 | 5859 | 5041 | 120 s limit | **5715** | 36.96 |
| multiply13 | 105 | 95 | 93 | 91 | 91 | **93** | 0.71 |
| rc_adder_6 | 191 | 185 | 172 | 147 | 163 | **157** | 1.31 |
| csla_mux_3 | 161 | 157 | 157 | 156 | 177 | **152** | 0.96 |
| qft_n002 | 8 | 7 | 7 | 8 | 8 | **8** | 0.19 |

The study-proposal column retains the original comparison data; it is not rerun or used by the planner. A separate historical study table lists multiply13 as 91 rather than the main comparison's 93; the native reference here is also 91. Study semantics and QFT layout freedom differ from fixed-wire hold. Stock and Qiskit columns reuse the saved paired measurements, rather than recompiling those tools during this tuning run. The stock native gate words were checked to match the frozen tuning inputs exactly. Compile times are single wall-time observations for normalization, synthesis and regtomem, excluding independent verification; no comparative compiler speedup or peak-RSS claim is made.

Every tuple below is **gates / T / two-qubit gates / depth / T-depth**. The preserved-network native optimizer remains a comparison only.

| Circuit | Stock CUDA-Q | Qiskit | Native Clifford+T | Tuned hold |
|---|---|---|---|---|
| adder4 | 24 / 8 / 10 / 12 / 2 | 25 / 8 / 10 / 13 / 2 | 22 / 8 / 10 / 11 / 2 | 20 / 8 / 8 / 14 / 5 |
| adder10 | 140 / 32 / 57 / 113 / 32 | 142 / 48 / 65 / 103 / 32 | 106 / 32 / 53 / 95 / 32 | 128 / 32 / 63 / 87 / 22 |
| adder28 | 418 / 96 / 171 / 221 / 64 | 424 / 144 / 195 / 199 / 64 | 316 / 96 / 159 / 185 / 64 | 370 / 96 / 192 / 180 / 52 |
| adder64 | 974 / 224 / 399 / 437 / 128 | 988 / 336 / 455 / 391 / 128 | 736 / 224 / 371 / 365 / 128 | 828 / 224 / 437 / 327 / 98 |
| adder118 | 1808 / 416 / 741 / 761 / 224 | 1834 / 624 / 845 / 679 / 224 | 1366 / 416 / 689 / 635 / 224 | 1506 / 416 / 804 / 497 / 151 |
| adder433 | 6673 / 1536 / 2736 / 2651 / 784 | 6769 / 2304 / 3120 / 2359 / 784 | 5041 / 1536 / 2544 / 2210 / 784 | 5715 / 1536 / 2995 / 2092 / 628 |
| multiply13 | 105 / 32 / 40 / 40 / 15 | 95 / 36 / 40 / 39 / 15 | 91 / 32 / 40 / 39 / 15 | 93 / 32 / 40 / 32 / 12 |
| rc_adder_6 | 191 / 63 / 91 / 111 / 40 | 185 / 67 / 83 / 103 / 39 | 147 / 47 / 76 / 91 / 31 | 157 / 47 / 88 / 87 / 30 |
| csla_mux_3 | 161 / 64 / 76 / 69 / 26 | 157 / 64 / 72 / 68 / 25 | 156 / 62 / 72 / 69 / 26 | 152 / 62 / 74 / 67 / 21 |
| qft_n002 | 8 / 3 / 3 / 6 / 3 | 7 / 3 / 2 / 5 / 3 | 8 / 3 / 3 / 6 / 3 | 8 / 3 / 3 / 6 / 3 |

Across these ten cases the tuned output has 8,977 gates, versus 10,502 stock, 10,626 Qiskit and 7,989 native-optimizer gates. This is a total-gate improvement, not five-metric dominance. For example, adder433 uses 2,995 entanglers versus stock's 2,736, while reducing gates from 6,673 to 5,715 and depth from 2,651 to 2,092. Adder4's T-depth grows from 2 to 5. The native optimizer still has fewer total gates on seven cases; multiply13's earlier hold result of 91 also remains better than this fixed portfolio's 93. Those gaps are retained visibly rather than replaced by weaker-contract results.


### Isolated validation and reproduction

The final ten-case compiler artifact is
`compilers/checkpoint-portfolio/cudaq-opt` inside the isolated workspace, SHA-256
`31cd8eb0f3dd7dbc37ba2dbd5b4bba254b0a87fd0d17c776152dd283420b23c3`.
`final-comparison.json` includes all five metrics, paired baseline hashes,
input/output hashes, measured compile times and independent checker results.
`runs/checkpoint-portfolio-ten/results.json` records exact commands and options.
`hold-tuning.patch` contains only the isolated experimental changes relative to
the frozen hold compiler; the complete built sources and build manifest are
under `compilers/checkpoint-portfolio/source/` and `manifest.json`.

All ten outputs pass QCEC 3.11.0 with simulation checking disabled, checking
the unitary gate portion on the same physical wires. The exact scalar contract
additionally follows from signed/phase replay of every synthesized block and
source-bound refinement certificates, composed across unchanged wire
interfaces. Dense matrices including the emitted scalar phases independently
pass for adder4 and QFT-2 at the unchanged `1e-10` tolerance. Two deterministic
100-gate mixed Clifford+T fixtures on four and seven wires, including a nonzero
input scalar phase and nonadjacent wire interactions, additionally pass dense
full-unitary checks with maximum errors `7.84e-16` and `2.37e-16`
(`boundary-checks/results.json`). These exercise the new local/global wire
mapping and block boundaries on arbitrary inputs.

The final compiler's release spot checks on adder4, multiply13 and QFT-2
produce identical native gate words, physical readouts and all five metrics
as the frozen pre-tuning compiler. All six pass independent ordered joint
measurement checks (`release-spot-final/results.json`). The paired totals
are 20, 73 and 7. These checks use explicit release semantics and the existing
release options; no hold tuning environment variables are set. The full
benchmark suite and the source tree's refactoring build are not run.

All paths below are inside `/tmp`; this experiment changes only this document
in the repository. Reproduce the fixed ten-case profile with:

```sh
PBC_HOLD_PRIMARY=1 PBC_HOLD_LOCAL_CLIFFORD=1 \
PBC_HOLD_CHECKPOINT_PORTFOLIO=1 PYTHONPATH=/tmp/pbc-hold-python \
python3 /tmp/pbc-hold-tuning-20261106-ob5kvhcm/tools/run.py checkpoint-repeat \
  --binary /tmp/pbc-hold-tuning-20261106-ob5kvhcm/compilers/checkpoint-portfolio/cudaq-opt \
  --cases adder4 adder10 adder28 adder64 adder118 adder433 \
          multiply13 rc_adder_6 csla_mux_3 qft_n002 \
  --local-input --timeout 120 --proof \
  --options 'synthesis-candidate-nodes=1 synthesis-lookahead=16 synthesis-discrepancy-steps=22'
```

`--local-input` retains source-local Clifford checkpoints through scalar-wire
conversion. Each block still undergoes PBC normalization and full graph/frame
resynthesis; it does not select the preserve-network path. The build and runner
scripts used for this artifact are frozen beside its manifest. The generic
hold profile and final table cover exactly the ten requested study cases.

## Integrated hold checkpoints: fifteen-case validation (2026-11-06)

The isolated checkpoint fix is now implemented in the source tree. Default
hold conversion retains local primitive Clifford/phase frames between Pauli
rotations. `SynthesisHold.h` partitions before absorption, compacts active
wires, and invokes full Pauli-graph **and** Clifford-frame reconstruction for
every block. Source-bound refinement consumes only the generated circuits.
The combined hold synthesis driver also starts from the original source,
without first selecting a preserved-network optimizer output. Conversion's
`local-frames=false` and synthesis's `synthesis-hold-checkpoints=false` expose
controlled global-route comparisons; release behavior is unchanged.

The first complete route uses three active wires and 64 imported operations
per block; an individual wider operation stays intact. A four-wire route then
has a separate 10B-work/30-second allowance, followed by a global route with a
1B-work/5-second allowance. All child work and deadlines remain inside the
parent. Optional exhaustion retains a verified completion; an invalid unitary
proof or source binding is fatal. Native continuation ranks generated circuits
by the configured objective with nonincreasing T count. Local factor completion
uses native H/S/Sdg/X/Y/Z. The experimental regional beam and joint-context
hooks from the isolated study are not included.

**All fifteen final hold outputs are independently verified.** The original
ten reproduce the isolated portfolio's exact five-metric vectors, including
nine strict total-gate wins over both paired stock CUDA-Q and Qiskit references.
QFT-2 remains at its proven fixed-wire eight-gate minimum; its seven-gate
Qiskit/study references elide the physical output swap. The recorded Qiskit and
study columns in the ten-case table above still apply. These remain reference
data, never synthesis inputs.

| Circuit | Stock gates | Native C+T gates | Earlier hold gates | Checkpoint hold gates | T | Two-qubit | Depth | T-depth | Compile wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adder4 | 24 | 22 | 20 | **20** | 8 | 8 | 14 | 5 | 0.30 |
| adder10 | 140 | 106 | 145 | **128** | 32 | 63 | 87 | 22 | 0.86 |
| adder28 | 418 | 316 | 434 | **370** | 96 | 192 | 180 | 52 | 2.70 |
| adder64 | 974 | 736 | 1031 | **828** | 224 | 437 | 327 | 98 | 6.94 |
| adder118 | 1808 | 1366 | 2216 | **1506** | 416 | 804 | 497 | 151 | 14.57 |
| adder433 | 6673 | 5041 | 120 s limit | **5715** | 1536 | 2995 | 2092 | 628 | 40.62 |
| multiply13 | 105 | 91 | 91 | **93** | 32 | 40 | 32 | 12 | 0.77 |
| rc_adder_6 | 191 | 147 | 163 | **157** | 47 | 88 | 87 | 30 | 1.35 |
| csla_mux_3 | 161 | 156 | 177 | **152** | 62 | 74 | 67 | 21 | 0.96 |
| qft_n002 | 8 | 8 | 8 | **8** | 3 | 3 | 6 | 3 | 0.19 |
| mod5_4 | 57 | 37 | 29 | **33** | 8 | 17 | 22 | 5 | 0.34 |
| basis_trotter_n4_transpiled | 20299 | 14816 | 17964 | **14888** | 7110 | 378 | 8060 | 3865 | 25.24 |
| basis_trotter_n4 | 19667 | 14410 | 17769 | **14573** | 6958 | 342 | 6542 | 3095 | 25.56 |
| basis_test_n4_transpiled | 885 | 654 | 650 | **648** | 310 | 26 | 480 | 228 | 1.37 |
| grover_5 | 668 | 500 | 493 | **510** | 166 | 228 | 395 | 146 | 5.96 |

Counts exclude scalar phases, allocations, measurements and sinks. Depth and
T-depth use the same dependency scheduling convention as the prior reports.
Compile wall time covers conversion, synthesis and regtomem, excluding
independent verification. Times are single local observations, not a comparative
compiler-speed benchmark. The stock/native columns are the frozen paired
references for these exact input circuits; those compilers were not rerun.
All fifteen new hold compilations completed within their 120-second allowance.

There are nine gate-count improvements over the earlier completed hold results,
two ties, three regressions, and one newly completed case (adder433). The
regressions are multiply13 **91 -> 93**, mod5_4 **29 -> 33**, and grover_5
**493 -> 510**. Native Clifford+T still wins ten of fifteen total-gate
comparisons. Secondary metrics also trade off: basis_test_n4_transpiled reduces
gates 650 -> 648 but increases depth 397 -> 480 and T-depth 188 -> 228;
adder433 uses 2,995 entanglers versus stock's 2,736. The table retains these
costs instead of claiming five-metric dominance or borrowing release results.
Across all fifteen, hold emits **39,629 gates**, compared with **52,078 stock**
and **38,406 native Clifford+T** gates.

### Configuration, verification and reproduction

One fixed `checkpoints` hold profile is used for every input: native CX/CZ/SWAP
and one-qubit Clifford+T at unit costs, total-gates objective,
`synthesis-hold-checkpoints=true`, `synthesis-simplify=true`,
`synthesis-candidate-nodes=1`, `synthesis-lookahead=16`,
`synthesis-discrepancy-steps=22`, `synthesis-algebraic-candidates=4`,
`synthesis-commuting-candidates=0`, `synthesis-joint-phase-candidates=0`,
no pruning, `max-gates=262144`, `max-dependencies=64000000`, a 512 MiB
compressed-dependency cap, `max-work=1000000000000`, and `timeout-ms=120000`.
Compiler processes use one worker and an 8 GiB address-space cap. Hold
conversion explicitly uses `semantics=hold local-frames=true`, retains original
measurements and quantum owners, and inserts no terminal measurements. Only
compiler-generated implicit `quake.evince` annotations are removed, as in the
existing workflow. No environment-variable tuning hooks are required.

MQT QCEC 3.11.0 independently verifies every final circuit's unitary gate
portion on identical physical wires, with simulation disabled; ordered readout
identities are checked separately. Exact scalar phase additionally follows
from block-wise signed/phase replay and compositional refinement certificates.
Dense full matrices including emitted scalar phases pass for all six inputs
of at most eight qubits; maximum entrywise error is `1.01e-13` at the unchanged
`1e-10` tolerance. Release regression spot checks on adder4, QFT-2 and mod5_4
retain identical native words, physical readouts and all five metrics; all six
before/after outputs pass independent ordered joint-measurement checks.

The final implementation passes **380 native tests and all 15 PBC lit tests**.
Added checks cover arbitrary entangled inputs, sparse physical-wire embeddings,
oversized Pauli operations, scalar-only identity rotations, optional budget
exhaustion, fatal invalid certificates, and local-frame retention through
Quake conversion. The broader benchmark suite was not run.

Final compiler SHA-256:
`5405c7f8351e4eab7051c45c12991c97e0162e77a21b47ba3877fca6971d565e`.
Artifacts are under `build/pbc-hold-checkpoints-2026-11-06/`:
`compiler-manifest.json`, frozen `source/`, `implementation.patch`,
`benchmark-final/results.json`, `comparison.json`, `tests.json` and `lit.log`.
The earlier `benchmark/` run is retained separately; its scalar-only corner
case review led to the final identity-rotation fix before this final run.
The final runner, compiler and every input/output have recorded hashes.

```sh
PYTHONPATH=/tmp/pbc-hold-python python3 utils/pbc/benchmark_contracts.py \
  --compiler build/pbc-hold-checkpoints-2026-11-06/cudaq-opt-hold \
  --before-compiler build/pbc-hold-2026-11-06/cudaq-opt-before \
  --snapshot build/pbc-synthesis-redesign-2026-11-05/benchmark \
  --output build/pbc-hold-checkpoints-2026-11-06/repeat \
  --hold-profile checkpoints --timeout-ms 120000
```

`--hold-profile legacy` selects the older globally absorbed profile on the new
compiler for controlled comparisons. Release spot checks retain their own
unchanged configuration. Full commands and proof results remain per-case in
the final JSON; checker inconclusiveness would remain unverified rather than
being substituted with original-circuit counts.

## Changelog

- **2026-11-06:** Integrated local-frame hold checkpoint synthesis; all fifteen cases independently verified with one fixed hold configuration, 380 native tests and 15 PBC lit tests passing.

- **2026-11-06:** Added isolated hold-only checkpoint tuning for the original ten study cases, complete metric comparisons, exact QFT-2 lower bound and final release spot checks. Experimental code and artifacts stay in `/tmp`.

- **2026-11-06:** Added default full-unitary hold synthesis, fifteen-case hold measurements, and unchanged before/after release spot checks.

- **2026-11-04:** Documented production synthesis, ten-case outcomes, native-SWAP QFT proof, runtime findings, and safe synthesis-limit fallback.
- **2026-11-05:** Integrated exact signed Clifford-frame refinement and recorded independently verified results for nine original and five additional cases.
- **2026-11-05:** Added the synthesis architecture proposal, literature review, fifteen-case structural inventory, reference-free frame experiments and dependency-oracle checks.
