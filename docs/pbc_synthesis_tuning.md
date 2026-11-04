# PBC synthesis tuning

This study tests whether normalized Pauli-based computation can be synthesized
competitively for a gate-based Clifford+T target. The production implementation
is in `cudaq/lib/Optimizer/Transforms/Pauli/GraphSynthesis.cpp`; its public
interface is `cudaq/Optimizer/Pauli/GraphSynthesis.h`. The new implementation
replaces the previous graph synthesizer.

## Contract and scope

Normalization absorbs the Clifford network into rotation axes and signed terminal
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
T-count decoding, circuit cleanup, Pauli propagation, phase placement and Clifford
window optimization. The dictionary uses the existing small Clifford optimizer
to construct actions; it does not clean up the resulting benchmark circuit.

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
  --quake-to-pbc='assume-measure-all=true' --quake-verify-pbc \
  --mlir-disable-threading -o normalized.qke
prlimit --as=8589934592 --cpu=3600 --core=0 -- \
  "$CUDAQ_OPT" normalized.qke \
  --quake-synthesize-pbc='objective=total-gates synthesis-native-entanglers=cx,cz,swap synthesis-swap-cost=1 synthesis-simplify=false synthesis-prune-unobserved=false synthesis-discrepancy-steps=22 max-work=1000000000000000 timeout-ms=0' \
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

| Case | CUDA-Q | Qiskit | Synthesis off | Production synthesis |
|---|---:|---:|---:|---:|
| adder_n4_transpiled | 24 | 25 | 22 | 23 |
| adder_n10_transpiled | 140 | 142 | 106 | 119 |
| adder_n28_transpiled | 418 | 424 | 316 | 392 |
| adder_n64_transpiled | 974 | 988 | 736 | 970 |
| adder_n118_transpiled | 1808 | 1834 | 1366 | 1885 |
| adder_n433_transpiled | 6673 | 6769 | 5041 | Timeout |
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
approximately ten minutes and is reported as a timeout; the configured one-hour
CPU cap was not reached. It produced no completed circuit, so no output metrics
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
| adder_n433_transpiled | 2.652 | Timeout (~10 min) | 0.166 |
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

## Changelog

- **2026-11-04:** Documented production synthesis, ten-case outcomes, native-SWAP QFT proof, runtime findings, and safe synthesis-limit fallback.
