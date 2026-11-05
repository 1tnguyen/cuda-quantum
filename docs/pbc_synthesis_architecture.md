# PBC synthesis architecture

## Scope and implementation model

The implementation reconstructs a native circuit from normalized Pauli-based computation (PBC): an ordered rotation word, signed terminal observables, and fixed physical readout wires. Its terminal contract preserves the **ordered joint measurement distribution for arbitrary input states**, including states entangled with an external reference. It does not preserve the post-measurement quantum state.

The top-level architecture is a **portfolio of complete-circuit routes**. An existing synthesis and optimization pipeline first produces a verified native incumbent. An optional joint-phase family then constructs additional complete programs from the normalized source. Each family contains bounded regional planners, but there is no single planner choosing freely among all strategies at each region. Candidate generation, proof, completion, and final selection are separate responsibilities.

The main entry point is `graph::synthesizeMeasured` in [GraphSynthesis.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/GraphSynthesis.cpp), with public controls in [GraphSynthesis.h](../cudaq/include/cudaq/Optimizer/Pauli/GraphSynthesis.h). The implementation and configuration described here are the integrated architecture used for the fixed-configuration reconciliation. Benchmark results and their current validation status belong in [the synthesis report](pbc_synthesis_tuning.md), rather than being duplicated in this document.

```mermaid
flowchart TB
    Q[Supported scalar-wire Quake] --> I[Import and validate]
    I --> N[PBC normalization]
    N --> S[Immutable normalized measured source]
    S --> E[Existing complete-circuit portfolio]
    E --> A[Verified native incumbent]
    S --> J[Optional joint-phase profiles]
    A -. immutable admission anchor .-> J
    J --> C[Checked complete candidates]
    A --> F[Final complete-circuit selection]
    C --> F
    F --> O[Clone, emit native Quake, verify, commit]
    P[Preserve-network reference] -. benchmark comparison only .-> R[Reported metrics and independent checks]
    O --> R
```

The preserve-network reference is not an input to any normalized-PBC synthesis provider. It is a separate optimization path and a comparison target. Generated native words may be refined after synthesis; this does not reintroduce the original or preserve-network circuit as a seed.

## 1. IR boundary and semantic contracts

### Quake ingress and transactional emission

[PBCPasses.cpp](../cudaq/lib/Optimizer/Transforms/PBCPasses.cpp) implements the importer, `Plan`, `read`, `emit`, and the pass drivers. The pass declarations and command-line names are in [Passes.td](../cudaq/include/cudaq/Optimizer/Transforms/Passes.td).

| Pass or entry point | Responsibility |
|---|---|
| `quake-verify-pbc` / `QuakeVerifyPBCPass` | Validate the actual IR contract; an attribute or marker is not proof. |
| `quake-to-pbc` / `QuakeToPBCPass` | Import supported primitive circuits and absorb their Clifford frame into rotations and terminal observables. |
| `pbc::normalize` | Native-library normalization, preserving ordered terminal measurement probabilities. |
| `quake-synthesize-pbc` / `QuakeSynthesizePBCPass` | Read already normalized PBC, call `graph::synthesizeMeasured`, and emit a verified native completion transactionally. |
| `quake-lower-pbc` / `QuakeLowerPBCPass` | Mechanical basis/parity and joint-readout lowering without the synthesis portfolio. |
| `quake-optimize-pbc`, `quake-try-pbc` / `PBCOptimizationPass` | Separate optimization/round-trip drivers, including the optional preserve-network output path. |

Verified PBC uses a single block, distinct scalar SSA wires, static Pauli words, uncontrolled arbitrary-angle `quake.exp_pauli` rotations followed by terminal commuting `quake.mpp` operations, and explicit sink closure. The importer rejects escaping quantum outputs, borrowed wire sets, control flow, and classical feed-forward into quantum operations. Supported classical operations, measurement discrimination, and classical returns remain connected to their original outcomes.

`QuakeSynthesizePBCPass` builds a shared `SearchBudget`, passes measurement readout identities explicitly, and consumes the synthesis result only on success. It clones the function, emits the candidate, runs MLIR verification, and commits the clone. Search never incrementally mutates the original function. The pass records synthesis work and outer-budget exhaustion as attributes; these are telemetry, not correctness certificates.

### Native Pauli representation

[Circuit.h](../cudaq/include/cudaq/Optimizer/Pauli/Circuit.h) defines `QubitSet`, `Pauli`, `Angle`, `Gate`, and `Kind`. `QubitSet` uses dynamically sized storage; a 64-bit machine word is not a circuit-width limit. A Pauli stores X and Z support plus its sign. A rotation represents

\[
R_P(\theta)=\exp(-i\theta P/2).
\]

Rational angles are exact multiples of π. Opaque angle identities remain symbolic and are not approximated into Clifford+T sequences. Explicit scalar phases matter to full-unitary subproofs even though a global phase is unobservable at the final measured boundary.

[PBC.h](../cudaq/include/cudaq/Optimizer/Pauli/PBC.h) and [PBC.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/PBC.cpp) implement `pbc::normalize` and `CommutingObservables`. Normalization absorbs Clifford operations into the signed axes and ordered terminal observables. It does not assume an all-zero initial state.

### Three different contracts

| Contract | What must be preserved | Main representation |
|---|---|---|
| Closed quantum region | Entire unitary, including scalar phase where required by the checker | Native gate word; `graph::equivalent` and checked backend witnesses |
| Open quantum region | Exact relation between body, source region, and an actual Clifford exit | `JointPhaseImplementation`, commuting/open-region implementations |
| Complete measured program | Ordered joint outcome probabilities for arbitrary inputs, with fixed physical readout identities | `MeasuredProblem`, `MeasuredCandidateSink`, `CertifiedCompletion` |

The measured contract cannot be substituted for an internal live quantum region. An internal region must account for its actual exit on every later quantum obligation. Conversely, the terminal measured contract intentionally permits changing an unobserved final quantum state.

## 2. Shared contracts, costs, and dependency structure

### Measured problems and checked candidates

[SynthesisCandidates.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisCandidates.h) defines:

- `MeasuredProblem`: a borrowed immutable view of width, ordered rotations, signed observables, and physical readout wires.
- `MeasuredCandidateSink`: independent whole-word measured replay followed by checked local signed-frame refinement and the existing selection policy.
- `VerifiedCandidate`: a privately constructed value indicating measured replay, optionally composed with checked local channels.

The sink's implementation is in [GraphSynthesis.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/GraphSynthesis.cpp). `submit` checks native basis, operands, size, measured equivalence, and the source T-count guard. `takeBest` retains the existing five-metric comparison against the best unrefined rollout. A proof enum is descriptive; it is not permission to reuse the candidate against another source.

`RegionalPlan`, in [SynthesisRegions.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisRegions.h), contains a native prefix and residual signed observables. It is a **proposal**, not a complete native circuit or transferable proof. A caller must construct the fixed-wire readout and verify the resulting whole program.

### Native cost and ranking

[SynthesisCost.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisCost.h) centralizes `nativeWord`, `evaluateNativeCost`, `evaluateCircuitCost`, `CostVector`, and `circuitScore`.

The five reported integral metrics are total gates, T count, two-qubit gates, depth, and T-depth. Native weights can distinguish CX, CZ, SWAP, one-qubit Clifford, and T costs. `nativeWord` materializes the selected entangling profile before final comparisons. For the total-gates objective, the common rank is weighted native cost, T count, depth, two-qubit count, then T-depth. With unit weights, the first component coincides with the supported native gate-count model.

Opaque rotations use the existing unresolved-rotation proxy for ranking; their eventual approximation cost is not known here. Cost evaluation does not prove equivalence, and a low prefix cost does not bound the cost of its terminal readout.

### Exact dependency index

[SynthesisDependencies.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisDependencies.h) provides `DependencyIndex::build` and `DependencyCursor::{create,ready,consume}`. It groups repeated unsigned Pauli axes into classes and stores conflicts between classes. Occurrence order still determines the exact readiness constraints. Ignoring signs for commutation does not remove signs from the quantum program.

The logical dependency cap retains the meaning of the original occurrence graph: the number of earlier/later anticommuting occurrence pairs. Compression changes storage, not which edges are enforced. If there are `m` occurrences, `K` distinct axes, and `A` stored class conflicts, storage is proportional to `m + K + A`; conflict discovery uses O(K²) Pauli comparisons. The index does not silently discard constraints to fit a cap.

A common Clifford conjugation preserves all pairwise commutation relations, so the joint planner can reuse the original dependency index while carrying exits. Rewrites that change the underlying obligation word require the appropriate reconstruction or a new index. The legacy `Synthesizer::run` also preserves its established occurrence traversal and newly unblocked-node behavior.

## 3. Existing complete-circuit portfolio

The existing family remains an independently useful pipeline. Its orchestration is in `runMeasuredPortfolio` and `synthesizeMeasured` in [GraphSynthesis.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/GraphSynthesis.cpp). Optional providers are gated by their public controls and remaining budget.

```mermaid
flowchart TB
    S[Normalized measured source] --> B[representationBaseline]
    B --> L[Mechanical bootstrap and/or legacy discrepancy search]
    L --> I[Verified incumbent]
    I --> R[Alternative signed readouts]
    R --> G[Commuting, regional, general, macro and open plans]
    G --> V[Certified sign/frame representation and full continuation]
    V --> N[Complete native continuation]
    N --> A[Checked algebra alternatives and fixed-word reconstruction]
    A --> F[Final native cleanup]
    F --> X[Outer Four, indexed, Native-to-Four and Clifford-window routes]
    X --> O[Existing portfolio result]
```

The diagram shows orchestration, not a requirement that every stage run. Resource limits can stop optional exploration after an earlier verified result exists.

### Feasibility and primitive reconstruction

`representationBaseline` selects the baseline schedule. With `legacyWorkLimit == 0`, it retains legacy-first behavior. A positive limit first attempts a checked mechanical completion, then bounds legacy search in a child slice so other providers retain resources.

[SynthesisBootstrap.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisBootstrap.h), `buildBootstrapPrefix`, lowers normalized rotations using independent basis/parity gadgets. It establishes a feasible prefix from PBC alone; the caller must still supply and prove readout. It is not a hidden original-circuit fallback.

`runMeasuredSearch` and the private `Synthesizer` explore primitive Clifford actions, ready Pauli obligations, frame costs, and bounded discrepancy rollouts. Candidate scoring can include emitted action cost, sampled residual support, ready-frontier weight, readout cost, and ASAP scheduling credit. These are heuristics. Completed circuits go through measured replay and the sink.

### Readout and regional routes

| Component | Role and proof boundary |
|---|---|
| [SynthesisReadout.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisReadout.h), `generateReadoutCandidates` | Alternative elimination orders and pivots for signed terminal observables. Each complete native tail independently maps every original row to positive Z on its specified physical wire. Output identities are never permuted to improve cost. |
| [SynthesisCommuting.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisCommuting.h), `solveCommutingRegion` | Bounded implementations of commuting regions, including small-rank alternatives and actual Clifford exits. |
| [SynthesisRegions.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisRegions.h), `generateRegionCandidates` | Completed commuting sweeps, then bounded alternatives over different carried boundaries. |
| [SynthesisGeneral.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisGeneral.h), `generateGeneralCandidates` | Compact adjacent-window reconstruction with a full-unitary synthesis baseline. |
| [SynthesisMacros.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisMacros.h), `solveMacroRegion` | Bounded multi-action body/exit proposals. The `SynthesisGeneral` adapter closes and independently verifies selected candidates; the standalone macro proposal is not a certificate. |
| [SynthesisOpen.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisOpen.h), `generateOpenRegionCandidates` | Open-region synthesis and exact propagation of actual exits through subsequent rotations and observables. |

`tryRegionalAlternatives` first attempts the small whole-word commuting case, then completed regional sweeps, a boundary beam, general reconstruction without and with macros, and open-region plans. `tryCompletedRegion` finishes each plan with readout and the measured sink. This is a portfolio of completed proposals, not unrestricted mixing of these implementations in one global search.

### Representation changes and algebraic alternatives

[SynthesisRepresentations.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisRepresentations.h) defines `RepresentationAlternative`, `FlipTSign`, `exploreRepresentations`, and `verifyRepresentation`. A sign/frame change is allowed only with an independently replayed certificate that also accounts for its effect on later rotations and terminal observables.

[SynthesisRepresentationIndex.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisRepresentationIndex.h), `exploreIndexedRepresentations`, accelerates a bounded representation search. Its proposal still needs the same independent verifier. `tryIndexedRepresentation` constructs a complete continuation before considering replacement. Its older dedicated continuation intentionally uses one greedy rollout, disables repeated readout-cost scoring, and bounds lookahead/frontier size. Those are route-specific algorithm choices, not benchmark-case-specific options.

[SynthesisAlgebra.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisAlgebra.h), `generateAlgebraAlternatives`, starts from normalized rotations, invokes checked T-count transformations, and renormalizes their results with the source observables. `CertifiedAlgebraAlternative` owns its source binding. The implementation composes backend proof with measured normalization; it does not claim a new independent invalid-proof detector beyond what the backend exposes.

Before algebraic exploration, the existing pipeline completes native refinement of its current result. Each algebra alternative then receives a fixed-word synthesis portfolio with recursive algebra generation disabled, followed by native refinement. This prevents an unfinished reconstruction from displacing the earlier completed continuation merely because its abstract T count is lower.

### Native refinement of generated words

These components operate on circuits already produced and checked by synthesis:

| Component | Implemented mechanism |
|---|---|
| [SignedCliffordFrames.h](../cudaq/lib/Optimizer/Transforms/Pauli/SignedCliffordFrames.h), `optimizeSignedFrames` | Local signed-frame optimization, followed by independently bounded primitive-Pauli and generic local-Clifford transport routes from the same immutable seed. Checked channel relations and metric guards retain safe completed results. |
| [SynthesisNativeRefinement.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisNativeRefinement.h), `refineSynthesizedNative` | Checked gate-count and T-count proposals, native lowering, then bounded linear-window refinement. A decode-size limit can select merge-only T refinement rather than suppressing inexpensive merges. |
| [SynthesisLinearWindows.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisLinearWindows.h), `optimizeLinearWindows` | Small linear CX/SWAP windows with exact binary-map checks and quantum-equivalence validation. |
| [SynthesisLinearFour.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisLinearFour.h), `buildLinearFourDictionary`, `optimizeLinearFourWindows` | Profile-bound four-wire linear dictionaries and checked window replacements; retained dictionaries are invocation-owned. |
| [SynthesisCliffordWindows.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisCliffordWindows.h), `optimizeCliffordWindows` | One-/two-wire Clifford windows extracted across exactly commuting interruptions, with signed-channel lookup and independent scalar-phase verification. |
| [CliffordDictionary.h](../cudaq/lib/Optimizer/Transforms/Pauli/CliffordDictionary.h) | Shared Clifford state algebra and dictionary mechanics. Existing [GateCount.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/GateCount.cpp) BFS behavior is preserved; the window provider uses its own native-profile search policy. |

The T-count backend is implemented in [TCountOptimizer.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/TCountOptimizer.cpp), [TCountMerge.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/TCountMerge.cpp), [TCountDecode.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/TCountDecode.cpp), [TCountAlgebra.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/TCountAlgebra.cpp), and [TCountVerification.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/TCountVerification.cpp).

The outer measured entry point appends Four refinement, the independent indexed route, Native-to-Four completion, and Clifford-window-to-native completion when enabled. Older portfolio admissions retain their existing all-five nonregression rules against their current incumbents. The joint family's opt-in G/T policy does not relax those upstream rules.

## 4. Joint ready-phase engine and planner

The joint family separates a finite local solver from a whole-program planner. It uses the same immutable normalized source as the existing family. The older final circuit supplies admission costs only.

### Local engine: parity demands with an actual exit

[SynthesisJointReadyPhase.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisJointReadyPhase.h), `solveJointReadyPhaseRegion`, accepts a compact commuting packet. The configured local limits are up to 64 rotations, 16 active wires, and rank three. Its implemented steps are:

1. Validate dimensions, native profile, signed axes, commutation, and resource bounds.
2. Construct and independently check a closed mechanical baseline. Retain it before optional exploration.
3. Try bounded independent generators and diagonalizers. Coordinate layouts include the original first-rank assignment and bounded injective pivots drawn from the generators' natural support.
4. Express the diagonal rotations as parity demands. Search finite states consisting of an invertible binary row map and the set of demands already visited.
5. Materialize phase rotations when their parities are available, reconstruct a complete native body and actual Clifford exit, and independently verify every selected survivor.

For rank three, there are 168 invertible binary maps and at most 128 visited-demand masks, giving a maximum table of 21,504 states per diagonalizer. This finite parity-walk search does not establish global circuit optimality. Basis/layout limits, endpoint shortlisting, and whole-program context remain bounded heuristics.

The local relation is `body = exitCliffords * source`, including scalar phase, with gate words interpreted in circuit order. `JointPhaseImplementation` remains a mutable proposal after this local check; the final measured program must still be verified after composition.

The optional `JointPhaseScorer` sees complete endpoint proposals before survivor verification. Its result affects shortlisting only. It must not treat that callback argument as a certificate; the engine checks selected survivors independently afterwards.

### Whole-program planner: exact obligations, bounded choices

[SynthesisJointPhasePlanner.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisJointPhasePlanner.h), `generateJointPhasePlans`, maintains:

- The current signed rotation axes and active occurrence mask.
- The immutable dependency index, with an exact cursor reconstructed from the active mask for each expansion.
- A native prefix, actual accumulated exit, and ordered residual observables.
- Cached per-row native rotation costs and signed readout potentials.

Packet selection considers independent source-order and low-support ready pools, then closes a chosen rank-three span over the full ready frontier. The closure does not add dependencies or reorder noncommuting obligations. A broad singleton can use the checked bootstrap fallback when compact local synthesis is unavailable.

The planner first finishes a control greedy chain with the original coordinate-layout policy. It then finishes one contextual greedy chain before optional beam search. The control stays pinned; the contextual completion has one separate final-scoring allowance, and the beam retains its eight completion-score allowance. The configured beam width is two and at most four complete plans are retained per source.

```mermaid
flowchart LR
    S[Current signed obligations] --> D[Exact ready frontier]
    D --> P[Bounded packet and span closure]
    P --> L[Joint local solver]
    L --> B[Native body plus actual exit]
    B --> T[Transport exit through all live axes and ordered observables]
    T --> S
    T --> C{All obligations consumed?}
    C -- no --> D
    C -- yes --> R[Complete RegionalPlan proposal]
    R --> K[Completion and independent measured proof]
```

### Transport and context scoring

[SynthesisCompactTransport.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisCompactTransport.h) contains only generic helpers: `add`, `mul`, `sameWord`, `exitImages`, `transport`, and `lift`. It has no dependency on an earlier experimental planner. It maps compact signed Clifford images back to physical supports, preserving untouched spectator factors and the sign of the full Pauli product.

Those conjugation images do not encode an overall Clifford scalar. The actual native exit word and the phase-aware local relation remain authoritative; the transport helper itself is not a proof-bearing type.

An exit is carried through every remaining affected rotation and every affected ordered observable. Rows disjoint from its support remain unchanged; empty exits require no transport. The fixed physical readout association never changes.

The context scorer estimates **prefix + residual rotation cost + terminal readout cost**. It chooses one scale per parent: exact readout with independent signed-row replay when the callback slice can fund it, or a cached signed-row potential otherwise. If exact rescoring misses for a retained sibling, the retained siblings use the common estimate rather than mixing incomparable scales. Only affected rows are transported and repriced.

Callback allowance is distributed across estimated remaining packets, prioritizing completion of the current contextual chain. A packet has at most 24 endpoint callbacks and, when funded, up to four exact survivor rescores. Wider beam exploration uses remaining quota or cheap estimates. Scores can change exploration order, but never certify a region or authorize final replacement.

## 5. Completion, immutable admission, and profile isolation

[SynthesisCompletion.h](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisCompletion.h) and [SynthesisCompletion.cpp](../cudaq/lib/Optimizer/Transforms/Pauli/SynthesisCompletion.cpp) implement the reusable `CompletionSession` service. The graph adapter does not reproduce completion logic or artifact/JSON handling.

### Ownership and certificates

`CompletionRoute` owns an immutable source. An original route binds directly to the session's normalized source; an indexed route is created only after `verifyRepresentation` succeeds against that original source. An unrelated session's route is rejected even if its dimensions match.

`CertifiedCompletion` has a private constructor and retains its route, native word, metrics, rank, and stage. `matchesSource` compares against the original normalized contract. Source identity, signed observables, and fixed readout wires are part of the binding; a route label or matching cost vector is insufficient.

### Ranked continuation and independent final admission

The production adapter enables `CompletionLimits::rankedContinuation`. Its bounded chain follows:

```mermaid
flowchart TB
    P[RegionalPlan proposal] --> R[Fixed-wire signed readout]
    R --> W[Whole measured replay of raw word]
    W --> S[Checked sink local-channel result]
    S --> N[One ranked native refinement source]
    N --> F[One ranked Four refinement source]
    W --> A[Immutable-anchor admission]
    S --> A
    N -- every checked alternative --> A
    F -- every checked alternative --> A
    A --> B[Best admissible complete word]
```

Local continuation uses the raw word's G/T envelope and strict common rank: raw → locally selected sink word → selected native alternative → one Four continuation. **This local choice is not final admission.** Every checked intermediate word reaches the session's immutable-anchor comparison before local ranking, including words not selected to continue. A raw word failing final admission can still generate a successful refinement.

The expanded private mode remains available and independently continues raw/sink words and native alternatives. The fixed reconciliation uses the bounded ranked mode so additional branches do not silently alter the validated planner-feedback and resource schedule.

`AllFive` is the default admission policy: gates, T count, two-qubit count, depth, and T-depth must not exceed the incoming anchor. Explicit `GatesAndT` requires only gates and T count to remain within that anchor; secondary metrics may trade off and are still reported. Either policy additionally requires strict improvement in the common rank. All candidates are compared to the **same immutable anchor**, not to a changing sequence of accepted candidates. The evolving best rank selects among admissible completed words.

### Frozen scoring feedback versus retryable completion

`CompletionSession::score` freezes the rank from a plan's first checked completion, including a first attempt whose optional refinement exhausted. `CompletionSession::complete` may later retry an exhausted entry with a larger allowance and improve final admission without changing that frozen scorer rank. Pre-proof misses are not cached as completed results.

```mermaid
sequenceDiagram
    participant P as Planner
    participant C as CompletionSession
    participant A as Immutable-anchor admission
    P->>C: score(route, complete plan, small child)
    C->>C: Readout and measured proof
    C->>A: Offer every completed checked stage
    C-->>P: First completed rank, frozen for this plan
    Note over P,C: Optional scoring miss does not certify or erase a complete plan
    P->>C: complete(route, retained plan, larger allowance)
    C->>C: Retry exhausted continuation
    C->>A: Offer further checked candidates
    P->>C: score(same route, same plan)
    C-->>P: Original frozen rank
```

The cache is bounded and keyed by immutable route identity plus the exact plan. Normal and wide profiles own **separate sessions and scoring caches**. Both use the same incoming anchor. Richer results in one profile cannot change another profile's heuristic feedback. At the end, the adapter promotes the best source-bound result across the profiles once.

## 6. Proof composition and failure handling

```mermaid
flowchart TB
    S[Original normalized source contract] --> O[Original source route]
    S --> V[Independent representation verification]
    V --> I[Owned indexed source route]
    O --> L[Locally checked region relations]
    I --> L
    L --> T[Exact signed exit transport]
    T --> R[Readout maps each ordered row to its fixed positive Z]
    R --> M[Independent whole-word measured replay]
    M --> C[Checked signed local channels]
    C --> U[Source-bound full-unitary Native and Four certificates]
    M --> A[Cost admission]
    C --> A
    U --> A
    H[Heuristic scores and cost estimates] -. ordering only .-> L
    H -. never proof .-> A
```

The proof chain is compositional. It does not require rerunning a restricted measured-word matcher after a checked T-count rewrite has changed the rotation word. Instead, the source-bound unitary certificate composes with the already established measured relation. Internal phase-aware checks remain necessary: equality up to global phase at an arbitrary region boundary is not automatically a substitute for the required exact relation.

An indexed sign/frame certificate establishes a relation between terminal measured programs, not necessarily equality of their complete unitaries. Raw completion is replayed against the route's own certified source; its relation to the original follows by composition. `graph::equivalent` is also a restricted checker based on partially commuting rotation words and exact Clifford phase, not a decision procedure for arbitrary quantum circuits. A failed or inconclusive check does not authorize publication.

Three outcomes are distinct:

| Outcome | Consequence |
|---|---|
| Unsupported input, unavailable optional strategy, or a cost rejection | No replacement from that attempt; an existing verified incumbent remains available. |
| Work, memory, candidate, or deadline exhaustion | Stop the affected exploration. Retain completed checked results; a complete but unverified plan still needs external completion/proof. |
| Invalid proof or source-binding mismatch | Fatal for the affected synthesis invocation. Clear the completion session's best/cache and propagate the error; do not silently use a failed proof as a heuristic miss. |

Some older checked backends return only successfully verified candidates and do not expose a distinct invalid-proof status. The adapters describe that limitation rather than inventing additional proof coverage. Where an explicit `InvalidProof` status exists, the graph orchestration propagates it.

The optional final planner scorer returns a rank, not a proof-bearing value. If its budget misses, the planner can retain the completed `RegionalPlan` in a separate heuristic tier; the caller must still run full completion. This preserves useful work without promoting a partial or unchecked program.

## 7. Work, memory, scheduling, and reproducibility

[SearchBudget.h](../cudaq/include/cudaq/Optimizer/Pauli/SearchBudget.h) implements hierarchical work and wall-clock limits. Successful child work is charged once to every ancestor. A failed local charge does not debit work, and local exhaustion does not stop siblings or the parent. Deadlines are checked at budget operations; they are not preemptive compiler-process timers. Work units are implementation estimates, not seconds, bytes, or a uniform bit-operation count.

### Fixed reconciliation configuration

The defaults keep the joint route disabled (`jointPhaseCandidates = 0`, `jointPhaseWideSearch = false`, acceptance `AllFive`). The reconciliation deliberately enables a common portfolio for every benchmark case:

| Level | Fixed configuration |
|---|---|
| Whole synthesis invocation | 1 trillion work units, 300-second parent deadline, 262,144 input/output gate cap, 64 million logical dependency cap |
| Existing portfolio | 180-second child; 100 billion work cap for the bounded legacy search; 22 discrepancy steps |
| Existing optional providers | 32 representation steps, 8 readout candidates, 8 commuting/regional candidates, 4 algebraic alternatives, 64 indexed-representation steps |
| Entire joint family | 100 billion aggregate work units, up to 120 seconds, capped by the parent |
| Normal profile | 3 billion planner work per source; 4,096 context calls and expansions; up to 60 seconds for the profile |
| Wide profile | 10 billion planner work per source; 16,384 context calls and expansions; up to 60 seconds for the profile |
| Both joint profiles | Original source then checked indexed source; four complete plans per source; beam width two; four coordinate layouts; 128 MiB planner envelope |
| Each completion session | 2 GiB accounting envelope; ranked continuation; native provider work cap 100 million and Four cap 50 million, subject to available parent reserve |
| Final joint admission | `gates-and-t`; same immutable incoming anchor for both profiles |
| External benchmark worker | One process at a time; 8 GiB address-space limit; separate process and independent-proof timeouts |

The complete command and compiler/input hashes are recorded in the report's reconciliation artifacts. These settings are an explicit validation configuration, not a change to all API or pass defaults. Uniform top-level options do not mean every internal route performs 22 discrepancy rollouts: the dedicated indexed continuation retains its bounded algorithm-specific settings.

```mermaid
flowchart TB
    P[Shared invocation: 1T work, 300 s] --> E[Existing family: at most 180 s]
    P --> J[Joint family: at most 100B work, 120 s]
    J --> N[Normal profile: at most 60 s]
    J --> W[Wide profile: at most 60 s]
    N -. runs before .-> W
    N --> NO[Original then indexed planners: 3B each]
    W --> WO[Original then indexed planners: 10B each]
    NO --> NC[Profile-local completion and proof slices]
    WO --> WC[Separate profile-local completion and proof slices]
    NC --> A[Retained complete candidates]
    WC --> A
    E --> A
```

These bounds nest; their advertised allowances are not additive guarantees of available work or wall time. The existing-family child leaves the joint aggregate work reserve when the parent can fund both. Final source binding and selection have an explicit reserve. Optional child deadlines can expire earlier than the outer deadline while leaving completed candidates usable.

### Memory and publication accounting

The joint planner admits its sequential and beam phases separately. A greedy expansion needs three live states: parent, retained next choice, and a newly constructed child. Completed plans are counted at their actual retained vector capacities and bounded Pauli payloads. The previous sequential state is destroyed before admission of the next phase. Beam accounting includes old/new frontier overlap and future retained outputs. If a later phase does not fit, earlier completed plans remain available. Increasing a gate cap is not treated as proof that its storage fits.

Completion keeps source/cache/final-candidate storage separate from scratch. Ranked native refinement separately accounts for temporary gates and retained output words: a smaller retained-output gate cap must not be reused as the temporary-expansion cap, or inexpensive cleanup can be suppressed before it runs. The fixed configuration funds the 262,144 temporary-gate allowance while retaining a bounded output cap derived from available scratch.

Raw publication is charged using known readout/prefix sizes before copying and proof. Optional providers reserve work for binding and publication, then charge actual returned sizes instead of billing every possible maximum-sized output. Their own nested caps remain effective rather than being unintentionally halved by a second small wrapper slice. A locally exhausted provider cannot consume its publication reserve; the enclosing deadline remains authoritative.

### Reproducibility boundary

Deterministic enumeration and fixed work bounds make search choices inspectable. Finite wall-clock deadlines can still change which complete candidates are reached, even with the same executable and input. Increasing a budget or adding a locally better candidate can change subsequent heuristic feedback. The architecture therefore preserves complete incumbents, isolates profile caches, and distinguishes final admission from search ranking. It does not promise monotonic quality under arbitrary budget changes.

## 8. Validation and code navigation

The integrated frozen build passed the **350-test native suite** and completed the fixed-configuration benchmark validation. Separately, 13 benchmark-runner policy tests passed. These tests exercise contracts and bounded behavior; they are separate from independent end-to-end QCEC checks of actual compiler output.

Key test groups are registered in [the optimizer unit-test CMake file](../cudaq/unittests/Optimizer/CMakeLists.txt):

| Tests | Coverage emphasis |
|---|---|
| [SynthesisDependenciesTest.cpp](../cudaq/unittests/Optimizer/SynthesisDependenciesTest.cpp), [SynthesisDependencyIntegrationTest.cpp](../cudaq/unittests/Optimizer/SynthesisDependencyIntegrationTest.cpp) | Exact readiness, compressed dependencies, and integration compatibility. |
| [SynthesisCandidatesTest.cpp](../cudaq/unittests/Optimizer/SynthesisCandidatesTest.cpp), [SynthesisReadoutTest.cpp](../cudaq/unittests/Optimizer/SynthesisReadoutTest.cpp) | Source replay, signed outcomes, fixed physical readouts, and proof-budget behavior. |
| [SynthesisJointReadyPhaseTest.cpp](../cudaq/unittests/Optimizer/SynthesisJointReadyPhaseTest.cpp) | Independent dense/full-phase local relations, signed axes, layouts, parity demands, and baseline retention. |
| [SynthesisJointPhasePlannerTest.cpp](../cudaq/unittests/Optimizer/SynthesisJointPhasePlannerTest.cpp), [SynthesisJointPhaseCostTest.cpp](../cudaq/unittests/Optimizer/SynthesisJointPhaseCostTest.cpp) | Whole-plan signed effects, readiness/transport, phase-specific memory, scorer misses, quotas, and analytic-cost comparisons. |
| [SynthesisCompactTransportTest.cpp](../cudaq/unittests/Optimizer/SynthesisCompactTransportTest.cpp) | Signed compact transport, spectator/reference behavior, high wires, phases and opaque rotations. |
| [SynthesisCompletionTest.cpp](../cudaq/unittests/Optimizer/SynthesisCompletionTest.cpp) | Immutable-anchor policy, every-stage admission, original/indexed ownership, fatal proof mismatch, retryable exhaustion, and frozen scorer feedback. |
| [SynthesisJointIntegrationTest.cpp](../cudaq/unittests/Optimizer/SynthesisJointIntegrationTest.cpp) | Public opt-in controls, disabled-route compatibility, acceptance policy, ordered effects and exhaustion retention. |
| [CliffordDictionaryTest.cpp](../cudaq/unittests/Optimizer/CliffordDictionaryTest.cpp), [SynthesisCliffordWindowsTest.cpp](../cudaq/unittests/Optimizer/SynthesisCliffordWindowsTest.cpp) | Existing dictionary-word compatibility, commuting-window extraction, native profiles and phase-aware proofs. |

The benchmark harness separately verifies normalized inputs, runs the frozen compiler, checks emitted/regtomem output, records all five metrics and resource usage, and performs independent equivalence checks with ordered measurement associations. All fifteen cases in the fixed-configuration reconciliation passed independent QCEC and retained or improved the previously verified best total-gate and T-count results. [The reconciliation report](pbc_synthesis_tuning.md) gives the measured results and links the configuration, hashes, commands, and proof artifacts. Every row comes from actual output of the same frozen compiler and identical top-level options; earlier draft measurements are retained separately as design evidence.

## 9. Architectural boundaries and remaining limitations

The implementation deliberately separates finite local search, whole-program planning, completion, proof, and admission. Its strongest reuse boundaries are the source/readout contracts, exact dependency index, compact signed transport, native cost adapter, source-bound refinement certificates, and completion service.

It remains a bounded heuristic portfolio. Region selection, basis/layout enumeration, finite endpoint quotas, beam width, and completion scheduling can miss a better circuit. Rank-three packets do not solve arbitrary large commuting blocks optimally. Native metric models do not include placement, routing, physical durations, error correction, or magic-state-factory scheduling. Opaque rotations remain opaque.

No earlier experimental ready-axis planner is imported into the production joint planner. Generic transport was extracted into its own module; the production engine is `SynthesisJointReadyPhase`. Further integration should preserve that boundary, rather than exposing draft-specific machinery through public APIs.

The architecture also does not collapse all successful strategies into one region-by-region optimizer. The implemented reconciliation is explicit: preserve the existing completed pipeline, run the two validated joint profiles with isolated heuristic state, compose proofs to the same normalized source, and select among complete native programs under a stated acceptance policy.
