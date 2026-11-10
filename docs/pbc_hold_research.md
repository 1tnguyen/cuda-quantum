# PBC hold synthesis: closing the preserved-network gap

The target is **an exact gate win/tie or at most three extra gates per T saved
against preserved network**, without regressing existing hold total gates or
T count. In particular, T may never exceed the current hold result. Exact gate
wins/ties and allowed T tradeoffs are reported separately. Preserved network is feasibility
evidence and comparison data only. Its output is never a synthesis seed,
incumbent or fallback. Release synthesis has a different semantic contract
and is excluded from admission.

## Baseline and evidence

The frozen 197-case run has **61 wins, 17 ties, 88 losses and 31 compilation
failures**. The aggregate contained 196 records; the missing triangular
Ising-9 five-step record was recovered from its per-case result. The reconciled
manifest, source hashes, compiler hash and fixed options are in
`build/pbc-hold-research-2026-11-10/baseline-197.json`.

The initial local audit found 186 exact matches. A later search of locally
archived remote artifacts recovered eight more, including the missing
`gf2_256_mult` tie. All 166 previously successful inputs now have exact hash
matches; the three remaining unmatched inputs belong to old failures. Five initial local
baseline outputs matched the frozen remote outputs byte-for-byte, despite a
different local compiler binary. All subsequent synthesis experiments run
locally with up to four concurrent compiler processes during initial research.
At the user’s request, final validation now uses **at most two concurrent,
single-threaded compiler processes**, each
with an 8 GiB address-space limit. Most spots have a 180-second CPU cap; the
HHL-12 follow-up used the existing benchmark 300-second cap and finished in
200 seconds. Its initial short-limit attempt did not complete.

Development used short spot checks. The final fixed-compiler audit covers
**all 166 previously successful cases**, with exact input/output hashes and
recounted emitted metrics. Its result is **152 wins, 14 ties and zero losses**
against preserved network, with **zero total-gate or T-count regressions against
frozen hold**. The 88 original losses become **80 wins and 8 ties**; the previous
78 wins/ties become **72 wins and 6 ties**. No T-tradeoff exception is needed.
The 31 old compilation failures were not rerun or reclassified.

The fixed compiler is
`1d413cbe5b52bcd69cf8bdd4f315825816d49c62723d1d8179edf94236e96256`.
The final audit also verifies that its saved source hashes match the current
implementation. See `current-compiler-report.json` in the evidence root for
coverage, secondary metric changes, runtime and proof scope. Earlier completed
results were retained when the user reduced concurrency from four jobs to two;
the remaining 115 cases ran with the hard two-job limit.

## Ranked findings and implemented experiments

1. **Checkpoint boundaries and completed-candidate ranking.** Three-wire closed
   regions can destroy short local realizations through frame completion.
   Added one/two-wire reconstruction routes in forward and adjoint order.
   Every block's graph and complete frame are synthesized, including individual
   wider operations. Native refinement runs before comparing complete results.
   This is the strongest demonstrated lead across arithmetic, QFT and Hamiltonians.
2. **Ranking a normal form before its completion.** On QFT-29, normalizing
   one-qubit chains alone can grow the word, causing premature rejection.
   Keeping that exact intermediate through native cleanup reduces the pilot
   from 13,870 to 13,825 gates, versus preserved network's 13,851, with unchanged
   T. The minimal regression fixture grows from eight gates to ten before
   cleanup reaches seven. Added a final optional route, with exact ring-matrix
   replay for each chain, so it cannot redirect the earlier successful search.
   The forward integrated QFT-29 result is 13,825 gates and independently
   verified. A reverse-direction pilot also closes the two-gate residual
   on FT QFT-18: 8,105 → 8,085 versus 8,103, with unchanged T. Both directions
   now follow the previous searches, each with a separate bounded allowance.
   The two-direction integrated compiler reaches **8,085/3,855 T** for FT
   QFT-18, **14,943/7,122 T** for FT QFT-29, and **13,808/6,528 T** for
   transpiled QFT-29. The QFT-18 result passes independent equivalence checking.
   Integrated spot results are recorded below as they complete.

3. **Large completion loses its useful candidate to nested budgets.** Native
   scratch limits and repeated time reservations can suppress gate cleanup
   before it publishes a result. The deferred reverse reconstruction now
   completes checked gate cleanup under its own stage allowance, then composes
   T merging and both one-qubit normal forms. It closes the final QFT-160 and
   Ising-420 variants. The retained output cap remains 262,144 operations;
   the enclosing deadline and incumbent T ceiling remain authoritative.

4. **Incomplete cleanup.** A single cleanup invocation can stop before further
   cancellations become available. Added up to four generated-word refinement
   rounds with separate bounded work. This removes much of the Trotter gap.
   It also avoids the original adapter's scratch-admission barrier for these
   bounded generated words; larger words still need investigation.
5. **Overlapping boundaries.** Added bounded reconstruction of overlapping
   24/48-operation windows on three/four wires of the generated incumbent.
   Each local full-unitary proof extends by identity to the other wires.
   This closes residual gaps after fine reconstruction.
6. **T savings must survive reconstruction.** A diagnostic source-T ceiling
   reached Cuccaro-8/16/64 gate ties by giving back one T. The user explicitly
   rejected that tradeoff: **those results are disqualified** and the source-T
   admission code has been removed. The repair route now closes all six tested
   Cuccaro sizes (2, 4, 8, 16, 32 and 64) under the strict T ceiling.
   Subsequent candidates may retain an excess of up to three gates per T saved
   against preserved network, while still respecting old hold's T ceiling.
   A local experiment found a 25-gate/8-T generated region reconstructed in
   24 gates/7 T. Added a bounded repair route with all cheapest ready nodes,
   local T-count ranking and nonmaximal window endings. Admission is deferred
   until the incumbent's searches finish, preserving its optimization path.
   Cuccaro-16 reaches **397 gates/127 T** versus preserved network's
   **401/128**; Cuccaro-64 reaches **1,597/511** versus **1,601/512**.
7. **Repeated reconstruction cost.** Added a bounded cache of verified compact
   local words. Reuse requires exact source and width equality after hashing.
   The cache is limited to 512 entries and 65,536 stored gates. This improves
   feasibility on repetitive circuits without substituting source gates.
   Checkpoint auditing found 4,343 regions but only 254 distinct words in the
   largest triangular Ising case. Caching now covers the initial checkpoint
   routes and overlapping windows, as well as fine reconstruction.
8. **A premature cleanup deadline.** The three-second nested native deadline
   discarded large candidates before their checks completed. Letting native
   rounds use the already bounded twelve-second reconstruction route closes
   the large triangular Ising-36 gap: 159,765 → 153,501 gates versus 154,181,
   with T reduced 73,180 → 72,914. This is an integrated result; external
   large-circuit validation remains pending.
9. **Reverse six-wire windows.** A bounded generated-word route closes the
   strict-T Cuccaro-8 gap at 201 gates/63 T and reduces Cuccaro-16/64 to
   409/1,657 gates without losing T savings. Smaller closed windows missed
   these opportunities. Repeating cleanup, repartitioning and increasing to
   eight wires did not close the remaining larger Cuccaro deficits; the
   constrained T-repair experiment addresses a different obstruction.

All additions follow the established three-wire, four-wire and global routes,
retaining their order and allowances. Optional exhaustion keeps a complete
incumbent; invalid proofs remain fatal. New fine routes each receive at most
20B work/12 seconds, with parent limits authoritative. Overlap receives at most
30B work/40 seconds. These routes are limited to 262,144 stored operations.
See [the architecture](pbc_synthesis_architecture.md) for proof composition.

The final completion experiment separates temporary native expansion from retained
output capacity for large generated words. A reverse one-wire QFT-160 pilot
followed by native and both one-qubit completions reaches 87,352 gates/41,247 T
versus 87,577/41,247. The first integrated route reached only 89,290/41,461. Stage diagnostics show that repeated budget reservations inside the adapter
can discard the large initial cleanup. Running checked gate cleanup directly
under the stage allowance, then composing the adapter, recovers 87,352/41,247
in a 36-second pilot. The corresponding Ising-420 pilot reaches 49,579/23,684
versus 49,981/23,684. The revised full compiler now closes both QFT-160 variants: transpiled
87,151/41,247 versus 87,577/41,247, and original 96,632/46,029 versus
96,773/46,029. Both Ising-420 variants also close in the full compiler: original
49,579/23,684 versus 49,981/23,684, and transpiled 49,519/23,684 versus
49,901/23,684. The largest runtime is approximately 282 seconds. The integrated route does improve QSVT-6
to 63,522/30,814 and modular-adder-1024 to 2,774/1,004. Its allowance is 30B
work/90 seconds after earlier searches, with a retained cap of 262,144 operations.

## Latest benchmark results

All 166 rows below use compiler `1d413cbe5b52bcd69cf8bdd4f315825816d49c62723d1d8179edf94236e96256`.
Metrics come from `build/pbc-hold-research-2026-11-10/completion-native-spots/results.json`.
Results compare total gates against preserved network: **152 wins, 14 ties, 0 losses**.
No T-for-total-gate exception is needed. The [main benchmark table](pbc-optimization-report.md#f2-measurements)
includes all five metrics and output-specific independent verification status.

<!-- SPOT_TABLE -->

| Case | Preserve gates | PBC hold gates | Preserve T | PBC hold T | Result |
|---|---:|---:|---:|---:|---|
| feynman:adder | 106 | **101** | 32 | 31 | win |
| feynman:adder_8 | 621 | **541** | 173 | 173 | win |
| feynman:barenco_tof_10 | 342 | **286** | 100 | 100 | win |
| feynman:barenco_tof_3 | 46 | **38** | 16 | 16 | win |
| feynman:barenco_tof_4 | 88 | **72** | 28 | 28 | win |
| feynman:barenco_tof_5 | 130 | **109** | 40 | 40 | win |
| feynman:bigadder | 209 | **205** | 64 | 62 | win |
| feynman:csla_mux_3 | 156 | **152** | 62 | 62 | win |
| feynman:csum_mux_9 | 308 | **252** | 84 | 84 | win |
| feynman:gf2_10_mult | 1088 | **1087** | 410 | 410 | win |
| feynman:gf2_128_mult | 165019 | **165018** | 65664 | 65664 | win |
| feynman:gf2_16_mult | 2703 | **2683** | 1040 | 1040 | win |
| feynman:gf2_256_mult | 657723 | **657723** | 262400 | 262400 | tie |
| feynman:gf2_32_mult | 10562 | **10550** | 4128 | 4128 | win |
| feynman:gf2_4_mult | 193 | **184** | 68 | 68 | win |
| feynman:gf2_5_mult | 302 | **294** | 115 | 115 | win |
| feynman:gf2_64_mult | 41547 | **41520** | 16448 | 16448 | win |
| feynman:gf2_6_mult | 411 | **400** | 150 | 150 | win |
| feynman:gf2_7_mult | 550 | **547** | 217 | 217 | win |
| feynman:gf2_8_mult | 705 | **696** | 264 | 264 | win |
| feynman:gf2_9_mult | 887 | **886** | 351 | 351 | win |
| feynman:grover_5 | 500 | **496** | 166 | 166 | win |
| feynman:ham15_high | 3410 | **3120** | 1019 | 1016 | win |
| feynman:ham15_low | 364 | **350** | 97 | 97 | win |
| feynman:ham15_med | 771 | **654** | 211 | 212 | win |
| feynman:hwb6 | 212 | **194** | 75 | 72 | win |
| feynman:hwb8 | 11583 | **11053** | 3517 | 3509 | win |
| feynman:mod5_4 | 37 | **24** | 7 | 8 | win |
| feynman:mod_adder_1024 | 2780 | **2774** | 1011 | 1004 | win |
| feynman:mod_mult_55 | 92 | **90** | 35 | 35 | win |
| feynman:mod_red_21 | 196 | **187** | 73 | 73 | win |
| feynman:qcla_adder_10 | 419 | **406** | 162 | 162 | win |
| feynman:qcla_com_7 | 298 | **288** | 95 | 95 | win |
| feynman:qcla_mod_7 | 678 | **648** | 237 | 237 | win |
| feynman:qft | 411 | **406** | 195 | 195 | win |
| feynman:qft_4 | 157 | **146** | 67 | 67 | win |
| feynman:rc_adder_6 | 147 | **147** | 47 | 47 | tie |
| feynman:tof_10 | 183 | **183** | 71 | 71 | tie |
| feynman:tof_3 | 36 | **35** | 15 | 15 | win |
| feynman:tof_4 | 57 | **57** | 23 | 23 | tie |
| feynman:tof_5 | 78 | **78** | 31 | 31 | tie |
| feynman:vbe_adder_3 | 97 | **79** | 24 | 24 | win |
| ftcircuitbench:adder_adder_10q | 106 | **101** | 32 | 31 | win |
| ftcircuitbench:adder_adder_28q | 316 | **304** | 96 | 93 | win |
| ftcircuitbench:adder_adder_4q | 22 | **20** | 8 | 8 | win |
| ftcircuitbench:adder_adder_64q | 736 | **694** | 224 | 217 | win |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_128q | 116152 | **113313** | 53056 | 53056 | win |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_18q | 17405 | **17102** | 7978 | 7978 | win |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_200q | 221837 | **186211** | 83720 | 83476 | win |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_1d_72q | 65396 | **63992** | 29876 | 29876 | win |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_18q | 32419 | **31900** | 14252 | 14252 | win |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_72q | 132646 | **131451** | 56360 | 56360 | win |
| ftcircuitbench:hamiltonians_5trotter_fermi_hubbard_2d_triangular_18q | 49225 | **48354** | 21620 | 21620 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_100q | 85031 | **83653** | 40584 | 40584 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_36q | 30028 | **29453** | 14344 | 14344 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_64q | 54109 | **53011** | 25824 | 25824 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_1d_9q | 6861 | **6728** | 3274 | 3274 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_100q | 210698 | **179794** | 85008 | 83378 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_36q | 62502 | **61242** | 29692 | 29692 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_64q | 110826 | **108769** | 52644 | 52644 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_9q | 15754 | **15444** | 7486 | 7486 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_36q | 91903 | **90133** | 43626 | 43626 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_64q | 202402 | **170672** | 80640 | 79058 | win |
| ftcircuitbench:hamiltonians_5trotter_heisenberg_2d_triangular_9q | 22962 | **22512** | 10896 | 10896 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_100q | 58069 | **58012** | 27990 | 27990 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_36q | 20885 | **20860** | 10070 | 10070 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_64q | 37154 | **37112** | 17910 | 17910 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_1d_9q | 5199 | **5176** | 2510 | 2510 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_100q | 84905 | **84772** | 41080 | 41080 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_36q | 30615 | **30545** | 14800 | 14800 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_64q | 54306 | **54266** | 26300 | 26300 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_9q | 7661 | **7639** | 3700 | 3700 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_100q | 111326 | **111285** | 54080 | 54080 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_36q | 40114 | **40081** | 19480 | 19480 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_64q | 71280 | **71224** | 34620 | 34620 | win |
| ftcircuitbench:hamiltonians_5trotter_ising_2d_triangular_9q | 10058 | **10020** | 4870 | 4870 | win |
| ftcircuitbench:hamiltonians_fermi_hubbard_1d_18q | 121693 | **119964** | 54602 | 54602 | win |
| ftcircuitbench:hamiltonians_heisenberg_1d_9q | 65391 | **64957** | 31136 | 31136 | win |
| ftcircuitbench:hamiltonians_heisenberg_2d_9q | 131867 | **130979** | 62616 | 62616 | win |
| ftcircuitbench:hamiltonians_heisenberg_2d_triangular_9q | 263321 | **212598** | 96600 | 96522 | win |
| ftcircuitbench:hamiltonians_ising_1d_100q | 273560 | **229875** | 107800 | 107798 | win |
| ftcircuitbench:hamiltonians_ising_1d_36q | 79846 | **79841** | 38766 | 38766 | win |
| ftcircuitbench:hamiltonians_ising_1d_64q | 141972 | **141960** | 68950 | 68950 | win |
| ftcircuitbench:hamiltonians_ising_1d_9q | 19944 | **19941** | 9660 | 9660 | win |
| ftcircuitbench:hamiltonians_ising_2d_36q | 117032 | **116500** | 55850 | 55850 | win |
| ftcircuitbench:hamiltonians_ising_2d_64q | 250920 | **214050** | 99584 | 99544 | win |
| ftcircuitbench:hamiltonians_ising_2d_9q | 29243 | **29055** | 13934 | 13934 | win |
| ftcircuitbench:hamiltonians_ising_2d_triangular_36q | 154181 | **153503** | 72914 | 72914 | win |
| ftcircuitbench:hamiltonians_ising_2d_triangular_9q | 38529 | **38211** | 18200 | 18200 | win |
| ftcircuitbench:hhl_hhl_12q | 103534 | **102789** | 50240 | 50240 | win |
| ftcircuitbench:hhl_hhl_4q | 734 | **723** | 352 | 352 | win |
| ftcircuitbench:hhl_hhl_7q | 12295 | **12227** | 5999 | 5999 | win |
| ftcircuitbench:qft_qft_18q | 8103 | **8085** | 3855 | 3855 | win |
| ftcircuitbench:qft_qft_29q | 14972 | **14943** | 7122 | 7122 | win |
| ftcircuitbench:qft_qft_4q | 411 | **407** | 195 | 195 | win |
| ftcircuitbench:qft_qft_63q | 36231 | **36002** | 17102 | 17102 | win |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_6q | 63919 | **63522** | 30814 | 30814 | win |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_7q | 118339 | **117435** | 56729 | 56729 | win |
| ftcircuitbench:qsvt_qsvt_bandedcirculant_8q | 255293 | **209940** | 101712 | 98550 | win |
| generated_adder:cuccaro_n002 | 51 | **51** | 16 | 16 | tie |
| generated_adder:cuccaro_n004 | 101 | **97** | 32 | 31 | win |
| generated_adder:cuccaro_n008 | 201 | **197** | 64 | 63 | win |
| generated_adder:cuccaro_n016 | 401 | **397** | 128 | 127 | win |
| generated_adder:cuccaro_n032 | 801 | **797** | 256 | 255 | win |
| generated_adder:cuccaro_n064 | 1601 | **1597** | 512 | 511 | win |
| generated_qft:qft_n002 | 8 | **8** | 3 | 3 | tie |
| generated_qft:qft_n004 | 416 | **412** | 195 | 195 | win |
| generated_qft:qft_n008 | 1946 | **1920** | 919 | 919 | win |
| generated_qft:qft_n016 | 6820 | **6754** | 3237 | 3237 | win |
| generated_qft:qft_n032 | 16857 | **16727** | 7957 | 7957 | win |
| generated_qft:qft_n064 | 36879 | **36569** | 17397 | 17397 | win |
| generated_single_rotation:a00_eps1em03 | 0 | **0** | 0 | 0 | tie |
| qasmbench:qasmbench_large_adder_n118_adder_n118 | 1366 | **1314** | 416 | 403 | win |
| qasmbench:qasmbench_large_adder_n118_adder_n118_transpiled | 1366 | **1366** | 416 | 416 | tie |
| qasmbench:qasmbench_large_adder_n28_adder_n28 | 316 | **304** | 96 | 93 | win |
| qasmbench:qasmbench_large_adder_n28_adder_n28_transpiled | 316 | **316** | 96 | 96 | tie |
| qasmbench:qasmbench_large_adder_n433_adder_n433 | 5041 | **4849** | 1536 | 1488 | win |
| qasmbench:qasmbench_large_adder_n433_adder_n433_transpiled | 5041 | **5041** | 1536 | 1536 | tie |
| qasmbench:qasmbench_large_adder_n64_adder_n64 | 736 | **708** | 224 | 217 | win |
| qasmbench:qasmbench_large_adder_n64_adder_n64_transpiled | 736 | **736** | 224 | 224 | tie |
| qasmbench:qasmbench_large_ising_n34_ising_n34 | 4018 | **3991** | 1906 | 1906 | win |
| qasmbench:qasmbench_large_ising_n34_ising_n34_transpiled | 4006 | **3966** | 1900 | 1900 | win |
| qasmbench:qasmbench_large_ising_n420_ising_n420 | 49981 | **49579** | 23684 | 23684 | win |
| qasmbench:qasmbench_large_ising_n420_ising_n420_transpiled | 49901 | **49519** | 23684 | 23684 | win |
| qasmbench:qasmbench_large_ising_n42_ising_n42 | 4926 | **4881** | 2328 | 2328 | win |
| qasmbench:qasmbench_large_ising_n66_ising_n66 | 7686 | **7631** | 3646 | 3646 | win |
| qasmbench:qasmbench_large_ising_n66_ising_n66_transpiled | 7651 | **7597** | 3644 | 3644 | win |
| qasmbench:qasmbench_large_ising_n98_ising_n98 | 11567 | **11491** | 5483 | 5483 | win |
| qasmbench:qasmbench_large_ising_n98_ising_n98_transpiled | 11536 | **11439** | 5481 | 5481 | win |
| qasmbench:qasmbench_large_multiplier_n45_multiplier_n45 | 2599 | **2437** | 634 | 634 | win |
| qasmbench:qasmbench_large_multiplier_n45_multiplier_n45_transpiled | 2611 | **2424** | 634 | 634 | win |
| qasmbench:qasmbench_large_multiplier_n75_multiplier_n75 | 7307 | **6853** | 1774 | 1774 | win |
| qasmbench:qasmbench_large_multiplier_n75_multiplier_n75_transpiled | 7320 | **6825** | 1774 | 1774 | win |
| qasmbench:qasmbench_large_qft_n160_qft_n160 | 96773 | **96632** | 46029 | 46029 | win |
| qasmbench:qasmbench_large_qft_n160_qft_n160_transpiled | 87577 | **87359** | 41247 | 41247 | win |
| qasmbench:qasmbench_large_qft_n29_qft_n29 | 15052 | **14969** | 7122 | 7122 | win |
| qasmbench:qasmbench_large_qft_n29_qft_n29_transpiled | 13851 | **13808** | 6528 | 6528 | win |
| qasmbench:qasmbench_large_qft_n320_qft_n320 | 239671 | **206610** | 95439 | 94809 | win |
| qasmbench:qasmbench_large_qft_n320_qft_n320_transpiled | 216825 | **187096** | 84913 | 84903 | win |
| qasmbench:qasmbench_large_qft_n63_qft_n63 | 36199 | **36144** | 17220 | 17220 | win |
| qasmbench:qasmbench_large_qft_n63_qft_n63_transpiled | 32643 | **32556** | 15424 | 15424 | win |
| qasmbench:qasmbench_medium_bigadder_n18_bigadder_n18 | 209 | **205** | 64 | 62 | win |
| qasmbench:qasmbench_medium_bigadder_n18_bigadder_n18_transpiled | 209 | **209** | 64 | 64 | tie |
| qasmbench:qasmbench_medium_ising_n26_ising_n26 | 3045 | **3023** | 1442 | 1442 | win |
| qasmbench:qasmbench_medium_ising_n26_ising_n26_transpiled | 3035 | **3011** | 1444 | 1444 | win |
| qasmbench:qasmbench_medium_multiplier_n15_multiplier_n15 | 280 | **237** | 70 | 70 | win |
| qasmbench:qasmbench_medium_multiplier_n15_multiplier_n15_transpiled | 281 | **252** | 70 | 70 | win |
| qasmbench:qasmbench_medium_multiply_n13_multiply_n13 | 93 | **91** | 32 | 32 | win |
| qasmbench:qasmbench_medium_multiply_n13_multiply_n13_transpiled | 91 | **91** | 32 | 32 | tie |
| qasmbench:qasmbench_medium_qft_n18_qft_n18 | 8104 | **8086** | 3855 | 3855 | win |
| qasmbench:qasmbench_medium_qft_n18_qft_n18_transpiled | 7594 | **7566** | 3589 | 3589 | win |
| qasmbench:qasmbench_small_adder_n10_adder_n10 | 106 | **101** | 32 | 31 | win |
| qasmbench:qasmbench_small_adder_n10_adder_n10_transpiled | 106 | **105** | 32 | 32 | win |
| qasmbench:qasmbench_small_adder_n4_adder_n4 | 22 | **20** | 8 | 8 | win |
| qasmbench:qasmbench_small_adder_n4_adder_n4_transpiled | 22 | **20** | 8 | 8 | win |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_test_n4 | 616 | **607** | 288 | 288 | win |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_test_n4_transpiled | 654 | **645** | 310 | 310 | win |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_trotter_n4 | 14410 | **14248** | 6886 | 6886 | win |
| qasmbench:qasmbench_small_basis_trotter_n4_basis_trotter_n4_transpiled | 14816 | **14688** | 7108 | 7108 | win |
| qasmbench:qasmbench_small_ising_n10_ising_n10 | 7336 | **7286** | 3578 | 3578 | win |
| qasmbench:qasmbench_small_ising_n10_ising_n10_transpiled | 7381 | **7327** | 3600 | 3600 | win |
| qasmbench:qasmbench_small_pea_n5_pea_n5 | 550 | **547** | 260 | 260 | win |
| qasmbench:qasmbench_small_pea_n5_pea_n5_transpiled | 578 | **569** | 270 | 270 | win |
| qasmbench:qasmbench_small_qft_n4_qft_n4 | 408 | **404** | 195 | 195 | win |
| qasmbench:qasmbench_small_qft_n4_qft_n4_transpiled | 430 | **427** | 205 | 205 | win |
| qasmbench:qasmbench_small_qpe_n9_qpe_n9_transpiled | 998 | **979** | 461 | 461 | win |

<!-- END_SPOT_TABLE -->

The short win/tie controls retain or improve their total gates and T counts:
CSLA mux3 remains 152 gates, QFT-2 remains 8, transpiled adder-4 remains 20,
tof-3 improves 36 → 35, and mod5-4 improves 33 → 24. Cuccaro-64 retains the
earlier repair result, 1,597 gates/511 T. Full emitted matrices for QFT-2
and mod5-4 agree including scalar phase (maximum error below 9e-16).
No tested case increases total gates. Some candidates increase depth or
T-depth. The source-T diagnostic Cuccaro-8/16/64 ties are excluded from accepted
closures because they increase T by one relative to old hold.

## Validation and remaining work

Eleven new regression tests cover adjoint reconstruction with scalar witnesses,
optional exhaustion and fatal proof errors, exact cache binding, and the
hard incumbent T ceiling, cached checkpoint reconstruction, exact one-qubit
normalization across barriers, and separate temporary/retained native limits. All **21 GraphSynthesis.Hold*** tests pass.
Dense original-to-emitted comparisons on Cuccaro-2, adder10 and Trotter preserve
full complex matrices including scalar phase (maximum error below 1e-12).
Independent arbitrary-input QCEC checks additionally pass the first arithmetic,
Grover, QFT-8 and Ising-1d spots. These QASM checks ignore global phase; exact
phase remains an internal certificate obligation. Cuccaro source-T diagnostic
checks also pass, but they do not qualify for admission. The final repair
compiler passes independent checks on Cuccaro-4/8/16/32/64, CSLA mux3 and
transpiled QFT-18. QSVT-6 passes full dense matrix comparison including scalar
phase (maximum error 1.77e-13).

Both Ising-420 closures pass independent equivalence checking.
The short independent checks for both QFT-160 variants, five-step triangular
Ising-36, adder-433, QFT-64, five-step Fermi-Hubbard-18 and HHL-12 returned
`no_information`. Both final QFT-160 outputs also return `no_information` in
the short ZX checks. Their internal
certificates pass, but their external validation is **inconclusive**, not a
proof failure or a verified external success. The final metric audit is not an
independent whole-program equivalence proof for every case. Validation records
bind results to exact output hashes. Eighteen earlier checks are additionally
reused only after complete IR equality is established, removing solely the
nonsemantic work-count and budget-exhaustion diagnostic attributes.

`utils/pbc/research_gate.py` requires complete coverage, matching source hashes,
no new failures, nonincreasing total gates and verified hold certificates for
changed outputs. The T ceiling is always old hold. `--all-metrics` additionally enforces the
stricter five-metric policy. The accepted reference tradeoff is at most three
extra gates per T saved; this is reported separately from exact gate wins/ties. Secondary increases are always reported. The script is
an evidence comparator, not a proof runner. All three comparator tests pass.
Runtime is measured separately: the fixed-compiler run has a median of
71.95 seconds, a 95th percentile of 230.63 seconds and a maximum of 280.33
seconds. This is a material added cost, not a controlled speedup comparison.

Fixed-compiler consolidation is complete without gate/T regression, reference
loss or new compilation failure among the 166 previously successful cases.
No selected output reports parent-budget exhaustion. Reducing repeated search
and runtime is the next research lead; retain the same admission checks and
use short spot tests for each change. The 31 old compilation failures remain
a separate category, and the short inconclusive external checks remain open.

## Reproduction

Evidence root: `build/pbc-hold-research-2026-11-10/`.
`build_completion_native.py` compiles the changed synthesis translation unit and links
an isolated compiler, leaving the baseline archive and executable untouched.
`candidate-completion-native/compiler-manifest.json` identifies the immutable compiler snapshot.
`test_completion_native.py` runs the focused hold tests. `completion_native_spots.py CASE_KEY...`
runs selected current-corpus inputs; result folders retain command lines,
logs, complete metrics and output hashes. Verification entry points include
`verify_integrated.py`, `verify_wide_spots.py`, `verify_family_spots.py`,
`verify_cached_spots.py`, `verify_large_zx.py` and `verify_policy_spots.py`.
The earlier standalone pilot and its historical log remain in the evidence
root; they are exploratory evidence, not the current integrated result.
`consolidate_current.py` resumes only unfinished cases with two workers.
`collect_current_report.py` checks coverage, compiler/source/output hashes and
actual emitted metrics; `inherit_verification.py` performs the restricted IR
equality check described above. These checks do not invent missing equivalence
certificates or convert an inconclusive checker result into a success.
