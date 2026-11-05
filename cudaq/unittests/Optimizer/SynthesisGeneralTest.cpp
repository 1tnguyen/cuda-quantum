/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisGeneral.h"
#include "gtest/gtest.h"
#include <complex>
#include <numbers>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
using namespace cudaq::pauli::graph::detail;

namespace {
using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

State pauliAction(const Pauli &word, const State &state) {
  State output(state.size());
  for (size_t basis = 0; basis < state.size(); ++basis) {
    size_t target = basis;
    Complex phase = word.sign ? -1. : 1.;
    for (unsigned q = 0; (size_t(1) << q) < state.size(); ++q) {
      bool bit = (basis >> q) & 1;
      switch (word.axis(q)) {
      case 1:
        target ^= size_t(1) << q;
        break;
      case 2:
        phase *= bit ? -1. : 1.;
        break;
      case 3:
        target ^= size_t(1) << q;
        phase *= bit ? -imaginary : imaginary;
        break;
      }
    }
    output[target] = phase * state[basis];
  }
  return output;
}

/// Direct primitive matrices, independent of production Pauli transport.
State apply(const Gate &gate, State state, bool inverse) {
  if (gate.kind == Kind::Rotate || gate.kind == Kind::Phase) {
    double angle = gate.angle.symbol ? .173 * gate.angle.symbol
                                     : std::numbers::pi * gate.angle.numerator /
                                           gate.angle.denominator;
    if (gate.angle.negative != inverse)
      angle = -angle;
    if (gate.kind == Kind::Phase) {
      for (auto &amplitude : state)
        amplitude *= std::exp(imaginary * angle);
    } else {
      auto product = pauliAction(gate.pauli, state);
      for (size_t i = 0; i < state.size(); ++i)
        state[i] = std::cos(angle / 2) * state[i] -
                   imaginary * std::sin(angle / 2) * product[i];
    }
  } else if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
             gate.kind == Kind::Swap) {
    auto input = state;
    for (size_t basis = 0; basis < state.size(); ++basis) {
      bool a = (basis >> gate.a) & 1, b = (basis >> gate.b) & 1;
      if (gate.kind == Kind::CX)
        state[basis ^ (size_t(a) << gate.b)] = input[basis];
      else if (gate.kind == Kind::CZ)
        state[basis] = a && b ? -input[basis] : input[basis];
      else
        state[a == b
                  ? basis
                  : basis ^ ((size_t(1) << gate.a) | (size_t(1) << gate.b))] =
            input[basis];
    }
  } else if (gate.kind == Kind::S || gate.kind == Kind::Sdg) {
    auto phase = (gate.kind == Kind::Sdg) != inverse ? -imaginary : imaginary;
    for (size_t basis = 0; basis < state.size(); ++basis)
      if ((basis >> gate.a) & 1)
        state[basis] *= phase;
  } else {
    Pauli x{QubitSet::single(gate.a), {}};
    Pauli z{{}, QubitSet::single(gate.a)};
    if (gate.kind == Kind::H) {
      auto a = pauliAction(x, state), b = pauliAction(z, state);
      for (size_t i = 0; i < state.size(); ++i)
        state[i] = (a[i] + b[i]) / std::sqrt(2.);
    } else if (gate.kind == Kind::X)
      state = pauliAction(x, state);
    else if (gate.kind == Kind::Z)
      state = pauliAction(z, state);
    else if (gate.kind == Kind::Y)
      state = pauliAction({x.x, z.z}, state);
    else
      ADD_FAILURE() << "unsupported primitive in independent oracle";
  }
  return state;
}

void expectUnitary(unsigned width, const std::vector<Gate> &original,
                   const std::vector<Gate> &output) {
  for (unsigned column = 0; column < (1u << width); ++column) {
    State a(1u << width), b(1u << width);
    a[column] = b[column] = 1;
    for (const auto &gate : original)
      a = apply(gate, std::move(a), false);
    for (const auto &gate : output)
      b = apply(gate, std::move(b), false);
    for (unsigned row = 0; row < a.size(); ++row)
      EXPECT_NEAR(std::abs(a[row] - b[row]), 0, 8e-12);
  }
}

struct Fixture {
  std::vector<Gate> word{Gate::phase({1, 7}),
                         Gate::rotation({0, 3, true}, {1, 4}),
                         Gate::rotation({0, 6}, {-1, 4}),
                         Gate::rotation({1, 0}, {1, 4}),
                         Gate::rotation({3, 0, true}, {1, 4}),
                         Gate::rotation({0, 1}, {-1, 4}),
                         Gate::rotation({0, 5}, {0, 1, 3, true}),
                         Gate::phase({0, 1, 5, true})};
  std::vector<Pauli> observables{{0, 1, true}, {0, 2}, {0, 4}};
  std::vector<unsigned> readouts{2, 0, 1};
  MeasuredProblem problem() const { return {3, word, observables, readouts}; }
};

Options localOptions() {
  Options options;
  options.discrepancySteps = 0;
  return options;
}
} // namespace

TEST(SynthesisGeneral, SignedNoncommutingWordsPreserveFullQuantumInterface) {
  Fixture f;
  auto options = localOptions();
  // These terminal-only freedoms must never leak into an internal region.
  options.release = true;
  options.pruneUnobserved = true;
  GeneralLimits limits;
  SearchBudget budget(200000000);
  auto result = generateGeneralCandidates(f.problem(), options, limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  EXPECT_EQ(result.windows, 3u);
  EXPECT_EQ(result.completedProposals, 2u);
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.work, budget.work());
  for (const auto &plan : result.candidates) {
    EXPECT_EQ(plan.observables, f.observables);
    EXPECT_TRUE(graph::equivalent(3, f.word, plan.prefix));
    expectUnitary(3, f.word, plan.prefix);
    // Equality of the extended matrix also preserves external entanglement.
    expectUnitary(4, f.word, plan.prefix);
  }
  EXPECT_EQ(f.readouts, (std::vector<unsigned>{2, 0, 1}));
}

TEST(SynthesisGeneral, NativeProfilesRemainSeparateAndExactlyEquivalent) {
  Fixture f;
  GeneralLimits limits;
  limits.maxPlans = 1;
  for (bool nativeCX : {false, true}) {
    auto options = localOptions();
    options.nativeCX = nativeCX;
    options.nativeCZ = !nativeCX;
    options.nativeSwap = false;
    SearchBudget budget(200000000);
    auto result =
        generateGeneralCandidates(f.problem(), options, limits, budget);
    ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
    ASSERT_EQ(result.candidates.size(), 1u);
    const auto &body = result.candidates[0].prefix;
    for (const auto &gate : body) {
      EXPECT_NE(gate.kind, Kind::Swap);
      EXPECT_NE(gate.kind, nativeCX ? Kind::CZ : Kind::CX);
    }
    expectUnitary(3, f.word, body);
  }
}

TEST(SynthesisGeneral, CacheBindsSignsAndAnglesOfRepeatedCompactWords) {
  std::vector<Gate> word;
  // First two pairs are identical; subsequent pairs change respectively the
  // Pauli sign and opaque angle identity. Every window contains one rotation.
  for (unsigned i = 0; i < 4; ++i) {
    word.push_back(Gate::rotation({0, 1, i == 2}, {0, 1, i == 3 ? 9u : 7u}));
    word.push_back(Gate::rotation({1, 0}, {1, 4}));
  }
  std::vector<Pauli> observables{{0, 1}};
  std::vector<unsigned> readouts{0};
  MeasuredProblem problem{1, word, observables, readouts};
  GeneralLimits limits;
  limits.maxPlans = 1;
  auto options = localOptions();
  SearchBudget budget(20000000);
  auto cached = generateGeneralCandidates(problem, options, limits, budget);
  ASSERT_EQ(cached.status, RegionStatus::Complete) << cached.error;
  EXPECT_EQ(cached.windows, 8u);
  EXPECT_EQ(cached.cacheHits, 1u);
  EXPECT_EQ(cached.cacheEntries, 3u);
  EXPECT_EQ(cached.regionsSolved, 3u);
  EXPECT_LE(cached.cacheBytes, limits.maxCacheBytes);
  ASSERT_EQ(cached.candidates.size(), 1u);
  expectUnitary(1, word, cached.candidates[0].prefix);

  limits.maxCacheEntries = 0;
  SearchBudget uncachedBudget(20000000);
  auto uncached =
      generateGeneralCandidates(problem, options, limits, uncachedBudget);
  ASSERT_EQ(uncached.status, RegionStatus::Complete) << uncached.error;
  EXPECT_EQ(uncached.cacheHits, 0u);
  EXPECT_EQ(uncached.cacheEntries, 0u);
  EXPECT_EQ(uncached.cacheBytes, 0u);
  EXPECT_EQ(uncached.regionsSolved, 4u);
  ASSERT_EQ(uncached.candidates.size(), 1u);
  EXPECT_TRUE(
      region_detail::samePlan(cached.candidates[0], uncached.candidates[0]));
}

TEST(SynthesisGeneral, CacheReusesExactWordsAcrossPhysicalRenaming) {
  std::vector<Gate> word;
  for (unsigned q = 0; q < 3; ++q) {
    auto support = QubitSet::single(q) | QubitSet::single(q + 1);
    word.push_back(Gate::rotation({{}, support}, {1, 4}));
    word.push_back(Gate::rotation({QubitSet::single(q + 1), {}}, {-1, 4}));
  }
  std::vector<Pauli> observables{{0, 1}};
  std::vector<unsigned> readouts{3};
  GeneralLimits limits;
  limits.maxPlans = 1;
  limits.maxCacheEntries = 1;
  SearchBudget budget(20000000);
  auto result = generateGeneralCandidates({4, word, observables, readouts},
                                          localOptions(), limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  EXPECT_EQ(result.windows, 6u);
  EXPECT_EQ(result.regionsSolved, 1u);
  EXPECT_EQ(result.cacheEntries, 1u);
  EXPECT_EQ(result.cacheHits, 2u);
  ASSERT_EQ(result.candidates.size(), 1u);
  expectUnitary(4, word, result.candidates[0].prefix);
}

TEST(SynthesisGeneral, HighPhysicalWiresUseOnlyExactActiveCoordinates) {
  Fixture compact;
  std::vector<unsigned> physical{2, 67, 130};
  auto word = compact.word;
  for (auto &gate : word)
    if (gate.kind == Kind::Rotate) {
      Pauli mapped{{}, {}, gate.pauli.sign};
      for (unsigned q = 0; q < 3; ++q) {
        mapped.x.set(physical[q], gate.pauli.x.test(q));
        mapped.z.set(physical[q], gate.pauli.z.test(q));
      }
      gate.pauli = mapped;
    }
  std::vector<Pauli> observables{{{}, QubitSet::single(130), true},
                                 {{}, QubitSet::single(67)}};
  std::vector<unsigned> readouts{67, 130};
  MeasuredProblem problem{131, word, observables, readouts};
  GeneralLimits limits;
  limits.maxPlans = 1;
  limits.maxActiveWires = 3;
  SearchBudget budget(200000000);
  auto result =
      generateGeneralCandidates(problem, localOptions(), limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  const auto &plan = result.candidates[0];
  EXPECT_EQ(plan.observables, observables);
  EXPECT_TRUE(graph::equivalent(131, word, plan.prefix));
  auto body = plan.prefix;
  auto toLocal = [&](unsigned q) {
    auto at = std::find(physical.begin(), physical.end(), q);
    EXPECT_NE(at, physical.end());
    return unsigned(at - physical.begin());
  };
  for (auto &gate : body) {
    if (gate.kind == Kind::Rotate) {
      Pauli mapped{{}, {}, gate.pauli.sign};
      for (unsigned q : gate.pauli.x.set_bits())
        mapped.x.set(toLocal(q));
      for (unsigned q : gate.pauli.z.set_bits())
        mapped.z.set(toLocal(q));
      gate.pauli = mapped;
    } else if (gate.kind != Kind::Phase) {
      gate.a = toLocal(gate.a);
      if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
          gate.kind == Kind::Swap)
        gate.b = toLocal(gate.b);
    }
  }
  expectUnitary(3, compact.word, body);
}

TEST(SynthesisGeneral, WidthAndRegionLimitsNeverPublishPartialPlans) {
  Fixture f;
  GeneralLimits limits;
  limits.maxActiveWires = 2;
  SearchBudget budget(2000000);
  auto result =
      generateGeneralCandidates(f.problem(), localOptions(), limits, budget);
  EXPECT_EQ(result.status, RegionStatus::Unsupported);
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_FALSE(budget.stopped());
  limits.maxActiveWires = 32;
  limits.maxRegionRotations = 1;
  SearchBudget regionBudget(2000000);
  result = generateGeneralCandidates(f.problem(), localOptions(), limits,
                                     regionBudget);
  EXPECT_EQ(result.status, RegionStatus::Unsupported);
  EXPECT_TRUE(result.candidates.empty());

  QubitSet support;
  for (unsigned q = 0; q < 33; ++q)
    support.set(q);
  std::vector<Gate> word{Gate::rotation({support, {}}, {1, 4})};
  std::vector<Pauli> observables{{0, 1}};
  std::vector<unsigned> readouts{0};
  limits.maxActiveWires = 100;
  SearchBudget wideBudget(2000000);
  result = generateGeneralCandidates({33, word, observables, readouts},
                                     localOptions(), limits, wideBudget);
  EXPECT_EQ(result.status, RegionStatus::Unsupported);
  EXPECT_TRUE(result.candidates.empty());
}

TEST(SynthesisGeneral, ExhaustionRetainsCompletedPlanAndParentReserve) {
  std::vector<Gate> word;
  for (unsigned i = 0; i < 4; ++i) {
    word.push_back(Gate::rotation({0, 1}, {0, 1, 3}));
    word.push_back(Gate::rotation({1, 0}, {1, 4}));
  }
  std::vector<Pauli> observables{{0, 1}};
  std::vector<unsigned> readouts{0};
  MeasuredProblem problem{1, word, observables, readouts};
  GeneralLimits limits;
  limits.maxCacheEntries = 8;
  limits.maxPlans = 1;
  SearchBudget baselineBudget(2000000);
  auto baseline = generateGeneralCandidates(problem, localOptions(), limits,
                                            baselineBudget);
  ASSERT_EQ(baseline.status, RegionStatus::Complete) << baseline.error;
  ASSERT_EQ(baseline.candidates.size(), 1u);
  limits.maxPlans = 2;
  bool retained = false;
  // Vary the small deterministic allowance because nested proof reserves mean
  // measured work alone is not a sufficient scheduling allowance. This bounded
  // scan checks actual exhaustion paths, rather than assuming a charge count.
  for (unsigned factor = 8; factor <= 64 && !retained; ++factor) {
    uint64_t allowance = std::max<uint64_t>(2, baseline.work * factor / 8);
    SearchBudget limited(allowance);
    auto result =
        generateGeneralCandidates(problem, localOptions(), limits, limited);
    EXPECT_FALSE(limited.stopped());
    EXPECT_GE(limited.remaining(), allowance - allowance / 2);
    if (result.status != RegionStatus::Exhausted || result.candidates.empty())
      continue;
    EXPECT_TRUE(
        region_detail::samePlan(baseline.candidates[0], result.candidates[0]));
    expectUnitary(2, word, result.candidates[0].prefix);
    retained = true;
  }
  EXPECT_TRUE(retained);

  SearchBudget tiny(10);
  auto result =
      generateGeneralCandidates(problem, localOptions(), limits, tiny);
  EXPECT_EQ(result.status, RegionStatus::Exhausted);
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_FALSE(tiny.stopped());
  EXPECT_GE(tiny.remaining(), 5u);
}

TEST(SynthesisGeneral, ScalarOnlyAndEmptyWordsKeepExactPhase) {
  std::vector<Gate> word{Gate::phase({1, 7}), Gate::phase({0, 1, 3, true})};
  std::vector<Pauli> observables{{0, 1}};
  std::vector<unsigned> readouts{0};
  GeneralLimits limits;
  SearchBudget budget(2000000);
  auto result = generateGeneralCandidates({1, word, observables, readouts},
                                          localOptions(), limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  expectUnitary(2, word, result.candidates[0].prefix);
  word.clear();
  SearchBudget emptyBudget(2000000);
  result = generateGeneralCandidates({1, word, observables, readouts},
                                     localOptions(), limits, emptyBudget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_TRUE(result.candidates[0].prefix.empty());
  EXPECT_EQ(result.candidates[0].observables, observables);
}

TEST(SynthesisGeneral, CacheByteAndGateCapsAreExplicit) {
  Fixture f;
  GeneralLimits limits;
  limits.maxPlans = 1;
  limits.maxCacheBytes = 1;
  SearchBudget budget(200000000);
  auto result =
      generateGeneralCandidates(f.problem(), localOptions(), limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  EXPECT_EQ(result.cacheBytes, 0u);
  EXPECT_EQ(result.cacheEntries, 0u);
  ASSERT_EQ(result.candidates.size(), 1u);
  expectUnitary(3, f.word, result.candidates[0].prefix);
  auto options = localOptions();
  options.maxGates = f.word.size();
  SearchBudget gateBudget(200000000);
  result = generateGeneralCandidates(f.problem(), options, limits, gateBudget);
  EXPECT_EQ(result.status, RegionStatus::Exhausted);
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_FALSE(gateBudget.stopped());
}

TEST(SynthesisGeneral, RejectsMalformedContractsWithoutPublication) {
  Fixture f;
  GeneralLimits limits;
  f.readouts[0] = 3;
  SearchBudget budget(2000000);
  auto result =
      generateGeneralCandidates(f.problem(), localOptions(), limits, budget);
  EXPECT_EQ(result.status, RegionStatus::Unsupported);
  EXPECT_TRUE(result.candidates.empty());
  f.readouts[0] = 2;
  f.word[1].angle.denominator = 0;
  SearchBudget angleBudget(2000000);
  result = generateGeneralCandidates(f.problem(), localOptions(), limits,
                                     angleBudget);
  EXPECT_EQ(result.status, RegionStatus::Unsupported);
  EXPECT_TRUE(result.candidates.empty());
}

TEST(SynthesisGeneral, MacroModeClosesQuantumExitBeforeCostAndProof) {
  Fixture f;
  // Exactly two commuting windows, so this plan has one full-unitary region.
  f.word.resize(5);
  f.word.push_back(Gate::rotation({2, 0}, {0, 1, 3, true}));
  f.word.push_back(Gate::phase({0, 1, 5, true}));
  auto options = localOptions();
  GeneralLimits limits;
  limits.maxPlans = 1;
  SearchBudget legacyBudget(200000000);
  auto legacy =
      generateGeneralCandidates(f.problem(), options, limits, legacyBudget);
  ASSERT_EQ(legacy.status, RegionStatus::Complete) << legacy.error;
  ASSERT_EQ(legacy.candidates.size(), 1u);
  EXPECT_EQ(legacy.windows, 2u);
  EXPECT_EQ(legacy.macroRegions, 0u);
  limits.useMacros = true;
  limits.macros.maxActionPairs = 64;
  limits.macros.maxRounds = 1;
  SearchBudget macroBudget(200000000);
  auto result =
      generateGeneralCandidates(f.problem(), options, limits, macroBudget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_EQ(result.macroRegions, 1u);
  EXPECT_GT(result.macroPairs, 0u);
  EXPECT_EQ(result.work, macroBudget.work());
  EXPECT_FALSE(macroBudget.stopped());
  EXPECT_EQ(result.candidates[0].observables, f.observables);
  EXPECT_LE(circuitScore(3, result.candidates[0].prefix, options),
            circuitScore(3, legacy.candidates[0].prefix, options));
  EXPECT_TRUE(graph::equivalent(3, f.word, result.candidates[0].prefix));
  expectUnitary(4, f.word, result.candidates[0].prefix);
}

TEST(SynthesisGeneral, MacroExhaustionAndUnsupportedWidthRetainLocalLegacy) {
  Fixture f;
  f.word.resize(5);
  auto options = localOptions();
  GeneralLimits limits;
  limits.maxPlans = 1;
  SearchBudget legacyBudget(200000000);
  auto legacy =
      generateGeneralCandidates(f.problem(), options, limits, legacyBudget);
  ASSERT_EQ(legacy.status, RegionStatus::Complete) << legacy.error;
  ASSERT_EQ(legacy.candidates.size(), 1u);
  limits.useMacros = true;
  limits.macros.maxWork = 8;
  SearchBudget macroBudget(200000000);
  auto result =
      generateGeneralCandidates(f.problem(), options, limits, macroBudget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_EQ(result.macroRegions, 1u);
  EXPECT_EQ(result.macroPairs, 0u);
  EXPECT_EQ(result.macroImprovements, 0u);
  EXPECT_FALSE(macroBudget.stopped());
  EXPECT_TRUE(
      region_detail::samePlan(legacy.candidates[0], result.candidates[0]));
  expectUnitary(3, f.word, result.candidates[0].prefix);

  limits.macros.maxWork = 1000000;
  limits.macros.maxActiveWires = 2;
  SearchBudget widthBudget(200000000);
  result = generateGeneralCandidates(f.problem(), options, limits, widthBudget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_EQ(result.macroRegions, 0u);
  EXPECT_TRUE(
      region_detail::samePlan(legacy.candidates[0], result.candidates[0]));
}
