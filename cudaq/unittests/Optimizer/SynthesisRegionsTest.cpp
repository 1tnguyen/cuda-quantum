/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisRegions.h"
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

State effect(const std::vector<Gate> &word, const Pauli &observable,
             State state) {
  for (const auto &gate : word)
    state = apply(gate, std::move(state), false);
  state = pauliAction(observable, state);
  for (auto gate = word.rbegin(); gate != word.rend(); ++gate)
    state = apply(*gate, std::move(state), true);
  return state;
}

void expectEffects(unsigned width, const std::vector<Gate> &original,
                   const std::vector<Pauli> &observables,
                   const RegionalPlan &plan) {
  ASSERT_EQ(plan.observables.size(), observables.size());
  unsigned dimension = 1u << width;
  for (size_t outcome = 0; outcome < observables.size(); ++outcome)
    for (unsigned column = 0; column < dimension; ++column) {
      State input(dimension);
      input[column] = 1;
      auto a = effect(original, observables[outcome], input);
      auto b = effect(plan.prefix, plan.observables[outcome], input);
      for (unsigned row = 0; row < dimension; ++row)
        EXPECT_NEAR(std::abs(a[row] - b[row]), 0, 6e-12);
    }
}

struct Fixture {
  std::vector<Gate> word{
      Gate::rotation({0, 3}, {1, 4}),  Gate::rotation({0, 6}, {-1, 4}),
      Gate::rotation({2, 0}, {1, 4}),  Gate::rotation({3, 0}, {1, 4}),
      Gate::rotation({0, 1}, {-1, 4}), Gate::rotation({0, 5}, {0, 1, 3, true})};
  std::vector<Pauli> observables{{0, 1, true}, {0, 2}, {0, 4}};
  std::vector<unsigned> readout{2, 0, 1};
  MeasuredProblem problem() const { return {3, word, observables, readout}; }
};
} // namespace

TEST(SynthesisRegions, FullQuantumExitsComposeAcrossNoncommutingWindows) {
  Fixture f;
  auto original = f.word;
  RegionLimits limits;
  limits.maxPlans = 3;
  SearchBudget budget(200000000);
  auto result = generateRegionCandidates(f.problem(), {}, limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.regions, 3u);
  EXPECT_LE(result.candidates.size(), 3u);
  EXPECT_EQ(result.work, budget.work());
  for (const auto &plan : result.candidates) {
    expectEffects(3, original, f.observables, plan);
    // An external reference is untouched; effect equality extends to it.
    expectEffects(4, original, f.observables, plan);
  }
  for (size_t i = 0; i < original.size(); ++i) {
    EXPECT_EQ(f.word[i].pauli, original[i].pauli);
    EXPECT_EQ(f.word[i].angle, original[i].angle);
  }
  EXPECT_EQ(f.readout, (std::vector<unsigned>{2, 0, 1}));
}

TEST(SynthesisRegions, FinalReadoutKeepsOrderedSignedPhysicalOutputs) {
  Fixture f;
  RegionLimits limits;
  limits.maxPlans = 1;
  SearchBudget budget(20000000);
  auto result = generateRegionCandidates(f.problem(), {}, limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  auto plan = result.candidates[0];
  auto readout = synthesizeReadout(3, plan.observables, f.readout, {});
  ASSERT_TRUE(readout) << readout.error;
  plan.prefix.insert(plan.prefix.end(), readout.gates.begin(),
                     readout.gates.end());
  plan.observables.clear();
  for (unsigned wire : f.readout)
    plan.observables.push_back({{}, QubitSet::single(wire)});
  expectEffects(3, f.word, f.observables, plan);
}

TEST(SynthesisRegions, CompletedFirstPlanSurvivesOptionalSweepExhaustion) {
  Fixture f;
  RegionLimits limits;
  limits.maxPlans = 1;
  SearchBudget baselineBudget(20000000);
  auto baseline =
      generateRegionCandidates(f.problem(), {}, limits, baselineBudget);
  ASSERT_EQ(baseline.status, RegionStatus::Complete) << baseline.error;
  ASSERT_EQ(baseline.candidates.size(), 1u);
  limits.maxPlans = 4;
  SearchBudget limited(baseline.work);
  auto result = generateRegionCandidates(f.problem(), {}, limits, limited);
  EXPECT_EQ(result.status, RegionStatus::Exhausted);
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_TRUE(
      region_detail::samePlan(baseline.candidates[0], result.candidates[0]));
  expectEffects(3, f.word, f.observables, result.candidates[0]);
  SearchBudget none(0);
  auto empty = generateRegionCandidates(f.problem(), {}, limits, none);
  EXPECT_EQ(empty.status, RegionStatus::Exhausted);
  EXPECT_TRUE(empty.candidates.empty());
}

TEST(SynthesisRegions, WholePrefixGateCapRejectsPartialPrograms) {
  Fixture f;
  Options options;
  options.maxGates = f.word.size();
  RegionLimits limits;
  limits.maxPlans = 1;
  SearchBudget budget(20000000);
  auto result = generateRegionCandidates(f.problem(), options, limits, budget);
  EXPECT_EQ(result.status, RegionStatus::Exhausted);
  EXPECT_TRUE(result.candidates.empty());
}

TEST(SynthesisRegions, NativeProfileAppliesToEveryRegion) {
  Fixture f;
  Options options;
  options.nativeCX = false;
  options.nativeSwap = false;
  RegionLimits limits;
  limits.maxPlans = 1;
  SearchBudget budget(20000000);
  auto result = generateRegionCandidates(f.problem(), options, limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  for (const auto &plan : result.candidates) {
    expectEffects(3, f.word, f.observables, plan);
    for (const auto &gate : plan.prefix) {
      EXPECT_NE(gate.kind, Kind::CX);
      EXPECT_NE(gate.kind, Kind::Swap);
    }
  }
}

TEST(SynthesisRegions, HighWireIndicesAndPartialMeasurement) {
  QubitSet a = QubitSet::single(64), b = QubitSet::single(130);
  std::vector<Gate> word{Gate::rotation({{}, a | b}, {1, 4}),
                         Gate::rotation({b, {}}, {-1, 4}),
                         Gate::rotation({{}, a}, {0, 1, 2})};
  std::vector<Pauli> observables{{{}, b, true}};
  std::vector<unsigned> readout{64};
  MeasuredProblem source{131, word, observables, readout};
  RegionLimits limits;
  limits.maxPlans = 1;
  SearchBudget budget(20000000);
  auto result = generateRegionCandidates(source, {}, limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  auto wire = [](unsigned q) {
    EXPECT_TRUE(q == 64 || q == 130);
    return q == 64 ? 0u : 1u;
  };
  auto pauli = [&](Pauli p) {
    Pauli out{{}, {}, p.sign};
    for (unsigned q : p.x.set_bits())
      out.x |= QubitSet::single(wire(q));
    for (unsigned q : p.z.set_bits())
      out.z |= QubitSet::single(wire(q));
    return out;
  };
  auto gates = [&](std::vector<Gate> input) {
    for (auto &gate : input)
      if (gate.kind == Kind::Rotate)
        gate.pauli = pauli(gate.pauli);
      else {
        gate.a = wire(gate.a);
        if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
            gate.kind == Kind::Swap)
          gate.b = wire(gate.b);
      }
    return input;
  };
  for (const auto &plan : result.candidates)
    expectEffects(2, gates(word), {pauli(observables[0])},
                  {gates(plan.prefix), {pauli(plan.observables[0])}});
  EXPECT_EQ(readout, (std::vector<unsigned>{64}));
}

TEST(SynthesisRegions, ValidatesWidthAndReadoutContractBeforeComposition) {
  Fixture f;
  SearchBudget budget(20000000);
  auto source = f.problem();
  source.width = 2;
  EXPECT_EQ(generateRegionCandidates(source, {}, {}, budget).status,
            RegionStatus::Unsupported);
  f.readout[0] = 3;
  EXPECT_EQ(generateRegionCandidates(f.problem(), {}, {}, budget).status,
            RegionStatus::Unsupported);
  f.readout = {0, 0, 2};
  EXPECT_EQ(generateRegionCandidates(f.problem(), {}, {}, budget).status,
            RegionStatus::Unsupported);
  f.readout = {0, 1};
  f.observables = {{1, 0}, {0, 1}};
  EXPECT_EQ(generateRegionCandidates(f.problem(), {}, {}, budget).status,
            RegionStatus::Unsupported);
}

TEST(SynthesisRegions, EmptyProgramAndRepeatedOutputIdentityRemainOrdered) {
  std::vector<Gate> word;
  std::vector<Pauli> observables{{0, 1, true}, {0, 1, true}};
  std::vector<unsigned> wires{1, 1};
  MeasuredProblem source{2, word, observables, wires};
  SearchBudget budget(1000000);
  auto result = generateRegionCandidates(source, {}, {}, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete);
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_EQ(result.regions, 0u);
  EXPECT_TRUE(result.candidates[0].prefix.empty());
  EXPECT_EQ(result.candidates[0].observables, observables);
}

TEST(SynthesisRegions, BoundaryBeamPreservesEffectsAndBoundsLiveState) {
  Fixture f;
  RegionLimits limits;
  limits.maxPlans = 4;
  limits.beamWidth = 2;
  limits.commuting.maxCandidates = 4;
  limits.commuting.maxDiagonalizers = 2;
  SearchBudget budget(200000000);
  auto result = generateRegionCandidates(f.problem(), {}, limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_GT(result.beamExpansions, result.regions);
  EXPECT_GT(result.completedProposals, 1u);
  EXPECT_LE(result.candidates.size(), 4u);
  EXPECT_GT(result.beamPeakStates, 0u);
  EXPECT_LE(result.beamPeakStates, 2u * limits.beamWidth + 1);
  EXPECT_EQ(result.work, budget.work());
  for (size_t i = 0; i < result.candidates.size(); ++i) {
    const auto &plan = result.candidates[i];
    expectEffects(3, f.word, f.observables, plan);
    for (size_t j = 0; j < i; ++j)
      EXPECT_FALSE(region_detail::samePlan(plan, result.candidates[j]));
  }
}

TEST(SynthesisRegions, CompletedPlanScorerRanksBeforeTheOutputCap) {
  Fixture f;
  RegionLimits limits;
  limits.maxPlans = 2; // First greedy plus the best admitted beam completion.
  limits.beamWidth = 2;
  limits.commuting.maxCandidates = 4;
  limits.commuting.maxDiagonalizers = 2;
  unsigned scored = 0;
  std::optional<CostRank> best;
  auto completeCost = [&](const RegionalPlan &plan) -> std::optional<CostRank> {
    auto readout = synthesizeReadout(3, plan.observables, f.readout, {});
    EXPECT_TRUE(readout) << readout.error;
    if (!readout)
      return std::nullopt;
    auto gates = plan.prefix;
    gates.insert(gates.end(), readout.gates.begin(), readout.gates.end());
    return circuitScore(3, gates, {});
  };
  RegionScorer scorer = [&](const RegionalPlan &plan,
                            SearchBudget &slice) -> std::optional<CostRank> {
    if (!slice.take(plan.prefix.size() + 100))
      return std::nullopt;
    ++scored;
    auto rank = completeCost(plan);
    if (rank && (!best || *rank < *best))
      best = rank;
    return rank;
  };
  SearchBudget budget(200000000);
  auto result =
      generateRegionCandidates(f.problem(), {}, limits, budget, scorer);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_GT(scored, 0u);
  ASSERT_TRUE(best);
  ASSERT_EQ(result.candidates.size(), 2u);
  EXPECT_EQ(completeCost(result.candidates[1]), best);
  expectEffects(3, f.word, f.observables, result.candidates[1]);
  EXPECT_EQ(result.work, budget.work());
}

TEST(SynthesisRegions, RejectedBeamCompletionsDoNotReplaceFirstGreedyPlan) {
  Fixture f;
  RegionLimits first;
  first.maxPlans = 1;
  SearchBudget firstBudget(20000000);
  auto original = generateRegionCandidates(f.problem(), {}, first, firstBudget);
  ASSERT_EQ(original.status, RegionStatus::Complete) << original.error;
  ASSERT_EQ(original.candidates.size(), 1u);
  RegionLimits limits = first;
  limits.maxPlans = 4;
  limits.beamWidth = 2;
  limits.commuting.maxCandidates = 4;
  limits.commuting.maxDiagonalizers = 2;
  unsigned calls = 0;
  RegionScorer reject = [&](const RegionalPlan &,
                            SearchBudget &slice) -> std::optional<CostRank> {
    EXPECT_TRUE(slice.take());
    ++calls;
    return std::nullopt;
  };
  SearchBudget budget(200000000);
  auto result =
      generateRegionCandidates(f.problem(), {}, limits, budget, reject);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  EXPECT_GT(calls, 0u);
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_TRUE(
      region_detail::samePlan(original.candidates[0], result.candidates[0]));
}

TEST(SynthesisRegions, BeamLocalExhaustionKeepsFirstPlanAndParentReserve) {
  Fixture f;
  RegionLimits first;
  first.maxPlans = 1;
  SearchBudget firstBudget(20000000);
  auto original = generateRegionCandidates(f.problem(), {}, first, firstBudget);
  ASSERT_EQ(original.status, RegionStatus::Complete) << original.error;
  RegionLimits limits = first;
  limits.maxPlans = 4;
  limits.beamWidth = 2;
  limits.maxWork = original.work;
  SearchBudget parent(2 * original.work + 1);
  auto result = generateRegionCandidates(f.problem(), {}, limits, parent);
  EXPECT_EQ(result.status, RegionStatus::Exhausted);
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_TRUE(
      region_detail::samePlan(original.candidates[0], result.candidates[0]));
  EXPECT_EQ(result.work, original.work);
  EXPECT_FALSE(parent.stopped());
  EXPECT_TRUE(parent.take());

  limits.maxPlans = 1;
  SearchBudget single(20000000);
  auto unchanged = generateRegionCandidates(f.problem(), {}, limits, single);
  ASSERT_EQ(unchanged.status, RegionStatus::Complete);
  EXPECT_EQ(unchanged.beamExpansions, 0u);
  EXPECT_EQ(unchanged.work, original.work);
}

TEST(SynthesisRegions, BeamNativeProfileAndGateBoundsApplyToEveryChoice) {
  Fixture f;
  Options options;
  options.nativeCX = false;
  options.nativeSwap = false;
  RegionLimits limits;
  limits.maxPlans = 3;
  limits.beamWidth = 2;
  limits.commuting.maxCandidates = 3;
  limits.commuting.maxDiagonalizers = 2;
  SearchBudget budget(200000000);
  auto result = generateRegionCandidates(f.problem(), options, limits, budget);
  ASSERT_EQ(result.status, RegionStatus::Complete) << result.error;
  ASSERT_GT(result.beamExpansions, 0u);
  for (const auto &plan : result.candidates) {
    EXPECT_LE(plan.prefix.size(), options.maxGates);
    expectEffects(3, f.word, f.observables, plan);
    for (const auto &gate : plan.prefix) {
      EXPECT_NE(gate.kind, Kind::CX);
      EXPECT_NE(gate.kind, Kind::Swap);
    }
  }
  options.maxGates = f.word.size();
  SearchBudget bounded(20000000);
  auto incomplete =
      generateRegionCandidates(f.problem(), options, limits, bounded);
  EXPECT_EQ(incomplete.status, RegionStatus::Exhausted);
  EXPECT_TRUE(incomplete.candidates.empty());
}
