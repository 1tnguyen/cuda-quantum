/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/
#include "Pauli/SynthesisJointPhasePlanner.h"
#include "gtest/gtest.h"
#include <complex>
#include <numbers>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
using namespace cudaq::pauli::graph::detail;

namespace {
using Complex = std::complex<double>;
using Dense = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

Dense pauliAction(const Pauli &word, const Dense &state) {
  Dense output(state.size());
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

// Direct primitive matrices, independent of production signed-Pauli algebra.
Dense apply(const Gate &gate, Dense state, bool inverse = false) {
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
    Complex phase =
        (gate.kind == Kind::Sdg) != inverse ? -imaginary : imaginary;
    for (size_t basis = 0; basis < state.size(); ++basis)
      if ((basis >> gate.a) & 1)
        state[basis] *= phase;
  } else {
    Pauli x{QubitSet::single(gate.a), {}}, z{{}, QubitSet::single(gate.a)};
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
      ADD_FAILURE() << "unsupported native primitive";
  }
  return state;
}

Dense effect(const std::vector<Gate> &word, const Pauli &observable,
             Dense state) {
  for (const auto &gate : word)
    state = apply(gate, std::move(state));
  state = pauliAction(observable, state);
  for (auto gate = word.rbegin(); gate != word.rend(); ++gate)
    state = apply(*gate, std::move(state), true);
  return state;
}

void expectEffects(unsigned width, const std::vector<Gate> &original,
                   const std::vector<Pauli> &observables,
                   const RegionalPlan &plan) {
  ASSERT_EQ(plan.observables.size(), observables.size());
  for (size_t outcome = 0; outcome < observables.size(); ++outcome)
    for (unsigned column = 0; column < (1u << width); ++column) {
      Dense input(1u << width);
      input[column] = 1;
      auto expected = effect(original, observables[outcome], input);
      auto actual = effect(plan.prefix, plan.observables[outcome], input);
      for (size_t row = 0; row < input.size(); ++row)
        EXPECT_NEAR(std::abs(actual[row] - expected[row]), 0, 8e-12);
    }
}

JointPhasePlannerCandidates solve(const MeasuredProblem &problem,
                                  JointPhasePlannerLimits limits = {}) {
  SearchBudget budget(2000000000);
  return generateJointPhasePlans(problem, {}, limits, budget);
}
} // namespace

TEST(SynthesisJointPhasePlanner,
     AffinePacketConsumesJointSpanAndPreservesEffects) {
  const std::vector<Gate> word{
      Gate::rotation({7, 6}, {1, 4}), Gate::rotation({7, 5}, {1, 4}),
      Gate::rotation({7, 3}, {-1, 4}), Gate::rotation({7, 0}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1, true}, {0, 2}, {0, 4}};
  const std::vector<unsigned> readouts{2, 0, 1};
  JointPhasePlannerLimits limits;
  limits.maxPlans = 1;
  auto result = solve({3, word, observables, readouts}, limits);
  ASSERT_EQ(result.status, OpenRegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_EQ(result.packetsSolved, 1u);
  EXPECT_EQ(result.packetRotations, 4u);
  expectEffects(4, word, observables, result.candidates[0]);
}

TEST(SynthesisJointPhasePlanner,
     NoncommutingResidualAndSignedReadoutsStayBound) {
  const std::vector<Gate> word{Gate::rotation({0, 1}, {1, 4}),
                               Gate::rotation({0, 2}, {1, 4}),
                               Gate::rotation({0, 3}, {-1, 4}),
                               Gate::rotation({1, 0, true}, {1, 4}),
                               Gate::rotation({1, 5}, {0, 1, 7, true}),
                               Gate::rotation({0, 6}, {-1, 4})};
  const std::vector<Pauli> observables{{0, 1, true}, {0, 2}, {0, 4}};
  const std::vector<unsigned> readouts{2, 0, 1};
  auto result = solve({3, word, observables, readouts});
  ASSERT_NE(result.status, OpenRegionStatus::InvalidProof) << result.error;
  ASSERT_FALSE(result.candidates.empty()) << result.error;
  for (const auto &candidate : result.candidates)
    expectEffects(4, word, observables, candidate);
  EXPECT_GT(result.packetsSolved, 1u);
  EXPECT_GT(result.transportedRows, 0u);
  EXPECT_LE(result.peakStates, 9u);
}

TEST(SynthesisJointPhasePlanner, BroadHighWirePacketUsesExactClosedFallback) {
  const std::vector<Gate> word{Gate::rotation(
      {QubitSet::single(130), QubitSet::single(0), true}, {0, 1, 3})};
  const std::vector<Pauli> observables{{{}, QubitSet::single(130), true}};
  const std::vector<unsigned> readouts{130};
  JointPhasePlannerLimits limits;
  limits.region.maxActiveWires = 1;
  limits.maxPlans = 1;
  auto result = solve({131, word, observables, readouts}, limits);
  ASSERT_EQ(result.status, OpenRegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_EQ(result.fallbackPackets, 1u);
  EXPECT_EQ(result.transportWork, 0u);
  EXPECT_EQ(result.transportedRows, 0u);
  ASSERT_EQ(result.trace.size(), 1u);
  EXPECT_TRUE(result.trace[0].physical.empty());
  EXPECT_EQ(result.trace[0].physicalSupport, 2u);
  EXPECT_EQ(result.candidates[0].observables, observables);
  EXPECT_TRUE(
      cudaq::pauli::graph::equivalent(131, word, result.candidates[0].prefix));
}

TEST(SynthesisJointPhasePlanner,
     CompleteGreedySurvivesLaterSiblingExpansionCap) {
  const std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1}, {0, 2, true}};
  const std::vector<unsigned> readouts{0, 1};
  JointPhasePlannerLimits limits;
  limits.maxExpansions = 1;
  auto result = solve({2, word, observables, readouts}, limits);
  ASSERT_EQ(result.status, OpenRegionStatus::Exhausted) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  expectEffects(3, word, observables, result.candidates[0]);
}

TEST(SynthesisJointPhasePlanner, ExhaustedFinalScorerRetainsCompleteGreedy) {
  const std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1}, {0, 2, true}};
  const std::vector<unsigned> readouts{0, 1};
  SearchBudget parent(2000000000);
  unsigned calls = 0;
  auto result = generateJointPhasePlans(
      {2, word, observables, readouts}, {}, {}, parent,
      [&](const RegionalPlan &plan,
          SearchBudget &local) -> std::optional<CostRank> {
        ++calls;
        expectEffects(3, word, observables, plan);
        EXPECT_FALSE(local.take(std::numeric_limits<uint64_t>::max()));
        return std::nullopt;
      });
  EXPECT_GT(calls, 0u);
  ASSERT_EQ(result.status, OpenRegionStatus::Exhausted) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  expectEffects(3, word, observables, result.candidates[0]);
  EXPECT_FALSE(parent.stopped());
  EXPECT_TRUE(parent.take(1));
}

TEST(SynthesisJointPhasePlanner, StorageAndMalformedInputFailTransactionally) {
  const std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1}};
  const std::vector<unsigned> readouts{0};
  JointPhasePlannerLimits limits;
  limits.maxLiveBytes = 1;
  auto memory = solve({2, word, observables, readouts}, limits);
  EXPECT_EQ(memory.status, OpenRegionStatus::Exhausted);
  EXPECT_TRUE(memory.candidates.empty());
  const std::vector<Gate> invalid{
      Gate::rotation({QubitSet::single(4095), {}}, {1, 4})};
  auto bad = solve({2, invalid, observables, readouts});
  EXPECT_EQ(bad.status, OpenRegionStatus::Unsupported);
  EXPECT_TRUE(bad.candidates.empty());
}

TEST(SynthesisJointPhasePlanner, ContextRankIncludesSignedFixedWireReadout) {
  SearchBudget budget(1000000);
  bool invalid = false;
  auto zero = joint_plan_detail::boundaryScore(2, {}, {{0, 1}}, {0}, 7, {},
                                               budget, invalid);
  auto negative = joint_plan_detail::boundaryScore(2, {}, {{0, 1, true}}, {0},
                                                   7, {}, budget, invalid);
  auto parity = joint_plan_detail::boundaryScore(2, {}, {{0, 3}}, {0}, 7, {},
                                                 budget, invalid);
  ASSERT_TRUE(zero);
  ASSERT_TRUE(negative);
  ASSERT_TRUE(parity);
  EXPECT_FALSE(invalid);
  EXPECT_DOUBLE_EQ(zero->readoutCost, 0);
  EXPECT_DOUBLE_EQ(negative->readoutCost, 1);
  EXPECT_DOUBLE_EQ(parity->readoutCost, 1);
  EXPECT_LT(zero->rank, negative->rank);
  EXPECT_LT(zero->rank, parity->rank);
}

TEST(SynthesisJointPhasePlanner,
     ContextRouteRetainsPriorGreedyAndExactEffects) {
  const std::vector<Gate> word{
      Gate::rotation({7, 6}, {1, 4}),          Gate::rotation({7, 5}, {1, 4}),
      Gate::rotation({7, 3}, {-1, 4}),         Gate::rotation({7, 0}, {1, 4}),
      Gate::rotation({0, 3, true}, {0, 1, 7}), Gate::rotation({1, 4}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1, true}, {0, 2}, {0, 4}};
  const std::vector<unsigned> readouts{2, 0, 1};
  JointPhasePlannerLimits original;
  original.beamWidth = 0;
  auto baseline = solve({3, word, observables, readouts}, original);
  ASSERT_FALSE(baseline.candidates.empty());
  JointPhasePlannerLimits contextual;
  contextual.contextScoring = true;
  contextual.region.maxCoordinateLayouts = 4;
  auto result = solve({3, word, observables, readouts}, contextual);
  ASSERT_NE(result.status, OpenRegionStatus::InvalidProof) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_TRUE(
      region_detail::samePlan(baseline.candidates[0], result.candidates[0]));
  EXPECT_GT(result.contextCalls, 0u);
  EXPECT_GT(result.contextWork, 0u);
  for (const auto &plan : result.candidates)
    expectEffects(4, word, observables, plan);
  bool scoredTrace = false;
  for (const auto &trace : result.trace) {
    scoredTrace |= trace.contextRanked;
    for (unsigned id : trace.packet) {
      ASSERT_LT(id, word.size());
      for (unsigned other : trace.packet)
        EXPECT_FALSE(word[id].pauli.anti(word[other].pauli));
    }
    for (unsigned wire : trace.physical)
      EXPECT_LT(wire, 3u);
  }
  EXPECT_TRUE(scoredTrace);
}

TEST(SynthesisJointPhasePlanner, OptionalContextExhaustionKeepsOldGreedy) {
  const std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1}, {0, 2, true}};
  const std::vector<unsigned> readouts{0, 1};
  JointPhasePlannerLimits limits;
  limits.contextScoring = true;
  limits.region.maxCoordinateLayouts = 4;
  limits.maxContextWork = 1;
  SearchBudget budget(2000000000);
  auto result = generateJointPhasePlans({2, word, observables, readouts}, {},
                                        limits, budget);
  ASSERT_NE(result.status, OpenRegionStatus::InvalidProof) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_GT(result.contextMisses, 0u);
  expectEffects(3, word, observables, result.candidates[0]);
  EXPECT_FALSE(budget.stopped());
  EXPECT_TRUE(budget.take(1));
}

TEST(SynthesisJointPhasePlanner, ReadySourceQuartetSurvivesLowWeightPool) {
  std::vector<Gate> word{
      Gate::rotation({31, 3}, {1, 4}), Gate::rotation({31, 5}, {1, 4}),
      Gate::rotation({31, 6}, {-1, 4}), Gate::rotation({31, 0}, {1, 4})};
  for (unsigned q = 5; q < 45; ++q)
    word.push_back(Gate::rotation({{}, QubitSet::single(q)}, {1, 4}));
  for (unsigned i = 0; i < 4; ++i)
    word.push_back(word[i]);
  const std::vector<Pauli> observables{{0, 1}};
  const std::vector<unsigned> readouts{0};
  JointPhasePlannerLimits limits;
  limits.maxPlans = 1;
  auto result = solve({45, word, observables, readouts}, limits);
  ASSERT_NE(result.status, OpenRegionStatus::InvalidProof) << result.error;
  ASSERT_FALSE(result.candidates.empty()) << result.error;
  ASSERT_FALSE(result.trace.empty());
  EXPECT_EQ(result.trace[0].packet,
            (std::vector<unsigned>{0, 1, 2, 3, 44, 45, 46, 47}));
  Options options;
  options.pruneUnobserved = false;
  auto readout = graph::synthesizeReadout(45, result.candidates[0].observables,
                                          readouts, options);
  ASSERT_TRUE(readout) << readout.error;
  auto completed = result.candidates[0].prefix;
  completed.insert(completed.end(), readout.gates.begin(), readout.gates.end());
  SearchBudget proof(100000000);
  MeasuredCandidateSink sink({45, word, observables, readouts}, options,
                             &proof);
  ASSERT_TRUE(sink.error().empty()) << sink.error();
  EXPECT_EQ(sink.submit(std::move(completed)).status,
            SubmissionStatus::Verified);
}

TEST(SynthesisJointPhasePlanner, WideReadoutUsesEstimateAndKeepsExactEffects) {
  const std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4}),
                               Gate::rotation({1, 0}, {1, 4})};
  std::vector<Pauli> observables;
  std::vector<unsigned> readouts;
  for (unsigned q = 0; q < 64; ++q) {
    observables.push_back({{}, QubitSet::single(q)});
    readouts.push_back(q);
  }
  JointPhasePlannerLimits limits;
  limits.contextScoring = true;
  limits.region.maxCoordinateLayouts = 4;
  auto result = solve({64, word, observables, readouts}, limits);
  ASSERT_NE(result.status, OpenRegionStatus::InvalidProof) << result.error;
  ASSERT_FALSE(result.candidates.empty()) << result.error;
  EXPECT_GT(result.estimatedContextScores, 0u);
  EXPECT_EQ(result.exactContextScores, 0u);
  bool contextual = false;
  for (const auto &trace : result.trace)
    contextual |= trace.contextRanked;
  EXPECT_TRUE(contextual);
  for (const auto &plan : result.candidates) {
    for (unsigned q = 2; q < 64; ++q)
      EXPECT_EQ(plan.observables[q], observables[q]);
    for (const auto &gate : plan.prefix) {
      if (gate.kind == Kind::Rotate) {
        EXPECT_TRUE(gate.pauli.x.fits(2));
        EXPECT_TRUE(gate.pauli.z.fits(2));
      } else if (gate.kind != Kind::Phase) {
        EXPECT_LT(gate.a, 2u);
        if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
            gate.kind == Kind::Swap)
          EXPECT_LT(gate.b, 2u);
      }
    }
    RegionalPlan projected{plan.prefix,
                           {plan.observables[0], plan.observables[1]}};
    expectEffects(3, word, {observables[0], observables[1]}, projected);
  }
}

TEST(SynthesisJointPhasePlanner,
     ContextualCompletePassSurvivesBeamExpansionCap) {
  const std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1, true}, {0, 2}};
  const std::vector<unsigned> readouts{0, 1};
  JointPhasePlannerLimits limits;
  limits.contextScoring = true;
  limits.region.maxCoordinateLayouts = 4;
  limits.beamWidth = 0;
  auto score = [&](const RegionalPlan &plan,
                   SearchBudget &) -> std::optional<CostRank> {
    expectEffects(3, word, observables, plan);
    auto tail = graph::synthesizeReadout(2, plan.observables, readouts, {});
    if (!tail)
      return std::nullopt;
    auto gates = plan.prefix;
    gates.insert(gates.end(), tail.gates.begin(), tail.gates.end());
    return circuitScore(2, gates, {});
  };
  SearchBudget fullBudget(2000000000);
  auto complete = generateJointPhasePlans({2, word, observables, readouts}, {},
                                          limits, fullBudget, score);
  ASSERT_EQ(complete.status, OpenRegionStatus::Complete) << complete.error;
  ASSERT_FALSE(complete.candidates.empty());
  EXPECT_EQ(complete.completeScores, 0u);
  EXPECT_LE(complete.contextCompleteScores, 1u);
  bool contextual = false;
  for (const auto &trace : complete.trace)
    contextual |= trace.contextualGreedy;
  EXPECT_TRUE(contextual);
  limits.beamWidth = 2;
  limits.maxExpansions = complete.expansions;
  SearchBudget cappedBudget(2000000000);
  auto capped = generateJointPhasePlans({2, word, observables, readouts}, {},
                                        limits, cappedBudget, score);
  EXPECT_EQ(capped.status, OpenRegionStatus::Exhausted);
  ASSERT_EQ(capped.candidates.size(), complete.candidates.size());
  EXPECT_EQ(capped.completeScores, 0u);
  EXPECT_EQ(capped.contextCompleteScores, complete.contextCompleteScores);
  for (unsigned i = 0; i < capped.candidates.size(); ++i) {
    EXPECT_TRUE(
        region_detail::samePlan(capped.candidates[i], complete.candidates[i]));
    expectEffects(3, word, observables, capped.candidates[i]);
  }
  EXPECT_FALSE(cappedBudget.stopped());
}

TEST(SynthesisJointPhasePlanner,
     TightMemoryKeepsCompletedGreedyWhenBeamCannotFit) {
  const std::vector<Gate> word{Gate::rotation({3, 3}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1, true}, {0, 2}};
  const std::vector<unsigned> readouts{0, 1};
  JointPhasePlannerLimits limits;
  limits.maxTraceEntries = 0;
  limits.maxLiveBytes = 224 * 1024;
  Options options;
  options.maxGates = 100;
  SearchBudget budget(2000000000);
  auto result = generateJointPhasePlans({2, word, observables, readouts},
                                        options, limits, budget);
  ASSERT_EQ(result.status, OpenRegionStatus::Exhausted) << result.error;
  ASSERT_FALSE(result.candidates.empty()) << result.error;
  EXPECT_TRUE(result.greedyMemory.admitted);
  // YY rotation's independently closed bootstrap needs11 native operations.
  EXPECT_GE(result.greedyMemory.gateCap, 11u);
  EXPECT_LT(result.beamMemory.gateCap, 11u);
  EXPECT_GT(result.beamMemory.retainedBytes, 0u);
  EXPECT_LE(result.greedyMemory.peakStates, 3u);
  EXPECT_LE(result.greedyMemory.reservedBytes, limits.maxLiveBytes);
  EXPECT_LE(result.beamMemory.reservedBytes, limits.maxLiveBytes);
  EXPECT_LE(result.reservedLiveBytes, limits.maxLiveBytes);
  for (const auto &plan : result.candidates)
    expectEffects(3, word, observables, plan);
  EXPECT_FALSE(budget.stopped());
}

TEST(SynthesisJointPhasePlanner,
     CallbackQuotaPreservesSmallPolicyAndSpreadsWideWork) {
  using joint_plan_detail::endpointQuota;
  using joint_plan_detail::packetCallAllowance;
  EXPECT_EQ(endpointQuota(4096, 32, 4, 2, true, true), 24u);
  EXPECT_EQ(endpointQuota(4096, 96, 4, 2, true, true), 24u);
  EXPECT_EQ(endpointQuota(4096, 1536, 4, 2, true, false), 10u);
  uint64_t calls = 4096;
  unsigned firstQuarterCalls = 0;
  for (unsigned remaining = 1536; remaining; remaining -= 4) {
    unsigned quota = endpointQuota(calls, remaining, 4, 2, true, false);
    ASSERT_LE(quota, calls);
    calls -= quota;
    if (remaining > 1152)
      firstQuarterCalls += quota;
  }
  // Spread work across the current complete path, instead of spending all4096
  // before halfway. The later beam consumes only leftovers or cheap estimates.
  EXPECT_LT(firstQuarterCalls, 4096u / 4);
  EXPECT_LE(calls, 24u);
  for (unsigned remaining = 1536; remaining; remaining -= 4)
    for (unsigned lane = 0; lane < 2; ++lane) {
      unsigned quota = endpointQuota(calls, remaining, 4, 2, false, false);
      ASSERT_LE(quota, calls);
      calls -= quota;
    }
  EXPECT_LE(calls, 4096u);
  EXPECT_EQ(packetCallAllowance(2, 8, 4, 2, true), 1u);
  EXPECT_EQ(endpointQuota(0, 8, 4, 2, true, false), 0u);
  EXPECT_EQ(endpointQuota(8, 4, 4, 0, true, true), 4u);
}

TEST(SynthesisJointPhasePlanner,
     CompletedContextSurvivesUnavailableFinalScore) {
  const std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1, true}, {0, 2}};
  const std::vector<unsigned> readouts{0, 1};
  JointPhasePlannerLimits limits;
  limits.contextScoring = true;
  limits.region.maxCoordinateLayouts = 4;
  limits.beamWidth = 0;
  auto reference = solve({2, word, observables, readouts}, limits);
  ASSERT_EQ(reference.status, OpenRegionStatus::Complete) << reference.error;
  ASSERT_GE(reference.candidates.size(), 2u);
  for (unsigned mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    unsigned calls = 0;
    auto bounded = limits;
    if (mode == 2)
      bounded.maxContextCompleteScores = 0;
    SearchBudget parent(2000000000);
    auto retained = generateJointPhasePlans(
        {2, word, observables, readouts}, {}, bounded, parent,
        [&](const RegionalPlan &plan,
            SearchBudget &local) -> std::optional<CostRank> {
          ++calls;
          expectEffects(3, word, observables, plan);
          if (mode == 1)
            EXPECT_FALSE(local.take(std::numeric_limits<uint64_t>::max()));
          return std::nullopt;
        });
    EXPECT_EQ(calls, mode == 2 ? 0u : 1u);
    EXPECT_EQ(retained.status, mode == 1 ? OpenRegionStatus::Exhausted
                                         : OpenRegionStatus::Complete);
    ASSERT_GE(retained.candidates.size(), 2u) << retained.error;
    EXPECT_EQ(retained.finalScoreFallbackPlans, 1u);
    EXPECT_TRUE(region_detail::samePlan(retained.candidates[0],
                                        reference.candidates[0]));
    EXPECT_TRUE(region_detail::samePlan(retained.candidates[1],
                                        reference.candidates[1]));
    for (const auto &plan : retained.candidates)
      expectEffects(3, word, observables, plan);
    EXPECT_FALSE(parent.stopped());
    EXPECT_TRUE(parent.take(1));
  }
}
