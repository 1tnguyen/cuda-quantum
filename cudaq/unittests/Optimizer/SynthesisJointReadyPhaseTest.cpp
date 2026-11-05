/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * Licensed under the Apache License, Version 2.0.                             *
 ******************************************************************************/

#include "Pauli/SynthesisJointReadyPhase.h"
#include "gtest/gtest.h"
#include <complex>
#include <numbers>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
using namespace cudaq::pauli::graph::detail;

namespace {
using Complex = std::complex<double>;
using Vector = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

Vector pauliMatrix(const Pauli &p, const Vector &state) {
  Vector result(state.size());
  for (size_t i = 0; i < state.size(); ++i) {
    size_t j = i;
    Complex factor = p.sign ? -1. : 1.;
    for (unsigned q = 0; (size_t(1) << q) < state.size(); ++q) {
      bool bit = (i >> q) & 1;
      if (p.x.test(q))
        j ^= size_t(1) << q;
      if (p.x.test(q) && p.z.test(q))
        factor *= bit ? -imaginary : imaginary;
      else if (p.z.test(q))
        factor *= bit ? -1. : 1.;
    }
    result[j] = factor * state[i];
  }
  return result;
}

Vector matrix(const Gate &gate, Vector state) {
  if (gate.kind == Kind::Phase || gate.kind == Kind::Rotate) {
    double angle = gate.angle.symbol ? .173 * gate.angle.symbol
                                     : std::numbers::pi * gate.angle.numerator /
                                           gate.angle.denominator;
    if (gate.angle.negative)
      angle = -angle;
    if (gate.kind == Kind::Phase) {
      for (auto &x : state)
        x *= std::exp(imaginary * angle);
    } else {
      auto other = pauliMatrix(gate.pauli, state);
      for (size_t i = 0; i < state.size(); ++i)
        state[i] = std::cos(angle / 2) * state[i] -
                   imaginary * std::sin(angle / 2) * other[i];
    }
  } else if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
             gate.kind == Kind::Swap) {
    auto input = state;
    for (size_t i = 0; i < state.size(); ++i) {
      bool a = (i >> gate.a) & 1, b = (i >> gate.b) & 1;
      if (gate.kind == Kind::CX)
        state[i ^ (size_t(a) << gate.b)] = input[i];
      else if (gate.kind == Kind::CZ)
        state[i] = a && b ? -input[i] : input[i];
      else
        state[a == b ? i : i ^ (size_t(1) << gate.a) ^ (size_t(1) << gate.b)] =
            input[i];
    }
  } else if (gate.kind == Kind::S || gate.kind == Kind::Sdg) {
    for (size_t i = 0; i < state.size(); ++i)
      if ((i >> gate.a) & 1)
        state[i] *= gate.kind == Kind::S ? imaginary : -imaginary;
  } else {
    Pauli x{QubitSet::single(gate.a), {}};
    Pauli z{{}, QubitSet::single(gate.a)};
    if (gate.kind == Kind::H) {
      auto a = pauliMatrix(x, state), b = pauliMatrix(z, state);
      for (size_t i = 0; i < state.size(); ++i)
        state[i] = (a[i] + b[i]) / std::sqrt(2.);
    } else if (gate.kind == Kind::X)
      state = pauliMatrix(x, state);
    else if (gate.kind == Kind::Z)
      state = pauliMatrix(z, state);
    else if (gate.kind == Kind::Y)
      state = pauliMatrix({x.x, z.z}, state);
    else
      ADD_FAILURE() << "unexpected native gate in dense oracle";
  }
  return state;
}

void expectExact(unsigned width, const std::vector<Gate> &source,
                 const JointPhaseCandidates &result) {
  ASSERT_NE(result.status, OpenRegionStatus::InvalidProof) << result.error;
  ASSERT_FALSE(result.candidates.empty()) << result.error;
  // One untouched external reference is included. Comparing every column
  // fixes the entire unitary and its scalar phase, not only measurement data.
  const unsigned dim = 1u << (width + 1);
  for (const auto &candidate : result.candidates) {
    auto target = source;
    target.insert(target.end(), candidate.exitCliffords.begin(),
                  candidate.exitCliffords.end());
    for (unsigned column = 0; column < dim; ++column) {
      Vector a(dim), b(dim);
      a[column] = b[column] = 1;
      for (const auto &gate : target)
        a = matrix(gate, std::move(a));
      for (const auto &gate : candidate.body)
        b = matrix(gate, std::move(b));
      for (unsigned row = 0; row < dim; ++row)
        EXPECT_NEAR(std::abs(a[row] - b[row]), 0., 8e-12);
    }
  }
}

JointPhaseCandidates solve(unsigned width, const std::vector<Gate> &source,
                           JointPhaseLimits limits = {}, Options options = {}) {
  SearchBudget budget(100000000);
  return solveJointReadyPhaseRegion(width, source, options, limits, budget);
}

std::vector<Gate> allParities() {
  std::vector<Gate> source;
  for (unsigned mask = 1; mask < 8; ++mask)
    source.push_back(Gate::rotation({0, mask}, {mask & 1 ? 1 : -1, 4}));
  return source;
}

unsigned entanglers(const std::vector<Gate> &word) {
  return std::count_if(word.begin(), word.end(), [](const Gate &gate) {
    return gate.kind == Kind::CX || gate.kind == Kind::CZ ||
           gate.kind == Kind::Swap;
  });
}
} // namespace

TEST(SynthesisJointReadyPhase, FullRankThreeDemandSetHasExactFourStepOpenTour) {
  auto source = allParities();
  auto result = solve(3, source);
  ASSERT_EQ(result.status, OpenRegionStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 4u);
  EXPECT_TRUE(result.candidates[0].exitCliffords.empty());
  EXPECT_EQ(result.rank, 3u);
  unsigned best = UINT_MAX;
  for (const auto &candidate : result.candidates)
    best = std::min(best, entanglers(candidate.body));
  // Three parities are resident initially; one CX can introduce at most one
  // new parity. Four further demands therefore require at least four CXs.
  EXPECT_EQ(best, 4u);
  EXPECT_GT(result.statesVisited, 0u);
  EXPECT_LE(result.statesVisited, uint64_t(result.diagonalizers) * 21504);
  expectExact(3, source, result);
}

TEST(SynthesisJointReadyPhase, AffineFourDemandPacketUsesThreeStepOpenTour) {
  const std::vector<Gate> source{
      Gate::rotation({0, 4}, {1, 4}), Gate::rotation({0, 5}, {-1, 4}),
      Gate::rotation({0, 6}, {1, 4}), Gate::rotation({0, 7}, {1, 4})};
  auto result = solve(3, source);
  ASSERT_EQ(result.status, OpenRegionStatus::Complete) << result.error;
  unsigned best = UINT_MAX;
  for (const auto &candidate : result.candidates)
    best = std::min(best, entanglers(candidate.body));
  EXPECT_EQ(best, 3u);
  expectExact(3, source, result);
}

TEST(SynthesisJointReadyPhase,
     SignedYAffinePacketPreservesOpaqueAndScalarPhase) {
  const std::vector<Gate> source{Gate::phase({1, 7}),
                                 Gate::rotation({7, 6, true}, {1, 4}),
                                 Gate::rotation({7, 5}, {-1, 4}),
                                 Gate::rotation({7, 3, true}, {0, 1, 17}),
                                 Gate::rotation({7, 0}, {1, 3}),
                                 Gate::phase({0, 1, 9, true})};
  const auto saved = source;
  auto result = solve(3, source);
  ASSERT_GT(result.candidates.size(), 1u) << result.error;
  EXPECT_GT(result.basisTrials, 1u);
  EXPECT_TRUE(joint_phase_detail::sameWord(source, saved));
  expectExact(3, source, result);
}

TEST(SynthesisJointReadyPhase, CZOnlyProfileAndDuplicateDemandsRemainExact) {
  auto source = allParities();
  source.push_back(Gate::rotation({0, 3, true}, {0, 1, 13, true}));
  Options options;
  options.nativeCX = false;
  options.nativeCZ = true;
  options.nativeSwap = false;
  auto result = solve(3, source, {}, options);
  ASSERT_GT(result.candidates.size(), 1u) << result.error;
  for (const auto &candidate : result.candidates)
    for (const auto *word : {&candidate.body, &candidate.exitCliffords})
      for (const auto &gate : *word)
        EXPECT_NE(gate.kind, Kind::CX);
  expectExact(3, source, result);
}

TEST(SynthesisJointReadyPhase, SingleCandidatePinsTheClosedMechanicalBaseline) {
  auto source = allParities();
  JointPhaseLimits limits;
  limits.maxCandidates = 1;
  auto baseline = solve(3, source, limits);
  ASSERT_EQ(baseline.candidates.size(), 1u);
  EXPECT_TRUE(baseline.candidates[0].exitCliffords.empty());
  EXPECT_EQ(baseline.statesVisited, 0u);
  auto explored = solve(3, source);
  ASSERT_FALSE(explored.candidates.empty());
  EXPECT_TRUE(joint_phase_detail::sameWord(baseline.candidates[0].body,
                                           explored.candidates[0].body));
  expectExact(3, source, baseline);
}

TEST(SynthesisJointReadyPhase,
     StateCapRetainsCheckedBaselineAndParentAllowance) {
  auto source = allParities();
  JointPhaseLimits limits;
  limits.maxStates = 1;
  SearchBudget shared(100000000);
  auto result = solveJointReadyPhaseRegion(3, source, {}, limits, shared);
  EXPECT_EQ(result.status, OpenRegionStatus::Exhausted);
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_FALSE(shared.stopped());
  EXPECT_TRUE(shared.take());
  expectExact(3, source, result);
}

TEST(SynthesisJointReadyPhase, LateWorkExhaustionRetainsCompletedBaseline) {
  auto source = allParities();
  JointPhaseLimits baselineLimits;
  baselineLimits.maxCandidates = 1;
  auto baseline = solve(3, source, baselineLimits);
  ASSERT_EQ(baseline.candidates.size(), 1u);
  JointPhaseLimits limited;
  limited.maxWork = baseline.work + 1;
  SearchBudget shared(100000000);
  auto result = solveJointReadyPhaseRegion(3, source, {}, limited, shared);
  EXPECT_EQ(result.status, OpenRegionStatus::Exhausted);
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_TRUE(joint_phase_detail::sameWord(baseline.candidates[0].body,
                                           result.candidates[0].body));
  EXPECT_FALSE(shared.stopped());
  expectExact(3, source, result);
}

TEST(SynthesisJointReadyPhase, DeterministicTiesAndBoundedNativeFrontier) {
  auto source = allParities();
  JointPhaseLimits limits;
  limits.maxCandidates = 2;
  limits.maxBasisTrials = 6;
  limits.maxDiagonalizers = 2;
  auto first = solve(3, source, limits), second = solve(3, source, limits);
  ASSERT_EQ(first.candidates.size(), 2u);
  ASSERT_EQ(first.candidates.size(), second.candidates.size());
  EXPECT_EQ(first.work, second.work);
  EXPECT_LE(first.basisTrials, 6u);
  EXPECT_LE(first.diagonalizers, 2u);
  for (size_t i = 0; i < first.candidates.size(); ++i) {
    EXPECT_TRUE(joint_phase_detail::sameWord(first.candidates[i].body,
                                             second.candidates[i].body));
    EXPECT_TRUE(joint_phase_detail::sameWord(
        first.candidates[i].exitCliffords, second.candidates[i].exitCliffords));
  }
}

TEST(SynthesisJointReadyPhase, ScalarOnlyAndEmptyWordsHaveExactClosedOutputs) {
  for (const std::vector<Gate> &source :
       {std::vector<Gate>{},
        std::vector<Gate>{Gate::phase({1, 7}),
                          Gate::phase({0, 1, 11, true})}}) {
    auto result = solve(2, source);
    ASSERT_EQ(result.candidates.size(), 1u);
    EXPECT_EQ(result.rank, 0u);
    EXPECT_TRUE(result.candidates[0].exitCliffords.empty());
    expectExact(2, source, result);
  }
}

TEST(SynthesisJointReadyPhase,
     RejectsNoncommutationRankWidthAndMalformedAngles) {
  const std::vector<Gate> noncommuting{Gate::rotation({1, 0}, {1, 4}),
                                       Gate::rotation({0, 1}, {1, 4})};
  EXPECT_EQ(solve(1, noncommuting).status, OpenRegionStatus::Unsupported);
  std::vector<Gate> rankFour;
  for (unsigned q = 0; q < 4; ++q)
    rankFour.push_back(Gate::rotation({{}, QubitSet::single(q)}, {1, 4}));
  EXPECT_EQ(solve(4, rankFour).status, OpenRegionStatus::Unsupported);
  EXPECT_EQ(solve(17, {}).status, OpenRegionStatus::Unsupported);
  auto bad = Gate::rotation({0, 1}, {1, 4});
  bad.angle.denominator = 0;
  EXPECT_EQ(solve(1, {bad}).status, OpenRegionStatus::Unsupported);
  auto scalar = Gate::phase({1, 3});
  scalar.pauli.x = QubitSet::single(1000);
  EXPECT_EQ(solve(2, {scalar}).status, OpenRegionStatus::Unsupported);
}

TEST(SynthesisJointReadyPhase, RejectsMemoryGateAndInvalidNativeCostBounds) {
  auto source = allParities();
  JointPhaseLimits limits;
  limits.maxLiveBytes = 1;
  EXPECT_EQ(solve(3, source, limits).status, OpenRegionStatus::Exhausted);
  Options options;
  options.maxGates = 1;
  auto tooSmall = solve(3, source, {}, options);
  EXPECT_TRUE(tooSmall.candidates.empty());
  // A valid source exceeding the native-gate allowance is resource exhaustion.
  EXPECT_EQ(tooSmall.status, OpenRegionStatus::Exhausted);
  options = {};
  options.objective = static_cast<SynthesisObjective>(999);
  EXPECT_EQ(solve(3, source, {}, options).status,
            OpenRegionStatus::Unsupported);
  options = {};
  options.tCost = std::numeric_limits<double>::max();
  EXPECT_EQ(solve(3, source, {}, options).status,
            OpenRegionStatus::Unsupported);
}

TEST(SynthesisJointReadyPhase,
     OptionalStorageCapCannotDiscardFeasibleBaseline) {
  auto source = allParities();
  JointPhaseLimits limits;
  // Enough for this small closed source/proof, below the optional DP envelope.
  limits.maxLiveBytes = 512 * 1024;
  auto result = solve(3, source, limits);
  ASSERT_EQ(result.status, OpenRegionStatus::Exhausted) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_TRUE(result.candidates[0].exitCliffords.empty());
  expectExact(3, source, result);
}

TEST(SynthesisJointReadyPhase,
     HighCompactWireUsesActualPhysicalExitCoordinates) {
  const std::vector<Gate> source{
      Gate::rotation({QubitSet::single(15), {}}, {1, 4}),
      Gate::rotation({QubitSet::single(15), QubitSet::single(0), true},
                     {-1, 4})};
  auto result = solve(16, source);
  ASSERT_FALSE(result.candidates.empty()) << result.error;
  ASSERT_NE(result.status, OpenRegionStatus::InvalidProof) << result.error;
  for (const auto &candidate : result.candidates) {
    auto target = source;
    target.insert(target.end(), candidate.exitCliffords.begin(),
                  candidate.exitCliffords.end());
    EXPECT_TRUE(graph::equivalent(16, target, candidate.body));
  }
}

TEST(SynthesisJointReadyPhase,
     NaturalLayoutsPreserveControlAndUseGeneratorSupport) {
  const std::vector<Pauli> basis{{0, 18}, {19, 0}};
  const std::vector<unsigned> original{0, 1};
  auto layouts = joint_phase_detail::naturalLayouts(basis, original, 5, 4);
  ASSERT_EQ(layouts.front(), original);
  ASSERT_GT(layouts.size(), 1u);
  EXPECT_LE(layouts.size(), 4u);
  bool usesHigherWire = false;
  for (size_t i = 1; i < layouts.size(); ++i) {
    ASSERT_EQ(layouts[i].size(), basis.size());
    EXPECT_NE(layouts[i][0], layouts[i][1]);
    for (unsigned row = 0; row < basis.size(); ++row) {
      EXPECT_NE(basis[row].axis(layouts[i][row]), 0u);
      usesHigherWire |= layouts[i][row] == 4;
    }
  }
  EXPECT_TRUE(usesHigherWire);
  EXPECT_EQ(joint_phase_detail::naturalLayouts(basis, original, 5, 1),
            (std::vector<std::vector<unsigned>>{original}));
}

TEST(SynthesisJointReadyPhase,
     NaturalCoordinateLayoutsKeepExactPhysicalMapping) {
  const std::vector<Gate> source{Gate::rotation({0, 18, true}, {1, 4}),
                                 Gate::rotation({19, 0}, {0, 1, 7}),
                                 Gate::rotation({19, 18}, {-1, 4})};
  JointPhaseLimits limits;
  limits.maxCoordinateLayouts = 4;
  auto result = solve(5, source, limits);
  ASSERT_GT(result.candidates.size(), 1u) << result.error;
  EXPECT_GT(result.coordinateLayouts, result.basisTrials);
  expectExact(5, source, result);
}

TEST(SynthesisJointReadyPhase, BoundaryScorerRunsBeforeLocalBodyTruncation) {
  auto source = allParities();
  JointPhaseLimits limits;
  limits.maxDiagonalizers = 1;
  limits.maxContextScores = 128;
  SearchBudget shared(100000000);
  unsigned calls = 0;
  auto result = solveJointReadyPhaseRegion(
      3, source, {}, limits, shared,
      [&](const JointPhaseImplementation &proposal,
          SearchBudget &context) -> std::optional<CostRank> {
        if (!context.take(proposal.body.size() + 1))
          return std::nullopt;
        ++calls;
        unsigned edges = entanglers(proposal.body);
        // A boundary oracle can reject every shortest local prefix. This is
        // deliberately an injected ranking fixture, not a physical cost bound:
        // the solver must expose longer complete prefixes BEFORE pruning them.
        if (edges <= 4)
          return std::nullopt;
        return CostRank{double(edges), 0, 0, 0, 0};
      });
  ASSERT_EQ(result.status, OpenRegionStatus::Complete) << result.error;
  ASSERT_GT(result.candidates.size(), 1u);
  EXPECT_GT(calls, limits.maxCandidates);
  EXPECT_EQ(result.contextScores, calls);
  EXPECT_LE(calls, limits.maxContextScores);
  EXPECT_TRUE(result.candidates[0].exitCliffords.empty());
  for (size_t i = 1; i < result.candidates.size(); ++i)
    EXPECT_GT(entanglers(result.candidates[i].body), 4u);
  expectExact(3, source, result);
}

TEST(SynthesisJointReadyPhase, ContextChildExhaustionKeepsVerifiedBaseline) {
  auto source = allParities();
  JointPhaseLimits limits;
  SearchBudget shared(100000000);
  auto result = solveJointReadyPhaseRegion(
      3, source, {}, limits, shared,
      [](const JointPhaseImplementation &,
         SearchBudget &context) -> std::optional<CostRank> {
        EXPECT_FALSE(context.take(context.remaining() + 1));
        return std::nullopt;
      });
  EXPECT_EQ(result.status, OpenRegionStatus::Exhausted);
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_EQ(result.contextScores, 1u);
  EXPECT_FALSE(shared.stopped());
  EXPECT_TRUE(shared.take());
  expectExact(3, source, result);
}
