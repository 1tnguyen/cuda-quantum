/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisLinearWindows.h"
#include "gtest/gtest.h"
#include <complex>
#include <numbers>
#include <unordered_set>

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

std::vector<Gate> triangle(unsigned a = 0, unsigned b = 1, unsigned c = 2) {
  return {{Kind::CX, a, b}, {Kind::CX, a, c}, {Kind::CX, b, c}};
}

const std::vector<Gate> &output(const LinearWindowResult &result,
                                const std::vector<Gate> &source) {
  return result.alternatives.empty() ? source : result.alternatives[0].gates();
}
} // namespace

TEST(SynthesisLinearWindows, TriangleUsesExactQuantumMapAndOwnedSource) {
  auto source = triangle();
  auto saved = source;
  Options options;
  LinearWindowLimits limits;
  SearchBudget budget(2000000);
  auto result = optimizeLinearWindows(3, source, options, limits, budget);
  ASSERT_EQ(result.status, LinearWindowStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  const auto &candidate = result.alternatives[0];
  EXPECT_EQ(candidate.gates().size(), 2u);
  EXPECT_TRUE(candidate.matchesSource(3, source));
  EXPECT_FALSE(candidate.matchesSource(4, source));
  EXPECT_EQ(result.work, budget.work());
  EXPECT_GE(result.mapStates, 168u);
  EXPECT_LE(result.mapStates, 174u);
  expectUnitary(3, source, candidate.gates());
  expectUnitary(4, source, candidate.gates());
  source.clear();
  EXPECT_TRUE(candidate.matchesSource(3, saved));
  EXPECT_FALSE(candidate.matchesSource(3, source));
}

TEST(SynthesisLinearWindows, SkippedGatesNeedCommuteWithOnlySelectedPrefix) {
  // The H on wire 2 precedes the later selected CX gates and must remain there.
  // It need commute only with the selected prefix CX(0,1), as do CX(3,1)
  // and CX(4,2). A static three-wire barrier would miss this exact rewrite.
  std::vector<Gate> source{{Kind::CX, 0, 1}, {Kind::CX, 3, 1},
                           {Kind::H, 2},     {Kind::CX, 4, 2},
                           {Kind::CX, 0, 2}, {Kind::CX, 1, 2}};
  SearchBudget budget(5000000);
  auto result = optimizeLinearWindows(5, source, {}, {}, budget);
  ASSERT_EQ(result.status, LinearWindowStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  EXPECT_EQ(result.alternatives[0].gates().size(), 5u);
  EXPECT_GT(result.commutationChecks, 0u);
  expectUnitary(5, source, result.alternatives[0].gates());
  expectUnitary(6, source, result.alternatives[0].gates());
}

TEST(SynthesisLinearWindows, NoncommutingQuantumBarriersAreNotCrossed) {
  for (Kind barrier : {Kind::H, Kind::X, Kind::Y}) {
    auto source = triangle();
    source.insert(source.begin() + 1, {barrier, 0});
    SearchBudget budget(2000000);
    auto result = optimizeLinearWindows(3, source, {}, {}, budget);
    ASSERT_NE(result.status, LinearWindowStatus::InvalidProof) << result.error;
    EXPECT_EQ(result.status, LinearWindowStatus::Unchanged);
    EXPECT_TRUE(result.alternatives.empty());
    expectUnitary(4, source, output(result, source));
  }
  // Signed and opaque rotations are genuine quantum barriers as well.
  auto source = triangle();
  source.insert(source.begin() + 1,
                Gate::rotation({1, 0, true}, {0, 1, 7, true}));
  SearchBudget budget(2000000);
  auto result = optimizeLinearWindows(3, source, {}, {}, budget);
  EXPECT_EQ(result.status, LinearWindowStatus::Unchanged);
  EXPECT_TRUE(result.alternatives.empty());
  expectUnitary(3, source, output(result, source));
}

TEST(SynthesisLinearWindows, ScalarAndCommutingSignedRotationsRemainExact) {
  auto source = triangle();
  source.insert(source.begin() + 1, Gate::phase({1, 7}));
  source.insert(source.begin() + 2,
                Gate::rotation({0, 1, true}, {0, 1, 3, true}));
  SearchBudget budget(2000000);
  auto result = optimizeLinearWindows(3, source, {}, {}, budget);
  ASSERT_EQ(result.status, LinearWindowStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  EXPECT_EQ(result.alternatives[0].gates().size(), source.size() - 1);
  expectUnitary(4, source, result.alternatives[0].gates());
}

TEST(SynthesisLinearWindows,
     All168MapsMatchIndependentUnweightedShortestPaths) {
  // Independent BFS acts on all eight bit strings, not production GF(2) rows.
  // In this test every CX and native SWAP has cost one, so BFS is exact.
  struct State {
    std::array<unsigned, 8> permutation;
    std::vector<Gate> word;
  };
  auto key = [](const std::array<unsigned, 8> &permutation) {
    uint32_t bits = 0;
    for (unsigned i = 0; i < 8; ++i)
      bits |= permutation[i] << (3 * i);
    return bits;
  };
  std::vector<Gate> actions;
  for (unsigned a = 0; a < 3; ++a)
    for (unsigned b = 0; b < 3; ++b) {
      if (a == b)
        continue;
      actions.push_back({Kind::CX, a, b});
      if (a < b)
        actions.push_back({Kind::Swap, a, b});
    }
  std::vector<State> states{{{0, 1, 2, 3, 4, 5, 6, 7}, {}}};
  std::unordered_set<uint32_t> seen{key(states[0].permutation)};
  for (size_t at = 0; at < states.size(); ++at)
    for (const auto &gate : actions) {
      auto next = states[at];
      for (auto &bits : next.permutation) {
        bool a = (bits >> gate.a) & 1, b = (bits >> gate.b) & 1;
        if (gate.kind == Kind::CX)
          bits ^= unsigned(a) << gate.b;
        else if (a != b)
          bits ^= (1u << gate.a) | (1u << gate.b);
      }
      if (seen.insert(key(next.permutation)).second) {
        next.word.push_back(gate);
        states.push_back(std::move(next));
      }
    }
  ASSERT_EQ(states.size(), 168u);
  for (const auto &state : states) {
    auto source = state.word;
    source.push_back({Kind::CX, 0, 1});
    source.push_back({Kind::CX, 0, 1});
    SearchBudget budget(2000000);
    auto result = optimizeLinearWindows(3, source, {}, {}, budget);
    ASSERT_NE(result.status, LinearWindowStatus::Exhausted) << result.error;
    ASSERT_NE(result.status, LinearWindowStatus::InvalidProof) << result.error;
    const auto &best = output(result, source);
    EXPECT_EQ(best.size(), state.word.size());
    expectUnitary(3, source, best);
  }
}

TEST(SynthesisLinearWindows, NativeSwapCostsChangeTheVerifiedShortestNetwork) {
  std::vector<Gate> source{
      {Kind::Swap, 0, 1}, {Kind::Swap, 1, 2}, {Kind::Swap, 0, 1}};
  for (double swapCost : {1., 10.}) {
    Options options;
    options.swapCost = swapCost;
    SearchBudget budget(2000000);
    auto result = optimizeLinearWindows(3, source, options, {}, budget);
    ASSERT_EQ(result.status, LinearWindowStatus::Complete) << result.error;
    ASSERT_EQ(result.alternatives.size(), 1u);
    const auto &best = result.alternatives[0].gates();
    EXPECT_EQ(best.size(), swapCost == 1 ? 1u : 3u);
    if (swapCost == 10)
      for (const auto &gate : best)
        EXPECT_EQ(gate.kind, Kind::CX);
    EXPECT_LT(evaluateNativeCost(best, options).value,
              evaluateNativeCost(source, options).value);
    expectUnitary(4, source, best);
  }
}

TEST(SynthesisLinearWindows, NativeCZLoweringHasExactQuantumPhase) {
  std::vector<Gate> source{
      {Kind::Swap, 0, 1}, {Kind::Swap, 1, 2}, {Kind::Swap, 0, 1}};
  Options options;
  options.nativeCX = false;
  options.nativeCZ = true;
  options.swapCost = 10;
  SearchBudget budget(2000000);
  auto result = optimizeLinearWindows(3, source, options, {}, budget);
  ASSERT_EQ(result.status, LinearWindowStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  const auto &best = result.alternatives[0].gates();
  EXPECT_EQ(best.size(), 9u);
  for (const auto &gate : best)
    EXPECT_TRUE(gate.kind == Kind::H || gate.kind == Kind::CZ);
  expectUnitary(4, source, best);
}

TEST(SynthesisLinearWindows, HighPhysicalWiresAreMappedBackWithoutAncillas) {
  auto source = triangle(2, 67, 130);
  SearchBudget budget(2000000);
  auto result = optimizeLinearWindows(131, source, {}, {}, budget);
  ASSERT_EQ(result.status, LinearWindowStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  const auto &best = result.alternatives[0].gates();
  EXPECT_EQ(best.size(), 2u);
  EXPECT_TRUE(graph::equivalent(131, source, best));
  auto compact = best;
  std::vector<unsigned> wires{2, 67, 130};
  for (auto &gate : compact) {
    EXPECT_NE(std::find(wires.begin(), wires.end(), gate.a), wires.end());
    EXPECT_NE(std::find(wires.begin(), wires.end(), gate.b), wires.end());
    gate = linear_window_detail::compact(gate, wires);
  }
  expectUnitary(3, triangle(), compact);
}

TEST(SynthesisLinearWindows, ExhaustionRetainsOnlyCompleteCheckedReplacements) {
  std::vector<Gate> source{{Kind::CX, 0, 1}, {Kind::CX, 0, 1}, {Kind::H, 0}};
  auto later = triangle();
  source.insert(source.end(), later.begin(), later.end());
  LinearWindowLimits limits;
  limits.maxRewrites = 1;
  SearchBudget budget(2000000);
  auto result = optimizeLinearWindows(3, source, {}, limits, budget);
  EXPECT_EQ(result.status, LinearWindowStatus::Exhausted);
  ASSERT_EQ(result.alternatives.size(), 1u);
  EXPECT_EQ(result.rewrites, 1u);
  EXPECT_FALSE(budget.stopped());
  EXPECT_GE(budget.remaining(), 1000000u);
  expectUnitary(4, source, result.alternatives[0].gates());

  SearchBudget tiny(10);
  result = optimizeLinearWindows(3, source, {}, {}, tiny);
  EXPECT_EQ(result.status, LinearWindowStatus::Exhausted);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_FALSE(tiny.stopped());
  EXPECT_GE(tiny.remaining(), 5u);
}

TEST(SynthesisLinearWindows, OutputCapRejectsAWeightedButLargerReplacement) {
  auto source = triangle();
  Options options;
  options.cxCost = 10;
  options.swapCost = 10;
  LinearWindowLimits limits;
  limits.maxGates = source.size();
  SearchBudget budget(2000000);
  auto result = optimizeLinearWindows(3, source, options, limits, budget);
  EXPECT_EQ(result.status, LinearWindowStatus::Unchanged);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_FALSE(budget.stopped());
}

TEST(SynthesisLinearWindows, MalformedSourceAndNativeProfileAreRejected) {
  auto source = triangle();
  source[0].b = 3;
  SearchBudget budget(2000000);
  auto result = optimizeLinearWindows(3, source, {}, {}, budget);
  EXPECT_EQ(result.status, LinearWindowStatus::Unsupported);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_EQ(result.work, budget.work());
  source = triangle();
  Options options;
  options.nativeCX = false;
  SearchBudget profileBudget(2000000);
  result = optimizeLinearWindows(3, source, options, {}, profileBudget);
  EXPECT_EQ(result.status, LinearWindowStatus::Unsupported);
  EXPECT_TRUE(result.alternatives.empty());
}
