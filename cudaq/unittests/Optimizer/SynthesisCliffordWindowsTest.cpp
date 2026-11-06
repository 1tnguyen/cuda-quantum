/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisCliffordWindows.h"
#include "Pauli/SynthesisNativeRefinement.h"
#include "gtest/gtest.h"
#include <complex>
#include <numbers>
#include <unordered_set>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
using namespace cudaq::pauli::graph::detail;

TEST(SynthesisCliffordWindows, WholeProgramCapacityKeepsLocalMemoryAdmission) {
  const std::vector<Gate> word{{Kind::CX, 0, 1}, {Kind::CX, 0, 1}};
  for (unsigned capacity : {262144u, 200000000u}) {
    Options options;
    options.maxGates = capacity;
    CliffordWindowLimits limits;
    limits.maxGates = nativeContinuationGateLimit(options, word.size());
    SearchBudget budget(100000000);
    auto result = optimizeCliffordWindows(2, word, options, limits, budget);
    ASSERT_NE(result.status, CliffordWindowStatus::InvalidProof)
        << result.error;
    EXPECT_LE(result.memoryBound, limits.maxBytes);
    EXPECT_GT(result.windowsChecked, 0u);
  }
}

TEST(SynthesisCliffordWindows, BoundedBorrowedCleanupKeepsItsSourceBuffer) {
  std::vector<Gate> word(20000, Gate{Kind::CX, 0, 1});
  const auto source = word;
  auto *buffer = word.data();
  Options options;
  CliffordWindowLimits limits;
  limits.maxGates = 200000000;
  limits.maxBytes = 8ull << 20;
  limits.maxRewrites = 4;
  SearchBudget budget(100000000);
  auto result =
      optimizeCliffordWindows(2, word, options, limits, budget, &word);
  ASSERT_NE(result.status, CliffordWindowStatus::InvalidProof) << result.error;
  EXPECT_EQ(result.rewrites, 4u);
  EXPECT_EQ(word.size(), source.size() - 8);
  EXPECT_EQ(word.data(), buffer);
  EXPECT_LE(result.memoryBound, limits.maxBytes);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_TRUE(graph::equivalent(2, source, word));
  EXPECT_FALSE(budget.stopped());
}

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

std::vector<Gate> mechanism(unsigned a = 0, unsigned b = 1) {
  return {{Kind::H, a},
          Gate::rotation({{}, QubitSet::single(b)}, {1, 4}),
          {Kind::CZ, a, b},
          {Kind::H, a}};
}
const std::vector<Gate> &output(const CliffordWindowResult &result,
                                const std::vector<Gate> &source) {
  return result.alternatives.empty() ? source : result.alternatives[0].gates();
}
CliffordWindowResult solve(unsigned width, const std::vector<Gate> &source,
                           Options options = {},
                           CliffordWindowLimits limits = {}) {
  SearchBudget budget(100000000);
  return optimizeCliffordWindows(width, source, options, limits, budget);
}
} // namespace

TEST(SynthesisCliffordWindows, CrossesCommutingTAndOwnsExactSource) {
  auto source = mechanism();
  const auto original = source;
  auto result = solve(2, source);
  ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  auto &candidate = result.alternatives[0];
  EXPECT_EQ(tcount::cost(2, candidate.gates()).gates, 2u);
  EXPECT_TRUE(candidate.matchesSource(2, source));
  EXPECT_FALSE(candidate.matchesSource(3, source));
  EXPECT_GE(result.mapStates, 11520u);
  EXPECT_LE(result.mapStates, 11544u);
  EXPECT_GT(result.commutationChecks, 0u);
  EXPECT_GT(result.proofChecks, 0u);
  expectUnitary(2, source, candidate.gates());
  // A reference wire makes this full-channel test include entangled inputs.
  expectUnitary(3, source, candidate.gates());
  source.clear();
  EXPECT_TRUE(candidate.matchesSource(2, original));
  EXPECT_FALSE(candidate.matchesSource(2, source));
}

TEST(SynthesisCliffordWindows, ScalarAnticommutationIsNotDiscarded) {
  std::vector<Gate> source{
      {Kind::X, 0}, {Kind::Z, 0}, {Kind::X, 0}, {Kind::Z, 0}};
  auto result = solve(1, source);
  ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
  const auto &word = output(result, source);
  EXPECT_EQ(tcount::cost(1, word).gates, 0u);
  ASSERT_EQ(word.size(), 1u);
  EXPECT_EQ(word[0].kind, Kind::Phase);
  // The tableau is identity, but the actual word is -I, not I.
  expectUnitary(2, source, word);
  EXPECT_FALSE(graph::equivalent(1, source, {}));
}

TEST(SynthesisCliffordWindows, SignedOpaqueRotationsAndScalarsArePreserved) {
  auto source = mechanism();
  source[1] = Gate::rotation({0, 2, true}, {0, 1, 7, true});
  source.insert(source.begin() + 2, Gate::phase({1, 7}));
  auto result = solve(2, source);
  ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
  const auto &word = output(result, source);
  EXPECT_EQ(word.size(), 3u);
  auto rotation = std::find_if(word.begin(), word.end(), [](const Gate &g) {
    return g.kind == Kind::Rotate;
  });
  ASSERT_NE(rotation, word.end());
  EXPECT_EQ(rotation->angle, source[1].angle);
  EXPECT_EQ(rotation->pauli, source[1].pauli);
  expectUnitary(3, source, word);
}

TEST(SynthesisCliffordWindows, SignedAxisReversalIsARealBarrier) {
  for (bool negative : {false, true}) {
    std::vector<Gate> source{{Kind::X, 0},
                             Gate::rotation({0, 1, negative}, {0, 1, 3}),
                             {Kind::X, 0}};
    auto result = solve(1, source);
    EXPECT_EQ(result.status, CliffordWindowStatus::Unchanged) << result.error;
    EXPECT_TRUE(result.alternatives.empty());
    expectUnitary(2, source, output(result, source));
  }
}

TEST(SynthesisCliffordWindows, NoncommutingTStopsExtraction) {
  auto source = mechanism();
  source[1] = Gate::rotation({0, 1}, {1, 4});
  auto result = solve(2, source);
  EXPECT_EQ(result.status, CliffordWindowStatus::Unchanged) << result.error;
  EXPECT_TRUE(result.alternatives.empty());
  expectUnitary(3, source, output(result, source));
}

TEST(SynthesisCliffordWindows, SkippedGateCrossesOnlySelectedPrefix) {
  // CX21 commutes with the selected prefix H0 but not with the future CZ01.
  // It need not commute with that future operation: extraction keeps it left
  // of the completed replacement, preserving the original relative order.
  std::vector<Gate> source{
      {Kind::H, 0}, {Kind::CX, 2, 1}, {Kind::CZ, 0, 1}, {Kind::H, 0}};
  auto result = solve(3, source);
  ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
  EXPECT_EQ(tcount::cost(3, output(result, source)).gates, 2u);
  expectUnitary(4, source, output(result, source));
}

TEST(SynthesisCliffordWindows, NativeCXOnlyAndCZOnlyProfiles) {
  for (bool cx : {true, false}) {
    Options options;
    options.nativeCX = cx;
    options.nativeCZ = !cx;
    options.nativeSwap = false;
    std::vector<Gate> source;
    if (cx)
      // H0 CX10 H0 = CZ01; cancellation of two extra Hs improves natively.
      source = {{Kind::H, 0},
                {Kind::CX, 1, 0},
                {Kind::H, 0},
                {Kind::H, 1},
                {Kind::H, 1}};
    else
      source = {{Kind::H, 0},
                {Kind::CZ, 0, 1},
                {Kind::H, 0},
                {Kind::H, 1},
                {Kind::H, 1}};
    auto result = solve(2, source, options);
    ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
    const auto &word = output(result, source);
    EXPECT_EQ(tcount::cost(2, word).gates, 3u);
    for (const auto &g : word) {
      EXPECT_NE(g.kind, cx ? Kind::CZ : Kind::CX);
      EXPECT_NE(g.kind, Kind::Swap);
    }
    expectUnitary(3, source, word);
  }
}

TEST(SynthesisCliffordWindows, WeightedNativeProfileCanPreferLongerWord) {
  Options options;
  options.swapCost = 20;
  options.czCost = 20;
  std::vector<Gate> source{{Kind::Swap, 0, 1}, {Kind::H, 0}, {Kind::H, 0}};
  auto result = solve(2, source, options);
  ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
  const auto &word = output(result, source);
  EXPECT_EQ(tcount::cost(2, word).gates, 3u);
  EXPECT_DOUBLE_EQ(evaluateNativeCost(word, options).value, 3.0);
  EXPECT_EQ(std::count_if(word.begin(), word.end(),
                          [](const Gate &g) { return g.kind == Kind::CX; }),
            3);
  expectUnitary(3, source, word);
}

TEST(SynthesisCliffordWindows, DynamicPhysicalWiresAndFixedOutputAssociation) {
  auto source = mechanism(65, 130);
  auto result = solve(131, source);
  ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
  const auto &word = output(result, source);
  EXPECT_EQ(tcount::cost(131, word).gates, 2u);
  std::vector<unsigned> wires{65, 130};
  std::vector<Gate> a, b;
  for (const auto &g : source)
    a.push_back(clifford_window_detail::compact(g, wires));
  for (const auto &g : word)
    b.push_back(clifford_window_detail::compact(g, wires));
  // No wire relabeling: output correction acts on exactly the same wires.
  expectUnitary(3, a, b);
  EXPECT_TRUE(result.alternatives[0].matchesSource(131, source));
}

TEST(SynthesisCliffordWindows, WorkExhaustionNeverPublishesAnUncheckedPrefix) {
  const auto source = mechanism();
  CliffordWindowLimits limits;
  limits.maxWork = 100;
  SearchBudget parent(100000000);
  auto result = optimizeCliffordWindows(2, source, {}, limits, parent);
  EXPECT_EQ(result.status, CliffordWindowStatus::Exhausted);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_FALSE(parent.stopped());
  EXPECT_LE(parent.work(), limits.maxWork);
}

TEST(SynthesisCliffordWindows, StateCapRetainsEarlierCompleteRewrite) {
  // The one-wire X/X cancellation completes before the two-wire table is
  // requested. Its certified partial sweep remains a COMPLETE full circuit.
  auto source = mechanism(1, 2);
  source.insert(source.begin(), {{Kind::X, 0}, {Kind::X, 0}});
  CliffordWindowLimits limits;
  limits.maxStates = 24;
  auto result = solve(3, source, {}, limits);
  ASSERT_EQ(result.status, CliffordWindowStatus::Exhausted) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  EXPECT_EQ(tcount::cost(3, output(result, source)).gates, 4u);
  expectUnitary(4, source, output(result, source));
}

TEST(SynthesisCliffordWindows, UnsupportedNativeSourceAndOutputCap) {
  Options options;
  options.nativeCZ = false;
  auto source = mechanism();
  auto unsupported = solve(2, source, options);
  EXPECT_EQ(unsupported.status, CliffordWindowStatus::Unsupported);
  EXPECT_TRUE(unsupported.alternatives.empty());
  CliffordWindowLimits limits;
  limits.maxGates = 3;
  auto capped = solve(2, source, {}, limits);
  EXPECT_EQ(capped.status, CliffordWindowStatus::Unsupported);
  EXPECT_TRUE(capped.alternatives.empty());
}

TEST(SynthesisCliffordWindows, ZeroCostEdgesTerminateAndOverflowIsExplicit) {
  Options zero;
  zero.singleQubitCost = zero.cxCost = zero.czCost = zero.swapCost = 0;
  const auto source = mechanism();
  auto result = solve(2, source, zero);
  ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
  EXPECT_GE(result.mapStates, 11520u);
  EXPECT_LE(result.mapStates, 11544u);
  EXPECT_EQ(tcount::cost(2, output(result, source)).gates, 2u);
  expectUnitary(3, source, output(result, source));
  Options overflow;
  overflow.singleQubitCost = overflow.cxCost = overflow.czCost =
      overflow.swapCost = std::numeric_limits<double>::max();
  auto rejected = solve(2, source, overflow);
  EXPECT_EQ(rejected.status, CliffordWindowStatus::Unsupported);
  EXPECT_TRUE(rejected.alternatives.empty());
}

TEST(SynthesisCliffordWindows, IgnoredStoredAxesAreBoundedBeforeSourceCopy) {
  for (Kind kind : {Kind::H, Kind::Phase}) {
    auto source = mechanism();
    Gate extra{kind, 0};
    extra.pauli.x.set(100000);
    source.push_back(extra);
    auto rejected = solve(2, source);
    EXPECT_EQ(rejected.status, CliffordWindowStatus::Unsupported);
    EXPECT_TRUE(rejected.alternatives.empty());
    EXPECT_EQ(rejected.mapStates, 0u);
  }
  // Small ignored fields are retained exactly in the immutable source binding.
  auto source = mechanism();
  source[0].pauli.x.set(1);
  auto result = solve(2, source);
  ASSERT_EQ(result.status, CliffordWindowStatus::Complete) << result.error;
  EXPECT_TRUE(result.alternatives[0].matchesSource(2, source));
  source[0].pauli.x = {};
  EXPECT_FALSE(result.alternatives[0].matchesSource(2, source));
}

TEST(SynthesisCliffordWindows, SourceOwnerAndScratchCountAgainstMemoryCap) {
  CliffordWindowLimits limits;
  limits.maxBytes = 1;
  auto result = solve(2, mechanism(), {}, limits);
  EXPECT_EQ(result.status, CliffordWindowStatus::Exhausted);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_GT(result.memoryBound, limits.maxBytes);
  EXPECT_EQ(result.mapStates, 0u);
}
