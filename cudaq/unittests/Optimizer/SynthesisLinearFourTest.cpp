/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisLinearFour.h"
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

std::vector<Gate> ringWord(unsigned repeats = 4) {
  std::vector<Gate> word;
  for (unsigned i = 0; i < repeats; ++i)
    for (unsigned q = 0; q < 4; ++q)
      word.push_back({Kind::CX, q, (q + 1) % 4});
  return word;
}

LinearFourDictionaryResult dictionary(const Options &options = {},
                                      const LinearFourLimits &limits = {}) {
  SearchBudget budget(100000000);
  return buildLinearFourDictionary(options, limits, budget);
}

unsigned applyBits(unsigned bits, const std::vector<Gate> &word) {
  for (const auto &gate : word) {
    bool a = (bits >> gate.a) & 1, b = (bits >> gate.b) & 1;
    if (gate.kind == Kind::CX)
      bits ^= unsigned(a) << gate.b;
    else if (gate.kind == Kind::Swap && a != b)
      bits ^= (1u << gate.a) | (1u << gate.b);
    else
      EXPECT_TRUE(gate.kind == Kind::Swap);
  }
  return bits;
}
} // namespace

TEST(SynthesisLinearFour, All20160MapsReachableAndSampledAgainstTruthTables) {
  auto built = dictionary();
  ASSERT_EQ(built.status, LinearFourStatus::Complete) << built.error;
  ASSERT_TRUE(built.dictionary);
  EXPECT_EQ(built.mapStates, 20160u);
  EXPECT_EQ(built.dictionary->reachableStates(), 20160u);
  EXPECT_LE(built.memoryBound, LinearFourLimits{}.maxTableBytes);
  EXPECT_LE(built.dictionary->storageBytes(), built.memoryBound);
  EXPECT_LE(built.peakHeapEntries, LinearFourLimits{}.maxHeapEntries);
  unsigned reachable = 0, sampled = 0;
  SearchBudget queries(20000000);
  for (unsigned key = 0; key < 65536; ++key) {
    if (!built.dictionary->contains(key))
      continue;
    ++reachable;
    if (key % 97)
      continue;
    ++sampled;
    auto word = built.dictionary->implementation(key, 128, queries);
    ASSERT_TRUE(word);
    for (unsigned bits = 0; bits < 16; ++bits) {
      unsigned expected = 0;
      for (unsigned q = 0; q < 4; ++q)
        expected |= (std::popcount(((key >> (4 * q)) & 15) & bits) & 1) << q;
      EXPECT_EQ(applyBits(bits, word->logical), expected);
    }
    expectUnitary(4, word->logical, word->native);
  }
  EXPECT_EQ(reachable, 20160u);
  EXPECT_GT(sampled, 100u);
  EXPECT_FALSE(queries.stopped());
}

TEST(SynthesisLinearFour, GenuineFourWireWindowEscapesThreeWireLocalMinimum) {
  auto source = ringWord();
  // Every consecutive three ring edges touch all four wires; a fixed
  // three-wire extractor meets a noncommuting boundary after at most two.
  SearchBudget threeBudget(5000000);
  auto three = optimizeLinearWindows(4, source, {}, {}, threeBudget);
  ASSERT_EQ(three.status, LinearWindowStatus::Unchanged) << three.error;
  ASSERT_TRUE(three.alternatives.empty());
  auto built = dictionary();
  ASSERT_TRUE(built.dictionary) << built.error;
  SearchBudget fourBudget(5000000);
  auto result = optimizeLinearFourWindows(4, source, {}, {}, built.dictionary,
                                          fourBudget);
  ASSERT_EQ(result.status, LinearFourStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  const auto &candidate = result.alternatives[0];
  EXPECT_LT(candidate.gates().size(), source.size());
  // This is not an identity-only cancellation example.
  EXPECT_NE(applyBits(1, source), 1u);
  EXPECT_FALSE(candidate.gates().empty());
  EXPECT_TRUE(candidate.matchesSource(4, source));
  EXPECT_FALSE(candidate.matchesSource(5, source));
  expectUnitary(5, source, candidate.gates());
}

TEST(SynthesisLinearFour,
     ReusableDictionaryDoesNotRebindOwnedSourceCertificates) {
  auto built = dictionary();
  ASSERT_TRUE(built.dictionary) << built.error;
  auto source = ringWord();
  auto saved = source;
  SearchBudget firstBudget(5000000);
  auto first = optimizeLinearFourWindows(4, source, {}, {}, built.dictionary,
                                         firstBudget);
  ASSERT_EQ(first.alternatives.size(), 1u);
  source.clear();
  EXPECT_TRUE(first.alternatives[0].matchesSource(4, saved));
  EXPECT_FALSE(first.alternatives[0].matchesSource(4, source));
  auto other = ringWord(5);
  SearchBudget secondBudget(5000000);
  auto second = optimizeLinearFourWindows(4, other, {}, {}, built.dictionary,
                                          secondBudget);
  ASSERT_EQ(second.alternatives.size(), 1u);
  EXPECT_TRUE(second.alternatives[0].matchesSource(4, other));
  EXPECT_FALSE(second.alternatives[0].matchesSource(4, saved));
  EXPECT_TRUE(first.alternatives[0].matchesSource(4, saved));
  expectUnitary(4, other, second.alternatives[0].gates());
  EXPECT_EQ(built.dictionary->reachableStates(), 20160u);

  Options changed;
  changed.cxCost = 2;
  SearchBudget mismatchBudget(5000000);
  auto mismatch = optimizeLinearFourWindows(4, saved, changed, {},
                                            built.dictionary, mismatchBudget);
  EXPECT_EQ(mismatch.status, LinearFourStatus::Unsupported);
  EXPECT_TRUE(mismatch.alternatives.empty());
  EXPECT_EQ(mismatchBudget.work(), 0u);
}

TEST(SynthesisLinearFour, PrefixCommutationKeepsNonlinearFutureWireOperations) {
  auto source = ringWord();
  source.insert(source.begin() + 1,
                {{Kind::CX, 4, 1}, {Kind::H, 2}, {Kind::CX, 5, 2}});
  auto built = dictionary();
  ASSERT_TRUE(built.dictionary) << built.error;
  SearchBudget budget(10000000);
  auto result =
      optimizeLinearFourWindows(6, source, {}, {}, built.dictionary, budget);
  ASSERT_EQ(result.status, LinearFourStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  EXPECT_LT(result.alternatives[0].gates().size(), source.size());
  EXPECT_GT(result.commutationChecks, 0u);
  expectUnitary(6, source, result.alternatives[0].gates());
}

TEST(SynthesisLinearFour, ScalarPhaseAndSignedOpaqueRotationsRemainExact) {
  auto source = ringWord();
  source.insert(source.begin() + 1, Gate::phase({1, 7}));
  source.insert(source.begin() + 2,
                Gate::rotation({0, 1, true}, {0, 1, 3, true}));
  auto built = dictionary();
  ASSERT_TRUE(built.dictionary) << built.error;
  SearchBudget budget(5000000);
  auto result =
      optimizeLinearFourWindows(4, source, {}, {}, built.dictionary, budget);
  ASSERT_EQ(result.status, LinearFourStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  expectUnitary(5, source, result.alternatives[0].gates());
}

TEST(SynthesisLinearFour, NativeProfilesAndSwapWeightsBindTheDictionary) {
  Options options;
  options.nativeCX = false;
  options.nativeCZ = true;
  options.swapCost = 10;
  auto built = dictionary(options);
  ASSERT_TRUE(built.dictionary) << built.error;
  // Three transpositions realize SWAP(0,3). The fourth physical wire stays a
  // live quantum output; the native implementation cannot assume its state.
  std::vector<Gate> logical{
      {Kind::Swap, 0, 1}, {Kind::Swap, 1, 3}, {Kind::Swap, 0, 1}};
  unsigned key = linear_four_detail::identity();
  for (const auto &gate : logical)
    key = linear_four_detail::advance(key, gate);
  SearchBudget proof(1000000);
  auto implementation = built.dictionary->implementation(key, 64, proof);
  ASSERT_TRUE(implementation);
  EXPECT_EQ(implementation->native.size(), 9u);
  for (const auto &gate : implementation->native)
    EXPECT_TRUE(gate.kind == Kind::H || gate.kind == Kind::CZ);
  expectUnitary(5, logical, implementation->native);
  EXPECT_TRUE(built.dictionary->matchesProfile(options));
  options.swapCost = 1;
  EXPECT_FALSE(built.dictionary->matchesProfile(options));
}

TEST(SynthesisLinearFour, HighWireCoordinatesAreRestoredWithoutNewWires) {
  auto compact = ringWord();
  std::vector<unsigned> wires{2, 67, 130, 257};
  auto source = compact;
  for (auto &gate : source) {
    gate.a = wires[gate.a];
    gate.b = wires[gate.b];
  }
  auto built = dictionary();
  ASSERT_TRUE(built.dictionary) << built.error;
  SearchBudget budget(5000000);
  auto result =
      optimizeLinearFourWindows(258, source, {}, {}, built.dictionary, budget);
  ASSERT_EQ(result.status, LinearFourStatus::Complete) << result.error;
  ASSERT_EQ(result.alternatives.size(), 1u);
  auto restored = result.alternatives[0].gates();
  for (auto &gate : restored) {
    EXPECT_NE(std::find(wires.begin(), wires.end(), gate.a), wires.end());
    EXPECT_NE(std::find(wires.begin(), wires.end(), gate.b), wires.end());
    gate = linear_window_detail::compact(gate, wires);
  }
  expectUnitary(4, compact, restored);
}

TEST(SynthesisLinearFour,
     HeapMemoryAndWorkLimitsPrecedeAllocationOrPublication) {
  for (unsigned mode = 0; mode < 3; ++mode) {
    LinearFourLimits limits;
    if (mode == 0)
      limits.maxTableBytes = 1024;
    else if (mode == 1)
      limits.maxHeapEntries = 1;
    else
      limits.maxTableWork = 1;
    SearchBudget budget(100000000);
    auto result = buildLinearFourDictionary({}, limits, budget);
    EXPECT_EQ(result.status, LinearFourStatus::Exhausted) << result.error;
    EXPECT_FALSE(result.dictionary);
    EXPECT_FALSE(budget.stopped());
    EXPECT_GE(budget.remaining(), 50000000u);
    if (mode == 0 || mode == 2)
      EXPECT_EQ(budget.work(), 0u);
  }
}

TEST(SynthesisLinearFour, ExhaustionRetainsCompleteProvedPrefixRewrites) {
  auto built = dictionary();
  ASSERT_TRUE(built.dictionary) << built.error;
  auto source = ringWord();
  source.push_back({Kind::H, 0});
  auto second = ringWord();
  source.insert(source.end(), second.begin(), second.end());
  LinearFourLimits limits;
  limits.maxRewrites = 1;
  SearchBudget budget(5000000);
  auto result = optimizeLinearFourWindows(4, source, {}, limits,
                                          built.dictionary, budget);
  EXPECT_EQ(result.status, LinearFourStatus::Exhausted);
  ASSERT_EQ(result.alternatives.size(), 1u);
  EXPECT_EQ(result.rewrites, 1u);
  EXPECT_FALSE(budget.stopped());
  EXPECT_GE(budget.remaining(), 2500000u);
  expectUnitary(5, source, result.alternatives[0].gates());

  SearchBudget tiny(10);
  result = optimizeLinearFourWindows(4, source, {}, {}, built.dictionary, tiny);
  EXPECT_EQ(result.status, LinearFourStatus::Exhausted);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_FALSE(tiny.stopped());
}

TEST(SynthesisLinearFour, ZeroCostEdgesTerminateAndCostOverflowIsUnsupported) {
  Options zero;
  zero.singleQubitCost = zero.cxCost = zero.czCost = zero.swapCost = 0;
  auto built = dictionary(zero);
  ASSERT_EQ(built.status, LinearFourStatus::Complete) << built.error;
  ASSERT_TRUE(built.dictionary);
  EXPECT_EQ(built.mapStates, 20160u);
  auto source = ringWord();
  unsigned key = linear_four_detail::identity();
  for (const auto &gate : source)
    key = linear_four_detail::advance(key, gate);
  SearchBudget proof(1000000);
  auto implementation = built.dictionary->implementation(key, 64, proof);
  ASSERT_TRUE(implementation);
  EXPECT_LT(implementation->native.size(), source.size());
  expectUnitary(4, source, implementation->native);

  Options overflow;
  overflow.singleQubitCost = overflow.cxCost = overflow.czCost =
      overflow.swapCost = std::numeric_limits<double>::max();
  SearchBudget budget(100000000);
  auto rejected = buildLinearFourDictionary(overflow, {}, budget);
  EXPECT_EQ(rejected.status, LinearFourStatus::Unsupported);
  EXPECT_FALSE(rejected.dictionary);
  EXPECT_FALSE(budget.stopped());
}
