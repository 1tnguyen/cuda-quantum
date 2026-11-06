/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisNativeRefinement.h"
#include "gtest/gtest.h"
#include <bit>
#include <cmath>
#include <complex>
#include <numbers>

using namespace cudaq::pauli;
namespace synthesis = cudaq::pauli::graph::detail;

TEST(SynthesisNativeRefinement, WholeProgramCapacityKeepsLocalCleanupPolicy) {
  for (unsigned gates : {20000u, 40000u}) {
    std::vector<Gate> word(gates, {Kind::CX, 0, 1});
    unsigned expectedCalls = gates == 20000 ? 1 : 0;
    for (unsigned capacity : {262144u, 200000000u}) {
      cudaq::pauli::graph::Options options;
      options.maxGates = capacity;
      auto limits =
          synthesis::boundedNativeRefinementLimits(options, word.size());
      limits.gateCountRounds = 1;
      limits.enableTCount = limits.enableLinearWindows = false;
      limits.maxBackendWork = 2000000;
      SearchBudget budget(50000000);
      auto result =
          synthesis::refineSynthesizedNative(2, word, options, limits, budget);
      ASSERT_NE(result.status, synthesis::NativeRefinementStatus::InvalidProof)
          << result.error;
      // Crossing the local temporary bound must have the same effect for
      // both whole-program capacities, rather than consuming linear cleanup's
      // allowance through newly admitted backend work.
      EXPECT_EQ(result.gateCountCalls, expectedCalls);
    }
  }
}

namespace {
using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

size_t mask(const QubitSet &bits) {
  size_t value = 0;
  for (unsigned bit : bits.set_bits()) {
    EXPECT_LT(bit, 6u);
    value |= size_t(1) << bit;
  }
  return value;
}

State pauli(const State &state, const Pauli &p) {
  size_t x = mask(p.x), z = mask(p.z);
  State result(state.size());
  for (size_t i = 0; i < state.size(); ++i) {
    Complex phase = std::pow(imaginary, std::popcount(x & z));
    if ((std::popcount(i & z) & 1) != p.sign)
      phase = -phase;
    result[i ^ x] = phase * state[i];
  }
  return result;
}

State apply(State state, const std::vector<Gate> &gates) {
  for (const auto &gate : gates) {
    State before = state;
    if (gate.kind == Kind::Rotate || gate.kind == Kind::Phase) {
      double angle = gate.angle.symbol
                         ? 0.137 * gate.angle.symbol
                         : std::numbers::pi * gate.angle.numerator /
                               gate.angle.denominator;
      if (gate.angle.negative)
        angle = -angle;
      auto transformed =
          gate.kind == Kind::Rotate ? pauli(state, gate.pauli) : state;
      for (size_t i = 0; i < state.size(); ++i)
        state[i] = gate.kind == Kind::Phase
                       ? std::exp(imaginary * angle) * before[i]
                       : std::cos(angle / 2) * before[i] -
                             imaginary * std::sin(angle / 2) * transformed[i];
      continue;
    }
    size_t a = size_t(1) << gate.a, b = size_t(1) << gate.b;
    for (size_t i = 0; i < state.size(); ++i) {
      switch (gate.kind) {
      case Kind::H:
        state[i] =
            ((i & a) ? before[i ^ a] - before[i] : before[i] + before[i ^ a]) /
            std::sqrt(2.0);
        break;
      case Kind::S:
      case Kind::Sdg:
        state[i] = before[i] *
                   ((i & a) ? (gate.kind == Kind::S ? imaginary : -imaginary)
                            : Complex{1, 0});
        break;
      case Kind::X:
        state[i] = before[i ^ a];
        break;
      case Kind::Y:
        state[i] = before[i ^ a] * ((i & a) ? imaginary : -imaginary);
        break;
      case Kind::Z:
        state[i] = before[i] * ((i & a) ? -1.0 : 1.0);
        break;
      case Kind::CX:
        state[i] = before[(i & a) ? i ^ b : i];
        break;
      case Kind::CZ:
        state[i] = before[i] * ((i & a) && (i & b) ? -1.0 : 1.0);
        break;
      case Kind::Swap:
        state[i] = before[bool(i & a) != bool(i & b) ? i ^ a ^ b : i];
        break;
      default:
        ADD_FAILURE() << "unsupported test gate";
      }
    }
  }
  return state;
}

void checkUnitary(unsigned width, const std::vector<Gate> &before,
                  const std::vector<Gate> &after) {
  for (size_t column = 0; column < (size_t(1) << width); ++column) {
    State state(size_t(1) << width);
    state[column] = 1;
    auto a = apply(state, before), b = apply(state, after);
    for (size_t row = 0; row < state.size(); ++row)
      EXPECT_LT(std::abs(a[row] - b[row]), 1e-10);
  }
}

} // namespace

TEST(SynthesisNativeRefinement, ScalarPhaseAndOwnedSourceArePreserved) {
  std::vector<Gate> source{
      {Kind::X, 0}, {Kind::Y, 0}, {Kind::X, 0}, {Kind::Y, 0}};
  auto saved = source;
  graph::Options options;
  synthesis::NativeRefinementLimits limits;
  limits.enableTCount = false;
  SearchBudget budget(100000000);
  auto result =
      synthesis::refineSynthesizedNative(1, source, options, limits, budget);
  ASSERT_FALSE(result.alternatives.empty()) << result.error;
  bool scalarOnly = false;
  for (const auto &candidate : result.alternatives) {
    EXPECT_TRUE(candidate.matchesSource(1, source));
    EXPECT_FALSE(candidate.matchesSource(2, source));
    checkUnitary(1, source, candidate.gates());
    scalarOnly |= candidate.cost().metrics.gates == 0;
    EXPECT_EQ(candidate.proof(), synthesis::CertifiedNativeRefinement::Proof::
                                     CheckedNativeUnitaryRewritesAndLowering);
  }
  EXPECT_TRUE(scalarOnly);
  source.clear();
  EXPECT_TRUE(result.alternatives.front().matchesSource(1, saved));
  EXPECT_FALSE(result.alternatives.front().matchesSource(1, source));
}

TEST(SynthesisNativeRefinement, CheckedTStageUsesOnlyTheGeneratedNetwork) {
  const std::vector<Gate> source{Gate::rotation({0, 1}, {1, 4}),
                                 {Kind::CX, 0, 1},
                                 Gate::rotation({0, 1}, {1, 4})};
  graph::Options options;
  synthesis::NativeRefinementLimits limits;
  SearchBudget budget(100000000);
  auto result =
      synthesis::refineSynthesizedNative(2, source, options, limits, budget);
  ASSERT_FALSE(result.alternatives.empty()) << result.error;
  EXPECT_EQ(result.tCountCalls, 1u);
  EXPECT_EQ(result.backend.rounds, 1u);
  EXPECT_EQ(result.backend.candidates, 6u);
  EXPECT_GT(result.gateCountCalls, 0u);
  bool reduced = false;
  for (const auto &candidate : result.alternatives) {
    checkUnitary(2, source, candidate.gates());
    reduced |= candidate.cost().metrics.t < 2;
    EXPECT_TRUE(candidate.matchesSource(2, source));
  }
  EXPECT_TRUE(reduced);
}

TEST(SynthesisNativeRefinement,
     EveryCompletedWordUsesTheRequestedNativeProfile) {
  for (unsigned profile = 0; profile < 3; ++profile) {
    graph::Options options;
    options.nativeCX = profile != 1;
    options.nativeCZ = profile != 0;
    options.nativeSwap = profile == 2;
    std::vector<Gate> source{{Kind::H, 0},
                             {Kind::H, 0},
                             {options.nativeCX ? Kind::CX : Kind::CZ, 0, 1},
                             {Kind::X, 1},
                             {Kind::X, 1}};
    synthesis::NativeRefinementLimits limits;
    limits.enableTCount = false;
    SearchBudget budget(100000000);
    auto result =
        synthesis::refineSynthesizedNative(2, source, options, limits, budget);
    ASSERT_FALSE(result.alternatives.empty()) << result.error;
    for (const auto &candidate : result.alternatives) {
      checkUnitary(2, source, candidate.gates());
      for (const auto &g : candidate.gates()) {
        EXPECT_TRUE(g.kind != Kind::CX || options.nativeCX);
        EXPECT_TRUE(g.kind != Kind::CZ || options.nativeCZ);
        EXPECT_TRUE(g.kind != Kind::Swap || options.nativeSwap);
      }
    }
  }
}

TEST(SynthesisNativeRefinement, OpaqueAnglesRemainExactAndSigned) {
  const std::vector<Gate> source{{Kind::X, 0},
                                 Gate::rotation({0, 1, true}, {0, 1, 7}),
                                 {Kind::X, 0},
                                 {Kind::H, 1},
                                 {Kind::H, 1}};
  graph::Options options;
  SearchBudget budget(100000000);
  auto result =
      synthesis::refineSynthesizedNative(2, source, options, {}, budget);
  ASSERT_FALSE(result.alternatives.empty()) << result.error;
  for (const auto &candidate : result.alternatives) {
    checkUnitary(2, source, candidate.gates());
    EXPECT_FALSE(candidate.cost().native.exact);
    unsigned opaque = 0;
    for (const auto &g : candidate.gates())
      if (g.kind == Kind::Rotate && g.angle.symbol) {
        ++opaque;
        EXPECT_EQ(g.angle.symbol, 7u);
      }
    EXPECT_EQ(opaque, 1u);
  }
}

TEST(SynthesisNativeRefinement, ExhaustionKeepsTheSourceAndParentReserve) {
  const std::vector<Gate> source{{Kind::H, 0}, {Kind::H, 0}};
  graph::Options options;
  synthesis::NativeRefinementLimits limits;
  SearchBudget budget(6);
  auto result =
      synthesis::refineSynthesizedNative(1, source, options, limits, budget);
  EXPECT_EQ(result.status, synthesis::NativeRefinementStatus::Exhausted);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_FALSE(budget.stopped());
  EXPECT_EQ(budget.remaining(), 6u);
  EXPECT_EQ(source.size(), 2u);
  EXPECT_TRUE(budget.take(6));

  limits.maxBackendWork = 1;
  SearchBudget local(1000000);
  auto incomplete =
      synthesis::refineSynthesizedNative(1, source, options, limits, local);
  EXPECT_EQ(incomplete.status, synthesis::NativeRefinementStatus::Exhausted);
  EXPECT_TRUE(incomplete.alternatives.empty());
  EXPECT_GT(incomplete.localExhaustions, 0u);
  EXPECT_FALSE(local.stopped());
  EXPECT_GT(local.remaining(), 500000u);
}

TEST(SynthesisNativeRefinement, RejectsNonNativeSourceBeforeBackendCosts) {
  graph::Options options;
  const std::vector<Gate> multiwire{Gate::rotation({0, 3}, {1, 4})};
  SearchBudget budget(1000000);
  auto result =
      synthesis::refineSynthesizedNative(2, multiwire, options, {}, budget);
  EXPECT_EQ(result.status, synthesis::NativeRefinementStatus::Unsupported);
  EXPECT_TRUE(result.alternatives.empty());
  EXPECT_EQ(result.gateCountCalls, 0u);
  EXPECT_EQ(result.tCountCalls, 0u);
  options.nativeCX = false;
  auto wrongProfile = synthesis::refineSynthesizedNative(2, {{Kind::CX, 0, 1}},
                                                         options, {}, budget);
  EXPECT_EQ(wrongProfile.status,
            synthesis::NativeRefinementStatus::Unsupported);
  EXPECT_EQ(wrongProfile.gateCountCalls, 0u);
}

TEST(SynthesisNativeRefinement, DynamicPhysicalWiresAreNotTruncated) {
  const std::vector<Gate> source{
      {Kind::H, 128}, {Kind::H, 128}, {Kind::CX, 64, 128}};
  graph::Options options;
  synthesis::NativeRefinementLimits limits;
  limits.enableTCount = false;
  SearchBudget budget(100000000);
  auto result =
      synthesis::refineSynthesizedNative(129, source, options, limits, budget);
  ASSERT_FALSE(result.alternatives.empty()) << result.error;
  for (const auto &candidate : result.alternatives) {
    EXPECT_TRUE(candidate.matchesSource(129, source));
    auto localized = candidate.gates();
    for (auto &g : localized) {
      if (g.kind == Kind::Phase)
        continue;
      ASSERT_NE(g.kind, Kind::Rotate);
      EXPECT_TRUE(g.a == 64 || g.a == 128);
      g.a = g.a == 128 ? 1 : 0;
      if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap) {
        EXPECT_TRUE(g.b == 64 || g.b == 128);
        g.b = g.b == 128 ? 1 : 0;
      }
    }
    checkUnitary(2, {{Kind::CX, 0, 1}}, localized);
  }
}

TEST(SynthesisNativeRefinement, DecodedExpansionIsBoundedBeforeBackendEntry) {
  const std::vector<Gate> source{Gate::rotation({0, 1}, {1, 4}),
                                 {Kind::H, 0},
                                 Gate::rotation({0, 1}, {1, 4})};
  graph::Options options;
  synthesis::NativeRefinementLimits limits;
  limits.maxTemporaryGates = 64;
  SearchBudget budget(100000000);
  auto result =
      synthesis::refineSynthesizedNative(1, source, options, limits, budget);
  EXPECT_GT(result.gateCountCalls, 0u);
  EXPECT_EQ(result.tCountCalls, 1u);
  EXPECT_EQ(result.backend.candidates, 2u);
  EXPECT_TRUE(result.decodeSkippedBySize);
  EXPECT_FALSE(result.tCountSkippedBySize);
  EXPECT_GT(result.decodedGateBound, limits.maxTemporaryGates);
  EXPECT_LE(result.materializedGateBound, limits.maxTemporaryGates);
  for (const auto &candidate : result.alternatives)
    checkUnitary(1, source, candidate.gates());
}

TEST(SynthesisNativeRefinement, OversizedDecodeStillPublishesCheckedMerges) {
  const std::vector<Gate> source{Gate::rotation({0, 1}, {1, 4}),
                                 {Kind::CX, 0, 1},
                                 Gate::rotation({0, 1}, {1, 4})};
  graph::Options options;
  synthesis::NativeRefinementLimits limits;
  limits.maxTemporaryGates = 64;
  // Too small for the fixed Clifford-window table charge, but sufficient for
  // both merge scans and their exact transport witnesses. Cleanup exhaustion
  // must neither disable the later cheap strategy nor erase its valid result.
  limits.maxBackendWork = 1000;
  SearchBudget budget(100000000);
  auto result =
      synthesis::refineSynthesizedNative(2, source, options, limits, budget);
  EXPECT_TRUE(result.decodeSkippedBySize);
  EXPECT_FALSE(result.tCountSkippedBySize);
  EXPECT_EQ(result.tCountCalls, 1u);
  EXPECT_EQ(result.backend.candidates, 2u);
  EXPECT_GT(result.backend.merges, 0u);
  EXPECT_EQ(result.backend.budgetExhaustions, 0u);
  EXPECT_EQ(result.status, synthesis::NativeRefinementStatus::Exhausted);
  EXPECT_FALSE(budget.stopped());
  ASSERT_FALSE(result.alternatives.empty());
  bool fewerT = false;
  for (const auto &candidate : result.alternatives) {
    checkUnitary(2, source, candidate.gates());
    EXPECT_LE(candidate.gates().size(), source.size());
    fewerT |= candidate.cost().metrics.t < 2;
  }
  EXPECT_TRUE(fewerT);
}

TEST(SynthesisNativeRefinement,
     LinearPostStageRetainsTheEntireEarlierPortfolio) {
  const std::vector<Gate> source{{Kind::H, 2},
                                 {Kind::H, 2},
                                 {Kind::CX, 0, 1},
                                 {Kind::CX, 0, 2},
                                 {Kind::CX, 1, 2}};
  graph::Options options;
  synthesis::NativeRefinementLimits oldLimits, extendedLimits;
  oldLimits.enableLinearWindows = false;
  SearchBudget oldBudget(100000000), extendedBudget(100000000);
  auto previous = synthesis::refineSynthesizedNative(3, source, options,
                                                     oldLimits, oldBudget);
  auto extended = synthesis::refineSynthesizedNative(
      3, source, options, extendedLimits, extendedBudget);
  ASSERT_FALSE(previous.alternatives.empty());
  ASSERT_GE(extended.alternatives.size(), previous.alternatives.size());
  for (size_t i = 0; i < previous.alternatives.size(); ++i) {
    EXPECT_TRUE(synthesis::native_refinement_detail::sameWord(
        previous.alternatives[i].gates(), extended.alternatives[i].gates()));
    EXPECT_EQ(previous.alternatives[i].cost().metrics,
              extended.alternatives[i].cost().metrics);
  }
  EXPECT_EQ(extended.gateCountCalls, previous.gateCountCalls);
  EXPECT_EQ(extended.tCountCalls, previous.tCountCalls);
  EXPECT_EQ(extended.proposals, previous.proposals);
  EXPECT_EQ(extended.backend.work, previous.backend.work);
  EXPECT_EQ(extended.backend.candidates, previous.backend.candidates);
  EXPECT_EQ(extended.backend.after, previous.backend.after);
  EXPECT_GT(extended.linearCalls, 0u);
  EXPECT_GT(extended.linearRewrites, 0u);
  bool reduced = false;
  for (const auto &candidate : extended.alternatives) {
    EXPECT_TRUE(candidate.matchesSource(3, source));
    checkUnitary(3, source, candidate.gates());
    reduced |= candidate.cost().metrics.gates == 2 &&
               candidate.cost().metrics.twoQubit == 2;
  }
  EXPECT_TRUE(reduced);
}

TEST(SynthesisNativeRefinement,
     LinearExhaustionCannotEraseGateCountCandidates) {
  const std::vector<Gate> source{{Kind::H, 2},
                                 {Kind::H, 2},
                                 {Kind::CX, 0, 1},
                                 {Kind::CX, 0, 2},
                                 {Kind::CX, 1, 2}};
  graph::Options options;
  synthesis::NativeRefinementLimits oldLimits, extendedLimits;
  oldLimits.enableLinearWindows = false;
  oldLimits.enableTCount = extendedLimits.enableTCount = false;
  extendedLimits.maxLinearWork = 1;
  SearchBudget oldBudget(100000000), extendedBudget(100000000);
  auto previous = synthesis::refineSynthesizedNative(3, source, options,
                                                     oldLimits, oldBudget);
  auto extended = synthesis::refineSynthesizedNative(
      3, source, options, extendedLimits, extendedBudget);
  ASSERT_FALSE(previous.alternatives.empty());
  ASSERT_EQ(extended.alternatives.size(), previous.alternatives.size());
  for (size_t i = 0; i < previous.alternatives.size(); ++i)
    EXPECT_TRUE(synthesis::native_refinement_detail::sameWord(
        previous.alternatives[i].gates(), extended.alternatives[i].gates()));
  EXPECT_EQ(extended.gateCountCalls, previous.gateCountCalls);
  EXPECT_EQ(extended.tCountCalls, 0u);
  EXPECT_GT(extended.linearCalls, 0u);
  EXPECT_GT(extended.linearExhaustions, 0u);
  EXPECT_EQ(extended.status, synthesis::NativeRefinementStatus::Exhausted);
  EXPECT_FALSE(extendedBudget.stopped());
}

TEST(SynthesisNativeRefinement, DecodeSizeFallbackStillReachesLinearPostStage) {
  const std::vector<Gate> source{Gate::rotation({0, 1}, {1, 4}),
                                 {Kind::H, 0},
                                 Gate::rotation({0, 1}, {1, 4}),
                                 {Kind::CX, 0, 1},
                                 {Kind::CX, 0, 2},
                                 {Kind::CX, 1, 2}};
  graph::Options options;
  synthesis::NativeRefinementLimits limits;
  limits.maxTemporaryGates = 64;
  SearchBudget budget(100000000);
  auto result =
      synthesis::refineSynthesizedNative(3, source, options, limits, budget);
  EXPECT_TRUE(result.decodeSkippedBySize);
  EXPECT_FALSE(result.tCountSkippedBySize);
  EXPECT_EQ(result.tCountCalls, 1u);
  EXPECT_GT(result.linearCalls, 0u);
  EXPECT_GT(result.linearRewrites, 0u);
  ASSERT_FALSE(result.alternatives.empty());
  bool fewerEntanglers = false;
  for (const auto &candidate : result.alternatives) {
    checkUnitary(3, source, candidate.gates());
    fewerEntanglers |= candidate.cost().metrics.twoQubit < 3;
  }
  EXPECT_TRUE(fewerEntanglers);
}
