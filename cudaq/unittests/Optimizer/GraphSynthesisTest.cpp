/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Correctness and resource-bound tests for Pauli graph synthesis.
///
/// Uses independent small-circuit simulation to check unitary phase, output
/// permutations, and terminal measurement distributions. Also exercises search
/// controls, malformed certificates, budget exhaustion, and the highest wire
/// bit.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include "gtest/gtest.h"
#include <bit>
#include <complex>
#include <numbers>
#include <random>

using namespace cudaq::pauli::graph;
namespace {
// The independent statevector oracle is deliberately restricted to small
// circuits; wide tests explicitly relabel their active support before using it.
size_t smallMask(const cudaq::pauli::QubitSet &bits) {
  assert(bits.fits(sizeof(size_t) * 8));
  size_t mask = 0;
  for (unsigned q : bits.set_bits())
    mask |= size_t(1) << q;
  return mask;
}
using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex I{0, 1};
State simulate(const std::vector<Gate> &gates, State state) {
  for (Gate g : gates) {
    if (g.kind == Kind::Phase || g.kind == Kind::Rotate) {
      double angle = g.angle.symbol ? 0.173 * g.angle.symbol
                                    : std::numbers::pi * g.angle.numerator /
                                          g.angle.denominator;
      if (g.angle.negative)
        angle = -angle;
      if (g.kind == Kind::Phase) {
        for (auto &a : state)
          a *= std::exp(I * angle);
      } else {
        State old = state;
        for (size_t b = 0; b < state.size(); ++b) {
          Complex phase =
              std::pow(I, cudaq::pauli::population(g.pauli.x & g.pauli.z));
          if ((cudaq::pauli::population(g.pauli.z & b) & 1) ^ g.pauli.sign)
            phase = -phase;
          state[b ^ smallMask(g.pauli.x)] =
              std::cos(angle / 2) * old[b ^ smallMask(g.pauli.x)] -
              I * std::sin(angle / 2) * phase * old[b];
        }
      }
      continue;
    }
    if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap) {
      State old = state;
      for (size_t b = 0; b < state.size(); ++b) {
        bool a = (b >> g.a) & 1, c = (b >> g.b) & 1;
        if (g.kind == Kind::CX)
          state[b ^ (size_t(a) << g.b)] = old[b];
        else if (g.kind == Kind::CZ)
          state[b] = (a && c) ? -old[b] : old[b];
        else
          state[a == c ? b : b ^ ((size_t(1) << g.a) | (size_t(1) << g.b))] =
              old[b];
      }
      continue;
    }
    for (size_t b = 0; b < state.size(); ++b)
      if (!((b >> g.a) & 1)) {
        size_t c = b | (size_t(1) << g.a);
        Complex a = state[b], d = state[c];
        switch (g.kind) {
        case Kind::H:
          state[b] = (a + d) / std::sqrt(2.0);
          state[c] = (a - d) / std::sqrt(2.0);
          break;
        case Kind::S:
          state[c] = I * d;
          break;
        case Kind::Sdg:
          state[c] = -I * d;
          break;
        case Kind::X:
          state[b] = d;
          state[c] = a;
          break;
        case Kind::Y:
          state[b] = -I * d;
          state[c] = I * a;
          break;
        case Kind::Z:
          state[c] = -d;
          break;
        default:
          break;
        }
      }
  }
  return state;
}
void assertUnitary(unsigned n, const std::vector<Gate> &a,
                   const std::vector<Gate> &b) {
  for (unsigned col = 0; col < (1u << n); ++col) {
    State state(1u << n);
    state[col] = 1;
    auto lhs = simulate(a, state), rhs = simulate(b, state);
    for (unsigned row = 0; row < state.size(); ++row)
      ASSERT_LT(std::abs(lhs[row] - rhs[row]), 1e-11)
          << "row=" << row << " col=" << col;
  }
}
} // namespace
TEST(GraphSynthesis, ExactPhaseAndPermutation) {
  std::vector<Gate> gates{{Kind::H, 0},       {Kind::S, 0}, {Kind::H, 0},
                          {Kind::S, 0},       {Kind::H, 0}, {Kind::S, 0},
                          {Kind::Swap, 0, 2}, {Kind::Y, 1}, {Kind::CX, 2, 1}};
  Options options;
  options.acceptOnlyImprovement = false;
  auto result = synthesize(3, gates, options);
  ASSERT_TRUE(result) << result.error;
  assertUnitary(3, gates, result.gates);
  auto corrupted = result.gates;
  corrupted.push_back(Gate::phase({1, 4}));
  EXPECT_FALSE(equivalent(3, gates, corrupted));
  corrupted = result.gates;
  corrupted.push_back({Kind::Swap, 0, 2});
  EXPECT_FALSE(equivalent(3, gates, corrupted));
}
TEST(GraphSynthesis, RandomUnitaryAndAllSearchControls) {
  std::mt19937 rng(0x50434f41);
  for (unsigned n = 1; n <= 4; ++n)
    for (unsigned test = 0; test < 24; ++test) {
      std::vector<Gate> gates;
      for (unsigned k = 0; k < 32; ++k) {
        unsigned a = rng() % n, b = (a + 1 + rng() % n) % n;
        if (a == b)
          b = (a + 1) % n;
        switch (rng() % 6) {
        case 0:
          gates.push_back({Kind::H, a});
          break;
        case 1:
          gates.push_back({Kind::S, a});
          break;
        case 2:
          gates.push_back({Kind::Y, a});
          break;
        case 3:
          gates.push_back({n > 1 ? Kind::CX : Kind::Sdg, a, b});
          break;
        case 4:
          gates.push_back({n > 1 ? Kind::CZ : Kind::Z, a, b});
          break;
        default:
          gates.push_back(
              Gate::rotation({uint64_t(1) << a, uint64_t(1) << a},
                             k % 2 ? Angle{1, 8} : Angle{0, 1, k + 1}));
        }
      }
      Options options;
      options.acceptOnlyImprovement = false;
      options.asymmetric = test % 2;
      options.simplify = test % 3;
      options.parallelizationCredit = (test % 3) * 0.7;
      options.frontierWeight = test % 4;
      options.frameWeight = (test % 5) * 0.2;
      options.lookahead = test % 2 ? 1 : 0;
      options.candidateNodes = test % 2 ? 1 : 0;
      auto result = synthesize(n, gates, options);
      ASSERT_TRUE(result) << "n=" << n << " test=" << test << " "
                          << result.error;
      assertUnitary(n, gates, result.gates);
    }
}
TEST(GraphSynthesis, CommutingMergeAndEntanglerReduction) {
  std::vector<Gate> input{
      {Kind::CX, 0, 1}, Gate::rotation({0, 2}, {1, 8}), {Kind::CX, 0, 1},
      {Kind::CX, 0, 1}, Gate::rotation({0, 2}, {1, 8}), {Kind::CX, 0, 1}};
  auto result = synthesize(2, input, {});
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.rotationsAfter, 1u);
  EXPECT_LT(result.stats.tqes, 4u);
  assertUnitary(2, input, result.gates);
}
TEST(GraphSynthesis, ReleasePreservesJointDistribution) {
  std::vector<Gate> input{{Kind::H, 0},
                          {Kind::CX, 0, 1},
                          Gate::rotation({0, 3}, {1, 8}),
                          Gate::rotation({1, 0}, {1, 4}),
                          Gate::rotation({0, 2}, {1, 4})};
  Options options;
  options.release = true;
  auto result = synthesize(2, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_GT(result.stats.released, 0u);
  // Arbitrary complex input, not just the allocated zero state.
  State state{{0.1, 0.3}, {0.2, -0.1}, {-0.4, 0.2}, {0.6, 0.7}};
  auto lhs = simulate(input, state), rhs = simulate(result.gates, state);
  for (unsigned i = 0; i < 4; ++i)
    EXPECT_NEAR(std::norm(lhs[i]), std::norm(rhs[i]), 1e-12);
}
TEST(GraphSynthesis, BudgetFailsWithoutReturningAnAcceptedCandidate) {
  std::vector<Gate> input{Gate::rotation({0, 3}, {1, 8})};
  Options options;
  options.maxSteps = 0;
  auto result = synthesize(2, input, options);
  EXPECT_FALSE(result);
  EXPECT_NE(result.error.find("step limit"), std::string::npos);
  options.maxSteps = 1000;
  options.frameWeight = -1;
  EXPECT_FALSE(synthesize(2, input, options));
}

TEST(GraphSynthesis, RejectMalformedCertificates) {
  std::vector<Gate> identity;
  EXPECT_FALSE(equivalent(1, identity, {{Kind::H, 64}}));
  EXPECT_FALSE(equivalent(1, identity, {{Kind::Swap, 0, 0}}));
  EXPECT_FALSE(equivalent(1, identity, {Gate::phase({1, 0})}));
  EXPECT_FALSE(equivalent(1, identity, {{static_cast<Kind>(255), 0}}));
}

TEST(GraphSynthesis, AcceptanceNeverIncreasesEntanglerCost) {
  std::vector<Gate> input{{Kind::CX, 0, 1}};
  auto result = synthesize(2, input, {});
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(result.stats.accepted);
  ASSERT_EQ(result.gates.size(), 1u);
  EXPECT_EQ(result.gates.front().kind, Kind::CX);
}

TEST(GraphSynthesis, RotationWitnessRejectsAnticommutingReorder) {
  auto x = Gate::rotation({1, 0}, {0, 1, 1});
  auto z = Gate::rotation({0, 1}, {0, 1, 2});
  auto zOther = Gate::rotation({0, 2}, {0, 1, 2});
  EXPECT_FALSE(equivalent(1, {x, z}, {z, x}));
  EXPECT_TRUE(equivalent(2, {x, zOther}, {zOther, x}));
  EXPECT_FALSE(equivalent(1, {x}, {Gate::rotation({1, 0}, {0, 1, 1, true})}));
}

TEST(GraphSynthesis, HighestSupportedQubit) {
  std::vector<Gate> input{{Kind::H, 63},
                          {Kind::CX, 63, 0},
                          Gate::rotation({0, 1}, {0, 1, 1}),
                          {Kind::CX, 63, 0},
                          {Kind::H, 63}};
  Options options;
  options.acceptOnlyImprovement = false;
  auto result = synthesize(64, input, options);
  ASSERT_TRUE(result) << result.error;
  auto compress = [](std::vector<Gate> gates) {
    for (auto &g : gates) {
      if (g.a == 63)
        g.a = 1;
      if (g.b == 63)
        g.b = 1;
      g.pauli.x =
          uint64_t(g.pauli.x.test(0)) | (uint64_t(g.pauli.x.test(63)) << 1);
      g.pauli.z =
          uint64_t(g.pauli.z.test(0)) | (uint64_t(g.pauli.z.test(63)) << 1);
    }
    return gates;
  };
  assertUnitary(2, compress(input), compress(result.gates));
  EXPECT_TRUE(synthesize(65, input, options));
}

TEST(GraphSynthesis, DynamicWidthAndSignedReplay) {
  for (unsigned width : {63u, 64u, 65u, 118u, 128u, 433u}) {
    unsigned q = width - 1;
    std::vector<Gate> input{{Kind::H, q},     {Kind::S, q},
                            {Kind::CX, q, 0}, Gate::rotation({0, 1}, {0, 1, 1}),
                            {Kind::CX, q, 0}, {Kind::Sdg, q},
                            {Kind::H, q}};
    Options options;
    options.acceptOnlyImprovement = false;
    auto result = synthesize(width, input, options);
    ASSERT_TRUE(result) << width << ": " << result.error;
    auto compress = [&](std::vector<Gate> gates) {
      for (auto &g : gates) {
        if (g.a == q)
          g.a = 1;
        if (g.b == q)
          g.b = 1;
        g.pauli.x =
            uint64_t(g.pauli.x.test(0)) | (uint64_t(g.pauli.x.test(q)) << 1);
        g.pauli.z =
            uint64_t(g.pauli.z.test(0)) | (uint64_t(g.pauli.z.test(q)) << 1);
      }
      return gates;
    };
    assertUnitary(2, compress(input), compress(result.gates));
    EXPECT_FALSE(equivalent(width - 1, input, result.gates));
  }
}

TEST(GraphSynthesis, WideWorkEstimateCannotWrapOrAllocateBeforeRejection) {
  // Four width-squared charges used to wrap to four at this width. The bounded
  // request must fail before allocating a tableau, not run out of memory.
  constexpr unsigned width = 1u << 31;
  cudaq::pauli::SearchBudget budget(100);
  std::vector<Gate> input(4, Gate{Kind::H, 0});
  auto result = synthesize(width, input, {}, &budget);
  EXPECT_FALSE(result);
  EXPECT_TRUE(result.stats.budgetExhausted);
  EXPECT_TRUE(budget.stopped());
  EXPECT_EQ(budget.work(), 0u);
  EXPECT_TRUE(synthesize(width, {}, {}));
}

TEST(GraphSynthesis, NativeReadoutPreservesPhysicalOutcomeOrder) {
  std::vector<Pauli> words{{0, 2}, {0, 1}};
  Options options;
  auto native = synthesizeReadout(2, words, {0, 1}, options);
  ASSERT_TRUE(native) << native.error;
  ASSERT_EQ(native.gates.size(), 1u);
  EXPECT_EQ(native.gates.front().kind, Kind::Swap);
  EXPECT_EQ(nativeGateCost(native.gates, options), 1);
  for (bool cx : {false, true}) {
    options.nativeCX = cx;
    options.nativeCZ = !cx;
    options.nativeSwap = false;
    auto decomposed = synthesizeReadout(2, words, {0, 1}, options);
    ASSERT_TRUE(decomposed) << decomposed.error;
    assertUnitary(2, native.gates, decomposed.gates);
    for (auto g : decomposed.gates) {
      EXPECT_NE(g.kind, Kind::Swap);
      EXPECT_NE(g.kind, cx ? Kind::CZ : Kind::CX);
    }
    EXPECT_EQ(nativeGateCost(native.gates, options), cx ? 3 : 9);
  }
}

TEST(GraphSynthesis, MeasuredSynthesisPreservesAllJointOutcomeEffects) {
  // Carrying X/Y basis changes and reconstructing a physical permutation must
  // preserve off-diagonal POVM entries as well as computational-basis inputs.
  std::vector<Gate> input{{Kind::H, 0},
                          {Kind::CX, 0, 1},
                          Gate::rotation({2, 2}, {1, 4}),
                          {Kind::Sdg, 0},
                          {Kind::Swap, 0, 1}};
  std::vector<Pauli> outcomes{{0, 1}, {0, 2}};
  auto normalized = cudaq::pauli::pbc::normalize(2, input, outcomes);
  ASSERT_TRUE(normalized) << normalized.error;
  for (unsigned profile = 0; profile < 3; ++profile) {
    Options options;
    options.simplify = false;
    options.discrepancySteps = 3;
    options.nativeCX = profile != 2;
    options.nativeCZ = profile != 1;
    options.nativeSwap = profile == 0;
    cudaq::pauli::SearchBudget budget(10000000);
    auto result =
        synthesizeMeasured(2, normalized.rotations, normalized.measurements,
                           {0, 1}, options, &budget);
    ASSERT_TRUE(result) << result.error;
    EXPECT_EQ(result.measurements, outcomes);
    std::vector<State> before, after;
    for (unsigned col = 0; col < 4; ++col) {
      State state(4);
      state[col] = 1;
      before.push_back(simulate(input, state));
      after.push_back(simulate(result.rotations, state));
    }
    for (unsigned outcome = 0; outcome < 4; ++outcome)
      for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = 0; j < 4; ++j)
          EXPECT_LT(
              std::abs(std::conj(before[i][outcome]) * before[j][outcome] -
                       std::conj(after[i][outcome]) * after[j][outcome]),
              1e-11);
  }
}

TEST(GraphSynthesis,
     MeasuredSynthesisRejectsInvalidProfilesAndInsufficientWork) {
  Options options;
  options.nativeCX = options.nativeCZ = false;
  EXPECT_FALSE(synthesizeMeasured(1, {}, {{0, 1}}, {0}, options));
  options = {};
  options.actionWeight = -1;
  EXPECT_FALSE(synthesizeMeasured(1, {}, {{0, 1}}, {0}, options));
  options = {};
  cudaq::pauli::SearchBudget budget(0);
  EXPECT_FALSE(
      synthesizeMeasured(1000000000, {}, {{0, 1}}, {0}, options, &budget));
  EXPECT_TRUE(budget.stopped());
}

TEST(GraphSynthesis, MeasuredSearchRetainsVerifiedIncumbentAtWorkLimit) {
  std::vector<Gate> rotations{Gate::rotation({0, 3}, {1, 4})};
  std::vector<Pauli> measured{{0, 1}, {0, 2}};
  Options options;
  options.discrepancySteps = 0;
  cudaq::pauli::SearchBudget first(10000000);
  auto greedy =
      synthesizeMeasured(2, rotations, measured, {0, 1}, options, &first);
  ASSERT_TRUE(greedy) << greedy.error;
  options.discrepancySteps = 22;
  cudaq::pauli::SearchBudget limited(first.work());
  auto result =
      synthesizeMeasured(2, rotations, measured, {0, 1}, options, &limited);
  ASSERT_TRUE(result) << result.error;
  EXPECT_TRUE(limited.stopped());
  assertUnitary(2, greedy.rotations, result.rotations);
}

TEST(GraphSynthesis, NativeReadoutUsesDynamicSupportStorage) {
  using cudaq::pauli::QubitSet;
  for (unsigned last : {64u, 128u}) {
    std::vector<Pauli> words{{0, QubitSet::single(last)}, {0, 1}};
    auto result = synthesizeReadout(last + 1, words, {0, last}, {});
    ASSERT_TRUE(result) << result.error;
    ASSERT_EQ(result.gates.size(), 1u);
    EXPECT_EQ(result.gates.front().kind, Kind::Swap);
    EXPECT_EQ(result.gates.front().a, last);
    EXPECT_EQ(result.gates.front().b, 0u);
    EXPECT_EQ(nativeGateCost(
                  {Gate::rotation({QubitSet::single(last), 0}, {1, 4})}, {}),
              3);
  }
}
