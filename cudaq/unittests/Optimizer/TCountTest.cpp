/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Validation of native T optimization, portfolio search, and PBC
/// frames.
///
/// Checks phase-sensitive unitaries and joint measurement effects with
/// independent small-circuit simulation. Covers signed axes, opaque angles,
/// decoding witnesses, strategy composition, deterministic bounds, and invalid
/// inputs.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Pauli/TCount.h"
#include "Pauli/TCountInternal.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Pauli/Optimizer.h"
#include "cudaq/Optimizer/Pauli/PBC.h"
#include "cudaq/Optimizer/Pauli/Telemetry.h"
#include <bit>
#include <complex>
#include <numbers>
#include <random>
#include <set>

using namespace cudaq::pauli;
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
namespace tc = cudaq::pauli::tcount;
namespace {
void rotation(std::vector<Gate> &gates, Pauli p, int k = 1) {
  gates.push_back(Gate::rotation(p, {k, 4}));
}
} // namespace
TEST(TCount, SignedPauliMerging) {
  for (bool sign : {false, true}) {
    std::vector<Gate> input;
    rotation(input, {0, 3});
    rotation(input, {3, 0});
    rotation(input, {0, 3, sign});
    auto result = tc::optimize(2, input);
    ASSERT_TRUE(result) << result.error;
    EXPECT_EQ(result.stats.before.t, 3u);
    EXPECT_EQ(result.stats.after.t, 1u);
    assertUnitary(2, input, result.gates);
  }
}
TEST(TCount, CliffordTransportAndOpaqueCommutation) {
  std::vector<Gate> input{Gate::rotation({0, 1}, {1, 4}),
                          {Kind::H, 0},
                          Gate::rotation({1, 0}, {0, 1, 1}),
                          Gate::rotation({1, 0}, {-1, 4})};
  auto result = tc::optimize(1, input);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.stats.after.t, 0u);
  assertUnitary(1, input, result.gates);
  input[2].pauli = {0, 1};
  result = tc::optimize(1, input);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.stats.after.t, 2u);
  assertUnitary(1, input, result.gates);
}
TEST(TCount, ExactCommutingBlockBeyondPairMerging) {
  // Algebraic regression only, never a new performance benchmark case.
  std::vector<Gate> input;
  for (unsigned mask = 1; mask < 16; ++mask) {
    rotation(input, {0, mask}, mask & 1 ? -1 : 1);
    input.push_back(Gate::phase({mask & 1 ? -1 : 1, 8}));
  }
  tc::Options mergeOnly;
  mergeOnly.decode = false;
  auto merged = tc::optimize(4, input, mergeOnly);
  ASSERT_TRUE(merged);
  EXPECT_EQ(merged.stats.after.t, 15u);
  tc::Options options;
  options.maxTwoQubitGrowth = 1000;
  auto result = tc::optimize(4, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.after.t, 0u);
  EXPECT_GT(result.stats.verifiedBlocks, 0u);
  assertUnitary(4, input, result.gates);
}
TEST(TCount, RandomFullUnitariesAndPhase) {
  std::mt19937 rng(20261023);
  for (unsigned n = 1; n <= 5; ++n)
    for (unsigned trial = 0; trial < 40; ++trial) {
      std::vector<Gate> input;
      for (unsigned j = 0; j < 60; ++j) {
        unsigned a = rng() % n, b = (a + 1) % n;
        switch (rng() % 12) {
        case 0:
          input.push_back({Kind::H, a});
          break;
        case 1:
          input.push_back({Kind::Sdg, a});
          break;
        case 2:
          input.push_back({Kind::Y, a});
          break;
        case 3:
          if (n > 1)
            input.push_back({Kind::CX, a, b});
          break;
        case 4:
          if (n > 1)
            input.push_back({Kind::CZ, a, b});
          break;
        case 5:
          if (n > 1)
            input.push_back({Kind::Swap, a, b});
          break;
        case 6:
          input.push_back(Gate::phase({int(rng() % 17) - 8, 8}));
          break;
        default:
          rotation(input,
                   {rng() % (1u << n), rng() % (1u << n), bool(rng() & 1)},
                   int(rng() % 31) - 15);
          break;
        }
      }
      tc::Options options;
      options.maxTwoQubitGrowth = 1000;
      auto result = tc::optimize(n, input, options);
      ASSERT_TRUE(result) << result.error;
      EXPECT_LE(result.stats.after.t, result.stats.before.t);
      assertUnitary(n, input, result.gates);
    }
}
TEST(TCount, BoundsAndDeterminism) {
  std::vector<Gate> input;
  for (unsigned i = 0; i < 100; ++i)
    rotation(input, {0, 1});
  tc::Options options;
  options.maxWork = 1;
  auto bounded = tc::optimize(1, input, options);
  ASSERT_TRUE(bounded);
  EXPECT_EQ(bounded.stats.after.t, 100u);
  EXPECT_EQ(bounded.stats.budgetExhaustions, 1u);
  assertUnitary(1, input, bounded.gates);
  auto a = tc::optimize(1, input), b = tc::optimize(1, input);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  EXPECT_EQ(a.stats.after, b.stats.after);
  EXPECT_EQ(a.stats.work, b.stats.work);
  EXPECT_EQ(a.stats.after.t, 0u);
  assertUnitary(1, input, a.gates);
  options.maxBlockRank = 6;
  EXPECT_FALSE(tc::optimize(1, input, options));
  EXPECT_TRUE(tc::optimize(65, input));
}
TEST(TCount, HighBitAndLargeExactAngles) {
  std::vector<Gate> input{
      Gate::rotation({0, uint64_t(1) << 63}, {INT64_MAX, 4}),
      Gate::rotation({0, uint64_t(1) << 63}, {INT64_MIN, 4})};
  auto result = tc::optimize(64, input);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.stats.after.t, 1u);
  input[0].angle = {1, 4};
  input[1].angle = {-1, 4};
  result = tc::optimize(64, input);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.stats.after.t, 0u);
}

TEST(TCount, DecodingSignedNonDiagonalGenerators) {
  // Independent commuting generators with Y and X support, rank five.
  // Product signs matter: (+XX)(+ZZ) = -YY.
  std::vector<Pauli> generators{{3, 0}, {0, 3}, {4, 4}, {8, 0}, {0, 16}};
  std::vector<Pauli> axes{{}};
  for (auto p : generators) {
    auto previous = axes;
    for (auto q : previous) {
      Pauli v{p.x ^ q.x, p.z ^ q.z};
      int phase = cudaq::pauli::population(p.x & p.z) +
                  cudaq::pauli::population(q.x & q.z) -
                  cudaq::pauli::population(v.x & v.z) +
                  2 * cudaq::pauli::population(p.z & q.x) +
                  2 * (p.sign + q.sign);
      v.sign = (phase & 2) != 0;
      axes.push_back(v);
    }
  }
  std::vector<Gate> input;
  for (unsigned i = 1; i < axes.size(); ++i)
    rotation(input, axes[i], i & 1 ? 3 : -1);
  tc::Options options;
  options.maxTwoQubitGrowth = 10000;
  auto result = tc::optimize(5, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.after.t, 0u);
  assertUnitary(5, input, result.gates);
}

TEST(TCount, RejectCorruptedRewriteWitnesses) {
  namespace d = cudaq::pauli::tcount::detail;
  tc::Options options;
  tc::Statistics stats;
  d::Budget budget(options, stats);
  std::vector<Gate> gates{Gate::rotation({0, 1}, {1, 4}),
                          {Kind::H, 0},
                          Gate::rotation({1, 0}, {1, 4})};
  ASSERT_EQ(d::verifyMerge(gates, 0, 2, budget, stats), 1);
  gates[2].pauli = {0, 1};
  EXPECT_FALSE(d::verifyMerge(gates, 0, 2, budget, stats));
  gates[2].pauli = {1, 0, true};
  ASSERT_EQ(d::verifyMerge(gates, 0, 2, budget, stats), -1);
  EXPECT_TRUE(d::verifyPhases({0, 1, 0, 1}, {0, 1, 0, 1}, 0, budget));
  EXPECT_FALSE(d::verifyPhases({0, 1, 0, 1}, {0, 1, 0, 1}, 1, budget));
  EXPECT_FALSE(d::verifyPhases({0, 1, 0, 1}, {0, 1, 0, 0}, 0, budget));
}

TEST(TCount, ExistingFramePlacement) {
  namespace d = cudaq::pauli::tcount::detail;
  // Check the bidirectional frame mapping independently over every signed
  // Pauli on three wires, including all Y products and high-support axes.
  d::Frame frame(3);
  std::vector<Gate> cliffords{{Kind::H, 0},     {Kind::S, 1},
                              {Kind::CX, 0, 2}, {Kind::Y, 2},
                              {Kind::CZ, 1, 2}, {Kind::Swap, 0, 1}};
  for (auto g : cliffords)
    frame.append(g);
  for (unsigned x = 0; x < 8; ++x)
    for (unsigned z = 0; z < 8; ++z)
      for (bool sign : {false, true}) {
        Pauli p{x, z, sign};
        EXPECT_EQ(frame.inverseLookup(frame.lookup(p)), p);
      }
}

TEST(TCount, IgnoreUnusedGateFields) {
  // Scalar and Pauli rotations have no primitive qubit operands. Unused
  // fields must not enter bit shifts or primitive-gate identity keys.
  Gate phase = Gate::phase({1, 4});
  phase.a = 999;
  Gate rotate = Gate::rotation({0, 1}, {1, 4});
  rotate.a = 1000;
  std::vector<Gate> input{rotate,       phase,           {Kind::H, 0, 4096},
                          {Kind::S, 0}, {Kind::H, 0, 0}, rotate};
  auto result = tc::optimize(1, input);
  ASSERT_TRUE(result) << result.error;
  assertUnitary(1, input, result.gates);
}

TEST(PauliOptimizer, RetainsTOnlyAndPreservesFullUnitary) {
  std::vector<Gate> input{Gate::rotation({0, 3}, {1, 4}),
                          Gate::rotation({0, 3}, {1, 4}),
                          {Kind::H, 0},
                          {Kind::S, 1},
                          Gate::phase({1, 4})};
  auto tOnly = tc::optimize(2, input);
  auto unified = cudaq::pauli::optimize(2, input);
  ASSERT_TRUE(tOnly);
  ASSERT_TRUE(unified) << unified.error;
  EXPECT_LE(unified.stats.after.t, tOnly.stats.after.t);
  EXPECT_GT(unified.stats.tcountCandidates, 0u);
  EXPECT_GT(unified.stats.graphSynthesisCandidates, 0u);
  EXPECT_LE(unified.stats.after.twoQubit, unified.stats.before.twoQubit);
  assertUnitary(2, input, unified.gates);
}

TEST(PauliOptimizer, GraphSynthesisOnlyAndObjectiveControls) {
  std::vector<Gate> input{
      {Kind::CX, 0, 1}, {Kind::CX, 0, 1}, Gate::phase({3, 4})};
  for (auto objective :
       {cudaq::pauli::Objective::TCount, cudaq::pauli::Objective::TDepth,
        cudaq::pauli::Objective::Entanglers, cudaq::pauli::Objective::Depth}) {
    cudaq::pauli::OptimizerOptions options;
    options.objective = objective;
    options.enableTCount = false;
    auto result = cudaq::pauli::optimize(2, input, options);
    ASSERT_TRUE(result) << result.error;
    EXPECT_EQ(result.stats.after.twoQubit, 0u);
    EXPECT_EQ(result.stats.tcountCandidates, 0u);
    EXPECT_GT(result.stats.graphSynthesisCandidates, 0u);
    assertUnitary(2, input, result.gates);
  }
}

TEST(PauliOptimizer, ComposesDecodingAndCliffordReconstruction) {
  // Decoding removes the phase polynomial; reconstruction removes the
  // nonlocal Clifford identity. Neither strategy alone performs both steps.
  std::vector<Gate> input;
  for (unsigned mask = 1; mask < 16; ++mask) {
    rotation(input, {0, mask});
    input.push_back(Gate::phase({1, 8}));
  }
  for (auto g :
       {Gate{Kind::CX, 0, 1}, Gate{Kind::CX, 1, 2}, Gate{Kind::CX, 0, 1},
        Gate{Kind::CX, 1, 2}, Gate{Kind::CX, 0, 2}})
    input.push_back(g);
  cudaq::pauli::OptimizerOptions options;
  options.enableGraphSynthesis = false;
  auto tOnly = cudaq::pauli::optimize(4, input, options);
  options.enableGraphSynthesis = true;
  options.enableTCount = false;
  auto pcOnly = cudaq::pauli::optimize(4, input, options);
  options.enableTCount = true;
  auto combined = cudaq::pauli::optimize(4, input, options);
  ASSERT_TRUE(tOnly) << tOnly.error;
  ASSERT_TRUE(pcOnly) << pcOnly.error;
  ASSERT_TRUE(combined) << combined.error;
  EXPECT_EQ(combined.stats.after.t, 0u);
  EXPECT_EQ(combined.stats.after.twoQubit, 0u);
  EXPECT_GT(tOnly.stats.after.twoQubit, combined.stats.after.twoQubit);
  EXPECT_GT(pcOnly.stats.after.t, combined.stats.after.t);
  assertUnitary(4, input, combined.gates);
}

TEST(PauliOptimizer, SharedBudgetAndSizeFallback) {
  std::vector<Gate> input{Gate::rotation({0, 1}, {1, 4}),
                          Gate::rotation({0, 1}, {1, 4})};
  cudaq::pauli::OptimizerOptions options;
  options.maxWork = 1;
  auto result = cudaq::pauli::optimize(1, input, options);
  ASSERT_TRUE(result);
  EXPECT_LE(result.stats.work, 1u);
  EXPECT_GT(result.stats.budgetExhaustions, 0u);
  EXPECT_EQ(result.stats.after.t, 2u);
  assertUnitary(1, input, result.gates);
  options.maxWork = 100000;
  options.maxGates = 1;
  result = cudaq::pauli::optimize(1, input, options);
  ASSERT_TRUE(result);
  EXPECT_TRUE(result.stats.skippedSize);
  assertUnitary(1, input, result.gates);
  options.maxGates = 8192;
  options.maxCandidates = 1;
  result = cudaq::pauli::optimize(1, input, options);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.stats.candidates, 1u);
  EXPECT_EQ(result.stats.after.t, 0u);
  assertUnitary(1, input, result.gates);
}

TEST(PauliOptimizer, TDepthTracksEntanglingDependencies) {
  std::vector<Gate> input{Gate::rotation({0, 1}, {1, 4}),
                          Gate::rotation({0, 2}, {1, 4}),
                          {Kind::CX, 0, 1},
                          Gate::rotation({0, 2}, {1, 4})};
  auto c = tc::cost(2, input);
  EXPECT_EQ(c.t, 3u);
  EXPECT_EQ(c.tDepth, 2u);
}

TEST(PauliOptimizer, RejectsReleaseAndInvalidControls) {
  cudaq::pauli::OptimizerOptions options;
  options.graphSynthesis.release = true;
  EXPECT_FALSE(cudaq::pauli::optimize(1, {}, options));
  options.graphSynthesis.release = false;
  options.graphSynthesis.frameWeight = -1;
  EXPECT_FALSE(cudaq::pauli::optimize(1, {}, options));
  options.graphSynthesis.frameWeight = 0.1;
  options.beamWidth = 65;
  EXPECT_FALSE(cudaq::pauli::optimize(1, {}, options));
}

TEST(PauliOptimizer, GraphSynthesisDependencyLimitAndProofBudget) {
  std::vector<Gate> input{Gate::rotation({1, 0}, {1, 4}),
                          Gate::rotation({0, 1}, {1, 4}),
                          Gate::rotation({1, 0}, {1, 4})};
  cudaq::pauli::graph::Options options;
  options.maxDependencies = 1;
  auto result = cudaq::pauli::graph::synthesize(1, input, options);
  EXPECT_FALSE(result);
  EXPECT_TRUE(result.stats.budgetExhausted);
  options.maxDependencies = 100;
  cudaq::pauli::SearchBudget budget(1);
  result = cudaq::pauli::graph::synthesize(1, input, options, &budget);
  EXPECT_FALSE(result);
  EXPECT_TRUE(result.stats.budgetExhausted);
  EXPECT_LE(budget.work(), 1u);
}

TEST(PauliOptimizer, RandomPhaseSensitiveUnitaries) {
  std::mt19937 rng(20261024);
  for (unsigned trial = 0; trial < 30; ++trial) {
    std::vector<Gate> input;
    for (unsigned i = 0; i < 16; ++i) {
      input.push_back(Gate::rotation({rng() % 8, rng() % 8, bool(rng() & 1)},
                                     {int(rng() % 7) - 3, 4}));
      input.push_back({Kind::H, unsigned(rng() % 3)});
    }
    input.push_back(Gate::phase({int(rng() % 8), 4}));
    cudaq::pauli::OptimizerOptions options;
    options.objective = static_cast<cudaq::pauli::Objective>(trial % 4);
    options.maxWork = 200000;
    auto result = cudaq::pauli::optimize(3, input, options);
    ASSERT_TRUE(result) << result.error;
    EXPECT_LE(result.stats.work, options.maxWork);
    EXPECT_LE(result.stats.after.twoQubit, result.stats.before.twoQubit);
    assertUnitary(3, input, result.gates);
  }
}

namespace {
State project(State state, Pauli p, bool outcome) {
  State result = state;
  for (size_t b = 0; b < state.size(); ++b) {
    Complex phase = std::pow(I, cudaq::pauli::population(p.x & p.z));
    if (bool(cudaq::pauli::population(p.z & b) & 1) ^ p.sign ^ outcome)
      phase = -phase;
    result[b ^ smallMask(p.x)] += phase * state[b];
  }
  for (auto &v : result)
    v /= 2;
  return result;
}
// Compare every entry of every joint-outcome POVM effect. Unlike comparing
// probabilities only on computational basis states, this detects errors in
// coherences and therefore covers arbitrary inputs and entangled references.
void assertInstrument(unsigned n, const std::vector<Gate> &input,
                      const std::vector<Pauli> &measurements,
                      const pbc::Result &output) {
  for (unsigned outcome = 0; outcome < (1u << measurements.size()); ++outcome) {
    std::vector<State> before, after;
    for (unsigned col = 0; col < (1u << n); ++col) {
      State state(1u << n);
      state[col] = 1;
      auto a = simulate(input, state), b = simulate(output.rotations, state);
      for (unsigned i = 0; i < measurements.size(); ++i) {
        a = project(a, measurements[i], (outcome >> i) & 1);
        b = project(b, output.measurements[i], (outcome >> i) & 1);
      }
      before.push_back(a);
      after.push_back(b);
    }
    for (unsigned a = 0; a < (1u << n); ++a)
      for (unsigned b = 0; b < (1u << n); ++b) {
        Complex lhs{}, rhs{};
        for (unsigned row = 0; row < (1u << n); ++row) {
          lhs += std::conj(before[a][row]) * before[b][row];
          rhs += std::conj(after[a][row]) * after[b][row];
        }
        ASSERT_LT(std::abs(lhs - rhs), 1e-11) << "outcome=" << outcome;
      }
  }
}
} // namespace

TEST(PBC, FrameAndSignedMeasurement) {
  std::vector<Gate> input{{Kind::CX, 0, 1}, Gate::rotation({0, 2}, {1, 4})};
  std::vector<Pauli> measurements{{1, 0}, {2, 0}};
  auto result = pbc::normalize(2, input, measurements);
  ASSERT_TRUE(result) << result.error;
  ASSERT_EQ(result.rotations.size(), 1u);
  EXPECT_EQ(result.rotations[0].pauli, (Pauli{0, 3}));
  EXPECT_EQ(result.measurements, (std::vector<Pauli>{{3, 0}, {2, 0}}));
  assertInstrument(2, input, measurements, result);
  input.push_back(Gate::rotation({0, 2}, {1, 4}));
  auto optimized = tc::optimize(2, input);
  ASSERT_TRUE(optimized);
  auto reduced = pbc::normalize(2, optimized.gates, measurements);
  ASSERT_TRUE(reduced) << reduced.error;
  EXPECT_TRUE(reduced.rotations.empty());
  EXPECT_EQ(reduced.measurements[1], (Pauli{2, 3, true}));
  assertInstrument(2, input, measurements, reduced);
}

TEST(PBC, RandomAllOutcomeEffects) {
  std::mt19937 rng(20261030);
  for (unsigned trial = 0; trial < 40; ++trial) {
    std::vector<Gate> input;
    for (unsigned i = 0; i < 18; ++i) {
      unsigned q = rng() % 3;
      switch (rng() % 5) {
      case 0:
        input.push_back({Kind::H, q});
        break;
      case 1:
        input.push_back({Kind::CX, q, (q + 1) % 3});
        break;
      default:
        input.push_back(
            Gate::rotation({rng() % 8, rng() % 8, bool(rng() & 1)},
                           {int(rng() % 15) - 7, i % 3 == 0 ? 7 : 4}));
      }
    }
    input.push_back(Gate::rotation({1, 2}, {0, 1, 1, bool(trial & 1)}));
    input.push_back(Gate::phase({3, 7}));
    std::vector<Pauli> measurements{{0, 1}, {0, 2, true}, {0, 4}};
    auto result = pbc::normalize(3, input, measurements);
    ASSERT_TRUE(result) << result.error;
    assertInstrument(3, input, measurements, result);
  }
}

TEST(PBC, RejectNoncommutingObservablesAndBudgetExhaustion) {
  EXPECT_FALSE(pbc::normalize(1, {}, {{1, 0}, {0, 1}}));
  EXPECT_FALSE(pbc::normalize(1, {}, {{0, 0}}));
  EXPECT_FALSE(pbc::normalize(1, {}, {{0, 2}}));
  SearchBudget budget(1);
  auto result = pbc::normalize(2, {{Kind::H, 0}}, {{0, 1}}, &budget);
  EXPECT_FALSE(result);
  EXPECT_LE(budget.work(), 1u);
}

TEST(PauliOptimizer, TelemetryPreservesSelectionAndTracksAncestry) {
  std::vector<Gate> input{{Kind::H, 0},
                          Gate::rotation({0, 1}, {1, 4}),
                          Gate::rotation({0, 1}, {1, 4}),
                          {Kind::CX, 0, 1}};
  OptimizerOptions options;
  auto baseline = optimize(2, input, options);
  Telemetry trace;
  auto observed = optimize(2, input, options, nullptr, &trace);
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(observed);
  EXPECT_EQ(baseline.stats.after, observed.stats.after);
  EXPECT_EQ(baseline.stats.work, observed.stats.work);
  ASSERT_EQ(baseline.gates.size(), observed.gates.size());
  for (size_t i = 0; i < baseline.gates.size(); ++i) {
    EXPECT_EQ(baseline.gates[i].kind, observed.gates[i].kind);
    EXPECT_EQ(baseline.gates[i].a, observed.gates[i].a);
    EXPECT_EQ(baseline.gates[i].b, observed.gates[i].b);
    EXPECT_EQ(baseline.gates[i].pauli, observed.gates[i].pauli);
    EXPECT_EQ(baseline.gates[i].angle, observed.gates[i].angle);
  }
  ASSERT_FALSE(trace.events.empty());
  std::set<uint64_t> ids{0};
  for (const auto &event : trace.events) {
    EXPECT_TRUE(ids.insert(event.id).second);
    EXPECT_GE(event.elapsedNs, event.verificationNs);
  }
  EXPECT_TRUE(ids.contains(trace.winner));
  for (const auto &event : trace.events) {
    EXPECT_TRUE(ids.contains(event.parent));
    EXPECT_TRUE(ids.contains(event.scope));
    EXPECT_TRUE(ids.contains(event.selectedChild));
  }
  Telemetry truncated;
  truncated.maxEvents = 1;
  auto bounded = optimize(2, input, options, nullptr, &truncated);
  ASSERT_TRUE(bounded);
  EXPECT_EQ(bounded.stats.after, baseline.stats.after);
  EXPECT_EQ(bounded.stats.work, baseline.stats.work);
  EXPECT_EQ(truncated.events.size(), 1);
  EXPECT_GT(truncated.droppedEvents, 0);
  EXPECT_EQ(truncated.methods.size(), trace.methods.size());
  for (const auto &[method, total] : trace.methods)
    EXPECT_EQ(total.calls, truncated.methods.at(method).calls);
}

TEST(PBC, DynamicSupportStorageAndHashing) {
  for (unsigned q :
       {0u, 55u, 56u, 57u, 63u, 64u, 65u, 127u, 128u, 432u, 1024u}) {
    auto bit = QubitSet::single(q);
    EXPECT_TRUE(bit.test(q));
    EXPECT_EQ(bit.count(), 1u);
    EXPECT_TRUE(bit.fits(q + 1));
    EXPECT_FALSE(bit.fits(q));
    Pauli x{bit, {}}, z{{}, bit}, y{bit, bit};
    EXPECT_EQ(y.axis(q), 3u);
    EXPECT_TRUE(x.anti(z));
    EXPECT_TRUE(y.anti(z));
    EXPECT_FALSE(y.anti(y));
    auto original = x;
    x.x.flip(q + 10);
    x.x.flip(q + 10);
    EXPECT_EQ(x, original);
    EXPECT_EQ(x.x.hash(), original.x.hash());
    EXPECT_EQ(x.x ^ original.x, QubitSet{});
    EXPECT_EQ((bit | QubitSet{1}).count(), q ? 2u : 1u);
  }
}

TEST(PBC, DynamicSupportOperationsMatchIndependentSets) {
  std::mt19937 rng(4721);
  for (unsigned trial = 0; trial < 100; ++trial) {
    QubitSet a, b;
    std::set<unsigned> left, right;
    unsigned widthA = 1 + rng() % 1025, widthB = 1 + rng() % 1025;
    for (unsigned i = 0; i < 50; ++i) {
      unsigned x = rng() % widthA, y = rng() % widthB;
      a.set(x);
      b.set(y);
      left.insert(x);
      right.insert(y);
    }
    std::set<unsigned> united, intersected, different;
    std::set_union(left.begin(), left.end(), right.begin(), right.end(),
                   std::inserter(united, united.end()));
    std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
                          std::inserter(intersected, intersected.end()));
    std::set_symmetric_difference(left.begin(), left.end(), right.begin(),
                                  right.end(),
                                  std::inserter(different, different.end()));
    auto check = [](const QubitSet &actual,
                    const std::set<unsigned> &expected) {
      auto range = actual.set_bits();
      EXPECT_EQ((std::set<unsigned>{range.begin(), range.end()}), expected);
      EXPECT_EQ(actual.count(), expected.size());
      QubitSet rebuilt;
      for (unsigned q : expected)
        rebuilt.set(q);
      EXPECT_EQ(actual, rebuilt);
      EXPECT_EQ(actual.hash(), rebuilt.hash());
    };
    check(a | b, united);
    check(a & b, intersected);
    check(a ^ b, different);
    EXPECT_EQ(a.intersectionCount(b), intersected.size());
    EXPECT_EQ(a.unionCount(b), united.size());
    for (unsigned q : left)
      a.set(q, false);
    check(a, {});
  }
}

TEST(PBC, WideObservableBasisAndNormalization) {
  constexpr unsigned width = 433;
  pbc::CommutingObservables observed;
  std::vector<Pauli> measurements;
  for (unsigned q = 0; q < width; ++q) {
    Pauli z{{}, QubitSet::single(q)};
    ASSERT_TRUE(observed.insert(z));
    measurements.push_back(z);
  }
  EXPECT_EQ(observed.size(), width);
  EXPECT_FALSE(observed.insert({QubitSet::single(432), {}}));
  auto result = pbc::normalize(width, {{Kind::H, 432}, {Kind::CX, 432, 64}},
                               measurements);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.measurements.size(), width);
  EXPECT_EQ(result.measurements[432], (Pauli{QubitSet::single(432), {}}));
  EXPECT_EQ(result.measurements[64],
            (Pauli{QubitSet::single(432), QubitSet::single(64)}));
}

TEST(PBC, WideFrameBudgetIsCheckedBeforeAllocation) {
  constexpr unsigned width = 1u << 31;
  SearchBudget budget(100);
  auto result = pbc::normalize(width, {{Kind::H, 0}}, {}, &budget);
  EXPECT_FALSE(result);
  EXPECT_TRUE(budget.stopped());
  EXPECT_EQ(budget.work(), 0u);
  EXPECT_TRUE(pbc::normalize(width, {}, {}));
  EXPECT_EQ(tc::cost(width, {}), tc::Cost{});
}

TEST(TCount, HighQubitGateKeysDoNotAlias) {
  // The former six-bit packed key conflated CX(0,64) and CX(1,0).
  std::vector<Gate> input{{Kind::CX, 0, 64}, {Kind::CX, 1, 0}};
  tc::Options options;
  options.decode = false;
  auto result = tc::optimize(65, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.after.twoQubit, 2u);
  ASSERT_TRUE(cudaq::pauli::graph::equivalent(65, input, result.gates));
}

TEST(TCount, WideCommutingBlockDecodesAndChecksExactPhases) {
  constexpr unsigned width = 433;
  unsigned indices[] = {64, 127, 255, 432};
  std::vector<Gate> input;
  for (unsigned mask = 1; mask < 16; ++mask) {
    QubitSet z;
    for (unsigned i = 0; i < 4; ++i)
      if (mask & (1u << i))
        z.set(indices[i]);
    input.push_back(Gate::rotation({{}, z}, {1, 4}));
  }
  tc::Options options;
  options.maxWork = 10000000;
  auto result = tc::optimize(width, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_LT(result.stats.after.t, result.stats.before.t);
  EXPECT_GT(result.stats.verifiedBlocks, 0u);
  auto compress = [&](std::vector<Gate> gates) {
    auto index = [&](unsigned q) {
      for (unsigned i = 0; i < 4; ++i)
        if (indices[i] == q)
          return i;
      ADD_FAILURE() << "Unexpected active wire " << q;
      return 0u;
    };
    for (auto &g : gates) {
      if (g.kind == Kind::Rotate) {
        Pauli p;
        p.sign = g.pauli.sign;
        for (unsigned q : g.pauli.x.set_bits())
          p.x.set(index(q));
        for (unsigned q : g.pauli.z.set_bits())
          p.z.set(index(q));
        g.pauli = p;
      } else if (g.kind != Kind::Phase) {
        g.a = index(g.a);
        if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
          g.b = index(g.b);
      }
    }
    return gates;
  };
  assertUnitary(4, compress(input), compress(result.gates));
}

TEST(GateCount, PhaseSensitiveRandomUnitariesAndStrategyControls) {
  // Algebraic tests, not performance workloads: compare every matrix entry,
  // including scalar phase, for each independently enabled strategy family.
  std::mt19937 rng(20261103);
  for (unsigned method = 0; method < 4; ++method)
    for (unsigned n = 1; n <= 3; ++n)
      for (unsigned trial = 0; trial < 12; ++trial) {
        std::vector<Gate> input;
        for (unsigned j = 0; j < 35; ++j) {
          unsigned a = rng() % n, b = (a + 1) % n;
          unsigned k = rng() % 12;
          if (k < 6)
            input.push_back({Kind(k), a});
          else if (k < 9 && n > 1)
            input.push_back({Kind(k), a, b});
          else if (k == 9)
            input.push_back(Gate::phase({int(rng() % 17) - 8, 8}));
          else
            input.push_back(
                Gate::rotation({rng() % (1u << n), 1u << a, bool(rng() & 1)},
                               {int(rng() % 31) - 15, 4}));
        }
        gatecount::Options options;
        options.rounds = 2;
        options.propagate = method == 0 || method == 3;
        options.phasePlacement = method == 1 || method == 3;
        options.cliffordWindows = method == 2 || method == 3;
        SearchBudget budget(100000000);
        auto output = gatecount::optimize(n, input, options, budget, 8192);
        ASSERT_FALSE(budget.stopped());
        EXPECT_LE(tc::cost(n, output).gates, tc::cost(n, input).gates);
        EXPECT_LE(tc::cost(n, output).t, tc::cost(n, input).t);
        assertUnitary(n, input, output);
      }
}

TEST(GateCount, NativeTotalGateObjectiveAndBudgetFallback) {
  // A Clifford window can become one CZ without changing phase or wire order.
  std::vector<Gate> input{{Kind::H, 1}, {Kind::CX, 0, 1}, {Kind::H, 1}};
  OptimizerOptions options;
  options.enableTCount = options.enableGraphSynthesis = false;
  options.enableGateCount = true;
  options.objective = Objective::TotalGates;
  auto result = optimize(2, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.after.gates, 1);
  EXPECT_EQ(result.stats.after.twoQubit, 1);
  assertUnitary(2, input, result.gates);
  SearchBudget exhausted(1);
  auto unchanged = gatecount::optimize(2, input, {}, exhausted, 8192);
  EXPECT_TRUE(exhausted.stopped());
  ASSERT_EQ(unchanged.size(), input.size());
  assertUnitary(2, input, unchanged);
}

TEST(GateCount, RetainsImprovementsUnderAnEntanglerCeiling) {
  // Some shortest two-wire Clifford words trade extra entanglers for fewer
  // local gates. Exercise that trade-off, then require a useful, independently
  // equivalent alternative under the outer portfolio's zero-growth policy.
  std::mt19937 rng(20261104);
  bool exercisedRetry = false;
  for (unsigned trial = 0; trial < 512 && !exercisedRetry; ++trial) {
    std::vector<Gate> input;
    for (unsigned i = 0; i < 16; ++i) {
      input.push_back({Kind(rng() % 6), unsigned(rng() % 2)});
      if (i == 7)
        input.push_back({Kind::CX, 0, 1});
    }
    SearchBudget unrestrictedBudget(100000000);
    auto unrestricted =
        gatecount::optimize(2, input, {}, unrestrictedBudget, 8192);
    ASSERT_FALSE(unrestrictedBudget.stopped());
    if (tc::cost(2, unrestricted).twoQubit <= 1)
      continue;
    exercisedRetry = true;
    OptimizerOptions options;
    options.enableTCount = options.enableGraphSynthesis = false;
    options.enableGateCount = true;
    options.objective = Objective::TotalGates;
    options.maxTwoQubitGrowth = 0;
    options.maxWork = 100000000;
    Telemetry telemetry;
    auto result = optimize(2, input, options, nullptr, &telemetry);
    ASSERT_TRUE(result) << result.error;
    EXPECT_LT(result.stats.after.gates, result.stats.before.gates);
    EXPECT_LE(result.stats.after.twoQubit, 1);
    assertUnitary(2, input, unrestricted);
    assertUnitary(2, input, result.gates);
  }
  EXPECT_TRUE(exercisedRetry);
}

TEST(GateCount, WideSupportsPreservePhaseAndNeverIncreaseT) {
  std::vector<Gate> input{{Kind::X, 130},
                          {Kind::CX, 130, 432},
                          Gate::rotation({0, QubitSet::single(432)}, {1, 4}),
                          {Kind::X, 130},
                          {Kind::X, 432},
                          {Kind::CX, 130, 432}};
  SearchBudget budget(100000000);
  auto output = gatecount::optimize(433, input, {}, budget, 8192);
  ASSERT_FALSE(budget.stopped());
  EXPECT_LT(tc::cost(433, output).gates, tc::cost(433, input).gates);
  EXPECT_LE(tc::cost(433, output).t, tc::cost(433, input).t);
  auto compact = [](std::vector<Gate> gates) {
    for (auto &g : gates) {
      if (g.kind == Kind::Phase)
        continue;
      if (g.kind == Kind::Rotate) {
        g.pauli.x = unsigned(g.pauli.x.test(130)) |
                    (unsigned(g.pauli.x.test(432)) << 1);
        g.pauli.z = unsigned(g.pauli.z.test(130)) |
                    (unsigned(g.pauli.z.test(432)) << 1);
      } else {
        g.a = g.a == 432;
        g.b = g.b == 432;
      }
    }
    return gates;
  };
  assertUnitary(2, compact(input), compact(output));
}

TEST(GateCount, MixedPhasesExposeCancelablePaulis) {
  // Sdg Tdg = Z T (and S T = Z Tdg), including the Rz/T scalar
  // correction. The explicit Z passes through H and cancels the X.
  for (int sign : {-1, 1}) {
    std::vector<Gate> input{{sign < 0 ? Kind::Sdg : Kind::S, 0},
                            Gate::rotation({0, 1}, {sign, 4}),
                            {Kind::H, 0},
                            {Kind::X, 0},
                            {Kind::H, 0}};
    gatecount::Options options;
    options.cliffordWindows = false;
    options.phasePlacement = false;
    SearchBudget budget(1000000);
    auto output = gatecount::optimize(1, input, options, budget, 128);
    ASSERT_FALSE(budget.stopped());
    EXPECT_EQ(tc::cost(1, output).gates, 1u);
    EXPECT_EQ(tc::cost(1, output).t, 1u);
    assertUnitary(1, input, output);
  }
}

TEST(GateCount, LastPlacementUsesEndOfParityLifetime) {
  // Wire 0's phase may move past both CXs, including the CX on which it is
  // an unchanged control. Merely recording where a row was created misses
  // this placement and the one-wire Clifford reduction it exposes.
  std::vector<Gate> input{{Kind::S, 0}, {Kind::CX, 1, 2}, {Kind::CX, 0, 1},
                          {Kind::H, 0}, {Kind::Sdg, 0},   {Kind::H, 0}};
  gatecount::Options options;
  options.propagate = false;
  options.rounds = 1;
  SearchBudget budget(1000000);
  Telemetry telemetry;
  auto output = gatecount::optimize(3, input, options, budget, 128, &telemetry);
  ASSERT_FALSE(budget.stopped());
  EXPECT_EQ(tc::cost(3, output).gates, 5u);
  assertUnitary(3, input, output);
  bool improvedLast = false;
  for (const auto &event : telemetry.events)
    if (event.method == "gate-count.phase-last")
      improvedLast = event.after->gates < event.before->gates;
  EXPECT_TRUE(improvedLast);
}

TEST(TCount, VisitsVerifiedAlternativesBeforeBeamPruning) {
  // Forward and reverse merging place the resulting S on different sides of
  // the CX. Even a one-entry T beam must expose both to the composing search.
  std::vector<Gate> input{Gate::rotation({0, 1}, {1, 4}),
                          {Kind::CX, 0, 1},
                          Gate::rotation({0, 1}, {1, 4})};
  tc::Options options;
  options.beamWidth = options.rounds = 1;
  options.decode = false;
  unsigned visited = 0;
  auto visit = [&](const std::vector<Gate> &gates, const tc::Cost &cost,
                   uint64_t) {
    ++visited;
    EXPECT_EQ(cost, tc::cost(2, gates));
    EXPECT_EQ(cost.t, 0u);
    assertUnitary(2, input, gates);
  };
  auto observed =
      tc::optimize(2, input, options, nullptr, nullptr, 0, 0, visit);
  auto plain = tc::optimize(2, input, options);
  ASSERT_TRUE(observed);
  ASSERT_TRUE(plain);
  EXPECT_EQ(visited, 2u);
  EXPECT_EQ(observed.stats.after, plain.stats.after);
  assertUnitary(2, observed.gates, plain.gates);
}
