/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisAlgebra.h"
#include "gtest/gtest.h"
#include <bit>
#include <cmath>
#include <complex>
#include <numbers>

using namespace cudaq::pauli;
namespace synthesis = cudaq::pauli::graph::detail;

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

State project(State state, const std::vector<Pauli> &observables,
              unsigned bits) {
  for (size_t i = 0; i < observables.size(); ++i) {
    auto transformed = pauli(state, observables[i]);
    double sign = (bits & (1u << i)) ? -1.0 : 1.0;
    for (size_t row = 0; row < state.size(); ++row)
      state[row] = (state[row] + sign * transformed[row]) / 2.0;
  }
  return state;
}

void checkEffects(const synthesis::MeasuredProblem &source,
                  const synthesis::CertifiedAlgebraAlternative &alternative) {
  size_t size = size_t(1) << source.width;
  std::vector<State> before, after;
  for (size_t column = 0; column < size; ++column) {
    State state(size);
    state[column] = 1;
    before.push_back(apply(state, source.rotations));
    after.push_back(apply(state, alternative.rotations()));
  }
  for (unsigned bits = 0; bits < (1u << source.observables.size()); ++bits)
    for (size_t column = 0; column < size; ++column) {
      auto a = project(before[column], source.observables, bits);
      auto b = project(after[column], alternative.observables(), bits);
      for (size_t row = 0; row < size; ++row) {
        Complex lhs = 0, rhs = 0;
        for (size_t k = 0; k < size; ++k) {
          lhs += std::conj(before[row][k]) * a[k];
          rhs += std::conj(after[row][k]) * b[k];
        }
        EXPECT_LT(std::abs(lhs - rhs), 1e-10);
      }
    }
}
} // namespace

TEST(SynthesisAlgebra, SignedMultiwireMergeOwnsSourceAndPreservesJointEffects) {
  std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4}),
                         Gate::rotation({3, 0}, {1, 4}),
                         Gate::rotation({0, 3, true}, {1, 4})};
  std::vector<Pauli> observables{{0, 2, true}, {0, 1}};
  std::vector<unsigned> wires{0, 1};
  synthesis::MeasuredProblem source{2, word, observables, wires};
  SearchBudget budget(100000000);
  auto result = synthesis::generateAlgebraAlternatives(source, {}, budget);
  ASSERT_EQ(result.status, synthesis::AlgebraStatus::Complete) << result.error;
  ASSERT_FALSE(result.alternatives.empty());
  for (const auto &alternative : result.alternatives) {
    EXPECT_TRUE(alternative.matchesSource(source));
    EXPECT_EQ(alternative.readoutWires(), wires);
    EXPECT_EQ(alternative.rotations().size(), 1u);
    checkUnitary(2, word, alternative.checkedWord());
    checkEffects(source, alternative);
  }
  word.clear();
  EXPECT_FALSE(result.alternatives.front().matchesSource(source));
  EXPECT_EQ(result.alternatives.front().source().rotations.size(), 3u);
}

TEST(SynthesisAlgebra, DecodingChecksScalarPhaseBeforeMeasuredNormalization) {
  std::vector<Gate> word;
  for (unsigned subset = 1; subset < 16; ++subset)
    word.push_back(Gate::rotation({0, subset}, {1, 4}));
  std::vector<Pauli> observables{{1, 0, true}, {8, 0}};
  std::vector<unsigned> wires{2, 0};
  synthesis::MeasuredProblem source{4, word, observables, wires};
  SearchBudget budget(100000000);
  auto result = synthesis::generateAlgebraAlternatives(source, {}, budget);
  ASSERT_EQ(result.status, synthesis::AlgebraStatus::Complete) << result.error;
  ASSERT_GT(result.backend.verifiedBlocks, 0u);
  ASSERT_FALSE(result.alternatives.empty());
  for (const auto &alternative : result.alternatives) {
    EXPECT_TRUE(alternative.rotations().empty());
    EXPECT_EQ(alternative.observables(), observables);
    checkUnitary(4, word, alternative.checkedWord());
    checkEffects(source, alternative);
  }
}

TEST(SynthesisAlgebra, OpaqueRotationsAreRetainedWhileCommutingTermsMerge) {
  std::vector<Gate> word{Gate::rotation({0, 3}, {1, 4}),
                         Gate::rotation({0, 1}, {0, 1, 3}),
                         Gate::rotation({0, 3}, {-1, 4})};
  std::vector<Pauli> observables{{1, 0, true}};
  std::vector<unsigned> wires{1};
  synthesis::MeasuredProblem source{2, word, observables, wires};
  SearchBudget budget(100000000);
  auto result = synthesis::generateAlgebraAlternatives(source, {}, budget);
  ASSERT_EQ(result.status, synthesis::AlgebraStatus::Complete) << result.error;
  ASSERT_FALSE(result.alternatives.empty());
  for (const auto &alternative : result.alternatives) {
    ASSERT_EQ(alternative.rotations().size(), 1u);
    EXPECT_EQ(alternative.rotations().front().angle.symbol, 3u);
    checkUnitary(2, word, alternative.checkedWord());
    checkEffects(source, alternative);
  }
}

TEST(SynthesisAlgebra, WideSupportAndBudgetFailureDoNotWeakenSourceContract) {
  Pauli axis{QubitSet::single(64), QubitSet::single(128)};
  std::vector<Gate> word{Gate::rotation(axis, {1, 4}),
                         Gate::rotation(axis, {-1, 4})};
  std::vector<Pauli> observables{{0, QubitSet::single(128), true}};
  std::vector<unsigned> wires{64};
  synthesis::MeasuredProblem source{129, word, observables, wires};
  SearchBudget budget(100000000);
  auto result = synthesis::generateAlgebraAlternatives(source, {}, budget);
  ASSERT_EQ(result.status, synthesis::AlgebraStatus::Complete) << result.error;
  ASSERT_FALSE(result.alternatives.empty());
  EXPECT_TRUE(result.alternatives.front().rotations().empty());
  EXPECT_EQ(result.alternatives.front().observables(), observables);
  EXPECT_EQ(result.alternatives.front().readoutWires(), wires);

  SearchBudget incompleteProof(result.work - 1);
  auto incomplete =
      synthesis::generateAlgebraAlternatives(source, {}, incompleteProof);
  EXPECT_EQ(incomplete.status, synthesis::AlgebraStatus::Exhausted);
  EXPECT_TRUE(incomplete.alternatives.empty());

  SearchBudget none(0);
  auto failed = synthesis::generateAlgebraAlternatives(source, {}, none);
  EXPECT_EQ(failed.status, synthesis::AlgebraStatus::Exhausted);
  EXPECT_TRUE(failed.alternatives.empty());
  EXPECT_EQ(word.size(), 2u);
  EXPECT_TRUE(observables.front().sign);

  synthesis::AlgebraLimits limits;
  limits.maxTemporaryGates = 1;
  SearchBudget bounded(100000000);
  auto oversized =
      synthesis::generateAlgebraAlternatives(source, limits, bounded);
  EXPECT_EQ(oversized.status, synthesis::AlgebraStatus::Unsupported);
  EXPECT_TRUE(oversized.alternatives.empty());
  EXPECT_EQ(oversized.backend.candidates, 0u);

  limits = {};
  limits.maxBackendWork = 1;
  SearchBudget limitedBackend(100000000);
  auto stopped =
      synthesis::generateAlgebraAlternatives(source, limits, limitedBackend);
  EXPECT_EQ(stopped.status, synthesis::AlgebraStatus::Exhausted);
  EXPECT_TRUE(stopped.alternatives.empty());
  EXPECT_FALSE(limitedBackend.stopped());
}
