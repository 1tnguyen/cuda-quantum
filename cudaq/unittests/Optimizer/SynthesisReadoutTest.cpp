/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisReadout.h"
#include "gtest/gtest.h"
#include <array>
#include <bit>
#include <cmath>
#include <complex>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
namespace synthesis = cudaq::pauli::graph::detail;

namespace {
using Complex = std::complex<double>;
using State = std::array<Complex, 4>;
constexpr Complex imaginary{0, 1};

// Independent two-qubit matrix oracle: no production Pauli conjugation.
State applyPauli(const State &state, const Pauli &p) {
  unsigned x = unsigned(p.x.test(0)) | (unsigned(p.x.test(1)) << 1);
  unsigned z = unsigned(p.z.test(0)) | (unsigned(p.z.test(1)) << 1);
  State result{};
  for (unsigned i = 0; i < 4; ++i) {
    Complex phase = std::pow(imaginary, std::popcount(x & z));
    if ((std::popcount(i & z) & 1) != p.sign)
      phase = -phase;
    result[i ^ x] = phase * state[i];
  }
  return result;
}

State applyCircuit(State state, const std::vector<Gate> &gates) {
  for (const auto &gate : gates) {
    unsigned a = 1u << gate.a, b = 1u << gate.b;
    State before = state;
    for (unsigned i = 0; i < 4; ++i) {
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
        ADD_FAILURE() << "unexpected readout gate";
        return {};
      }
    }
  }
  return state;
}

void checkEffects(const std::vector<Pauli> &observables,
                  const std::vector<unsigned> &readoutWires,
                  const std::vector<Gate> &gates) {
  std::array<State, 4> columns;
  for (unsigned column = 0; column < 4; ++column) {
    State basis{};
    basis[column] = 1;
    columns[column] = applyCircuit(basis, gates);
  }
  for (unsigned outcome = 0; outcome < (1u << observables.size()); ++outcome)
    for (unsigned column = 0; column < 4; ++column) {
      State effect{};
      effect[column] = 1;
      for (unsigned i = 0; i < observables.size(); ++i) {
        auto pauliEffect = applyPauli(effect, observables[i]);
        double sign = (outcome & (1u << i)) ? -1 : 1;
        for (unsigned row = 0; row < 4; ++row)
          effect[row] = (effect[row] + sign * pauliEffect[row]) / 2.0;
      }
      for (unsigned row = 0; row < 4; ++row) {
        Complex reconstructed = 0;
        for (unsigned output = 0; output < 4; ++output) {
          bool matches = true;
          for (unsigned i = 0; i < observables.size(); ++i)
            matches &= bool(output & (1u << readoutWires[i])) ==
                       bool(outcome & (1u << i));
          if (matches)
            reconstructed +=
                std::conj(columns[row][output]) * columns[column][output];
        }
        EXPECT_LT(std::abs(reconstructed - effect[row]), 1e-12);
      }
    }
}
} // namespace

TEST(SynthesisReadout, EveryNativeProfilePreservesOrderedSignedJointEffects) {
  // Ordered -XX, ZZ outcomes go to physical wires 1, 0, respectively.
  std::vector<Pauli> observables{{3, 0, true}, {0, 3}};
  std::vector<unsigned> wires{1, 0};
  for (unsigned profile = 0; profile < 3; ++profile) {
    Options options;
    options.nativeCX = profile != 2;
    options.nativeCZ = profile != 1;
    options.nativeSwap = profile == 0;
    SearchBudget budget(1000000);
    auto result = synthesis::generateReadoutCandidates(2, observables, wires,
                                                       options, {}, budget);
    ASSERT_EQ(result.status, synthesis::ReadoutStatus::Complete)
        << result.error;
    ASSERT_FALSE(result.candidates.empty());
    EXPECT_LE(result.candidates.size(), 8u);
    for (const auto &candidate : result.candidates) {
      for (const auto &gate : candidate.gates()) {
        if (!options.nativeCX)
          EXPECT_NE(gate.kind, Kind::CX);
        if (!options.nativeCZ)
          EXPECT_NE(gate.kind, Kind::CZ);
        if (!options.nativeSwap)
          EXPECT_NE(gate.kind, Kind::Swap);
      }
      checkEffects(observables, wires, candidate.gates());
    }
  }
}

TEST(SynthesisReadout,
     PartialReadoutPreservesEffectsWithoutRestoringOtherWire) {
  std::vector<Pauli> observables{{1, 3, true}}; // -YZ
  SearchBudget budget(1000000);
  auto result =
      synthesis::generateReadoutCandidates(2, observables, {1}, {}, {}, budget);
  ASSERT_EQ(result.status, synthesis::ReadoutStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  for (const auto &candidate : result.candidates)
    checkEffects(observables, {1}, candidate.gates());
}

TEST(SynthesisReadout, OneCandidateCapTriesMinimumSupportBeforeOriginalOrder) {
  const std::vector<Pauli> observables{{0, 3}, {0, 1}};
  const std::vector<unsigned> wires{0, 1};
  synthesis::ReadoutLimits limits;
  limits.maxCandidates = 1;
  SearchBudget budget(1000000);
  auto result = synthesis::generateReadoutCandidates(2, observables, wires, {},
                                                     limits, budget);
  ASSERT_EQ(result.status, synthesis::ReadoutStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 1u);
  const auto &tail = result.candidates.front().gates();
  ASSERT_EQ(tail.size(), 2u);
  // Resolve Z0 -> Z1 first with SWAP, then ZZ -> Z0 with CX(1,0).
  EXPECT_EQ(tail[0].kind, Kind::Swap);
  EXPECT_EQ(tail[1].kind, Kind::CX);
  EXPECT_EQ(tail[1].a, 1u);
  EXPECT_EQ(tail[1].b, 0u);
  auto original = synthesizeReadout(2, observables, wires, {});
  ASSERT_TRUE(original) << original.error;
  ASSERT_EQ(original.gates.size(), 2u);
  EXPECT_EQ(original.gates.front().kind, Kind::CX);
  checkEffects(observables, wires, tail);
}

TEST(SynthesisReadout, DuplicateCompletedWordsDoNotConsumeCandidateCap) {
  synthesis::ReadoutLimits limits;
  limits.maxCandidates = 2;
  SearchBudget budget(1000000);
  auto result = synthesis::generateReadoutCandidates(2, {{0, 1}}, {0}, {},
                                                     limits, budget);
  ASSERT_EQ(result.status, synthesis::ReadoutStatus::Complete) << result.error;
  // Every policy and pivot variant has the same empty completion.
  ASSERT_EQ(result.candidates.size(), 1u);
  EXPECT_TRUE(result.candidates.front().gates().empty());
  EXPECT_GT(result.work, 0u);

  limits.maxCandidates = 0;
  SearchBudget noCandidates(1000000);
  auto unsupported = synthesis::generateReadoutCandidates(2, {{0, 1}}, {0}, {},
                                                          limits, noCandidates);
  EXPECT_EQ(unsupported.status, synthesis::ReadoutStatus::Unsupported);
  EXPECT_TRUE(unsupported.candidates.empty());
}

TEST(SynthesisReadout, ExhaustionNeverPublishesPartialOrUnverifiedTail) {
  std::vector<Pauli> observables{{3, 0, true}, {0, 3}};
  std::vector<unsigned> wires{1, 0};
  SearchBudget none(0);
  auto empty =
      synthesis::generateReadoutCandidates(2, observables, wires, {}, {}, none);
  EXPECT_EQ(empty.status, synthesis::ReadoutStatus::Exhausted);
  EXPECT_TRUE(empty.candidates.empty());

  synthesis::ReadoutLimits limits;
  limits.maxCandidates = 1;
  SearchBudget complete(1000000);
  auto first = synthesis::generateReadoutCandidates(2, observables, wires, {},
                                                    limits, complete);
  ASSERT_EQ(first.status, synthesis::ReadoutStatus::Complete) << first.error;
  ASSERT_EQ(first.candidates.size(), 1u);
  // One less unit exhausts during the independent final replay.
  SearchBudget shortProof(first.work - 1);
  auto unfinished = synthesis::generateReadoutCandidates(
      2, observables, wires, {}, limits, shortProof);
  EXPECT_EQ(unfinished.status, synthesis::ReadoutStatus::Exhausted);
  EXPECT_TRUE(unfinished.candidates.empty());

  limits.maxCandidates = 8;
  SearchBudget oneOnly(first.work);
  auto retained = synthesis::generateReadoutCandidates(2, observables, wires,
                                                       {}, limits, oneOnly);
  EXPECT_EQ(retained.status, synthesis::ReadoutStatus::Exhausted);
  ASSERT_EQ(retained.candidates.size(), 1u);
  checkEffects(observables, wires, retained.candidates.front().gates());

  limits.maxWork = first.work;
  SearchBudget largerShared(1000000);
  auto local = synthesis::generateReadoutCandidates(2, observables, wires, {},
                                                    limits, largerShared);
  EXPECT_EQ(local.status, synthesis::ReadoutStatus::Exhausted);
  EXPECT_FALSE(largerShared.stopped());
  ASSERT_EQ(local.candidates.size(), 1u);
}

TEST(SynthesisReadout, DynamicWidthAndRepeatedPhysicalOutcomes) {
  unsigned last = 128;
  std::vector<Pauli> observables{{0, QubitSet::single(last), true},
                                 {0, 1},
                                 {0, QubitSet::single(last), true}};
  std::vector<unsigned> wires{0, last, 0};
  SearchBudget budget(1000000);
  auto result = synthesis::generateReadoutCandidates(last + 1, observables,
                                                     wires, {}, {}, budget);
  ASSERT_EQ(result.status, synthesis::ReadoutStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  for (const auto &candidate : result.candidates)
    for (size_t i = 0; i < observables.size(); ++i) {
      Pauli row = observables[i];
      for (const auto &gate : candidate.gates())
        row = tcount::detail::conjugate(row, gate);
      EXPECT_EQ(row, (Pauli{0, QubitSet::single(wires[i])}));
    }
}

TEST(SynthesisReadout, IndependentReplayRejectsChangedSignsAndOutputOrder) {
  const std::vector<Pauli> observables{{0, 2, true}, {0, 1}};
  const std::vector<unsigned> wires{0, 1};
  const std::vector<Gate> correct{{Kind::Swap, 0, 1}, {Kind::X, 0}};
  for (const auto &wrong :
       {std::vector<Gate>{{Kind::Swap, 0, 1}},
        std::vector<Gate>{
            {Kind::Swap, 0, 1}, {Kind::X, 0}, {Kind::Swap, 0, 1}}}) {
    synthesis::ReadoutCandidates result;
    SearchBudget shared(1000000);
    synthesis::readout_detail::Budget budget(shared, 1000000, result);
    EXPECT_TRUE(synthesis::readout_detail::verify(2, observables, wires,
                                                  correct, budget));
    EXPECT_FALSE(synthesis::readout_detail::verify(2, observables, wires, wrong,
                                                   budget));
    EXPECT_FALSE(shared.stopped());
  }
}

TEST(SynthesisReadout, UnsupportedContractsAndStructuralLimitsAreExplicit) {
  for (const auto &observables :
       {std::vector<Pauli>{{1, 0}, {0, 1}}, std::vector<Pauli>{{0, 1}, {0, 1}},
        std::vector<Pauli>{{0, 1}, {0, 1, true}}}) {
    SearchBudget budget(1000000);
    auto result = synthesis::generateReadoutCandidates(2, observables, {0, 1},
                                                       {}, {}, budget);
    EXPECT_EQ(result.status, synthesis::ReadoutStatus::Unsupported);
    EXPECT_TRUE(result.candidates.empty());
  }
  Options options;
  options.maxGates = 1;
  SearchBudget budget(1000000);
  auto oversized = synthesis::generateReadoutCandidates(
      2, {{3, 0, true}, {0, 3}}, {1, 0}, options, {}, budget);
  EXPECT_EQ(oversized.status, synthesis::ReadoutStatus::Exhausted);
  EXPECT_TRUE(oversized.candidates.empty());

  SearchBudget tiny(1);
  auto wide = synthesis::generateReadoutCandidates(1000000000, {{0, 1}}, {0},
                                                   {}, {}, tiny);
  EXPECT_EQ(wide.status, synthesis::ReadoutStatus::Exhausted);
  EXPECT_TRUE(wide.candidates.empty());
}
