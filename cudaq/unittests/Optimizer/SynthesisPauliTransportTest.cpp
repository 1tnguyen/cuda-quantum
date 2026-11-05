/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SignedCliffordFrames.h"
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

void checkChannel(const std::vector<Gate> &before,
                  const std::vector<Gate> &after) {
  std::array<State, 4> a, b;
  for (unsigned column = 0; column < 4; ++column) {
    State basis(4);
    basis[column] = 1;
    a[column] = apply(basis, before);
    b[column] = apply(basis, after);
  }
  for (unsigned i = 0; i < 4; ++i)
    for (unsigned j = 0; j < 4; ++j)
      for (unsigned row = 0; row < 4; ++row)
        for (unsigned column = 0; column < 4; ++column)
          EXPECT_LT(std::abs(a[i][row] * std::conj(a[j][column]) -
                             b[i][row] * std::conj(b[j][column])),
                    1e-11);
}

// Compare every matrix entry of each ordered output effect, including its
// off-diagonal entries. This checks arbitrary inputs, not only basis-state
// probabilities, and retains the physical identity of each readout bit.
void checkTerminalEffects(const std::vector<Gate> &before,
                          const std::vector<Gate> &after,
                          const std::vector<unsigned> &readouts) {
  std::array<State, 4> a, b;
  for (unsigned column = 0; column < 4; ++column) {
    State basis(4);
    basis[column] = 1;
    a[column] = apply(basis, before);
    b[column] = apply(basis, after);
  }
  for (unsigned outcome = 0; outcome < (1u << readouts.size()); ++outcome)
    for (unsigned i = 0; i < 4; ++i)
      for (unsigned j = 0; j < 4; ++j) {
        Complex left = 0, right = 0;
        for (unsigned row = 0; row < 4; ++row) {
          unsigned observed = 0;
          for (unsigned bit = 0; bit < readouts.size(); ++bit)
            observed |= ((row >> readouts[bit]) & 1u) << bit;
          if (observed == outcome) {
            left += std::conj(a[i][row]) * a[j][row];
            right += std::conj(b[i][row]) * b[j][row];
          }
        }
        EXPECT_LT(std::abs(left - right), 1e-11);
      }
}

bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}

void checkRefinement(const std::vector<Gate> &input, unsigned expectedSize) {
  SearchBudget budget(1000000);
  std::string error;
  auto result =
      synthesis::optimizeSignedFrames(2, input, {0, 1}, &budget, error);
  ASSERT_TRUE(result) << error;
  EXPECT_EQ(result->size(), expectedSize);
  auto before = tcount::cost(2, input), after = tcount::cost(2, *result);
  EXPECT_LE(after.gates, before.gates);
  EXPECT_LE(after.t, before.t);
  EXPECT_LE(after.twoQubit, before.twoQubit);
  EXPECT_LE(after.depth, before.depth);
  EXPECT_LE(after.tDepth, before.tDepth);
  checkChannel(input, *result);
}
} // namespace

TEST(SynthesisPauliTransport, TargetXCompletesTheSdgHCXIdentity) {
  const std::vector<Gate> input{
      {Kind::Sdg, 1}, {Kind::H, 1}, {Kind::CX, 0, 1}, {Kind::X, 1}};
  const std::vector<Gate> expected{
      {Kind::S, 1}, {Kind::H, 1}, {Kind::CX, 0, 1}};
  // X(target) commutes with CX, X H = H Z, and Z Sdg = S.
  checkUnitary(2, input, expected);
  checkRefinement(input, 3);
}

TEST(SynthesisPauliTransport, CompatibleCXRolesAndBothCZOperandsFuseCliffords) {
  auto opaqueX = [](unsigned q) {
    return Gate::rotation({QubitSet::single(q), 0}, {0, 1, 3});
  };
  checkRefinement({{Kind::Sdg, 0}, {Kind::CX, 0, 1}, {Kind::Z, 0}, opaqueX(0)},
                  3);
  checkRefinement({{Kind::Sdg, 1},
                   {Kind::H, 1},
                   {Kind::CX, 0, 1},
                   {Kind::X, 1},
                   Gate::rotation({0, 2}, {0, 1, 3})},
                  4);
  for (unsigned q : {0u, 1u})
    checkRefinement(
        {{Kind::Sdg, q}, {Kind::CZ, 0, 1}, {Kind::Z, q}, opaqueX(q)}, 3);
}

TEST(SynthesisPauliTransport, SignedOpaqueAxesCommuteWithoutChangingAngles) {
  for (bool negative : {false, true}) {
    std::vector<Gate> input{{Kind::Sdg, 0},
                            Gate::rotation({0, 1, negative}, {0, 1, 3}),
                            {Kind::Z, 0},
                            Gate::rotation({1, 0}, {0, 1, 5})};
    checkRefinement(input, 3);
    std::string error;
    auto transported = synthesis::transportPaulis(2, input, nullptr, error);
    ASSERT_TRUE(transported) << error;
    ASSERT_EQ(transported->size(), input.size());
    EXPECT_EQ((*transported)[2].pauli, input[1].pauli);
    EXPECT_EQ((*transported)[2].angle, input[1].angle);
  }
}

TEST(SynthesisPauliTransport, NoncommutingOperationsRemainBarriers) {
  for (const auto &[boundary, correction] :
       {std::pair{Gate{Kind::CX, 0, 1}, Gate{Kind::X, 0}},
        std::pair{Gate{Kind::CX, 0, 1}, Gate{Kind::Z, 1}},
        std::pair{Gate{Kind::CZ, 0, 1}, Gate{Kind::X, 0}},
        std::pair{Gate{Kind::CZ, 0, 1}, Gate{Kind::Y, 1}},
        std::pair{Gate::rotation({1, 0, true}, {0, 1, 9}), Gate{Kind::Z, 0}}}) {
    std::vector<Gate> input{{Kind::Sdg, correction.a}, boundary, correction};
    std::string error;
    auto transported = synthesis::transportPaulis(2, input, nullptr, error);
    ASSERT_TRUE(transported) << error;
    EXPECT_TRUE(sameWord(input, *transported));
  }
}

TEST(SynthesisPauliTransport, IndependentReplayRejectsCorruptedMovesAndWords) {
  std::vector<Gate> input{{Kind::Sdg, 0}, {Kind::CX, 0, 1}, {Kind::Z, 0}};
  std::vector<Gate> moved{{Kind::Sdg, 0}, {Kind::Z, 0}, {Kind::CX, 0, 1}};
  std::string error;
  EXPECT_TRUE(synthesis::verifyPauliTransport(2, input, moved, {{2, 1}},
                                              nullptr, error));
  moved[1].a = 1;
  EXPECT_FALSE(synthesis::verifyPauliTransport(2, input, moved, {{2, 1}},
                                               nullptr, error));
  EXPECT_FALSE(error.empty());
  error.clear();
  input[2].kind = Kind::X;
  moved[1] = {Kind::X, 0};
  EXPECT_FALSE(synthesis::verifyPauliTransport(2, input, moved, {{2, 1}},
                                               nullptr, error));
  EXPECT_FALSE(error.empty());
  error.clear();
  EXPECT_FALSE(synthesis::verifyPauliTransport(2, input, moved, {{9, 0}},
                                               nullptr, error));
  EXPECT_FALSE(error.empty());
}

TEST(SynthesisPauliTransport,
     OptionalExhaustionRetainsCheckedResultAndParentReserve) {
  std::vector<Gate> input{
      {Kind::Sdg, 1}, {Kind::H, 1}, {Kind::CX, 0, 1}, {Kind::X, 1}};
  std::string error;
  SearchBudget initial(1000000);
  auto baseline =
      synthesis::optimizeLocalSignedFrames(2, input, {0, 1}, &initial, error);
  ASSERT_TRUE(baseline) << error;
  SearchBudget limited(initial.work() + 2);
  auto result =
      synthesis::optimizeSignedFrames(2, input, {0, 1}, &limited, error);
  ASSERT_TRUE(result) << error;
  EXPECT_TRUE(error.empty());
  EXPECT_TRUE(sameWord(*result, *baseline));
  EXPECT_FALSE(limited.stopped());
  EXPECT_EQ(limited.remaining(), 2u);
  EXPECT_TRUE(limited.take(2));
  checkChannel(input, *result);
}

TEST(SynthesisPauliTransport, HighWireIndicesKeepTheirPhysicalIdentity) {
  std::vector<Gate> input{
      {Kind::Sdg, 128}, {Kind::H, 128}, {Kind::CX, 64, 128}, {Kind::X, 128}};
  std::string error;
  SearchBudget budget(1000000);
  auto result =
      synthesis::optimizeSignedFrames(129, input, {64, 128}, &budget, error);
  ASSERT_TRUE(result) << error;
  EXPECT_EQ(result->size(), 3u);
  auto two = *result;
  for (auto &gate : two) {
    gate.a = gate.a == 128 ? 1 : 0;
    if (gate.kind == Kind::CX)
      gate.b = gate.b == 128 ? 1 : 0;
  }
  checkChannel({{Kind::Sdg, 1}, {Kind::H, 1}, {Kind::CX, 0, 1}, {Kind::X, 1}},
               two);
}

TEST(SynthesisPauliTransport, SwapRelocatesCorrectionsBeforeLocalFusion) {
  const std::vector<Gate> input{
      {Kind::Sdg, 0}, {Kind::H, 0}, {Kind::Swap, 0, 1}, {Kind::X, 1}};
  const std::vector<Gate> expected{
      {Kind::S, 0}, {Kind::H, 0}, {Kind::Swap, 0, 1}};
  checkUnitary(2, input, expected);
  checkRefinement(input, 3);

  std::vector<Gate> moved{
      {Kind::Sdg, 0}, {Kind::H, 0}, {Kind::X, 0}, {Kind::Swap, 0, 1}};
  std::string error;
  EXPECT_TRUE(synthesis::verifyPauliTransport(2, input, moved, {{3, 2}},
                                              nullptr, error));
  moved[2].a = 1;
  EXPECT_FALSE(synthesis::verifyPauliTransport(2, input, moved, {{3, 2}},
                                               nullptr, error));
  EXPECT_FALSE(error.empty());
}

TEST(SynthesisPauliTransport, CommutingPhaseCliffordsCancelAcrossEntanglers) {
  const std::vector<Gate> input{{Kind::S, 0},
                                {Kind::CZ, 0, 1},
                                {Kind::CX, 0, 1},
                                {Kind::Sdg, 0},
                                {Kind::H, 0}};
  const std::vector<Gate> expected{
      {Kind::CZ, 0, 1}, {Kind::CX, 0, 1}, {Kind::H, 0}};
  checkUnitary(2, input, expected);
  checkRefinement(input, 3);
  // S on a CX target does not commute, even though its control counterpart
  // does. The complete signed generator replay must distinguish these roles.
  EXPECT_FALSE(synthesis::verifyLocalCliffordInterchange(
      {Kind::S, 1}, {Kind::CX, 0, 1}, {Kind::S, 1}, true));
}

TEST(SynthesisPauliTransport, ForwardTransportUsesFixedTerminalEffects) {
  struct Case {
    std::vector<Gate> gates;
    std::vector<unsigned> readouts;
    unsigned expectedSize;
  };
  const std::vector<Case> cases{
      {{{Kind::S, 0}, {Kind::CX, 0, 1}}, {1}, 1},
      {{{Kind::S, 0}, {Kind::CX, 0, 1}}, {1, 0}, 1},
      {{{Kind::S, 0}, {Kind::Swap, 0, 1}}, {1}, 1},
      {{{Kind::X, 0}, {Kind::Swap, 0, 1}}, {0}, 1},
      // The relocated X flips a requested bit. It cannot be discarded or
      // silently absorbed by changing the readout association.
      {{{Kind::X, 0}, {Kind::Swap, 0, 1}}, {1}, 2},
      {{{Kind::X, 0}, {Kind::Swap, 0, 1}}, {1, 0}, 2}};
  for (const auto &test : cases) {
    SearchBudget budget(1000000);
    std::string error;
    auto result = synthesis::optimizeSignedFrames(2, test.gates, test.readouts,
                                                  &budget, error);
    ASSERT_TRUE(result) << error;
    EXPECT_EQ(result->size(), test.expectedSize);
    auto before = tcount::cost(2, test.gates), after = tcount::cost(2, *result);
    EXPECT_LE(after.gates, before.gates);
    EXPECT_LE(after.t, before.t);
    EXPECT_LE(after.twoQubit, before.twoQubit);
    EXPECT_LE(after.depth, before.depth);
    EXPECT_LE(after.tDepth, before.tDepth);
    checkTerminalEffects(test.gates, *result, test.readouts);
  }
}

TEST(SynthesisPauliTransport,
     PrimitiveCompletionPrecedesBroaderCostlierSearch) {
  const std::vector<Gate> input{
      {Kind::Sdg, 1}, {Kind::H, 1}, {Kind::CX, 0, 1}, {Kind::X, 1}};
  std::string error;
  SearchBudget first(1000000);
  auto baseline =
      synthesis::optimizeLocalSignedFrames(2, input, {0, 1}, &first, error);
  ASSERT_TRUE(baseline) << error;

  // Reproduce the old primitive-only stage's exact work, including its proof,
  // repeated local DP and final metric charge. This is the compatibility
  // allowance the broader provider must not spend before trying that stage.
  SearchBudget primitive(1000000);
  auto moved = synthesis::transportPaulis(2, *baseline, &primitive, error);
  ASSERT_TRUE(moved) << error;
  auto oldCandidate = synthesis::optimizeLocalSignedFrames(2, *moved, {0, 1},
                                                           &primitive, error);
  ASSERT_TRUE(oldCandidate) << error;
  uint64_t comparisonWork = 3;
  for (const auto *word : {&*baseline, &*oldCandidate})
    for (const auto &gate : *word)
      comparisonWork +=
          gate.kind == Kind::Rotate ? 6 * uint64_t(gate.pauli.weight()) + 1 : 1;
  ASSERT_TRUE(primitive.take(comparisonWork));
  ASSERT_EQ(oldCandidate->size(), 3u);
  const uint64_t legacyAllowance = primitive.work();

  // Here generic transport proposes the SAME reordered word, but its more
  // expensive full-generator proof exhausts the old completion allowance.
  // Replacing the primitive generator therefore loses a valid incumbent even
  // without any semantic error or difference in the proposed gate sequence.
  SearchBudget generic(legacyAllowance);
  auto genericMoved =
      synthesis::transportLocalCliffords(2, *baseline, false, &generic, error);
  ASSERT_TRUE(genericMoved) << error;
  EXPECT_TRUE(sameWord(*genericMoved, *moved));
  auto genericCandidate = synthesis::optimizeLocalSignedFrames(
      2, *genericMoved, {0, 1}, &generic, error);
  bool genericCompleted = genericCandidate && generic.take(comparisonWork);
  EXPECT_FALSE(genericCompleted);
  EXPECT_TRUE(error.empty());

  SearchBudget routesParent(2 * legacyAllowance);
  SearchBudget primitiveRoute(routesParent, legacyAllowance);
  std::optional<tcount::Cost> completedCost;
  auto retained = synthesis::refineSignedFrameRoute(
      2, *baseline, {0, 1}, true, primitiveRoute, completedCost, error);
  ASSERT_TRUE(retained) << error;
  ASSERT_TRUE(completedCost);
  EXPECT_TRUE(sameWord(*retained, *oldCandidate));
  EXPECT_EQ(*completedCost, tcount::cost(2, *oldCandidate));
  EXPECT_TRUE(primitiveRoute.stopped());
  EXPECT_FALSE(routesParent.stopped());
  EXPECT_EQ(routesParent.remaining(), legacyAllowance);

  SearchBudget parent(first.work() + 2 * legacyAllowance);
  auto result =
      synthesis::optimizeSignedFrames(2, input, {0, 1}, &parent, error);
  ASSERT_TRUE(result) << error;
  EXPECT_TRUE(sameWord(*result, *oldCandidate));
  EXPECT_FALSE(parent.stopped());
  EXPECT_GE(parent.remaining(), legacyAllowance);
  checkChannel(input, *result);
}

TEST(SynthesisPauliTransport, CompleteRouteGuardAcceptsEqualGateDepthGain) {
  const tcount::Cost incumbent{7, 5, 12, 19, 4};
  auto candidate = incumbent;
  EXPECT_FALSE(synthesis::signedFrameRouteDominates(candidate, incumbent));
  --candidate.depth;
  EXPECT_TRUE(synthesis::signedFrameRouteDominates(candidate, incumbent));
  EXPECT_FALSE(synthesis::signedFrameRouteDominates(incumbent, candidate));

  // A gain in any one field cannot hide a regression in another. In
  // particular, equal total gates is a legitimate cross-route comparison.
  for (auto field : {&tcount::Cost::gates, &tcount::Cost::t,
                     &tcount::Cost::twoQubit, &tcount::Cost::tDepth}) {
    auto regressed = candidate;
    ++(regressed.*field);
    EXPECT_FALSE(synthesis::signedFrameRouteDominates(regressed, incumbent));
  }
  candidate = incumbent;
  --candidate.tDepth;
  EXPECT_TRUE(synthesis::signedFrameRouteDominates(candidate, incumbent));
  ++candidate.depth;
  EXPECT_FALSE(synthesis::signedFrameRouteDominates(candidate, incumbent));
}

TEST(SynthesisPauliTransport,
     IndependentGenericRouteRetainsEqualGateDepthGain) {
  // A small Clifford-only counterexample generated independently of benchmark
  // circuits. Mutating the generic route's seed with primitive transport loses
  // its seven-gate/depth-five completion, leaving seven gates at depth six.
  const std::vector<Gate> input{
      {Kind::X, 1},   {Kind::X, 0},     {Kind::H, 0}, {Kind::CX, 1, 0},
      {Kind::Sdg, 1}, {Kind::CZ, 0, 1}, {Kind::Z, 1}, {Kind::CX, 0, 1}};
  const std::vector<unsigned> readouts{1, 0};
  std::string error;
  SearchBudget local(1000000);
  auto seed =
      synthesis::optimizeLocalSignedFrames(2, input, readouts, &local, error);
  ASSERT_TRUE(seed) << error;
  const auto immutableSeed = *seed;

  // Independently verify full channels for both transport constructions. Local
  // terminal DP may discard phases invisible to the fixed readout effects, so
  // verify those complete effects separately for each completed route below.
  auto primitiveMoved = synthesis::transportPaulis(2, *seed, nullptr, error);
  auto genericMoved =
      synthesis::transportLocalCliffords(2, *seed, false, nullptr, error);
  ASSERT_TRUE(primitiveMoved) << error;
  ASSERT_TRUE(genericMoved) << error;
  checkChannel(*seed, *primitiveMoved);
  checkChannel(*seed, *genericMoved);

  SearchBudget primitiveBudget(1000000), genericBudget(1000000);
  std::optional<tcount::Cost> primitiveCost, genericCost;
  auto primitive = synthesis::refineSignedFrameRoute(
      2, *seed, readouts, true, primitiveBudget, primitiveCost, error);
  auto generic = synthesis::refineSignedFrameRoute(
      2, *seed, readouts, false, genericBudget, genericCost, error);
  ASSERT_TRUE(primitive) << error;
  ASSERT_TRUE(generic) << error;
  ASSERT_TRUE(primitiveCost);
  ASSERT_TRUE(genericCost);
  const tcount::Cost expectedPrimitive{0, 3, 6, 7, 0};
  const tcount::Cost expectedGeneric{0, 3, 5, 7, 0};
  EXPECT_EQ(*primitiveCost, expectedPrimitive);
  EXPECT_EQ(*genericCost, expectedGeneric);
  EXPECT_TRUE(sameWord(*seed, immutableSeed));
  checkTerminalEffects(input, *primitive, readouts);
  checkTerminalEffects(input, *generic, readouts);

  SearchBudget parent(1000000);
  auto selected =
      synthesis::optimizeSignedFrames(2, input, readouts, &parent, error);
  ASSERT_TRUE(selected) << error;
  EXPECT_TRUE(error.empty());
  EXPECT_TRUE(sameWord(*selected, *generic));
  EXPECT_EQ(tcount::cost(2, *selected), expectedGeneric);
  EXPECT_FALSE(parent.stopped());
  EXPECT_GE(parent.remaining(), (1000000 - local.work()) / 2);
  checkTerminalEffects(input, *selected, readouts);
}
