/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisRepresentations.h"
#include "gtest/gtest.h"
#include <complex>
#include <limits>
#include <numbers>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph::detail;

namespace {
using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

/// Independent dense action on small registers, including an untouched
/// external reference. No production conjugation/tableau code.
State pauliAction(const Pauli &p, const State &state) {
  State out(state.size());
  for (size_t basis = 0; basis < state.size(); ++basis) {
    size_t target = basis;
    Complex phase = p.sign ? -1. : 1.;
    for (unsigned wire = 0; (state.size() >> wire) > 1; ++wire) {
      bool bit = (basis >> wire) & 1;
      switch (p.axis(wire)) {
      case 1:
        target ^= size_t(1) << wire;
        break;
      case 2:
        phase *= bit ? -1. : 1.;
        break;
      case 3:
        target ^= size_t(1) << wire;
        phase *= bit ? -imaginary : imaginary;
        break;
      }
    }
    out[target] = phase * state[basis];
  }
  return out;
}

State rotate(const Gate &gate, State state, bool inverse = false) {
  double angle = gate.angle.symbol ? .173 * gate.angle.symbol
                                   : std::numbers::pi * gate.angle.numerator /
                                         gate.angle.denominator;
  if (gate.angle.negative != inverse)
    angle = -angle;
  State product = pauliAction(gate.pauli, state);
  for (size_t i = 0; i < state.size(); ++i)
    state[i] = std::cos(angle / 2) * state[i] -
               imaginary * std::sin(angle / 2) * product[i];
  return state;
}

State effect(const std::vector<Gate> &word, const Pauli &observable,
             State state) {
  for (const auto &gate : word)
    state = rotate(gate, std::move(state));
  state = pauliAction(observable, state);
  for (auto i = word.rbegin(); i != word.rend(); ++i)
    state = rotate(*i, std::move(state), true);
  return state;
}

void expectSameEffects(const MeasuredProblem &source,
                       const RepresentationAlternative &alternative,
                       unsigned dimension = 4) {
  for (size_t j = 0; j < source.observables.size(); ++j)
    for (unsigned column = 0; column < dimension; ++column) {
      State basis(dimension);
      basis[column] = 1;
      auto a = effect(source.rotations, source.observables[j], basis);
      auto b = effect(alternative.rotations, alternative.observables[j], basis);
      for (unsigned row = 0; row < dimension; ++row)
        EXPECT_NEAR(std::abs(a[row] - b[row]), 0, 2e-12);
    }
}

/// Construct a witness using the explorer's algebra; the checker uses its own
/// signed product calculation, and dense effects independently check both.
RepresentationAlternative flipped(const MeasuredProblem &source, size_t i) {
  RepresentationAlternative alternative{source.width,
                                        source.rotations,
                                        source.observables,
                                        source.readoutWires,
                                        {{i}}};
  auto &pivot = alternative.rotations[i];
  int sign = representation::pivotSign(pivot);
  Pauli axis = pivot.pauli;
  axis.sign = false;
  Gate compensation = Gate::rotation(axis, {-sign, 2});
  pivot.angle = pivot.angle.negated();
  for (size_t j = i + 1; j < alternative.rotations.size(); ++j) {
    auto &gate = alternative.rotations[j];
    gate.pauli = tcount::detail::conjugate(gate.pauli, compensation);
    if (gate.pauli.sign)
      gate.angle = gate.angle.negated();
    gate.pauli.sign = false;
  }
  for (auto &word : alternative.observables)
    word = tcount::detail::conjugate(word, compensation);
  return alternative;
}

struct Fixture {
  std::vector<Gate> rotations{Gate::rotation({3, 0}, {1, 4}),
                              Gate::rotation({3, 1}, {1, 4})};
  std::vector<Pauli> observables{{3, 1}};
  std::vector<unsigned> wires{0};
  MeasuredProblem problem() const { return {2, rotations, observables, wires}; }
};
} // namespace

TEST(SynthesisRepresentations,
     SupportProviderReturnsUnverifiedExactAlternative) {
  Fixture f;
  auto result = exploreRepresentations(f.problem());
  ASSERT_EQ(result.status, RepresentationSearchStatus::Complete);
  ASSERT_TRUE(result.alternative);
  ASSERT_EQ(result.alternative->certificate.size(), 1u);
  EXPECT_EQ(result.alternative->certificate[0].rotation, 0u);
  EXPECT_EQ(result.before.rotations, 4u);
  EXPECT_EQ(result.after.rotations, 3u);
  EXPECT_EQ(result.before.observables, 2u);
  EXPECT_EQ(result.after.observables, 1u);
  EXPECT_GT(result.work, 0u);
  EXPECT_TRUE(verifyRepresentation(f.problem(), *result.alternative));
  expectSameEffects(f.problem(), *result.alternative);
  // Identity is retained as the original immutable problem.
  EXPECT_EQ(f.rotations[0].angle.numerator, 1);
  EXPECT_EQ(f.rotations[1].pauli, (Pauli{3, 1}));
}

TEST(SynthesisRepresentations, ExhaustiveSignedTwoQubitCertificatesAndEffects) {
  for (unsigned x = 0; x < 4; ++x)
    for (unsigned z = 0; z < 4; ++z) {
      if (!(x | z))
        continue;
      for (unsigned u = 0; u < 4; ++u)
        for (unsigned v = 0; v < 4; ++v) {
          if (!(u | v))
            continue;
          for (int sign : {-1, 1})
            for (bool negativeAxis : {false, true}) {
              std::vector<Gate> word{
                  Gate::rotation({x, z, negativeAxis}, {sign, 4}),
                  Gate::rotation({u, v, true}, {0, 1, 3, true})};
              std::vector<Pauli> observables{{u, v, true}};
              std::vector<unsigned> readout{1};
              MeasuredProblem source{2, word, observables, readout};
              auto alternative = flipped(source, 0);
              ASSERT_TRUE(verifyRepresentation(source, alternative));
              expectSameEffects(source, alternative);
            }
        }
    }
}

TEST(SynthesisRepresentations, JointOrderedMeasurementsAndExternalReference) {
  Fixture f;
  f.observables = {{3, 1}, {0, 3, true}}; // YX and -ZZ commute.
  f.wires = {1, 0};
  auto alternative = flipped(f.problem(), 0);
  ASSERT_TRUE(verifyRepresentation(f.problem(), alternative));
  // Equality of each commuting Heisenberg observable implies equality of
  // their ordered joint projectors; the extra qubit is an external reference.
  expectSameEffects(f.problem(), alternative, 8);
  std::swap(alternative.observables[0], alternative.observables[1]);
  EXPECT_EQ(verifyRepresentation(f.problem(), alternative).status,
            RepresentationCheckStatus::InvalidProof);
}

TEST(SynthesisRepresentations, RejectsForgedWitnessAndChangedObligations) {
  Fixture f;
  const auto valid = flipped(f.problem(), 0);
  auto expectInvalid = [&](RepresentationAlternative alternative) {
    EXPECT_EQ(verifyRepresentation(f.problem(), alternative).status,
              RepresentationCheckStatus::InvalidProof);
  };
  auto alternative = valid;
  alternative.certificate[0].rotation = 200;
  expectInvalid(alternative);
  alternative = valid;
  alternative.certificate.clear();
  expectInvalid(alternative);
  alternative = valid;
  alternative.rotations[1].angle = alternative.rotations[1].angle.negated();
  expectInvalid(alternative);
  alternative = valid;
  alternative.observables[0].sign = !alternative.observables[0].sign;
  expectInvalid(alternative);
  alternative = valid;
  alternative.readoutWires[0] = 1;
  expectInvalid(alternative);
  alternative = valid;
  alternative.width = 3;
  expectInvalid(alternative);
  alternative = valid;
  alternative.rotations[0].kind = Kind::Phase;
  expectInvalid(alternative);
}

TEST(SynthesisRepresentations, SequentialCertificatesAndIdentity) {
  Fixture f;
  auto once = flipped(f.problem(), 0);
  MeasuredProblem next{once.width, once.rotations, once.observables,
                       once.readoutWires};
  auto twice = flipped(next, 0);
  twice.certificate.insert(twice.certificate.begin(), {0});
  EXPECT_TRUE(verifyRepresentation(f.problem(), twice));
  expectSameEffects(f.problem(), twice);
  RepresentationAlternative identity{
      2, f.rotations, f.observables, f.wires, {}};
  EXPECT_TRUE(verifyRepresentation(f.problem(), identity));
  EXPECT_FALSE(exploreRepresentations(f.problem(), 0).alternative);
}

TEST(SynthesisRepresentations, CertificateCannotChangeEarlierRotations) {
  Fixture f;
  f.rotations.insert(f.rotations.begin(), Gate::rotation({0, 1}, {3, 7}));
  auto alternative = flipped(f.problem(), 1);
  ASSERT_TRUE(verifyRepresentation(f.problem(), alternative));
  expectSameEffects(f.problem(), alternative);
  alternative.rotations[0].angle = {4, 7};
  EXPECT_EQ(verifyRepresentation(f.problem(), alternative).status,
            RepresentationCheckStatus::InvalidProof);
}

TEST(SynthesisRepresentations, ExactAnglesOpaqueSuffixAndHighWireIndices) {
  Fixture f;
  f.rotations[0].angle = {2, 8, 0, true};
  f.rotations[1].angle = {0, 1, 7, true};
  auto result = exploreRepresentations(f.problem());
  ASSERT_TRUE(result.alternative);
  EXPECT_TRUE(verifyRepresentation(f.problem(), *result.alternative));
  expectSameEffects(f.problem(), *result.alternative);
  // The sign flip must not assume bounded integer negation for suffix angles.
  f.rotations[1].angle = {std::numeric_limits<int64_t>::min(), 3};
  result = exploreRepresentations(f.problem());
  ASSERT_TRUE(result.alternative);
  EXPECT_TRUE(verifyRepresentation(f.problem(), *result.alternative));

  QubitSet a = QubitSet::single(64), b = QubitSet::single(130);
  f.rotations = {Gate::rotation({a | b, {}}, {1, 4}),
                 Gate::rotation({a | b, a}, {1, 4})};
  f.observables = {{a | b, a}};
  f.wires = {130};
  MeasuredProblem wide{131, f.rotations, f.observables, f.wires};
  result = exploreRepresentations(wide);
  ASSERT_TRUE(result.alternative);
  EXPECT_TRUE(verifyRepresentation(wide, *result.alternative));
  EXPECT_EQ(result.after.rotations, 3u);
  EXPECT_EQ(result.alternative->readoutWires[0], 130u);
}

TEST(SynthesisRepresentations, UnsupportedInputAndNonPivotAngles) {
  Fixture f;
  f.rotations[0].kind = Kind::H;
  EXPECT_EQ(exploreRepresentations(f.problem()).status,
            RepresentationSearchStatus::Unsupported);
  f = Fixture{};
  f.rotations[0].angle = {9, 4}; // No implicit modulo-8 scalar-phase loss.
  auto result = exploreRepresentations(f.problem());
  EXPECT_EQ(result.status, RepresentationSearchStatus::Unchanged);
  auto certificate = flipped(f.problem(), 0);
  EXPECT_EQ(verifyRepresentation(f.problem(), certificate).status,
            RepresentationCheckStatus::InvalidProof);
  f.rotations[0].angle = {0, 1, 1};
  EXPECT_EQ(exploreRepresentations(f.problem()).status,
            RepresentationSearchStatus::Unchanged);
}

TEST(SynthesisRepresentations, ExhaustionPublishesNoPartialAlternative) {
  Fixture f;
  auto completed = exploreRepresentations(f.problem());
  auto firstMove = exploreRepresentations(f.problem(), 1);
  ASSERT_TRUE(completed.alternative);
  ASSERT_TRUE(firstMove.alternative);
  bool retained = false;
  // Before the first move commits only identity remains. A later failed
  // search charge must preserve that entire move and its exact certificate.
  for (uint64_t limit = 0; limit < completed.work; ++limit) {
    SearchBudget budget(limit);
    auto result = exploreRepresentations(f.problem(), 32, &budget);
    EXPECT_EQ(result.status, RepresentationSearchStatus::BudgetExhausted);
    EXPECT_EQ(bool(result.alternative), limit >= firstMove.work);
    if (result.alternative) {
      retained = true;
      EXPECT_TRUE(verifyRepresentation(f.problem(), *result.alternative));
      EXPECT_EQ(result.alternative->certificate.size(), 1u);
      EXPECT_EQ(result.after.rotations, firstMove.after.rotations);
      EXPECT_EQ(result.after.observables, firstMove.after.observables);
    }
    EXPECT_EQ(result.work, budget.work());
  }
  EXPECT_TRUE(retained);
  SearchBudget proof(0);
  EXPECT_EQ(
      verifyRepresentation(f.problem(), *completed.alternative, &proof).status,
      RepresentationCheckStatus::BudgetExhausted);
  SearchBudget fullProof(std::numeric_limits<uint64_t>::max());
  ASSERT_TRUE(
      verifyRepresentation(f.problem(), *completed.alternative, &fullProof));
  for (uint64_t limit = 0; limit < fullProof.work(); ++limit) {
    SearchBudget limitedProof(limit);
    EXPECT_EQ(
        verifyRepresentation(f.problem(), *completed.alternative, &limitedProof)
            .status,
        RepresentationCheckStatus::BudgetExhausted);
  }
  EXPECT_TRUE(verifyRepresentation(f.problem(), *completed.alternative));
  EXPECT_EQ(f.rotations[0].angle.numerator, 1);
}

TEST(SynthesisRepresentations, ExhaustionRetainsLastCompleteCertificatePrefix) {
  std::vector<Gate> word{Gate::rotation({3, 0}, {1, 4}),
                         Gate::rotation({3, 1}, {1, 4}),
                         Gate::rotation({12, 0}, {-1, 4}),
                         Gate::rotation({12, 4}, {0, 1, 7, true})};
  std::vector<Pauli> observations{{3, 1}, {12, 4}};
  std::vector<unsigned> wires{1, 3};
  MeasuredProblem source{4, word, observations, wires};
  auto full = exploreRepresentations(source);
  ASSERT_TRUE(full.alternative);
  ASSERT_EQ(full.alternative->certificate.size(), 2u);
  bool sawOne = false, sawTwo = false;
  size_t previousSteps = 0;
  for (uint64_t limit = 0; limit < full.work; ++limit) {
    SearchBudget parent(full.work + 1);
    SearchBudget child(parent, limit);
    auto partial = exploreRepresentations(source, 32, &child);
    EXPECT_EQ(partial.status, RepresentationSearchStatus::BudgetExhausted);
    EXPECT_EQ(partial.work, child.work());
    EXPECT_EQ(partial.work, parent.work());
    EXPECT_FALSE(parent.stopped());
    EXPECT_TRUE(parent.take());
    if (!partial.alternative)
      continue;
    auto &alternative = *partial.alternative;
    EXPECT_TRUE(verifyRepresentation(source, alternative));
    EXPECT_GE(alternative.certificate.size(), previousSteps);
    previousSteps = alternative.certificate.size();
    EXPECT_LE(previousSteps, 2u);
    auto actual =
        representation::support(alternative.rotations, alternative.observables);
    EXPECT_EQ(partial.after.rotations, actual.rotations);
    EXPECT_EQ(partial.after.observables, actual.observables);
    if (previousSteps == 1 && !sawOne) {
      expectSameEffects(source, alternative, 16);
      sawOne = true;
    }
    if (previousSteps == 2 && !sawTwo) {
      expectSameEffects(source, alternative, 16);
      sawTwo = true;
    }
  }
  EXPECT_TRUE(sawOne);
  EXPECT_TRUE(sawTwo);
  EXPECT_EQ(word[0].angle.numerator, 1);
  EXPECT_EQ(word[3].angle.symbol, 7u);
  EXPECT_EQ(observations[1], (Pauli{12, 4}));
}
