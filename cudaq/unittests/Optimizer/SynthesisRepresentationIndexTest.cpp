/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisRepresentationIndex.h"
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

} // namespace

namespace {
void expectSameSearch(const MeasuredProblem &source,
                      const RepresentationSearchResult &expected,
                      const RepresentationSearchResult &actual) {
  EXPECT_EQ(actual.status, expected.status);
  EXPECT_EQ(actual.before.rotations, expected.before.rotations);
  EXPECT_EQ(actual.before.observables, expected.before.observables);
  EXPECT_EQ(actual.after.rotations, expected.after.rotations);
  EXPECT_EQ(actual.after.observables, expected.after.observables);
  ASSERT_EQ(bool(actual.alternative), bool(expected.alternative));
  if (!actual.alternative)
    return;
  const auto &a = *actual.alternative, &b = *expected.alternative;
  EXPECT_EQ(a.width, b.width);
  EXPECT_EQ(a.readoutWires, b.readoutWires);
  EXPECT_EQ(a.observables, b.observables);
  ASSERT_EQ(a.rotations.size(), b.rotations.size());
  for (size_t i = 0; i < a.rotations.size(); ++i) {
    EXPECT_EQ(a.rotations[i].kind, b.rotations[i].kind);
    EXPECT_EQ(a.rotations[i].a, b.rotations[i].a);
    EXPECT_EQ(a.rotations[i].b, b.rotations[i].b);
    EXPECT_EQ(a.rotations[i].pauli, b.rotations[i].pauli);
    EXPECT_EQ(a.rotations[i].angle, b.rotations[i].angle);
  }
  ASSERT_EQ(a.certificate.size(), b.certificate.size());
  for (size_t i = 0; i < a.certificate.size(); ++i)
    EXPECT_EQ(a.certificate[i].rotation, b.certificate[i].rotation);
  EXPECT_TRUE(verifyRepresentation(source, a));
}

struct Fixture {
  std::vector<Gate> word{Gate::rotation({3, 0}, {1, 4}),
                         Gate::rotation({3, 1}, {1, 4}),
                         Gate::rotation({12, 0}, {-1, 4}),
                         Gate::rotation({12, 4}, {0, 1, 7, true})};
  std::vector<Pauli> observables{{3, 1}, {12, 4}};
  std::vector<unsigned> wires{1, 3};
  MeasuredProblem problem() const { return {4, word, observables, wires}; }
};
} // namespace

TEST(SynthesisRepresentationIndex, EveryBoundedGreedyPrefixMatchesOriginal) {
  // Exhaust all pairs of unsigned two-wire axes, with signs, supported
  // quarter-turn encodings, nonpivot angles and opaque suffixes mixed in.
  for (unsigned p = 1; p < 16; ++p)
    for (unsigned q = 1; q < 16; ++q)
      for (unsigned signs = 0; signs < 4; ++signs) {
        unsigned last = 1 + (p + 3 * q) % 15;
        std::vector<Gate> word{
            Gate::rotation({p & 3, p >> 2, bool(signs & 1)}, {2, 8}),
            Gate::rotation({q & 3, q >> 2, bool(signs & 2)}, {-1, 4}),
            Gate::rotation({p & 3, p >> 2, true}, {9, 4}),
            Gate::rotation({last & 3, last >> 2, true}, {0, 1, 3, true})};
        std::vector<Pauli> observables{{q & 3, q >> 2, bool(signs & 1)}};
        std::vector<unsigned> wires{1};
        MeasuredProblem source{2, word, observables, wires};
        for (unsigned steps : {0u, 1u, 2u, 5u}) {
          auto expected = exploreRepresentations(source, steps);
          IndexedRepresentationLimits limits;
          limits.maxSteps = steps;
          SearchBudget budget(limits.maxWork);
          auto actual = exploreIndexedRepresentations(source, limits, budget);
          expectSameSearch(source, expected, actual);
          EXPECT_EQ(actual.work, budget.work());
          if (steps == 5 && signs == 3 && p == q && actual.alternative)
            expectSameEffects(source, *actual.alternative);
        }
      }
}

TEST(SynthesisRepresentationIndex, ChangedCrossCutCommutatorsUpdateScores) {
  // Pivot one wins on its two measurement reductions. Its compensation changes
  // the anticommutation of predecessor zero with suffix two. A cached conflict
  // graph, or an update assuming all commutators invariant, gives stale scores.
  std::vector<Gate> word{Gate::rotation({2, 1}, {1, 4}),
                         Gate::rotation({3, 0}, {1, 4}),
                         Gate::rotation({3, 1}, {1, 4})};
  std::vector<Pauli> observables{{3, 1}, {3, 2}};
  std::vector<unsigned> wires{0, 1};
  MeasuredProblem source{2, word, observables, wires};
  IndexedRepresentationLimits limits;
  SearchBudget budget(limits.maxWork);
  auto result = exploreIndexedRepresentations(source, limits, budget);
  auto expected = exploreRepresentations(source, limits.maxSteps);
  expectSameSearch(source, expected, result);
  ASSERT_TRUE(result.alternative);
  ASSERT_EQ(result.alternative->certificate.size(), 1u);
  EXPECT_EQ(result.alternative->certificate[0].rotation, 1u);
  EXPECT_TRUE(word[0].pauli.anti(word[2].pauli));
  EXPECT_FALSE(result.alternative->rotations[0].pauli.anti(
      result.alternative->rotations[2].pauli));
  expectSameEffects(source, *result.alternative);
}

TEST(SynthesisRepresentationIndex,
     DuplicateMultiplicitiesAndSignedCommutingRows) {
  std::vector<Gate> word{
      Gate::rotation({3, 0, true}, {-1, 4}), Gate::rotation({3, 1}, {1, 4}),
      Gate::rotation({3, 1, true}, {-1, 4}), Gate::rotation({3, 1}, {-1, 4}),
      Gate::rotation({0, 3, true}, {0, 1, 7})};
  std::vector<Pauli> observables{{3, 1}};
  std::vector<unsigned> wires{0};
  MeasuredProblem source{2, word, observables, wires};
  IndexedRepresentationLimits limits;
  SearchBudget budget(limits.maxWork);
  auto result = exploreIndexedRepresentations(source, limits, budget);
  expectSameSearch(source, exploreRepresentations(source, limits.maxSteps),
                   result);
  ASSERT_TRUE(result.alternative);
  EXPECT_FALSE(result.alternative->rotations.back().pauli.sign);
  EXPECT_TRUE(result.alternative->rotations.back().angle.negative);
  EXPECT_EQ(result.alternative->rotations.back().angle.symbol, 7u);
  expectSameEffects(source, *result.alternative);
}

TEST(SynthesisRepresentationIndex,
     FortyEightIndependentMovesHonorConfiguredCap) {
  std::vector<Gate> word;
  std::vector<Pauli> observables;
  std::vector<unsigned> wires;
  for (unsigned copy = 0; copy < 48; ++copy) {
    auto x = QubitSet::single(2 * copy) | QubitSet::single(2 * copy + 1);
    auto z = QubitSet::single(2 * copy);
    word.push_back(Gate::rotation({x, {}}, {1, 4}));
    word.push_back(Gate::rotation({x, z}, {-1, 4}));
    observables.push_back({x, z});
    wires.push_back(2 * copy);
  }
  MeasuredProblem source{96, word, observables, wires};
  IndexedRepresentationLimits limits;
  limits.maxSteps = 32;
  SearchBudget limitedBudget(limits.maxWork);
  auto limited = exploreIndexedRepresentations(source, limits, limitedBudget);
  ASSERT_TRUE(limited.alternative);
  EXPECT_EQ(limited.alternative->certificate.size(), 32u);
  limits.maxSteps = 64;
  SearchBudget budget(limits.maxWork);
  auto all = exploreIndexedRepresentations(source, limits, budget);
  auto expected = exploreRepresentations(source, limits.maxSteps);
  expectSameSearch(source, expected, all);
  ASSERT_TRUE(all.alternative);
  ASSERT_EQ(all.alternative->certificate.size(), 48u);
  for (unsigned i = 0; i < 48; ++i)
    EXPECT_EQ(all.alternative->certificate[i].rotation, 2 * i);
  EXPECT_EQ(all.after.rotations, 3u * 48);
  EXPECT_EQ(all.after.observables, 48u);
  // Deterministic charged work, not a runtime or native-cost claim.
  EXPECT_LT(all.work, expected.work);
  EXPECT_EQ(word.size(), 96u);
  EXPECT_EQ(word[0].angle.numerator, 1);
}

TEST(SynthesisRepresentationIndex,
     ExhaustionCommitsOnlyWholeScoreAndWordUpdates) {
  Fixture f;
  IndexedRepresentationLimits limits;
  SearchBudget ample(limits.maxWork);
  auto complete = exploreIndexedRepresentations(f.problem(), limits, ample);
  ASSERT_EQ(complete.status, RepresentationSearchStatus::Complete);
  ASSERT_TRUE(complete.alternative);
  ASSERT_EQ(complete.alternative->certificate.size(), 2u);
  bool sawOne = false, sawTwo = false;
  for (uint64_t cap = 0; cap < complete.work; ++cap) {
    limits.maxWork = cap;
    SearchBudget parent(complete.work + 1);
    auto partial = exploreIndexedRepresentations(f.problem(), limits, parent);
    EXPECT_EQ(partial.status, RepresentationSearchStatus::BudgetExhausted);
    EXPECT_EQ(partial.work, parent.work());
    EXPECT_FALSE(parent.stopped());
    EXPECT_TRUE(parent.take());
    if (!partial.alternative)
      continue;
    auto steps = partial.alternative->certificate.size();
    EXPECT_LE(steps, 2u);
    auto expected = exploreRepresentations(f.problem(), steps);
    // Only completion status differs when a later optional charge fails.
    partial.status = expected.status;
    expectSameSearch(f.problem(), expected, partial);
    if (steps == 1 && !sawOne) {
      expectSameEffects(f.problem(), *partial.alternative, 16);
      sawOne = true;
    }
    if (steps == 2 && !sawTwo) {
      expectSameEffects(f.problem(), *partial.alternative, 16);
      sawTwo = true;
    }
  }
  EXPECT_TRUE(sawOne);
  EXPECT_TRUE(sawTwo);
  SearchBudget tiny(1);
  limits.maxWork = 1000000;
  auto noStart = exploreIndexedRepresentations(f.problem(), limits, tiny);
  EXPECT_EQ(noStart.status, RepresentationSearchStatus::BudgetExhausted);
  EXPECT_FALSE(noStart.alternative);
  EXPECT_FALSE(tiny.stopped());
  EXPECT_TRUE(tiny.take());
}

TEST(SynthesisRepresentationIndex,
     HighWiresAndIndependentCertificateRejection) {
  auto x = QubitSet::single(64) | QubitSet::single(130);
  std::vector<Gate> word{Gate::rotation({x, {}}, {1, 4}),
                         Gate::rotation({x, QubitSet::single(64)}, {1, 4})};
  std::vector<Pauli> observables{{x, QubitSet::single(64)}};
  std::vector<unsigned> wires{130};
  MeasuredProblem source{131, word, observables, wires};
  IndexedRepresentationLimits limits;
  limits.maxSteps = std::numeric_limits<unsigned>::max();
  SearchBudget budget(limits.maxWork);
  auto result = exploreIndexedRepresentations(source, limits, budget);
  expectSameSearch(source, exploreRepresentations(source), result);
  ASSERT_TRUE(result.alternative);
  EXPECT_EQ(result.alternative->readoutWires, wires);
  auto forged = *result.alternative;
  forged.certificate.push_back({word.size()});
  EXPECT_EQ(verifyRepresentation(source, forged).status,
            RepresentationCheckStatus::InvalidProof);
  forged = *result.alternative;
  forged.rotations[0].angle = {9, 4};
  EXPECT_EQ(verifyRepresentation(source, forged).status,
            RepresentationCheckStatus::InvalidProof);
}

TEST(SynthesisRepresentationIndex,
     DimensionsStorageAndUnsupportedWordsAreBounded) {
  Fixture f;
  IndexedRepresentationLimits limits;
  SearchBudget budget(10000000);
  limits.maxTemporaryBytes = 1;
  auto storage = exploreIndexedRepresentations(f.problem(), limits, budget);
  EXPECT_EQ(storage.status, RepresentationSearchStatus::Unsupported);
  EXPECT_FALSE(storage.alternative);
  limits = {};
  limits.maxRotations = 1;
  auto dimensions = exploreIndexedRepresentations(f.problem(), limits, budget);
  EXPECT_EQ(dimensions.status, RepresentationSearchStatus::Unsupported);
  EXPECT_EQ(dimensions.work, 0u);
  limits = {};
  MeasuredProblem enormous{std::numeric_limits<unsigned>::max(), f.word,
                           f.observables, f.wires};
  EXPECT_EQ(exploreIndexedRepresentations(enormous, limits, budget).status,
            RepresentationSearchStatus::Unsupported);
  f.word[0].kind = Kind::Phase;
  EXPECT_EQ(exploreIndexedRepresentations(f.problem(), limits, budget).status,
            RepresentationSearchStatus::Unsupported);
  f = Fixture{};
  f.wires[1] = f.wires[0];
  EXPECT_EQ(exploreIndexedRepresentations(f.problem(), limits, budget).status,
            RepresentationSearchStatus::Unsupported);
  f = Fixture{};
  f.word.clear();
  EXPECT_EQ(exploreIndexedRepresentations(f.problem(), limits, budget).status,
            RepresentationSearchStatus::Unchanged);
}
