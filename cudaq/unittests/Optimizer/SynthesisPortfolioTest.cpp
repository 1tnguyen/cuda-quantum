/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisCost.h"
#include "Pauli/SynthesisDependencies.h"
#include "Pauli/SynthesisRepresentationIndex.h"
#include "Pauli/SynthesisRepresentations.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include <bit>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

using namespace cudaq::pauli;
namespace synthesis = cudaq::pauli::graph;
namespace local = cudaq::pauli::graph::detail;

namespace {
struct PortfolioProblem {
  unsigned width;
  std::vector<Gate> rotations;
  std::vector<Pauli> observables;
  std::vector<unsigned> wires;
  local::MeasuredProblem contract() const {
    return {width, rotations, observables, wires};
  }
};

PortfolioProblem signGaugeProblem() {
  return {2,
          {Gate::rotation({0, 3}, {1, 4}), Gate::rotation({1, 2}, {1, 4}),
           Gate::rotation({0, 3}, {1, 4})},
          {{0, 3}},
          {0}};
}

synthesis::Options baselineOptions() {
  synthesis::Options options;
  options.simplify = false;
  options.discrepancySteps = 0;
  options.representationSteps = 0;
  options.readoutCandidates = 0;
  options.commutingCandidates = 0;
  options.algebraicCandidates = 0;
  options.legacyWorkLimit = 0;
  return options;
}

pbc::Result run(const PortfolioProblem &problem,
                const synthesis::Options &options,
                SearchBudget *budget = nullptr) {
  return synthesis::synthesizeMeasured(problem.width, problem.rotations,
                                       problem.observables, problem.wires,
                                       options, budget);
}

void sameCircuit(const std::vector<Gate> &actual,
                 const std::vector<Gate> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (unsigned i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i].kind, expected[i].kind) << "gate " << i;
    EXPECT_EQ(actual[i].a, expected[i].a) << "gate " << i;
    EXPECT_EQ(actual[i].b, expected[i].b) << "gate " << i;
    EXPECT_EQ(actual[i].pauli, expected[i].pauli) << "gate " << i;
    EXPECT_EQ(actual[i].angle, expected[i].angle) << "gate " << i;
  }
}

using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

State applyPauli(const State &state, const Pauli &pauli) {
  unsigned x = 0, z = 0;
  for (unsigned q : pauli.x.set_bits())
    x |= 1u << q;
  for (unsigned q : pauli.z.set_bits())
    z |= 1u << q;
  Complex phase = pauli.sign ? -1. : 1.;
  for (unsigned count = std::popcount(x & z); count; --count)
    phase *= imaginary;
  State result(state.size());
  for (unsigned i = 0; i < state.size(); ++i)
    result[i ^ x] = state[i] * phase * (std::popcount(i & z) % 2 ? -1. : 1.);
  return result;
}

State evolve(State state, const std::vector<Gate> &gates) {
  for (const auto &gate : gates) {
    if (gate.kind == Kind::Rotate || gate.kind == Kind::Phase) {
      double angle = gate.angle.symbol
                         ? 0.131 * gate.angle.symbol + 0.037
                         : std::numbers::pi * double(gate.angle.numerator) /
                               double(gate.angle.denominator);
      if (gate.angle.negative)
        angle = -angle;
      if (gate.kind == Kind::Phase) {
        for (auto &amplitude : state)
          amplitude *= std::exp(imaginary * angle);
      } else {
        auto pauli = applyPauli(state, gate.pauli);
        for (unsigned i = 0; i < state.size(); ++i)
          state[i] = std::cos(angle / 2) * state[i] -
                     imaginary * std::sin(angle / 2) * pauli[i];
      }
      continue;
    }
    if (gate.kind == Kind::X || gate.kind == Kind::Y || gate.kind == Kind::Z) {
      state = applyPauli(
          state,
          {gate.kind != Kind::Z ? QubitSet::single(gate.a) : QubitSet{},
           gate.kind != Kind::X ? QubitSet::single(gate.a) : QubitSet{}});
      continue;
    }
    State next(state.size());
    for (unsigned i = 0; i < state.size(); ++i) {
      bool a = i & (1u << gate.a), b = i & (1u << gate.b);
      switch (gate.kind) {
      case Kind::H:
        next[i] = (state[i & ~(1u << gate.a)] +
                   (a ? -1. : 1.) * state[i | (1u << gate.a)]) /
                  std::sqrt(2.);
        break;
      case Kind::S:
      case Kind::Sdg:
        next[i] =
            state[i] * (a ? (gate.kind == Kind::S ? imaginary : -imaginary)
                          : Complex{1, 0});
        break;
      case Kind::CX:
        next[i ^ (a ? 1u << gate.b : 0u)] = state[i];
        break;
      case Kind::CZ:
        next[i] = state[i] * (a && b ? -1. : 1.);
        break;
      case Kind::Swap:
        next[i ^ (a != b ? (1u << gate.a) | (1u << gate.b) : 0u)] = state[i];
        break;
      default:
        ADD_FAILURE() << "unsupported portfolio test gate";
        return state;
      }
    }
    state = std::move(next);
  }
  return state;
}

double probability(State state, const std::vector<Pauli> &observables,
                   unsigned outcome) {
  for (unsigned j = 0; j < observables.size(); ++j) {
    auto pauli = applyPauli(state, observables[j]);
    double sign = outcome & (1u << j) ? -1. : 1.;
    for (unsigned i = 0; i < state.size(); ++i)
      state[i] = (state[i] + sign * pauli[i]) / 2.;
  }
  double result = 0;
  for (Complex amplitude : state)
    result += std::norm(amplitude);
  return result;
}

void checkMeasuredEffects(const PortfolioProblem &problem,
                          const pbc::Result &candidate) {
  ASSERT_LE(problem.width, 4u);
  ASSERT_EQ(candidate.measurements.size(), problem.wires.size());
  for (unsigned j = 0; j < problem.wires.size(); ++j)
    EXPECT_EQ(candidate.measurements[j],
              (Pauli{{}, QubitSet::single(problem.wires[j]), false}));
  unsigned dimension = 1u << problem.width;
  // Basis states and real/imaginary pair superpositions determine every entry
  // of every Hermitian outcome effect. This tests arbitrary input states, not
  // just zero-state probabilities, and fixes ordered physical Z readouts.
  for (unsigned first = 0; first < dimension; ++first)
    for (unsigned second = first; second < dimension; ++second)
      for (unsigned phase = 0; phase < (first == second ? 1u : 2u); ++phase) {
        State initial(dimension);
        if (first == second)
          initial[first] = 1.;
        else {
          initial[first] = 1. / std::sqrt(2.);
          initial[second] = (phase ? imaginary : Complex{1, 0}) / std::sqrt(2.);
        }
        auto before = evolve(initial, problem.rotations);
        auto after = evolve(initial, candidate.rotations);
        for (unsigned outcome = 0; outcome < (1u << problem.wires.size());
             ++outcome)
          EXPECT_NEAR(probability(before, problem.observables, outcome),
                      probability(after, candidate.measurements, outcome),
                      1e-11);
      }
  EXPECT_LE(tcount::cost(problem.width, candidate.rotations).t,
            tcount::cost(problem.width, problem.rotations).t);
}

void noMetricRegression(unsigned width, const pbc::Result &candidate,
                        const pbc::Result &baseline) {
  auto before = tcount::cost(width, baseline.rotations);
  auto after = tcount::cost(width, candidate.rotations);
  EXPECT_LE(after.gates, before.gates);
  EXPECT_LE(after.t, before.t);
  EXPECT_LE(after.twoQubit, before.twoQubit);
  EXPECT_LE(after.depth, before.depth);
  EXPECT_LE(after.tDepth, before.tDepth);
}
} // namespace

TEST(SynthesisPortfolio, OptionalProvidersCannotRegressAnyMeasuredMetric) {
  const PortfolioProblem problems[] = {
      signGaugeProblem(),
      {2, {}, {{3, 0}, {0, 3}}, {0, 1}},
      {3,
       {Gate::rotation({0, 3}, {1, 4}), Gate::rotation({0, 6}, {-1, 4}),
        Gate::rotation({0, 5}, {1, 4})},
       {{1, 0}, {2, 0}, {4, 0}},
       {0, 1, 2}}};
  for (const auto &problem : problems) {
    auto options = baselineOptions();
    auto baseline = run(problem, options);
    ASSERT_TRUE(baseline) << baseline.error;
    options.representationSteps = 1;
    options.readoutCandidates = 4;
    options.commutingCandidates = 4;
    auto improved = run(problem, options);
    ASSERT_TRUE(improved) << improved.error;
    checkMeasuredEffects(problem, improved);
    EXPECT_EQ(improved.measurements, baseline.measurements);
    auto before = tcount::cost(problem.width, baseline.rotations);
    auto after = tcount::cost(problem.width, improved.rotations);
    EXPECT_LE(after.gates, before.gates);
    EXPECT_LE(after.t, before.t);
    EXPECT_LE(after.twoQubit, before.twoQubit);
    EXPECT_LE(after.depth, before.depth);
    EXPECT_LE(after.tDepth, before.tDepth);
    EXPECT_LE(local::circuitScore(problem.width, improved.rotations, options),
              local::circuitScore(problem.width, baseline.rotations, options));
  }
}

TEST(SynthesisPortfolio, ExhaustedOptionalSearchRetainsTheVerifiedBaseline) {
  const auto problem = signGaugeProblem();
  auto options = baselineOptions();
  SearchBudget reference(std::numeric_limits<uint64_t>::max());
  auto baseline = run(problem, options, &reference);
  ASSERT_TRUE(baseline) << baseline.error;
  ASSERT_FALSE(reference.stopped());
  ASSERT_GT(reference.work(), 0u);

  // Supply only the observed baseline work. Optional refinements can leave
  // unused reserve when their child slice cannot finish; the checked baseline
  // must remain available without stopping the parent.
  options.representationSteps = 1;
  SearchBudget limited(reference.work());
  auto result = run(problem, options, &limited);
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(limited.stopped());
  EXPECT_LE(limited.work(), reference.work());
  sameCircuit(result.rotations, baseline.rotations);
  EXPECT_EQ(result.measurements, baseline.measurements);
}

TEST(SynthesisPortfolio, UnprunedRegionalPlansUseTheirFullWordProof) {
  const PortfolioProblem problems[] = {
      {1, {Gate::rotation({0, 1}, {1, 4})}, {{0, 1}}, {0}},
      {2, {Gate::rotation({0, 2}, {-1, 4})}, {{0, 1}}, {0}}};
  for (const auto &problem : problems) {
    auto options = baselineOptions();
    options.pruneUnobserved = true;
    auto baseline = run(problem, options);
    ASSERT_TRUE(baseline) << baseline.error;
    ASSERT_TRUE(baseline.rotations.empty());

    // Regional providers keep the complete original rotation word, including
    // an outcome-invisible Z rotation or a rotation on an unobserved wire.
    // Their proof must check that full word instead of deleting only the source
    // side and reporting a legitimate regional implementation as InvalidProof.
    // Alternative readout tails for that prefix require the same proof mode.
    options.commutingCandidates = 2;
    options.readoutCandidates = 2;
    auto result = run(problem, options);
    ASSERT_TRUE(result) << result.error;
    sameCircuit(result.rotations, baseline.rotations);
    EXPECT_EQ(result.measurements, baseline.measurements);
  }
}

TEST(SynthesisPortfolio,
     AlternateOnlyDependencyLimitRetainsTheVerifiedBaseline) {
  const auto problem = signGaugeProblem();
  auto proposal = local::exploreRepresentations(problem.contract(), 1);
  ASSERT_TRUE(proposal.alternative);
  const auto &alternative = *proposal.alternative;
  ASSERT_EQ(alternative.certificate.size(), 1u);
  ASSERT_EQ(alternative.certificate.front().rotation, 1u);
  ASSERT_TRUE(local::verifyRepresentation(problem.contract(), alternative));

  // Original axes ZZ, XZ, ZZ have two occurrence edges. The proved middle
  // flip lowers support but changes the suffix axis to +/-YI, creating a third
  // edge. Support improvement must not bypass or invalidate the original cap.
  std::vector<Pauli> originalAxes, alternativeAxes;
  for (const auto &gate : problem.rotations)
    originalAxes.push_back(gate.pauli);
  for (const auto &gate : alternative.rotations)
    alternativeAxes.push_back(gate.pauli);
  auto originalIndex = local::DependencyIndex::build(originalAxes, 2);
  ASSERT_TRUE(originalIndex);
  EXPECT_EQ(originalIndex.value->statistics().logicalEdges, 2u);
  auto alternateIndex = local::DependencyIndex::build(alternativeAxes, 3);
  ASSERT_TRUE(alternateIndex);
  EXPECT_EQ(alternateIndex.value->statistics().logicalEdges, 3u);
  EXPECT_EQ(local::DependencyIndex::build(alternativeAxes, 2).error,
            local::DependencyError::DependencyLimit);

  auto options = baselineOptions();
  options.maxDependencies = 2;
  auto baseline = run(problem, options);
  ASSERT_TRUE(baseline) << baseline.error;
  options.representationSteps = 1;
  auto result = run(problem, options);
  ASSERT_TRUE(result) << result.error;
  sameCircuit(result.rotations, baseline.rotations);
  EXPECT_EQ(result.measurements, baseline.measurements);
}

TEST(SynthesisPortfolio, InvalidMeasuredSourceFailsBeforeOptionalProviders) {
  auto problem = signGaugeProblem();
  auto options = baselineOptions();
  options.representationSteps = 1;
  options.readoutCandidates = 4;
  options.commutingCandidates = 4;
  problem.rotations.front().pauli.x.set(problem.width);
  auto invalidSource = run(problem, options);
  EXPECT_FALSE(invalidSource);
  EXPECT_FALSE(invalidSource.error.empty());
  EXPECT_TRUE(invalidSource.rotations.empty());

  problem = signGaugeProblem();
  problem.wires.front() = problem.width;
  auto invalidReadout = run(problem, options);
  EXPECT_FALSE(invalidReadout);
  EXPECT_FALSE(invalidReadout.error.empty());
  EXPECT_TRUE(invalidReadout.rotations.empty());
}

TEST(SynthesisPortfolio, TinyLegacySliceReturnsACheckedBootstrap) {
  auto problem = signGaugeProblem();
  // Include a signed partial observable to test outcome identity as well as
  // probability preservation when the other quantum output is released.
  problem.observables.front().sign = true;
  auto options = baselineOptions();
  options.legacyWorkLimit = 1;
  SearchBudget shared(100000000);
  auto result = run(problem, options, &shared);
  ASSERT_TRUE(result) << result.error;
  EXPECT_GT(shared.work(), 1u);
  EXPECT_FALSE(shared.stopped());
  EXPECT_GT(shared.remaining(), 0u);
  // The legacy child's first input charge cannot fit one work unit. Its
  // failure must leave the already checked bootstrap and parent reserve usable.
  EXPECT_TRUE(shared.take());
  checkMeasuredEffects(problem, result);
  EXPECT_EQ(tcount::cost(problem.width, problem.rotations).t, 3u);
  EXPECT_FALSE(result.rotations.empty());
}

TEST(SynthesisPortfolio, SuccessfulLegacyDefinesTheQualityBaseline) {
  const auto problem = signGaugeProblem();
  auto options = baselineOptions();
  SearchBudget reference(std::numeric_limits<uint64_t>::max());
  auto baseline = run(problem, options, &reference);
  ASSERT_TRUE(baseline) << baseline.error;
  ASSERT_FALSE(reference.stopped());
  ASSERT_GT(reference.work(), 0u);
  options.legacyWorkLimit = reference.work();
  SearchBudget shared(std::numeric_limits<uint64_t>::max());
  auto result = run(problem, options, &shared);
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(shared.stopped());
  noMetricRegression(problem.width, result, baseline);
  checkMeasuredEffects(problem, result);
  EXPECT_EQ(result.measurements, baseline.measurements);
}

TEST(SynthesisPortfolio, BootstrapSchedulingCannotBypassSourceValidation) {
  auto options = baselineOptions();
  options.legacyWorkLimit = 1;
  options.representationSteps = 1;
  options.readoutCandidates = 2;
  options.commutingCandidates = 2;
  options.algebraicCandidates = 2;
  for (unsigned invalid = 0; invalid < 3; ++invalid) {
    auto problem = signGaugeProblem();
    if (invalid == 0)
      problem.rotations.front().pauli.x.set(problem.width);
    else if (invalid == 1)
      problem.rotations.front().angle.denominator = 0;
    else {
      problem.observables = {{0, 1}, {1, 0}};
      problem.wires = {0, 1};
    }
    SearchBudget shared(100000000);
    auto result = run(problem, options, &shared);
    EXPECT_FALSE(result);
    EXPECT_FALSE(result.error.empty());
    EXPECT_TRUE(result.rotations.empty());
    EXPECT_TRUE(result.measurements.empty());
  }
}

TEST(SynthesisPortfolio, AlgebraicProviderIsAnOptInFullOptimizationTrack) {
  // Fixed-word synthesis retains both T rotations. Algebraic merging can
  // replace them with a Clifford and transport it into the signed observable.
  // X readout makes the rotation observable, preventing trivial terminal-Z
  // deletion from standing in for the algebraic optimization under test.
  const PortfolioProblem problem{
      1,
      {Gate::rotation({0, 1}, {1, 4}), Gate::rotation({0, 1}, {1, 4})},
      {{1, 0}},
      {0}};
  auto options = baselineOptions();
  auto baseline = run(problem, options);
  ASSERT_TRUE(baseline) << baseline.error;
  EXPECT_EQ(tcount::cost(problem.width, problem.rotations).t, 2u);
  ASSERT_EQ(tcount::cost(problem.width, baseline.rotations).t, 2u);
  checkMeasuredEffects(problem, baseline);

  options.algebraicCandidates = 4;
  SearchBudget shared(100000000);
  auto result = run(problem, options, &shared);
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(shared.stopped());
  EXPECT_EQ(tcount::cost(problem.width, result.rotations).t, 0u);
  noMetricRegression(problem.width, result, baseline);
  checkMeasuredEffects(problem, result);
  EXPECT_EQ(result.measurements, baseline.measurements);
}

TEST(SynthesisPortfolio,
     FullOptimizationKeepsFourWireEffectsAndAllFiveMetrics) {
  const PortfolioProblem problem{
      4,
      {Gate::rotation({0, 3}, {1, 4}), Gate::rotation({0, 6}, {-1, 4}),
       Gate::rotation({0, 12}, {1, 4}), Gate::rotation({0, 9, true}, {-1, 4})},
      {{1, 0}, {2, 0}, {4, 0}, {8, 0, true}},
      {2, 0, 3, 1}};
  auto options = baselineOptions();
  SearchBudget baselineBudget(100000000);
  auto baseline = run(problem, options, &baselineBudget);
  ASSERT_TRUE(baseline) << baseline.error;
  ASSERT_GE(baseline.rotations.size(), 4u);
  checkMeasuredEffects(problem, baseline);

  // Full mode includes the final four-wire extension. Its full-unitary proof
  // must compose with signed, ordered measurements on these physical wires.
  options.algebraicCandidates = 1;
  SearchBudget fullBudget(100000000);
  auto result = run(problem, options, &fullBudget);
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(fullBudget.stopped());
  EXPECT_EQ(result.measurements, baseline.measurements);
  noMetricRegression(problem.width, result, baseline);
  checkMeasuredEffects(problem, result);
}

TEST(SynthesisPortfolio,
     SmallFullOptimizationBudgetPreservesCompletedFourWireResult) {
  PortfolioProblem problem{4, {}, {}, {0, 1, 2, 3}};
  for (unsigned q = 0; q < problem.width; ++q) {
    problem.rotations.push_back(
        Gate::rotation({{}, QubitSet::single(q)}, {1, 4}));
    problem.observables.push_back({QubitSet::single(q), {}});
  }
  auto options = baselineOptions();
  auto baseline = run(problem, options);
  ASSERT_TRUE(baseline) << baseline.error;
  const auto baselineCost = tcount::cost(problem.width, baseline.rotations);
  ASSERT_EQ(baselineCost.gates, 8u);
  ASSERT_EQ(baselineCost.t, 4u);
  ASSERT_EQ(baselineCost.twoQubit, 0u);

  // Four independent observed T rotations already have a short completed
  // implementation. A small overall allowance cannot fund the optional
  // four-wire dictionary; failure of that extension must retain the complete
  // result. This uses a fixed resource envelope, not measured internal charges.
  options.algebraicCandidates = 1;
  SearchBudget limited(200000);
  auto result = run(problem, options, &limited);
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(limited.stopped());
  EXPECT_GT(limited.remaining(), 0u);
  sameCircuit(result.rotations, baseline.rotations);
  EXPECT_EQ(result.measurements, baseline.measurements);
  checkMeasuredEffects(problem, result);
}

TEST(SynthesisPortfolio, IndexedRoutePreservesSignedPermutedJointEffects) {
  // The third wire is an untouched reference. The existing dense oracle tests
  // all Hermitian effect entries, including inputs entangled with that wire.
  const PortfolioProblem problem{3,
                                 {Gate::rotation({2, 1}, {1, 4}),
                                  Gate::rotation({3, 0}, {1, 4}),
                                  Gate::rotation({3, 1}, {1, 4})},
                                 {{3, 1, true}, {3, 2}},
                                 {1, 0}};
  const auto original = problem;
  local::IndexedRepresentationLimits limits;
  SearchBudget generation(limits.maxWork);
  auto proposal = local::exploreIndexedRepresentations(problem.contract(),
                                                       limits, generation);
  ASSERT_TRUE(proposal.alternative);
  ASSERT_FALSE(proposal.alternative->certificate.empty());
  ASSERT_TRUE(
      local::verifyRepresentation(problem.contract(), *proposal.alternative));

  auto options = baselineOptions();
  EXPECT_EQ(options.indexedRepresentationSteps, 0u);
  // Keep the same regional/algebraic completion providers enabled on both
  // runs. The indexed branch must compose its certificate through that full
  // continuation and final four-wire refinement before native selection.
  options.readoutCandidates = options.commutingCandidates = 2;
  options.algebraicCandidates = 1;
  SearchBudget baselineBudget(100000000);
  auto baseline = run(problem, options, &baselineBudget);
  ASSERT_TRUE(baseline) << baseline.error;
  options.indexedRepresentationSteps = 64;
  SearchBudget indexedBudget(100000000);
  auto result = run(problem, options, &indexedBudget);
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(indexedBudget.stopped());
  EXPECT_EQ(result.measurements, baseline.measurements);
  noMetricRegression(problem.width, result, baseline);
  checkMeasuredEffects(problem, result);
  sameCircuit(problem.rotations, original.rotations);
  EXPECT_EQ(problem.observables, original.observables);
  EXPECT_EQ(problem.wires, original.wires);
}

TEST(SynthesisPortfolio, IndexedExhaustionRetainsThePriorVerifiedCircuit) {
  const PortfolioProblem problem{3,
                                 {Gate::rotation({2, 1}, {1, 4}),
                                  Gate::rotation({3, 0}, {1, 4}),
                                  Gate::rotation({3, 1}, {1, 4})},
                                 {{3, 1, true}, {3, 2}},
                                 {1, 0}};
  auto options = baselineOptions();
  SearchBudget reference(100000000);
  auto baseline = run(problem, options, &reference);
  ASSERT_TRUE(baseline) << baseline.error;
  ASSERT_FALSE(reference.stopped());
  ASSERT_GT(reference.work(), 0u);

  // Earlier optional slices depend on the enclosing allowance. Compare work
  // against a control with the identical envelope, not the large-budget run.
  const uint64_t allowance = reference.work() + 100;
  SearchBudget controlBudget(allowance);
  auto control = run(problem, options, &controlBudget);
  ASSERT_TRUE(control) << control.error;
  sameCircuit(control.rotations, baseline.rotations);

  // Give the optional route enough allowance to start, but not enough to
  // complete its proposal/proof/continuation. Its child must not invalidate
  // the old final result or stop the parent's remaining allowance.
  options.indexedRepresentationSteps = 64;
  SearchBudget limited(allowance);
  auto result = run(problem, options, &limited);
  ASSERT_TRUE(result) << result.error;
  EXPECT_GT(limited.work(), controlBudget.work());
  EXPECT_FALSE(limited.stopped());
  EXPECT_GT(limited.remaining(), 0u);
  EXPECT_TRUE(limited.take());
  sameCircuit(result.rotations, baseline.rotations);
  EXPECT_EQ(result.measurements, baseline.measurements);
  checkMeasuredEffects(problem, result);
}
