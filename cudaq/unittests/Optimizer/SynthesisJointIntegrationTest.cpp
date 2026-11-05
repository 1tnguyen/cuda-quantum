/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/

#include "gtest/gtest.h"
#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include "cudaq/Optimizer/Pauli/TCount.h"
#include <bit>
#include <cmath>
#include <complex>
#include <numbers>

using namespace cudaq::pauli;
namespace graph = cudaq::pauli::graph;

namespace {
struct Problem {
  unsigned width = 3;
  std::vector<Gate> rotations{
      Gate::phase({1, 8}), Gate::rotation({0, 3, true}, {1, 4}),
      Gate::rotation({3, 0}, {-1, 4}), Gate::rotation({3, 3}, {1, 4}),
      Gate::rotation({1, 2}, {1, 4})};
  // Wire 2 is an untouched reference. The signed, permuted physical readouts
  // expose sign loss and association errors that zero-input checks can miss.
  std::vector<Pauli> observables{{3, 0, true}, {0, 3}};
  std::vector<unsigned> readouts{1, 0};
};

graph::Options options() {
  graph::Options result;
  result.discrepancySteps = 0;
  result.simplify = false;
  result.maxGates = 256;
  result.jointPhaseCandidates = 4;
  result.jointPhasePlannerWork = 20000000;
  result.jointPhaseMaxWork = 100000000;
  result.jointPhaseTimeoutMs = 60000;
  return result;
}

pbc::Result run(const Problem &problem, const graph::Options &settings,
                SearchBudget &budget) {
  return graph::synthesizeMeasured(problem.width, problem.rotations,
                                   problem.observables, problem.readouts,
                                   settings, &budget);
}

void sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  ASSERT_EQ(a.size(), b.size());
  for (unsigned i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].kind, b[i].kind) << i;
    EXPECT_EQ(a[i].a, b[i].a) << i;
    EXPECT_EQ(a[i].b, b[i].b) << i;
    EXPECT_EQ(a[i].pauli, b[i].pauli) << i;
    EXPECT_EQ(a[i].angle, b[i].angle) << i;
  }
}

using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

State pauliAction(const State &state, const Pauli &pauli) {
  unsigned x = 0, z = 0;
  for (unsigned q : pauli.x.set_bits())
    x |= 1u << q;
  for (unsigned q : pauli.z.set_bits())
    z |= 1u << q;
  Complex phase = pauli.sign ? -1. : 1.;
  for (unsigned n = std::popcount(x & z); n; --n)
    phase *= imaginary;
  State result(state.size());
  for (unsigned i = 0; i < state.size(); ++i)
    result[i ^ x] = state[i] * phase * (std::popcount(i & z) % 2 ? -1. : 1.);
  return result;
}

State evolve(State state, const std::vector<Gate> &gates) {
  for (const Gate &gate : gates) {
    if (gate.kind == Kind::Rotate || gate.kind == Kind::Phase) {
      double theta = gate.angle.symbol
                         ? 0.173 * gate.angle.symbol + 0.037
                         : std::numbers::pi * double(gate.angle.numerator) /
                               double(gate.angle.denominator);
      theta *= gate.angle.negative ? -1. : 1.;
      if (gate.kind == Kind::Phase) {
        for (auto &value : state)
          value *= std::exp(imaginary * theta);
      } else {
        auto transformed = pauliAction(state, gate.pauli);
        for (unsigned i = 0; i < state.size(); ++i)
          state[i] = std::cos(theta / 2) * state[i] -
                     imaginary * std::sin(theta / 2) * transformed[i];
      }
      continue;
    }
    if (gate.kind == Kind::X || gate.kind == Kind::Y || gate.kind == Kind::Z) {
      state = pauliAction(
          state,
          {gate.kind == Kind::Z ? QubitSet{} : QubitSet::single(gate.a),
           gate.kind == Kind::X ? QubitSet{} : QubitSet::single(gate.a)});
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
        ADD_FAILURE() << "unexpected native gate " << unsigned(gate.kind);
        return state;
      }
    }
    state = std::move(next);
  }
  return state;
}

double probability(State state, const std::vector<Pauli> &rows,
                   unsigned outcome) {
  for (unsigned j = 0; j < rows.size(); ++j) {
    auto transformed = pauliAction(state, rows[j]);
    for (unsigned i = 0; i < state.size(); ++i)
      state[i] =
          (state[i] + (outcome & (1u << j) ? -1. : 1.) * transformed[i]) / 2.;
  }
  double result = 0;
  for (auto amplitude : state)
    result += std::norm(amplitude);
  return result;
}

void expectEffects(const Problem &problem, const pbc::Result &candidate) {
  ASSERT_TRUE(candidate) << candidate.error;
  ASSERT_EQ(candidate.measurements.size(), problem.readouts.size());
  for (unsigned i = 0; i < problem.readouts.size(); ++i)
    EXPECT_EQ(candidate.measurements[i],
              (Pauli{{}, QubitSet::single(problem.readouts[i]), false}));
  ASSERT_LE(problem.width, 3u);
  unsigned dimension = 1u << problem.width;
  // These inputs span every Hermitian matrix entry of each joint outcome
  // effect, including coherences with the reference wire.
  for (unsigned a = 0; a < dimension; ++a)
    for (unsigned b = a; b < dimension; ++b)
      for (unsigned phase = 0; phase < (a == b ? 1u : 2u); ++phase) {
        State initial(dimension);
        initial[a] = a == b ? 1. : 1. / std::sqrt(2.);
        if (a != b)
          initial[b] = (phase ? imaginary : Complex{1, 0}) / std::sqrt(2.);
        auto source = evolve(initial, problem.rotations);
        auto output = evolve(initial, candidate.rotations);
        for (unsigned outcome = 0; outcome < (1u << problem.readouts.size());
             ++outcome)
          EXPECT_NEAR(probability(source, problem.observables, outcome),
                      probability(output, candidate.measurements, outcome),
                      1e-11);
      }
}
} // namespace

TEST(SynthesisJointIntegration, DisabledProviderPreservesCircuitAndWork) {
  Problem problem;
  auto control = options();
  control.jointPhaseCandidates = 0;
  SearchBudget before(100000000), after(100000000);
  auto baseline = run(problem, control, before);
  auto changed = control;
  changed.jointPhaseAcceptance = graph::JointPhaseAcceptance::GatesAndT;
  changed.jointPhasePlannerWork = 1;
  changed.jointPhaseMaxWork = 1;
  auto result = run(problem, changed, after);
  ASSERT_TRUE(baseline) << baseline.error;
  ASSERT_TRUE(result) << result.error;
  sameWord(result.rotations, baseline.rotations);
  EXPECT_EQ(result.measurements, baseline.measurements);
  EXPECT_EQ(before.work(), after.work());
  expectEffects(problem, result);
}

TEST(SynthesisJointIntegration,
     BothPoliciesPreserveFullOrderedMeasuredEffects) {
  Problem problem;
  auto settings = options();
  settings.jointPhaseCandidates = 0;
  SearchBudget baselineBudget(100000000);
  auto baseline = run(problem, settings, baselineBudget);
  ASSERT_TRUE(baseline) << baseline.error;
  auto anchor = tcount::cost(problem.width, baseline.rotations);
  settings.jointPhaseCandidates = 4;
  for (auto policy : {graph::JointPhaseAcceptance::AllFive,
                      graph::JointPhaseAcceptance::GatesAndT}) {
    settings.jointPhaseAcceptance = policy;
    SearchBudget budget(200000000);
    auto result = run(problem, settings, budget);
    ASSERT_TRUE(result) << result.error;
    expectEffects(problem, result);
    auto cost = tcount::cost(problem.width, result.rotations);
    EXPECT_LE(cost.gates, anchor.gates);
    EXPECT_LE(cost.t, anchor.t);
    if (policy == graph::JointPhaseAcceptance::AllFive) {
      EXPECT_LE(cost.twoQubit, anchor.twoQubit);
      EXPECT_LE(cost.depth, anchor.depth);
      EXPECT_LE(cost.tDepth, anchor.tDepth);
    }
    EXPECT_FALSE(budget.stopped());
  }
}

TEST(SynthesisJointIntegration, OptionalExhaustionRetainsCheckedIncumbent) {
  Problem problem;
  auto settings = options();
  settings.jointPhaseCandidates = 0;
  SearchBudget controlBudget(100000000);
  auto baseline = run(problem, settings, controlBudget);
  ASSERT_TRUE(baseline) << baseline.error;
  settings.jointPhaseCandidates = 4;
  settings.jointPhaseAcceptance = graph::JointPhaseAcceptance::GatesAndT;
  settings.jointPhaseMaxWork = 1;
  SearchBudget budget(100000000);
  auto result = run(problem, settings, budget);
  ASSERT_TRUE(result) << result.error;
  sameWord(result.rotations, baseline.rotations);
  EXPECT_EQ(result.measurements, baseline.measurements);
  EXPECT_FALSE(budget.stopped());
  EXPECT_TRUE(budget.take());
  expectEffects(problem, result);
}

TEST(SynthesisJointIntegration, LegacyReserveExhaustionDoesNotStopParent) {
  auto settings = options();
  settings.legacyPortfolioTimeoutMs = 180000;
  settings.jointPhaseMaxWork = 1000000;
  // Deliberately leave too little legacy work even to establish a verified
  // incumbent. Failure must remain local: the reserved parent allowance is
  // still usable, and an unproved partial result must never reach joint search.
  SearchBudget budget(settings.jointPhaseMaxWork + 1);
  auto result = run(Problem{}, settings, budget);
  EXPECT_FALSE(result);
  EXPECT_FALSE(result.error.empty());
  EXPECT_TRUE(result.rotations.empty());
  EXPECT_FALSE(budget.stopped());
  EXPECT_GE(budget.remaining(), settings.jointPhaseMaxWork);
  EXPECT_TRUE(budget.take(settings.jointPhaseMaxWork));
}

TEST(SynthesisJointIntegration, InvalidAcceptanceIsRejectedWhenEnabled) {
  auto settings = options();
  settings.jointPhaseAcceptance = static_cast<graph::JointPhaseAcceptance>(99);
  SearchBudget budget(100000000);
  auto result = run(Problem{}, settings, budget);
  EXPECT_FALSE(result);
  EXPECT_FALSE(result.error.empty());
  EXPECT_TRUE(result.rotations.empty());
}

TEST(SynthesisJointIntegration, InvalidEnabledLimitsAreRejected) {
  for (unsigned corruption = 0; corruption < 4; ++corruption) {
    auto settings = options();
    if (corruption == 0)
      settings.jointPhaseCandidates = 5;
    else if (corruption == 1)
      settings.jointPhasePlannerWork = 0;
    else if (corruption == 2)
      settings.jointPhaseMaxWork = 0;
    else
      settings.jointPhaseTimeoutMs = 0;
    SearchBudget budget(100000000);
    auto result = run(Problem{}, settings, budget);
    EXPECT_FALSE(result) << corruption;
    EXPECT_FALSE(result.error.empty()) << corruption;
    EXPECT_TRUE(result.rotations.empty()) << corruption;
  }
}

TEST(SynthesisJointIntegration, WideExtensionRetainsPrimaryGuardsAndEffects) {
  Problem problem;
  auto settings = options();
  settings.jointPhaseCandidates = 0;
  SearchBudget baselineBudget(100000000);
  auto baseline = run(problem, settings, baselineBudget);
  ASSERT_TRUE(baseline) << baseline.error;
  auto anchor = tcount::cost(problem.width, baseline.rotations);
  settings.jointPhaseCandidates = 4;
  settings.jointPhaseWideSearch = true;
  settings.jointPhaseAcceptance = graph::JointPhaseAcceptance::GatesAndT;
  SearchBudget budget(200000000);
  auto result = run(problem, settings, budget);
  ASSERT_TRUE(result) << result.error;
  expectEffects(problem, result);
  auto cost = tcount::cost(problem.width, result.rotations);
  EXPECT_LE(cost.gates, anchor.gates);
  EXPECT_LE(cost.t, anchor.t);
  EXPECT_FALSE(budget.stopped());
}

TEST(SynthesisJointIntegration,
     OptionalCompletionCannotBypassSourceValidation) {
  auto settings = options();
  settings.jointPhaseAcceptance = graph::JointPhaseAcceptance::GatesAndT;
  for (unsigned corruption = 0; corruption < 3; ++corruption) {
    Problem problem;
    if (corruption == 0)
      problem.rotations[1].pauli.x.set(problem.width);
    else if (corruption == 1)
      problem.readouts[1] = problem.readouts[0];
    else
      problem.observables[1] = {0, 1}; // Anticommutes with the first row.
    SearchBudget budget(100000000);
    auto result = run(problem, settings, budget);
    EXPECT_FALSE(result) << corruption;
    EXPECT_FALSE(result.error.empty()) << corruption;
    EXPECT_TRUE(result.rotations.empty()) << corruption;
  }
}
