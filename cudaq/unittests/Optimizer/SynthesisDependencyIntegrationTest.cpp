/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "gtest/gtest.h"
#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;

namespace {
Options isolatedSynthesis() {
  Options options;
  options.simplify = false;
  options.discrepancySteps = 0;
  options.acceptOnlyImprovement = false;
  return options;
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
} // namespace

TEST(SynthesisDependencyIntegration, ExactConstraintsSurviveCarriedCliffords) {
  // XX conflicts with both ZI and IZ; the latter commute with each other.
  // Repeated/signed occurrences must share classes, and entangling/basis
  // changes must preserve the original occurrence ordering constraints.
  const std::vector<Gate> input{
      Gate::rotation({3, 0}, {1, 8}), Gate::rotation({0, 1}, {1, 8}),
      Gate::rotation({3, 0, true}, {1, 8}), Gate::rotation({0, 2}, {-1, 8}),
      Gate::rotation({0, 1}, {3, 8})};
  auto options = isolatedSynthesis();
  auto result = synthesize(2, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.rotationsAfter, 5u);
  EXPECT_EQ(result.stats.dependencies, 6u);
  EXPECT_EQ(result.stats.dependencyAxisClasses, 3u);
  EXPECT_EQ(result.stats.dependencyConflictEntries, 4u);
  EXPECT_GT(result.stats.tqes, 0u);
  EXPECT_TRUE(equivalent(2, input, result.gates));
  // A sufficient shared budget changes accounting, not the chosen circuit.
  SearchBudget budget(100000000);
  auto bounded = synthesize(2, input, options, &budget);
  ASSERT_TRUE(bounded) << bounded.error;
  EXPECT_FALSE(budget.stopped());
  sameCircuit(bounded.gates, result.gates);
}

TEST(SynthesisDependencyIntegration,
     RepeatedAxesKeepLogicalCapAndUseTwoConflicts) {
  constexpr unsigned repeats = 96;
  std::vector<Gate> input(repeats, Gate::rotation({1, 0}, {1, 8}));
  input.insert(input.end(), repeats, Gate::rotation({0, 1}, {1, 8}));
  auto options = isolatedSynthesis();
  options.maxDependencies = repeats * repeats;
  auto result = synthesize(1, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.rotationsAfter, 2 * repeats);
  EXPECT_EQ(result.stats.dependencies, repeats * repeats);
  EXPECT_EQ(result.stats.dependencyAxisClasses, 2u);
  EXPECT_EQ(result.stats.dependencyConflictEntries, 2u);
  EXPECT_TRUE(equivalent(1, input, result.gates));

  // Compressed storage must not silently reinterpret or bypass the public
  // occurrence-edge limit. No candidate has been emitted at this failure.
  --options.maxDependencies;
  auto rejected = synthesize(1, input, options);
  EXPECT_FALSE(rejected);
  EXPECT_TRUE(rejected.stats.budgetExhausted);
  EXPECT_NE(rejected.error.find("dependency limit"), std::string::npos);
  EXPECT_TRUE(rejected.gates.empty());
}

TEST(SynthesisDependencyIntegration, CommutingSingletsKeepOccurrenceOrder) {
  const std::vector<Gate> input{
      Gate::rotation({0, 2}, {0, 1, 11}), Gate::rotation({0, 1}, {0, 1, 12}),
      Gate::rotation({0, 2}, {0, 1, 13}), Gate::rotation({0, 1}, {0, 1, 14})};
  auto result = synthesize(2, input, isolatedSynthesis());
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.dependencies, 0u);
  EXPECT_EQ(result.stats.dependencyAxisClasses, 2u);
  EXPECT_EQ(result.stats.dependencyConflictEntries, 0u);
  sameCircuit(result.gates, input);
}

TEST(SynthesisDependencyIntegration,
     IndexUsesTheWordAfterSimplificationAndPruning) {
  const std::vector<Gate> input{Gate::rotation({0, 1}, {1, 4}),
                                Gate::rotation({0, 1}, {-1, 4})};
  auto options = isolatedSynthesis();
  options.simplify = true;
  auto simplified = synthesize(1, input, options);
  ASSERT_TRUE(simplified) << simplified.error;
  EXPECT_EQ(simplified.stats.rotationsBefore, 2u);
  EXPECT_EQ(simplified.stats.rotationsAfter, 0u);
  EXPECT_EQ(simplified.stats.dependencyAxisClasses, 0u);
  EXPECT_EQ(simplified.stats.dependencyConflictEntries, 0u);
  EXPECT_TRUE(simplified.gates.empty());

  options.simplify = false;
  options.release = true;
  auto released = synthesize(1, input, options);
  ASSERT_TRUE(released) << released.error;
  EXPECT_EQ(released.stats.rotationsBefore, 2u);
  EXPECT_EQ(released.stats.rotationsAfter, 0u);
  EXPECT_EQ(released.stats.released, 2u);
  EXPECT_EQ(released.stats.dependencyAxisClasses, 0u);
  EXPECT_EQ(released.stats.dependencyConflictEntries, 0u);
  EXPECT_TRUE(released.gates.empty());
}

TEST(SynthesisDependencyIntegration,
     IndexWorkExhaustionCannotPublishACandidate) {
  const std::vector<Gate> input{Gate::rotation({1, 0}, {1, 8}),
                                Gate::rotation({0, 1}, {1, 8})};
  // Admit the public input import, temporary axis copy and classification,
  // then exhaust during dependency indexing, before any gate is emitted.
  SearchBudget budget(9);
  auto result = synthesize(1, input, isolatedSynthesis(), &budget);
  EXPECT_FALSE(result);
  EXPECT_TRUE(result.stats.budgetExhausted);
  EXPECT_TRUE(budget.stopped());
  EXPECT_EQ(result.stats.rotationsAfter, input.size());
  EXPECT_NE(result.error.find("shared search budget exhausted"),
            std::string::npos);
  EXPECT_TRUE(result.gates.empty());
}
