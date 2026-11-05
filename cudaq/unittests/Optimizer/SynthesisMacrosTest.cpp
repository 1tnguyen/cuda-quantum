/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "Pauli/SynthesisMacros.h"
#include "gtest/gtest.h"
#include <complex>
#include <limits>
#include <numbers>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
using namespace cudaq::pauli::graph::detail;

namespace {
using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

State applyPauli(const Pauli &pauli, const State &state) {
  State out(state.size());
  for (size_t basis = 0; basis < state.size(); ++basis) {
    Complex phase = pauli.sign ? -1. : 1.;
    size_t target = basis;
    for (unsigned wire = 0; (state.size() >> wire) > 1; ++wire) {
      bool bit = (basis >> wire) & 1;
      switch (pauli.axis(wire)) {
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

/// Independent primitive matrices, including scalar phase. No production
/// conjugation, tableau or equivalence helper participates in this oracle.
State simulate(const std::vector<Gate> &word, State state) {
  for (const auto &gate : word) {
    if (gate.kind == Kind::Rotate || gate.kind == Kind::Phase) {
      double angle = gate.angle.symbol
                         ? .173 * gate.angle.symbol
                         : std::numbers::pi * gate.angle.numerator /
                               gate.angle.denominator;
      if (gate.angle.negative)
        angle = -angle;
      if (gate.kind == Kind::Phase) {
        for (auto &amplitude : state)
          amplitude *= std::exp(imaginary * angle);
      } else {
        auto product = applyPauli(gate.pauli, state);
        for (size_t i = 0; i < state.size(); ++i)
          state[i] = std::cos(angle / 2) * state[i] -
                     imaginary * std::sin(angle / 2) * product[i];
      }
      continue;
    }
    if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
        gate.kind == Kind::Swap) {
      State old = state;
      for (size_t basis = 0; basis < state.size(); ++basis) {
        bool a = (basis >> gate.a) & 1, b = (basis >> gate.b) & 1;
        if (gate.kind == Kind::CX)
          state[basis ^ (size_t(a) << gate.b)] = old[basis];
        else if (gate.kind == Kind::CZ)
          state[basis] = a && b ? -old[basis] : old[basis];
        else
          state[a == b
                    ? basis
                    : basis ^ ((size_t(1) << gate.a) | (size_t(1) << gate.b))] =
              old[basis];
      }
      continue;
    }
    for (size_t basis = 0; basis < state.size(); ++basis)
      if (!((basis >> gate.a) & 1)) {
        size_t other = basis | (size_t(1) << gate.a);
        auto a = state[basis], b = state[other];
        switch (gate.kind) {
        case Kind::H:
          state[basis] = (a + b) / std::sqrt(2.);
          state[other] = (a - b) / std::sqrt(2.);
          break;
        case Kind::S:
          state[other] = imaginary * b;
          break;
        case Kind::Sdg:
          state[other] = -imaginary * b;
          break;
        case Kind::X:
          state[basis] = b;
          state[other] = a;
          break;
        case Kind::Y:
          state[basis] = -imaginary * b;
          state[other] = imaginary * a;
          break;
        case Kind::Z:
          state[other] = -b;
          break;
        default:
          ADD_FAILURE() << "unexpected native gate";
        }
      }
  }
  return state;
}

void expectContract(unsigned width, const std::vector<Gate> &source,
                    const MacroImplementation &candidate) {
  auto expected = source;
  expected.insert(expected.end(), candidate.exitCliffords.begin(),
                  candidate.exitCliffords.end());
  const unsigned dimension = 1u << width;
  for (unsigned column = 0; column < dimension; ++column) {
    State basis(dimension);
    basis[column] = 1;
    auto a = simulate(expected, basis), b = simulate(candidate.body, basis);
    for (unsigned row = 0; row < dimension; ++row)
      EXPECT_NEAR(std::abs(a[row] - b[row]), 0, 5e-12);
  }
}

std::vector<Gate> closed(const MacroImplementation &candidate) {
  auto word = candidate.body;
  for (auto gate = candidate.exitCliffords.rbegin();
       gate != candidate.exitCliffords.rend(); ++gate) {
    Gate inverse = *gate;
    if (inverse.kind == Kind::S)
      inverse.kind = Kind::Sdg;
    else if (inverse.kind == Kind::Sdg)
      inverse.kind = Kind::S;
    word.push_back(inverse);
  }
  return word;
}

std::vector<Gate> example() {
  return {Gate::rotation({0, 3, true}, {-1, 4}), Gate::rotation({2, 0}, {1, 4}),
          Gate::rotation({1, 2}, {0, 1, 3, true})};
}

MacroLimits smallLimits() {
  MacroLimits limits;
  limits.maxRounds = 1;
  limits.beamWidth = 2;
  limits.maxCandidates = 6;
  limits.maxWork = 10000000;
  return limits;
}
} // namespace

TEST(SynthesisMacros, AtomicPairsExhaustTinyFamilyAndPreserveQuantumExit) {
  auto word = example();
  const auto original = word;
  Options options;
  auto limits = smallLimits();
  SearchBudget budget(limits.maxWork);
  auto result = solveMacroRegion(2, word, options, limits, budget);
  ASSERT_EQ(result.status, MacroStatus::Complete) << result.error;
  ASSERT_GT(result.candidates.size(), 1u);
  EXPECT_EQ(result.actions, 10u); // Identity, six local, three entanglers.
  EXPECT_EQ(result.expandedPairs, 99u);
  EXPECT_EQ(result.completedProposals, 100u);
  EXPECT_EQ(result.work, budget.work());
  EXPECT_LE(result.peakStates, 2 * limits.beamWidth + 1);
  EXPECT_TRUE(result.candidates[0].exitCliffords.empty());
  for (size_t i = 0; i < result.candidates.size(); ++i) {
    // Third wire is an untouched external reference; arbitrary superpositions
    // and entanglement follow from comparing the complete linear operator.
    expectContract(3, word, result.candidates[i]);
    for (size_t j = 0; j < i; ++j)
      EXPECT_FALSE(macro_detail::sameImplementation(result.candidates[i],
                                                    result.candidates[j]));
  }
  EXPECT_TRUE(macro_detail::sameWord(word, original));
}

TEST(SynthesisMacros, SupportIncreasingFirstActionRemainsEligible) {
  // CX(0,1) first spreads Z1 to Z0Z1. There is deliberately no pruning before
  // the second CX restores it and makes the leading rotation emit-ready.
  std::vector<Gate> word{Gate::rotation({0, 2}, {1, 4}),
                         Gate::rotation({3, 0}, {-1, 4})};
  Options options;
  auto limits = smallLimits();
  limits.maxCandidates = 2;
  unsigned eligible = 0;
  auto scorer = [&](const MacroImplementation &candidate,
                    SearchBudget &work) -> std::optional<CostRank> {
    if (!work.take(2 + 16 * (candidate.body.size() +
                             candidate.exitCliffords.size() + 1)))
      return std::nullopt;
    const auto &exit = candidate.exitCliffords;
    if (exit.size() != 2 || exit[0].kind != Kind::CX ||
        exit[1].kind != Kind::CX || exit[0].a != 0 || exit[0].b != 1 ||
        exit[1].a != 0 || exit[1].b != 1)
      return std::nullopt;
    ++eligible;
    return circuitScore(2, closed(candidate), options);
  };
  SearchBudget budget(limits.maxWork);
  auto result = solveMacroRegion(2, word, options, limits, budget, scorer);
  ASSERT_EQ(result.status, MacroStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 2u);
  EXPECT_EQ(eligible, 1u);
  expectContract(2, word, result.candidates[1]);
}

TEST(SynthesisMacros, SignedOpaqueAnglesAndScalarPhasesAreNotDiscarded) {
  std::vector<Gate> word{Gate::phase({3, 8}),
                         Gate::rotation({3, 5, true}, {-1, 4}),
                         Gate::rotation({4, 3, true}, {0, 1, 5, true}),
                         Gate::rotation({2, 4}, {3, 8})};
  auto limits = smallLimits();
  Options options;
  SearchBudget budget(limits.maxWork);
  auto result = solveMacroRegion(3, word, options, limits, budget);
  ASSERT_EQ(result.status, MacroStatus::Complete) << result.error;
  ASSERT_GT(result.candidates.size(), 1u);
  for (const auto &candidate : result.candidates)
    expectContract(4, word, candidate);
}

TEST(SynthesisMacros, CompletedScoringRanksBeforeTheOutputCap) {
  auto word = example();
  auto limits = smallLimits();
  limits.maxCandidates = 2;
  Options options;
  std::optional<CostRank> best;
  auto scorer = [&](const MacroImplementation &candidate,
                    SearchBudget &work) -> std::optional<CostRank> {
    if (!work.take(2 + 16 * (candidate.body.size() +
                             candidate.exitCliffords.size() + 1)))
      return std::nullopt;
    auto rank = circuitScore(2, closed(candidate), options);
    if (!best || rank < *best)
      best = rank;
    return rank;
  };
  SearchBudget budget(limits.maxWork);
  auto result = solveMacroRegion(2, word, options, limits, budget, scorer);
  ASSERT_EQ(result.status, MacroStatus::Complete) << result.error;
  ASSERT_EQ(result.candidates.size(), 2u);
  ASSERT_TRUE(best);
  EXPECT_EQ(circuitScore(2, closed(result.candidates[1]), options), *best);
  EXPECT_EQ(result.work, budget.work());
}

TEST(SynthesisMacros, ExhaustionRetainsBaselineAndCompletedAlternatives) {
  auto word = example();
  Options options;
  auto firstLimits = smallLimits();
  firstLimits.maxCandidates = 1;
  SearchBudget firstBudget(firstLimits.maxWork);
  auto first = solveMacroRegion(2, word, options, firstLimits, firstBudget);
  ASSERT_EQ(first.status, MacroStatus::Complete);
  ASSERT_EQ(first.candidates.size(), 1u);
  auto limits = smallLimits();
  limits.maxWork = first.work;
  SearchBudget parent(first.work + 1000);
  auto exhausted = solveMacroRegion(2, word, options, limits, parent);
  EXPECT_EQ(exhausted.status, MacroStatus::Exhausted);
  ASSERT_EQ(exhausted.candidates.size(), 1u);
  EXPECT_TRUE(macro_detail::sameImplementation(first.candidates[0],
                                               exhausted.candidates[0]));
  EXPECT_EQ(exhausted.work, first.work);
  EXPECT_FALSE(parent.stopped());
  EXPECT_TRUE(parent.take());

  // A caller with less remaining allowance than maxWork must not become
  // sticky-exhausted when the provider's next atomic charge cannot fit.
  SearchBudget smallerParent(1);
  auto noStart =
      solveMacroRegion(2, word, options, smallLimits(), smallerParent);
  EXPECT_EQ(noStart.status, MacroStatus::Exhausted);
  EXPECT_TRUE(noStart.candidates.empty());
  EXPECT_FALSE(smallerParent.stopped());
  EXPECT_TRUE(smallerParent.take());

  limits = smallLimits();
  unsigned calls = 0;
  auto scorer = [&](const MacroImplementation &candidate,
                    SearchBudget &work) -> std::optional<CostRank> {
    if (++calls == 2) {
      EXPECT_FALSE(work.take(std::numeric_limits<uint64_t>::max()));
      return std::nullopt;
    }
    if (!work.take(2 + 16 * (candidate.body.size() +
                             candidate.exitCliffords.size() + 1)))
      return std::nullopt;
    return circuitScore(2, closed(candidate), options);
  };
  SearchBudget reserved(2 * limits.maxWork);
  auto partial = solveMacroRegion(2, word, options, limits, reserved, scorer);
  EXPECT_EQ(partial.status, MacroStatus::Exhausted);
  ASSERT_EQ(partial.candidates.size(), 2u);
  EXPECT_EQ(partial.work, reserved.work());
  EXPECT_FALSE(reserved.stopped());
  EXPECT_TRUE(reserved.take());
  for (const auto &candidate : partial.candidates)
    expectContract(2, word, candidate);
}

TEST(SynthesisMacros, NativeProfileGateLimitsAndPairCapsApply) {
  auto word = example();
  Options options;
  options.nativeCX = false;
  options.nativeSwap = false;
  options.maxGates = 64;
  auto limits = smallLimits();
  limits.maxActionPairs = 17;
  SearchBudget budget(limits.maxWork);
  auto result = solveMacroRegion(2, word, options, limits, budget);
  ASSERT_EQ(result.status, MacroStatus::Exhausted);
  EXPECT_EQ(result.expandedPairs, 17u);
  ASSERT_GT(result.candidates.size(), 1u);
  for (const auto &candidate : result.candidates) {
    EXPECT_LE(candidate.body.size(), options.maxGates);
    for (const auto *part : {&candidate.body, &candidate.exitCliffords})
      for (const auto &gate : *part) {
        EXPECT_NE(gate.kind, Kind::CX);
        EXPECT_NE(gate.kind, Kind::Swap);
      }
    expectContract(2, word, candidate);
  }
  options.maxGates = 1;
  SearchBudget tiny(limits.maxWork);
  auto missing = solveMacroRegion(2, word, options, limits, tiny);
  EXPECT_EQ(missing.status, MacroStatus::Exhausted);
  EXPECT_TRUE(missing.candidates.empty());
}

TEST(SynthesisMacros, HighPhysicalWiresAndInvalidDimensions) {
  auto word = example();
  for (auto &gate : word) {
    auto &p = gate.pauli;
    p.x = QubitSet::single(0, p.x.test(0)) | QubitSet::single(130, p.x.test(1));
    p.z = QubitSet::single(0, p.z.test(0)) | QubitSet::single(130, p.z.test(1));
  }
  auto limits = smallLimits();
  limits.maxActionPairs = 5;
  Options options;
  SearchBudget budget(limits.maxWork);
  auto result = solveMacroRegion(131, word, options, limits, budget);
  EXPECT_EQ(result.status, MacroStatus::Exhausted);
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.actions, 10u); // Active support, not the register width.
  auto compact = [](std::vector<Gate> gates) {
    for (auto &gate : gates) {
      if (gate.kind == Kind::Rotate) {
        gate.pauli.x = QubitSet::single(0, gate.pauli.x.test(0)) |
                       QubitSet::single(1, gate.pauli.x.test(130));
        gate.pauli.z = QubitSet::single(0, gate.pauli.z.test(0)) |
                       QubitSet::single(1, gate.pauli.z.test(130));
      } else if (gate.kind != Kind::Phase) {
        EXPECT_TRUE(gate.a == 0 || gate.a == 130);
        gate.a = gate.a == 130;
        if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
            gate.kind == Kind::Swap) {
          EXPECT_TRUE(gate.b == 0 || gate.b == 130);
          gate.b = gate.b == 130;
        }
      }
    }
    return gates;
  };
  for (const auto &candidate : result.candidates)
    expectContract(2, compact(word),
                   {compact(candidate.body), compact(candidate.exitCliffords)});
  SearchBudget invalid(limits.maxWork);
  EXPECT_EQ(solveMacroRegion(130, word, options, limits, invalid).status,
            MacroStatus::Unsupported);
  limits.maxActiveWires = 1;
  EXPECT_EQ(solveMacroRegion(131, word, options, limits, invalid).status,
            MacroStatus::Unsupported);
  limits.maxActiveWires = 8;
  options.objective = static_cast<SynthesisObjective>(999);
  auto invalidObjective = solveMacroRegion(131, word, options, limits, invalid);
  EXPECT_EQ(invalidObjective.status, MacroStatus::Unsupported);
  EXPECT_TRUE(invalidObjective.candidates.empty());
  EXPECT_EQ(invalidObjective.work, 0u);
  options.objective = SynthesisObjective::TotalGates;
  word[0].angle.denominator = 0;
  EXPECT_EQ(solveMacroRegion(131, word, options, limits, invalid).status,
            MacroStatus::Unsupported);
}

TEST(SynthesisMacros, EmptyAndScalarWordsNeedNoMacroSearch) {
  Options options;
  auto limits = smallLimits();
  SearchBudget budget(limits.maxWork);
  auto empty = solveMacroRegion(2, {}, options, limits, budget);
  ASSERT_EQ(empty.status, MacroStatus::Complete);
  ASSERT_EQ(empty.candidates.size(), 1u);
  EXPECT_TRUE(empty.candidates[0].body.empty());
  std::vector<Gate> scalars{Gate::phase({3, 8}),
                            Gate::rotation({0, 0, true}, {1, 4})};
  auto scalar = solveMacroRegion(2, scalars, options, limits, budget);
  ASSERT_EQ(scalar.status, MacroStatus::Complete);
  ASSERT_EQ(scalar.candidates.size(), 1u);
  EXPECT_EQ(scalar.expandedPairs, 0u);
  expectContract(2, scalars, scalar.candidates[0]);
}
