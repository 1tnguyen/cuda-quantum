/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisCommuting.h"
#include "gtest/gtest.h"
#include <complex>
#include <numbers>
#include <queue>

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
    for (unsigned wire = 0; (size_t(1) << wire) < state.size(); ++wire) {
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

/// Independent dense matrices: no production frame, conjugation or replay.
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

void expectEqual(unsigned width, const std::vector<Gate> &a,
                 const std::vector<Gate> &b) {
  unsigned dimension = 1u << width;
  for (unsigned column = 0; column < dimension; ++column) {
    State input(dimension);
    input[column] = 1;
    auto left = simulate(a, input), right = simulate(b, input);
    for (unsigned row = 0; row < dimension; ++row)
      EXPECT_NEAR(std::abs(left[row] - right[row]), 0, 4e-12);
  }
}

void expectContract(unsigned width, const std::vector<Gate> &original,
                    const CommutingImplementation &candidate) {
  auto expected = original;
  expected.insert(expected.end(), candidate.exitCliffords.begin(),
                  candidate.exitCliffords.end());
  expectEqual(width, expected, candidate.body);
}

std::vector<Gate> diagonalExample() {
  return {Gate::rotation({0, 5}, {1, 4}), Gate::rotation({0, 6}, {-1, 4}),
          Gate::rotation({0, 7}, {1, 4})};
}
} // namespace

TEST(SynthesisCommuting, ExactStarToursMatchIndependentHypercubeSearch) {
  // Every subset of the seven nonzero vertices of a three-control star.
  // BFS explores individual CX edges; Held--Karp uses metric distances.
  for (unsigned required = 1; required < 128; ++required) {
    commuting_detail::Star star{3, {0, 1, 2}, {}};
    for (unsigned vertex = 1; vertex < 8; ++vertex)
      if ((required >> (vertex - 1)) & 1)
        star.demands.push_back({QubitSet(vertex), {{1, 4}}});
    CommutingCandidates accounting;
    SearchBudget shared(10000000);
    commuting_detail::Budget budget(shared, 10000000, accounting);
    auto tours = commuting_detail::exactTours(star, {}, budget);
    ASSERT_TRUE(tours);
    ASSERT_EQ(tours->size(), 8u);
    std::vector<int> distances(8 * 128, -1);
    std::queue<std::pair<unsigned, unsigned>> pending;
    distances[0] = 0;
    pending.push({0, 0});
    while (!pending.empty()) {
      auto [vertex, seen] = pending.front();
      pending.pop();
      for (unsigned control = 0; control < 3; ++control) {
        unsigned next = vertex ^ (1u << control);
        unsigned nextSeen = seen | (next ? (1u << (next - 1)) & required : 0);
        auto at = nextSeen * 8 + next;
        if (distances[at] < 0) {
          distances[at] = distances[seen * 8 + vertex] + 1;
          pending.push({next, nextSeen});
        }
      }
    }
    for (const auto &tour : *tours) {
      unsigned endpoint = 0;
      for (unsigned bit = 0; bit < 3; ++bit)
        endpoint |= unsigned(tour.waypoints.back().test(bit)) << bit;
      EXPECT_EQ(tour.cx, uint64_t(distances[required * 8 + endpoint]));
    }
  }
}

TEST(SynthesisCommuting, JointParityWalkRetainsUsefulExitFrames) {
  auto word = diagonalExample();
  Options options;
  CommutingLimits limits;
  limits.maxDiagonalizers = 0;
  SearchBudget budget(10000000);
  auto result = solveCommutingRegion(3, word, options, limits, budget);
  ASSERT_EQ(result.status, CommutingStatus::Complete) << result.error;
  ASSERT_GT(result.candidates.size(), 1u);
  bool threeCx = false, nonemptyExit = false;
  for (const auto &candidate : result.candidates) {
    expectContract(3, word, candidate);
    unsigned cx =
        std::count_if(candidate.body.begin(), candidate.body.end(),
                      [](const Gate &g) { return g.kind == Kind::CX; });
    threeCx |= cx == 3;
    nonemptyExit |= !candidate.exitCliffords.empty();
    EXPECT_EQ(
        std::count_if(candidate.body.begin(), candidate.body.end(),
                      [](const Gate &g) { return g.kind == Kind::Rotate; }),
        word.size());
  }
  EXPECT_TRUE(threeCx);
  EXPECT_TRUE(nonemptyExit);
}

TEST(SynthesisCommuting, SignedDiagonalizationPreservesFullUnitaryAndPhase) {
  std::vector<Gate> word{Gate::rotation({3, 0, true}, {1, 4}),
                         Gate::rotation({0, 3}, {-2, 8}),
                         Gate::rotation({3, 3}, {0, 1, 5, true})};
  SearchBudget budget(20000000);
  auto result = solveCommutingRegion(2, word, {}, {}, budget);
  ASSERT_EQ(result.status, CommutingStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.rank, 2u);
  for (const auto &candidate : result.candidates) {
    expectContract(2, word, candidate);
    for (const auto &gate : candidate.body)
      if (gate.kind == Kind::Rotate) {
        EXPECT_FALSE(bool(gate.pauli.x));
        EXPECT_EQ(gate.pauli.weight(), 1u);
      }
  }
}

TEST(SynthesisCommuting, ExitFrameTransportsNoncommutingQuantumSuffix) {
  auto word = diagonalExample();
  CommutingLimits limits;
  limits.maxDiagonalizers = 0;
  SearchBudget budget(10000000);
  auto result = solveCommutingRegion(3, word, {}, limits, budget);
  ASSERT_FALSE(result.candidates.empty());
  Gate suffix = Gate::rotation({1, 0}, {2, 7});
  for (const auto &candidate : result.candidates) {
    Gate transported = suffix;
    for (const auto &clifford : candidate.exitCliffords)
      transported.pauli =
          tcount::detail::conjugate(transported.pauli, clifford);
    auto actual = candidate.body;
    actual.push_back(transported);
    auto expected = word;
    expected.push_back(suffix);
    expected.insert(expected.end(), candidate.exitCliffords.begin(),
                    candidate.exitCliffords.end());
    expectEqual(3, expected, actual);
  }
}

TEST(SynthesisCommuting, NativeCzProfileAndBoundedGreedyFallback) {
  auto word = diagonalExample();
  Options options;
  options.nativeCX = false;
  CommutingLimits limits;
  limits.maxDiagonalizers = 0;
  limits.maxDpEntries = 0; // Exact search cannot allocate; use bounded greedy.
  SearchBudget budget(10000000);
  auto result = solveCommutingRegion(3, word, options, limits, budget);
  ASSERT_EQ(result.status, CommutingStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  for (const auto &candidate : result.candidates) {
    expectContract(3, word, candidate);
    for (const auto &gate : candidate.body)
      EXPECT_NE(gate.kind, Kind::CX);
  }
}

TEST(SynthesisCommuting, HighWireIndicesUseExistingQuantumWiresOnly) {
  QubitSet a = QubitSet::single(64), b = QubitSet::single(130);
  std::vector<Gate> word{Gate::rotation({{}, a}, {1, 4}),
                         Gate::rotation({{}, b}, {-1, 4}),
                         Gate::rotation({{}, a | b}, {0, 1, 2})};
  CommutingLimits limits;
  limits.maxDiagonalizers = 0;
  SearchBudget budget(10000000);
  auto result = solveCommutingRegion(131, word, {}, limits, budget);
  ASSERT_EQ(result.status, CommutingStatus::Complete) << result.error;
  ASSERT_FALSE(result.candidates.empty());
  auto relabel = [](std::vector<Gate> gates) {
    auto wire = [](unsigned q) {
      EXPECT_TRUE(q == 64 || q == 130);
      return q == 64 ? 0u : 1u;
    };
    for (auto &gate : gates) {
      if (gate.kind == Kind::Rotate) {
        Pauli p;
        p.sign = gate.pauli.sign;
        for (unsigned q : gate.pauli.x.set_bits())
          p.x |= QubitSet::single(wire(q));
        for (unsigned q : gate.pauli.z.set_bits())
          p.z |= QubitSet::single(wire(q));
        gate.pauli = p;
      } else {
        gate.a = wire(gate.a);
        if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
            gate.kind == Kind::Swap)
          gate.b = wire(gate.b);
      }
    }
    return gates;
  };
  for (const auto &candidate : result.candidates)
    expectContract(2, relabel(word),
                   {relabel(candidate.body), relabel(candidate.exitCliffords)});
}

TEST(SynthesisCommuting, RejectsNoncommutingOrMalformedRegions) {
  std::vector<Gate> word{Gate::rotation({1, 0}, {1, 4}),
                         Gate::rotation({0, 1}, {1, 4})};
  SearchBudget budget(1000000);
  EXPECT_EQ(solveCommutingRegion(1, word, {}, {}, budget).status,
            CommutingStatus::Unsupported);
  word = {{Kind::H, 0}};
  EXPECT_EQ(solveCommutingRegion(1, word, {}, {}, budget).status,
            CommutingStatus::Unsupported);
  word = {Gate::rotation({0, 2}, {1, 4})};
  EXPECT_EQ(solveCommutingRegion(1, word, {}, {}, budget).status,
            CommutingStatus::Unsupported);
  auto empty = solveCommutingRegion(2, {}, {}, {}, budget);
  ASSERT_EQ(empty.candidates.size(), 1u);
  EXPECT_TRUE(empty.candidates[0].body.empty());
}

TEST(SynthesisCommuting, LimitsRetainOnlyCompleteRegionImplementations) {
  auto word = diagonalExample();
  CommutingLimits limits;
  limits.maxDiagonalizers = 0;
  SearchBudget completeBudget(10000000);
  auto complete = solveCommutingRegion(3, word, {}, limits, completeBudget);
  ASSERT_EQ(complete.status, CommutingStatus::Complete);
  ASSERT_GT(complete.work, 1u);
  EXPECT_EQ(complete.work, completeBudget.work());
  for (uint64_t allowance :
       {uint64_t(0), complete.work / 2, complete.work - 1}) {
    SearchBudget bounded(allowance);
    auto result = solveCommutingRegion(3, word, {}, limits, bounded);
    EXPECT_EQ(result.status, CommutingStatus::Exhausted);
    for (const auto &candidate : result.candidates)
      expectContract(3, word, candidate);
  }
  SearchBudget nearEnd(complete.work - 1);
  auto retained = solveCommutingRegion(3, word, {}, limits, nearEnd);
  EXPECT_FALSE(retained.candidates.empty());
}

TEST(SynthesisCommuting,
     SmallRankEnumeratesOrdersAndExitFramesBeforeShortlist) {
  std::vector<Gate> word;
  for (unsigned parity = 1; parity < 16; ++parity)
    word.push_back(Gate::rotation({0, parity}, {1, 4}));
  CommutingLimits limits;
  limits.exhaustiveSmallRank = true;
  limits.maxDiagonalizers = 0;
  limits.maxCandidates = 3;
  SearchBudget budget(200000000);
  auto result = solveCommutingRegion(4, word, {}, limits, budget);
  ASSERT_EQ(result.status, CommutingStatus::Complete) << result.error;
  EXPECT_EQ(result.rank, 4u);
  EXPECT_EQ(result.wireOrders, 24u);
  // Four ordered stars have 1*2*4*8 exits, irrespective of shortlist size.
  EXPECT_EQ(result.completedImplementations, 24u * 64u);
  ASSERT_EQ(result.candidates.size(), 3u);
  EXPECT_EQ(result.work, budget.work());
  std::vector<std::array<unsigned, 4>> frames;
  for (const auto &candidate : result.candidates) {
    expectContract(4, word, candidate);
    std::array<unsigned, 4> rows{1, 2, 4, 8};
    for (const auto &gate : candidate.exitCliffords) {
      ASSERT_EQ(gate.kind, Kind::CX);
      rows[gate.b] ^= rows[gate.a];
    }
    // Equivalent linear exit maps do not consume separate shortlist slots.
    EXPECT_EQ(std::find(frames.begin(), frames.end(), rows), frames.end());
    frames.push_back(rows);
  }
}

TEST(SynthesisCommuting, CompletionScorerCanSelectALongerNativePrefix) {
  auto word = diagonalExample();
  CommutingLimits limits;
  limits.exhaustiveSmallRank = true;
  limits.maxDiagonalizers = 0;
  limits.maxCandidates = 1;
  Options options;
  SearchBudget prefixBudget(20000000);
  auto prefixRanked =
      solveCommutingRegion(3, word, options, limits, prefixBudget);
  ASSERT_EQ(prefixRanked.status, CommutingStatus::Complete);
  ASSERT_EQ(prefixRanked.candidates.size(), 1u);

  // A generic feasible full-quantum completion: omit an identity CNOT frame,
  // otherwise reverse its recorded CNOT word. This is deliberately not a
  // minimum readout solver. Its actual complete native cost must still override
  // the shorter prefix when selecting among regional proposals.
  auto completion = [](const CommutingImplementation &candidate) {
    std::array<unsigned, 3> rows{1, 2, 4};
    for (const auto &gate : candidate.exitCliffords) {
      EXPECT_EQ(gate.kind, Kind::CX);
      rows[gate.b] ^= rows[gate.a];
    }
    auto full = candidate.body;
    if (rows != std::array<unsigned, 3>{1, 2, 4})
      full.insert(full.end(), candidate.exitCliffords.rbegin(),
                  candidate.exitCliffords.rend());
    return full;
  };
  CommutingScorer scorer = [&](const CommutingImplementation &candidate,
                               SearchBudget &slice) -> std::optional<CostRank> {
    if (!slice.take(candidate.body.size() + 3 * candidate.exitCliffords.size() +
                    4))
      return std::nullopt;
    return circuitScore(3, completion(candidate), options);
  };
  SearchBudget completeBudget(20000000);
  auto completeRanked =
      solveCommutingRegion(3, word, options, limits, completeBudget, scorer);
  ASSERT_EQ(completeRanked.status, CommutingStatus::Complete)
      << completeRanked.error;
  ASSERT_EQ(completeRanked.candidates.size(), 1u);
  const auto &shortPrefix = prefixRanked.candidates[0];
  const auto &betterCompletion = completeRanked.candidates[0];
  EXPECT_GT(betterCompletion.body.size(), shortPrefix.body.size());
  EXPECT_LT(circuitScore(3, completion(betterCompletion), options),
            circuitScore(3, completion(shortPrefix), options));
  expectContract(3, word, betterCompletion);
  expectEqual(3, word, completion(betterCompletion));
  EXPECT_EQ(completeRanked.work, completeBudget.work());
}

TEST(SynthesisCommuting, ScoringExhaustionRetainsPreviouslyCompletedProposal) {
  auto word = diagonalExample();
  CommutingLimits limits;
  limits.exhaustiveSmallRank = true;
  limits.maxDiagonalizers = 0;
  unsigned calls = 0;
  CommutingScorer scorer = [&](const CommutingImplementation &candidate,
                               SearchBudget &slice) -> std::optional<CostRank> {
    if (++calls == 2) {
      EXPECT_FALSE(slice.take(slice.remaining() + 1));
      return std::nullopt;
    }
    if (!slice.take(candidate.body.size() + 1))
      return std::nullopt;
    return circuitScore(3, candidate.body, {});
  };
  SearchBudget shared(100000000);
  auto result = solveCommutingRegion(3, word, {}, limits, shared, scorer);
  EXPECT_EQ(result.status, CommutingStatus::Exhausted);
  EXPECT_EQ(calls, 2u);
  ASSERT_EQ(result.candidates.size(), 1u);
  expectContract(3, word, result.candidates[0]);
  EXPECT_EQ(result.work, shared.work());
  EXPECT_FALSE(shared.stopped());
  EXPECT_TRUE(shared.take()); // Caller still has completion/proof allowance.
}

TEST(SynthesisCommuting, UnavailableCompletionIsNotAVerificationFailure) {
  std::vector<Gate> word{Gate::rotation({0, 1}, {1, 4})};
  CommutingLimits limits;
  limits.exhaustiveSmallRank = true;
  limits.maxDiagonalizers = 0;
  CommutingScorer unavailable =
      [](const CommutingImplementation &,
         SearchBudget &slice) -> std::optional<CostRank> {
    EXPECT_TRUE(slice.take());
    return std::nullopt;
  };
  SearchBudget budget(10000);
  auto result = solveCommutingRegion(1, word, {}, limits, budget, unavailable);
  EXPECT_EQ(result.status, CommutingStatus::Complete);
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_EQ(result.completedImplementations, 1u);
  EXPECT_FALSE(budget.stopped());

  limits.maxDpEntries = 0;
  SearchBudget bounded(10000);
  auto unsupported =
      solveCommutingRegion(3, diagonalExample(), {}, limits, bounded);
  EXPECT_EQ(unsupported.status, CommutingStatus::Unsupported);
  EXPECT_TRUE(unsupported.candidates.empty());
}
