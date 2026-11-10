/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Correctness and resource-bound tests for Pauli graph synthesis.
///
/// Uses independent small-circuit simulation to check unitary phase, output
/// permutations, and terminal measurement distributions. Also exercises search
/// controls, malformed certificates, budget exhaustion, and the highest wire
/// bit.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include "Pauli/SignedCliffordFrames.h"
#include "Pauli/SynthesisHold.h"
#include "gtest/gtest.h"
#include <bit>
#include <complex>
#include <limits>
#include <numbers>
#include <random>

using namespace cudaq::pauli::graph;

TEST(GraphSynthesis, NativeNormalizationBindingChecksOrderedSigns) {
  const std::vector<Gate> input{
      {Kind::H, 0}, {Kind::CX, 0, 1}, Gate::rotation({0, 2}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1}, {0, 2}};
  auto normalized = cudaq::pauli::pbc::normalize(2, input, observables);
  ASSERT_TRUE(normalized) << normalized.error;
  EXPECT_TRUE(verifyMeasured(2, input, observables, normalized.rotations,
                             normalized.measurements));
  auto wrong = normalized.measurements;
  wrong[0].sign = !wrong[0].sign;
  EXPECT_FALSE(
      verifyMeasured(2, input, observables, normalized.rotations, wrong));
  wrong = normalized.measurements;
  std::swap(wrong[0], wrong[1]);
  EXPECT_FALSE(
      verifyMeasured(2, input, observables, normalized.rotations, wrong));
  cudaq::pauli::SearchBudget none(0);
  EXPECT_FALSE(verifyMeasured(2, input, observables, normalized.rotations,
                              normalized.measurements, &none));
  EXPECT_TRUE(none.stopped());
}

TEST(GraphSynthesis, IncomingAnchorSurvivesReconstructionLimits) {
  const std::vector<Gate> incoming{
      {Kind::H, 0}, {Kind::CX, 0, 1}, Gate::rotation({0, 2}, {1, 4})};
  const std::vector<Pauli> observables{{0, 1}, {0, 2}};
  const std::vector<unsigned> readouts{0, 1};
  auto source = cudaq::pauli::pbc::normalize(2, incoming, observables);
  ASSERT_TRUE(source) << source.error;
  Options options;
  options.maxGates = incoming.size();
  options.maxSteps = 0;
  options.legacyWorkLimit = 100000;
  cudaq::pauli::SearchBudget budget(10000000);
  SearchTermination status;
  auto result =
      synthesizeMeasured(2, source.rotations, source.measurements, readouts,
                         options, &budget, &status, &incoming);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(status, SearchTermination::Complete);
  EXPECT_EQ(result.rotations.size(), incoming.size());
  EXPECT_TRUE(verifyMeasured(2, incoming, observables, result.rotations,
                             result.measurements));
  auto memoryLimited = options;
  memoryLimited.maxSteps = 100;
  memoryLimited.maxDependencyBytes = 0;
  auto retained =
      synthesizeMeasured(2, source.rotations, source.measurements, readouts,
                         memoryLimited, nullptr, &status, &incoming);
  ASSERT_TRUE(retained) << retained.error;
  EXPECT_EQ(status, SearchTermination::Complete);
  EXPECT_TRUE(verifyMeasured(2, incoming, observables, retained.rotations,
                             retained.measurements));
  auto wrong = incoming;
  wrong[0].a = 1;
  auto invalid =
      synthesizeMeasured(2, source.rotations, source.measurements, readouts,
                         options, nullptr, &status, &wrong);
  EXPECT_FALSE(invalid);
  EXPECT_EQ(status, SearchTermination::InvalidProof);
}

namespace {
// The independent statevector oracle is deliberately restricted to small
// circuits; wide tests explicitly relabel their active support before using it.
size_t smallMask(const cudaq::pauli::QubitSet &bits) {
  assert(bits.fits(sizeof(size_t) * 8));
  size_t mask = 0;
  for (unsigned q : bits.set_bits())
    mask |= size_t(1) << q;
  return mask;
}
using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex I{0, 1};
State simulate(const std::vector<Gate> &gates, State state) {
  for (Gate g : gates) {
    if (g.kind == Kind::Phase || g.kind == Kind::Rotate) {
      double angle = g.angle.symbol ? 0.173 * g.angle.symbol
                                    : std::numbers::pi * g.angle.numerator /
                                          g.angle.denominator;
      if (g.angle.negative)
        angle = -angle;
      if (g.kind == Kind::Phase) {
        for (auto &a : state)
          a *= std::exp(I * angle);
      } else {
        State old = state;
        for (size_t b = 0; b < state.size(); ++b) {
          Complex phase =
              std::pow(I, cudaq::pauli::population(g.pauli.x & g.pauli.z));
          if ((cudaq::pauli::population(g.pauli.z & b) & 1) ^ g.pauli.sign)
            phase = -phase;
          state[b ^ smallMask(g.pauli.x)] =
              std::cos(angle / 2) * old[b ^ smallMask(g.pauli.x)] -
              I * std::sin(angle / 2) * phase * old[b];
        }
      }
      continue;
    }
    if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap) {
      State old = state;
      for (size_t b = 0; b < state.size(); ++b) {
        bool a = (b >> g.a) & 1, c = (b >> g.b) & 1;
        if (g.kind == Kind::CX)
          state[b ^ (size_t(a) << g.b)] = old[b];
        else if (g.kind == Kind::CZ)
          state[b] = (a && c) ? -old[b] : old[b];
        else
          state[a == c ? b : b ^ ((size_t(1) << g.a) | (size_t(1) << g.b))] =
              old[b];
      }
      continue;
    }
    for (size_t b = 0; b < state.size(); ++b)
      if (!((b >> g.a) & 1)) {
        size_t c = b | (size_t(1) << g.a);
        Complex a = state[b], d = state[c];
        switch (g.kind) {
        case Kind::H:
          state[b] = (a + d) / std::sqrt(2.0);
          state[c] = (a - d) / std::sqrt(2.0);
          break;
        case Kind::S:
          state[c] = I * d;
          break;
        case Kind::Sdg:
          state[c] = -I * d;
          break;
        case Kind::X:
          state[b] = d;
          state[c] = a;
          break;
        case Kind::Y:
          state[b] = -I * d;
          state[c] = I * a;
          break;
        case Kind::Z:
          state[c] = -d;
          break;
        default:
          break;
        }
      }
  }
  return state;
}
void assertUnitary(unsigned n, const std::vector<Gate> &a,
                   const std::vector<Gate> &b) {
  for (unsigned col = 0; col < (1u << n); ++col) {
    State state(1u << n);
    state[col] = 1;
    auto lhs = simulate(a, state), rhs = simulate(b, state);
    for (unsigned row = 0; row < state.size(); ++row)
      ASSERT_LT(std::abs(lhs[row] - rhs[row]), 1e-11)
          << "row=" << row << " col=" << col;
  }
}
} // namespace
TEST(GraphSynthesis, ExactPhaseAndPermutation) {
  std::vector<Gate> gates{{Kind::H, 0},       {Kind::S, 0}, {Kind::H, 0},
                          {Kind::S, 0},       {Kind::H, 0}, {Kind::S, 0},
                          {Kind::Swap, 0, 2}, {Kind::Y, 1}, {Kind::CX, 2, 1}};
  Options options;
  options.acceptOnlyImprovement = false;
  auto result = synthesize(3, gates, options);
  ASSERT_TRUE(result) << result.error;
  assertUnitary(3, gates, result.gates);
  auto corrupted = result.gates;
  corrupted.push_back(Gate::phase({1, 4}));
  EXPECT_FALSE(equivalent(3, gates, corrupted));
  corrupted = result.gates;
  corrupted.push_back({Kind::Swap, 0, 2});
  EXPECT_FALSE(equivalent(3, gates, corrupted));
}

TEST(GraphSynthesis, LosslessPauliNormalizationRetainsFrameAndScalar) {
  std::vector<Gate> input{{Kind::H, 0},
                          {Kind::CX, 0, 1},
                          Gate::rotation({0, 2}, {1, 4}),
                          {Kind::S, 1},
                          Gate::rotation({1, 1, true}, {1, 7}),
                          Gate::phase({3, 8}),
                          {Kind::Swap, 0, 1}};
  auto form = cudaq::pauli::pbc::normalizeUnitary(2, input);
  ASSERT_TRUE(form) << form.error;
  ASSERT_FALSE(form.frame.empty());
  auto word = form.rotations;
  word.insert(word.end(), form.frame.begin(), form.frame.end());
  assertUnitary(2, input, word);
  EXPECT_TRUE(equivalent(2, input, word));
}

TEST(GraphSynthesis, HoldReconstructsUnmeasuredCliffords) {
  for (const auto &input : std::vector<std::vector<Gate>>{
           {{Kind::S, 0}},
           {{Kind::CZ, 0, 1}},
           {{Kind::Swap, 0, 1}, Gate::phase({1, 8})}}) {
    Options options;
    options.discrepancySteps = 0;
    auto result = synthesizeUnitary(2, input, options);
    ASSERT_TRUE(result) << result.error;
    assertUnitary(2, input, result.gates);
    EXPECT_FALSE(equivalent(2, {}, result.gates));
  }
}

namespace {
std::vector<Gate> holdThreeWireGadget(unsigned a, unsigned b, unsigned c) {
  auto rz = [](unsigned q, int sign) {
    return Gate::rotation({0, cudaq::pauli::QubitSet::single(q)}, {sign, 4});
  };
  return {{Kind::CX, b, a}, {Kind::CZ, b, c}, {Kind::CX, b, c}, {Kind::H, b},
          {Kind::CX, b, a}, rz(a, -1),        {Kind::CX, c, a}, rz(b, 1),
          {Kind::CX, c, b}, rz(a, 1),         {Kind::CX, b, a}, rz(b, -1),
          {Kind::H, b},     {Kind::Z, b}};
}
} // namespace

TEST(GraphSynthesis, HoldThreeWireTemplatePreservesPhaseAndPhysicalWires) {
  std::array<unsigned, 3> physical{0, 2, 4};
  do {
    auto source = holdThreeWireGadget(physical[0], physical[1], physical[2]);
    // Entangle a spectator, and retain both rational and opaque scalar phases.
    source.insert(source.begin() + 5, Gate::phase({3, 8}));
    source.insert(source.begin() + 9, Gate::phase({0, 1, 7, true}));
    source.insert(source.begin(), {{Kind::H, 5}, {Kind::CX, 5, physical[0]}});
    Result result;
    result.gates = source;
    Options options;
    options.algebraicCandidates = 4;
    cudaq::pauli::SearchBudget budget(1000000);
    ASSERT_TRUE(
        detail::refineHoldThreeWireTemplates(6, result, options, budget));
    EXPECT_EQ(result.gates.size(), source.size() - 2);
    EXPECT_EQ(cudaq::pauli::tcount::cost(6, result.gates).t, 4u);
    assertUnitary(6, source, result.gates);
  } while (std::next_permutation(physical.begin(), physical.end()));
}

TEST(GraphSynthesis, HoldThreeWireTemplateHonorsObjectiveAndBudget) {
  auto source = holdThreeWireGadget(0, 1, 2);
  Options options;
  options.algebraicCandidates = 4;
  options.objective = SynthesisObjective::TDepth;
  Result result;
  result.gates = source;
  cudaq::pauli::SearchBudget budget(1000000);
  ASSERT_TRUE(detail::refineHoldThreeWireTemplates(3, result, options, budget));
  EXPECT_EQ(result.gates.size(), source.size());
  options.objective = SynthesisObjective::TotalGates;
  cudaq::pauli::SearchBudget empty(0);
  ASSERT_TRUE(detail::refineHoldThreeWireTemplates(3, result, options, empty));
  EXPECT_EQ(result.gates.size(), source.size());
  // A 2*pi rotation offset changes scalar phase and must not match pi/4.
  source[5].angle = {7, 4};
  result.gates = source;
  cudaq::pauli::SearchBudget reserve(1000000);
  ASSERT_TRUE(
      detail::refineHoldThreeWireTemplates(3, result, options, reserve));
  EXPECT_EQ(result.gates.size(), source.size());
  assertUnitary(3, source, result.gates);
}

TEST(GraphSynthesis,
     HoldNormalizesExactCliffordRotationsWithIndependentReplay) {
  std::vector<Gate> input{Gate::rotation({3, 2}, {1, 2}),
                          Gate::rotation({1, 3}, {1, 4}),
                          Gate::rotation({0, 1}, {-1, 2}),
                          Gate::rotation({3, 1}, {1, 7}),
                          {Kind::CX, 1, 0},
                          Gate::phase({1, 8})};
  Options options;
  options.discrepancySteps = 0;
  auto result = synthesizeUnitary(2, input, options);
  ASSERT_TRUE(result) << result.error;
  assertUnitary(2, input, result.gates);
}

TEST(GraphSynthesis, HoldRegionalAndJointExitsPreserveCompleteUnitary) {
  std::mt19937 rng(83127);
  for (unsigned trial = 0; trial < 12; ++trial) {
    std::vector<Gate> input;
    for (unsigned i = 0; i < 8; ++i) {
      unsigned x = rng() % 4, z = rng() % 4;
      if (!(x | z))
        z = 1;
      input.push_back(
          Gate::rotation({x, z, bool(rng() & 1)}, {rng() & 1 ? 1 : -1, 4}));
    }
    input.push_back({Kind::CX, 1, 0});
    input.push_back({Kind::Swap, 0, 1});
    input.push_back(Gate::phase({1, 8}));
    Options options;
    options.holdCheckpoints = false; // Exercise the global providers directly.
    options.discrepancySteps = 0;
    options.commutingCandidates = 2;
    options.algebraicCandidates = trial % 3 == 0 ? 2 : 0;
    options.jointPhaseCandidates = 2;
    options.jointPhasePlannerWork = 1000000;
    options.jointPhaseMaxWork = 10000000;
    options.jointPhaseTimeoutMs = 10000;
    options.nativeSwap = false;
    options.nativeCX = trial % 2;
    options.nativeCZ = !options.nativeCX;
    cudaq::pauli::SearchBudget budget(100000000);
    auto result = synthesizeUnitary(2, input, options, &budget);
    ASSERT_TRUE(result) << result.error;
    assertUnitary(2, input, result.gates);
    for (const auto &gate : result.gates) {
      EXPECT_NE(gate.kind, Kind::Swap);
      EXPECT_NE(gate.kind, options.nativeCX ? Kind::CZ : Kind::CX);
    }
  }
}

TEST(GraphSynthesis, HoldCheckpointsPreserveSparseWiresOversizedAxesAndPhase) {
  std::vector<Gate> input{
      Gate::phase({3, 8}), {Kind::H, 4},
      {Kind::CX, 4, 1},    Gate::rotation({0, 2}, {1, 4}),
      {Kind::Swap, 1, 3},  {Kind::Sdg, 0},
      {Kind::CZ, 0, 2},    Gate::rotation({0b10101, 0b01010, true}, {-1, 4}),
      {Kind::H, 3},        Gate::rotation({8, 0}, {1, 7}),
      {Kind::X, 4}};
  Options options;
  options.discrepancySteps = 0;
  options.candidateNodes = 1;
  options.lookahead = 16;
  options.algebraicCandidates = 4;
  Diagnostics diagnostics;
  options.diagnostics = &diagnostics;
  cudaq::pauli::SearchBudget budget(1000000000);
  auto result = synthesizeUnitary(5, input, options, &budget);
  ASSERT_TRUE(result) << result.error;
  assertUnitary(5, input, result.gates);
  EXPECT_TRUE(std::any_of(diagnostics.events.begin(), diagnostics.events.end(),
                          [](const auto &event) {
                            return event.route == "hold-checkpoint-3" &&
                                   event.status == "verified";
                          }));
}

TEST(GraphSynthesis, HoldCheckpointOptionalExhaustionRetainsVerifiedUnitary) {
  const std::vector<Gate> input{{Kind::H, 0}, Gate::phase({1, 8})};
  unsigned calls = 0;
  auto callback = [&](unsigned width, const std::vector<Gate> &word,
                      const Options &options,
                      cudaq::pauli::SearchBudget *work) {
    if (++calls > 1) {
      EXPECT_FALSE(work->take(std::numeric_limits<uint64_t>::max()));
      return detail::holdFailure("injected optional work exhaustion",
                                 SearchTermination::Exhausted);
    }
    return synthesize(width, word, options, work);
  };
  Options options;
  options.acceptOnlyImprovement = false;
  options.discrepancySteps = 0;
  cudaq::pauli::SearchBudget budget(10000000);
  auto result =
      detail::synthesizeHoldPortfolio(2, input, options, budget, callback);
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(budget.stopped());
  assertUnitary(2, input, result.gates);
}

TEST(GraphSynthesis, HoldCheckpointScalarOnlyAndIdentityRotations) {
  Options options;
  options.discrepancySteps = 0;
  for (const auto &input : std::vector<std::vector<Gate>>{
           {Gate::phase({1, 8})},
           {Gate::rotation({}, {1, 7}), Gate::phase({3, 8})},
           {Gate::rotation({}, {0, 1})}}) {
    auto result = synthesizeUnitary(2, input, options);
    ASSERT_TRUE(result) << result.error;
    assertUnitary(2, input, result.gates);
    for (const auto &gate : result.gates)
      EXPECT_NE(gate.kind, Kind::Rotate);
  }
}

TEST(GraphSynthesis, HoldOneWireCompletionRanksAfterCleanup) {
  // Tdg H T H Tdg H Sdg H is stuck under ordinary native cleanup. Its normal
  // form grows from eight to ten gates before cleanup reduces it to seven.
  const std::vector<Gate> input{Gate::rotation({0, 16}, {-1, 4}),
                                Gate::phase({-1, 8}),
                                {Kind::H, 4},
                                {Kind::H, 1},
                                Gate::rotation({0, 16}, {1, 4}),
                                {Kind::H, 4},
                                Gate::rotation({0, 16}, {-1, 4}),
                                {Kind::H, 4},
                                {Kind::Sdg, 4},
                                Gate::phase({0, 1, 7, true}),
                                {Kind::H, 4}};
  Options options;
  options.algebraicCandidates = 4;
  cudaq::pauli::SearchBudget budget(1000000000);
  auto candidate = detail::normalizeHoldOneQubit(5, input, options, budget);
  ASSERT_TRUE(candidate) << candidate.error;
  assertUnitary(5, input, candidate.gates);
  EXPECT_GT(cudaq::pauli::tcount::cost(5, candidate.gates).gates, 9u);
  ASSERT_TRUE(detail::continueHoldNative(5, candidate, options, budget));
  EXPECT_EQ(cudaq::pauli::tcount::cost(5, candidate.gates).gates, 8u);
  EXPECT_EQ(cudaq::pauli::tcount::cost(5, candidate.gates).t, 3u);
  assertUnitary(5, input, candidate.gates);
}

TEST(GraphSynthesis, HoldOneWireExactPhaseBarriersAndBounds) {
  std::mt19937 rng(0x314354);
  Options options;
  for (unsigned trial = 0; trial < 12; ++trial) {
    std::vector<Gate> input;
    for (unsigned i = 0; i < 80; ++i) {
      unsigned q = rng() % 2 ? 0 : 2;
      if (i % 17 == 0)
        input.push_back({Kind::CX, 0, 2});
      if (i % 19 == 0) {
        input.push_back(Gate::phase({0, 1, 3, true}));
        input.push_back(Gate::rotation({0, 5}, {0, 1, 2}));
      }
      if (rng() % 2)
        input.push_back({static_cast<Kind>(rng() % 6), q});
      else {
        unsigned axis = 1 + rng() % 3;
        input.push_back(Gate::rotation(
            {axis & 1 ? 1u << q : 0, axis & 2 ? 1u << q : 0, bool(rng() % 2)},
            {int(rng() % 31) - 15, 4}));
      }
    }
    cudaq::pauli::SearchBudget budget(1000000000);
    auto candidate = detail::normalizeHoldOneQubit(3, input, options, budget);
    ASSERT_TRUE(candidate) << candidate.error;
    assertUnitary(3, input, candidate.gates);
    auto reversed = detail::normalizeHoldOneQubit(
        3, cudaq::pauli::tcount::detail::adjoint(input), options, budget);
    ASSERT_TRUE(reversed) << reversed.error;
    assertUnitary(3, input,
                  cudaq::pauli::tcount::detail::adjoint(reversed.gates));
    cudaq::pauli::SearchBudget none(0);
    auto exhausted = detail::normalizeHoldOneQubit(3, input, options, none);
    EXPECT_FALSE(exhausted);
    EXPECT_EQ(exhausted.termination, SearchTermination::Exhausted);
    EXPECT_TRUE(exhausted.gates.empty());
    auto small = options;
    small.maxGates = 1;
    auto capped = detail::normalizeHoldOneQubit(3, input, small, budget);
    EXPECT_FALSE(capped);
    EXPECT_EQ(capped.termination, SearchTermination::Unsupported);
  }
  // A long uninterrupted chain must be cut without changing its full unitary.
  std::vector<Gate> longWord;
  for (unsigned i = 0; i < 300; ++i) {
    longWord.push_back({Kind::H, 0});
    longWord.push_back(Gate::rotation({0, 1}, {i % 2 ? 1 : -1, 4}));
  }
  cudaq::pauli::SearchBudget budget(1000000000);
  auto split = detail::normalizeHoldOneQubit(1, longWord, options, budget);
  ASSERT_TRUE(split) << split.error;
  assertUnitary(1, longWord, split.gates);
}

TEST(GraphSynthesis, HoldLargeCompletionSeparatesScratchAndRetainedLimits) {
  const auto phase = Gate::phase({0, 1, 7, true});
  std::vector<Gate> source(40000, {Kind::CX, 0, 64});
  source.insert(source.begin() + 20000, phase);
  Options options;
  options.algebraicCandidates = 4;
  options.maxGates = 262144;
  Result result;
  result.gates = source;
  cudaq::pauli::SearchBudget budget(1000000000);
  ASSERT_TRUE(detail::completeHoldLargeNative(65, result, options, budget));
  EXPECT_EQ(cudaq::pauli::tcount::cost(65, result.gates).gates, 0u);
  EXPECT_EQ(cudaq::pauli::tcount::cost(65, result.gates).t, 0u);
  std::vector<Gate> compactSource, compactResult;
  for (const auto &gate : source)
    compactSource.push_back(
        detail::linear_window_detail::compact(gate, {0, 64}));
  for (const auto &gate : result.gates)
    compactResult.push_back(
        detail::linear_window_detail::compact(gate, {0, 64}));
  assertUnitary(2, compactSource, compactResult);
  cudaq::pauli::SearchBudget none(0);
  result.gates = source;
  ASSERT_TRUE(detail::completeHoldLargeNative(65, result, options, none));
  EXPECT_TRUE(detail::native_refinement_detail::sameWord(source, result.gates));

  // Cleanup may introduce another entangler; lower it before admission.
  for (bool useCX : {false, true}) {
    options.nativeCX = useCX;
    options.nativeCZ = !useCX;
    options.nativeSwap = false;
    std::vector<Gate> word{{Kind::H, 1},
                           {Kind::CZ, 0, 1},
                           {Kind::H, 1},
                           {Kind::Swap, 0, 1},
                           {Kind::Swap, 0, 1}};
    word = detail::nativeWord(word, options);
    result.gates = word;
    cudaq::pauli::SearchBudget nativeBudget(1000000000);
    ASSERT_TRUE(
        detail::completeHoldLargeNative(2, result, options, nativeBudget));
    assertUnitary(2, word, result.gates);
    for (const auto &gate : result.gates) {
      EXPECT_NE(gate.kind, useCX ? Kind::CZ : Kind::CX);
      EXPECT_NE(gate.kind, Kind::Swap);
    }
  }
}

TEST(GraphSynthesis, HoldFineCheckpointsReconstructReverseAndScalarWitnesses) {
  const std::vector<Gate> input{{Kind::H, 4},
                                {Kind::CX, 4, 1},
                                Gate::phase({3, 8}),
                                Gate::rotation({0, 2}, {-1, 4}),
                                Gate::phase({0, 1, 7, true}),
                                {Kind::Swap, 1, 3},
                                Gate::rotation({0b01010, 0b10000}, {1, 4})};
  Options options;
  options.algebraicCandidates = 4;
  options.discrepancySteps = 0;
  auto reconstruct = [](unsigned width, const std::vector<Gate> &word,
                        const Options &settings,
                        cudaq::pauli::SearchBudget *budget) {
    auto local = settings;
    local.holdCheckpoints = false;
    return synthesizeUnitary(width, word, local, budget);
  };
  for (unsigned cap : {1u, 2u})
    for (bool reverse : {false, true}) {
      cudaq::pauli::SearchBudget budget(1000000000);
      auto result = detail::synthesizeHoldFineBlocks(5, input, options, budget,
                                                     cap, reverse, reconstruct);
      ASSERT_TRUE(result) << result.error;
      assertUnitary(5, input, result.gates);
    }
}

TEST(GraphSynthesis, HoldOverlapRejectsInvalidProofAndRetainsExhaustedWord) {
  auto input = holdThreeWireGadget(0, 2, 4);
  Options options;
  options.algebraicCandidates = 4;
  for (auto status :
       {SearchTermination::Exhausted, SearchTermination::InvalidProof}) {
    auto callback = [&](unsigned, const std::vector<Gate> &, const Options &,
                        cudaq::pauli::SearchBudget *) {
      return detail::holdFailure("injected overlap failure", status);
    };
    Result result;
    result.gates = input;
    cudaq::pauli::SearchBudget budget(1000000);
    bool valid =
        detail::refineHoldOverlapping(5, result, options, budget, callback);
    if (status == SearchTermination::InvalidProof) {
      EXPECT_FALSE(valid);
      EXPECT_EQ(result.termination, status);
    } else {
      ASSERT_TRUE(valid);
      assertUnitary(5, input, result.gates);
    }
  }
}

TEST(GraphSynthesis, HoldFineCacheBindsCompleteCompactSource) {
  const std::vector<Gate> input{{Kind::H, 0}, Gate::rotation({0, 1}, {1, 4}),
                                {Kind::H, 4}, Gate::rotation({0, 16}, {1, 4}),
                                {Kind::H, 0}, Gate::rotation({0, 1}, {1, 4}),
                                {Kind::H, 4}, Gate::rotation({0, 16}, {-1, 4})};
  Options options;
  options.discrepancySteps = 0;
  unsigned rebuilt = 0;
  auto reconstruct = [&](unsigned width, const std::vector<Gate> &word,
                         const Options &settings,
                         cudaq::pauli::SearchBudget *budget) {
    ++rebuilt;
    return synthesizeUnitary(width, word, settings, budget);
  };
  cudaq::pauli::SearchBudget budget(100000000);
  auto result = detail::synthesizeHoldFineBlocks(5, input, options, budget, 1,
                                                 false, reconstruct);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(rebuilt, 2u); // Same local word on different wires; distinct angle.
  assertUnitary(5, input, result.gates);
}

TEST(GraphSynthesis, HoldWindowCacheBindsSearchAndScalarWitnesses) {
  detail::HoldReconstructionCache cache;
  unsigned calls = 0;
  detail::HoldSynthesis reconstruct =
      [&](unsigned width, const std::vector<Gate> &word, const Options &options,
          cudaq::pauli::SearchBudget *work) {
        ++calls;
        return synthesizeUnitary(width, word, options, work);
      };
  Options options;
  options.holdCheckpoints = false;
  options.discrepancySteps = 2;
  cudaq::pauli::SearchBudget budget(100000000);
  std::vector<Gate> word{
      {Kind::CX, 0, 1}, {Kind::CX, 0, 1}, Gate::phase({1, 4})};
  auto first = cache.reconstruct(2, word, options, &budget, reconstruct);
  ASSERT_TRUE(first) << first.error;
  auto reused = cache.reconstruct(2, word, options, &budget, reconstruct);
  ASSERT_TRUE(reused) << reused.error;
  EXPECT_EQ(calls, 1u);
  assertUnitary(2, word, reused.gates);
  options.discrepancySteps = 0;
  auto otherSearch = cache.reconstruct(2, word, options, &budget, reconstruct);
  ASSERT_TRUE(otherSearch) << otherSearch.error;
  EXPECT_EQ(calls, 2u);
  word.back() = Gate::phase({-1, 4});
  auto otherPhase = cache.reconstruct(2, word, options, &budget, reconstruct);
  ASSERT_TRUE(otherPhase) << otherPhase.error;
  EXPECT_EQ(calls, 3u);
  assertUnitary(2, word, otherPhase.gates);
}

TEST(GraphSynthesis, HoldCheckpointCacheRetainsReconstruction) {
  std::vector<Gate> input;
  for (unsigned i = 0; i < 64; ++i) {
    input.push_back({Kind::H, 0});
    input.push_back({Kind::CX, 0, 1});
    input.push_back(Gate::rotation({0, 2}, {1, 4}));
    input.push_back({Kind::Sdg, 1});
  }
  Options options;
  options.algebraicCandidates = 0;
  options.discrepancySteps = 0;
  unsigned calls = 0;
  auto reconstruct = [&](unsigned width, const std::vector<Gate> &word,
                         const Options &settings,
                         cudaq::pauli::SearchBudget *work) {
    ++calls;
    return synthesizeUnitary(width, word, settings, work);
  };
  cudaq::pauli::SearchBudget directBudget(100000000), cachedBudget(100000000);
  auto direct = detail::synthesizeHoldBlocks(2, input, options, directBudget, 3,
                                             reconstruct);
  ASSERT_TRUE(direct) << direct.error;
  EXPECT_EQ(calls, 4u);
  calls = 0;
  auto cached = detail::synthesizeHoldCachedBlocks(
      2, input, options, cachedBudget, 3, reconstruct);
  ASSERT_TRUE(cached) << cached.error;
  EXPECT_EQ(calls, 1u);
  EXPECT_TRUE(
      detail::native_refinement_detail::sameWord(direct.gates, cached.gates));
  assertUnitary(2, input, cached.gates);
}

TEST(GraphSynthesis, HoldWideReverseWindowsRetainTAndScalarWitnesses) {
  auto t = [](unsigned q, int sign) {
    return Gate::rotation({0, uint64_t(1) << q}, {sign, 4});
  };
  const std::vector<Gate> input{t(2, -1),
                                {Kind::CZ, 1, 4},
                                {Kind::CX, 1, 2},
                                {Kind::CZ, 1, 5},
                                {Kind::CX, 1, 5},
                                {Kind::H, 1},
                                {Kind::CX, 1, 2},
                                {Kind::CX, 4, 2},
                                t(1, -1),
                                {Kind::H, 2},
                                {Kind::CX, 2, 5},
                                {Kind::CX, 0, 1},
                                t(1, 1),
                                {Kind::CX, 4, 1},
                                t(1, -1),
                                {Kind::CX, 0, 1},
                                t(1, 1),
                                {Kind::CZ, 0, 3},
                                {Kind::CZ, 0, 4},
                                {Kind::CX, 0, 4},
                                {Kind::H, 0},
                                t(0, -1),
                                {Kind::CX, 0, 1},
                                {Kind::CX, 3, 1},
                                {Kind::H, 1},
                                {Kind::CX, 1, 4},
                                Gate::phase({-1, 4}),
                                Gate::phase({0, 1, 19})};
  Options options;
  options.algebraicCandidates = 4;
  options.candidateNodes = 1;
  options.lookahead = 16;
  auto reconstruct = [](unsigned width, const std::vector<Gate> &word,
                        const Options &settings,
                        cudaq::pauli::SearchBudget *work) {
    return synthesizeUnitary(width, word, settings, work);
  };
  Result result;
  result.gates = input;
  cudaq::pauli::SearchBudget budget(1000000000);
  ASSERT_TRUE(
      detail::refineHoldWideReverse(6, result, options, budget, reconstruct));
  ASSERT_TRUE(result) << result.error;
  auto before = cudaq::pauli::tcount::cost(6, input);
  auto after = cudaq::pauli::tcount::cost(6, result.gates);
  EXPECT_LT(after.gates, before.gates);
  EXPECT_LE(after.t, before.t);
  assertUnitary(6, input, result.gates);
}

TEST(GraphSynthesis, HoldTRepairExploresInteriorCompletions) {
  auto t = [](int sign) { return Gate::rotation({0, 2}, {sign, 4}); };
  const std::vector<Gate> input{{Kind::CX, 1, 2},
                                {Kind::CX, 1, 0},
                                {Kind::H, 1},
                                {Kind::CX, 2, 1},
                                t(-1),
                                {Kind::CX, 0, 1},
                                t(1),
                                {Kind::CX, 2, 1},
                                t(-1),
                                {Kind::CX, 0, 1},
                                t(1),
                                {Kind::H, 1},
                                {Kind::CX, 1, 3},
                                {Kind::H, 1},
                                {Kind::CX, 2, 1},
                                t(-1),
                                {Kind::CX, 0, 1},
                                t(1),
                                {Kind::CX, 2, 1},
                                t(-1),
                                {Kind::CX, 0, 1},
                                t(1),
                                {Kind::CZ, 0, 2},
                                {Kind::H, 1},
                                {Kind::CX, 1, 0},
                                {Kind::CX, 0, 2},
                                {Kind::H, 0},
                                Gate::phase({1, 8}),
                                Gate::phase({0, 1, 19})};
  Options options;
  options.algebraicCandidates = 4;
  options.candidateNodes = 1;
  options.lookahead = 16;
  auto reconstruct = [](unsigned width, const std::vector<Gate> &word,
                        const Options &settings,
                        cudaq::pauli::SearchBudget *work) {
    EXPECT_EQ(settings.objective, SynthesisObjective::TCount);
    EXPECT_EQ(settings.candidateNodes, 0u);
    return synthesizeUnitary(width, word, settings, work);
  };
  Result result;
  result.gates = input;
  cudaq::pauli::SearchBudget budget(1000000000);
  ASSERT_TRUE(detail::repairHoldT(4, result, 7, options, budget, reconstruct));
  ASSERT_TRUE(result) << result.error;
  const auto before = cudaq::pauli::tcount::cost(4, input);
  const auto after = cudaq::pauli::tcount::cost(4, result.gates);
  EXPECT_EQ(after.t, 7u);
  EXPECT_LE(after.gates, before.gates + 3);
  assertUnitary(4, input, result.gates);

  Result exhausted;
  exhausted.gates = input;
  cudaq::pauli::SearchBudget noWork(0);
  EXPECT_TRUE(
      detail::repairHoldT(4, exhausted, 7, options, noWork, reconstruct));
  EXPECT_TRUE(
      detail::native_refinement_detail::sameWord(input, exhausted.gates));
  Result invalid;
  invalid.gates = input;
  cudaq::pauli::SearchBudget proofWork(1000000);
  auto reject = [](unsigned, const std::vector<Gate> &, const Options &,
                   cudaq::pauli::SearchBudget *) {
    return detail::holdFailure("injected T-repair proof failure",
                               SearchTermination::InvalidProof);
  };
  EXPECT_FALSE(detail::repairHoldT(4, invalid, 7, options, proofWork, reject));
  EXPECT_EQ(invalid.termination, SearchTermination::InvalidProof);
  EXPECT_TRUE(invalid.gates.empty());
}

TEST(GraphSynthesis, HoldTotalGatesRetainsIncumbentTSavings) {
  // Eight quarter turns give -I. Both proposed circuits implement that same
  // full unitary: a long Clifford word versus two inverse T rotations.
  std::vector<Gate> input(8, Gate::rotation({0, 1}, {1, 4}));
  for (auto objective :
       {SynthesisObjective::TotalGates, SynthesisObjective::TCount}) {
    unsigned calls = 0;
    auto callback = [&](unsigned, const std::vector<Gate> &, const Options &,
                        cudaq::pauli::SearchBudget *) {
      Result result;
      result.gates.push_back(Gate::phase({1, 1}));
      if (++calls == 1)
        result.gates.insert(result.gates.end(), 60, {Kind::H, 0});
      else {
        result.gates.push_back(Gate::rotation({0, 1}, {1, 4}));
        result.gates.push_back(Gate::rotation({0, 1}, {-1, 4}));
      }
      return result;
    };
    Options options;
    options.objective = objective;
    options.discrepancySteps = 0;
    cudaq::pauli::SearchBudget budget(1000000);
    auto result =
        detail::synthesizeHoldPortfolio(1, input, options, budget, callback);
    ASSERT_TRUE(result) << result.error;
    assertUnitary(1, input, result.gates);
    auto cost = cudaq::pauli::tcount::cost(1, result.gates);
    EXPECT_EQ(cost.gates, 60u);
    EXPECT_EQ(cost.t, 0u);
    EXPECT_LE(cost.t, cudaq::pauli::tcount::cost(1, input).t);
  }
}

TEST(GraphSynthesis, HoldCheckpointInvalidProofIsFatalAfterAnIncumbentExists) {
  const std::vector<Gate> input{{Kind::H, 0}};
  unsigned calls = 0;
  auto callback = [&](unsigned width, const std::vector<Gate> &word,
                      const Options &options,
                      cudaq::pauli::SearchBudget *work) {
    if (++calls > 1) {
      EXPECT_FALSE(work->take(std::numeric_limits<uint64_t>::max()));
      return detail::holdFailure("injected invalid unitary certificate",
                                 SearchTermination::InvalidProof);
    }
    return synthesize(width, word, options, work);
  };
  Options options;
  options.acceptOnlyImprovement = false;
  options.discrepancySteps = 0;
  cudaq::pauli::SearchBudget budget(10000000);
  auto result =
      detail::synthesizeHoldPortfolio(2, input, options, budget, callback);
  EXPECT_FALSE(result);
  EXPECT_EQ(result.termination, SearchTermination::InvalidProof);
  EXPECT_TRUE(result.gates.empty());
}

TEST(GraphSynthesis, HoldRejectsReleaseFreedomAndUnfinishedProofs) {
  std::vector<Gate> input{Gate::rotation({0, 1}, {1, 4})};
  Options options;
  options.pruneUnobserved = true;
  EXPECT_FALSE(synthesizeUnitary(1, input, options));
  options.pruneUnobserved = false;
  options.release = true;
  EXPECT_FALSE(synthesizeUnitary(1, input, options));
  options.release = false;
  cudaq::pauli::SearchBudget exhausted(1);
  EXPECT_FALSE(synthesizeUnitary(1, input, options, &exhausted));
  auto hold = synthesizeUnitary(1, input, options);
  ASSERT_TRUE(hold) << hold.error;
  assertUnitary(1, input, hold.gates);
  options.pruneUnobserved = true;
  auto release = synthesizeMeasured(1, input, {{0, 1}}, {0}, options);
  ASSERT_TRUE(release) << release.error;
  EXPECT_TRUE(release.rotations.empty());
}
TEST(GraphSynthesis, RandomUnitaryAndAllSearchControls) {
  std::mt19937 rng(0x50434f41);
  for (unsigned n = 1; n <= 4; ++n)
    for (unsigned test = 0; test < 24; ++test) {
      std::vector<Gate> gates;
      for (unsigned k = 0; k < 32; ++k) {
        unsigned a = rng() % n, b = (a + 1 + rng() % n) % n;
        if (a == b)
          b = (a + 1) % n;
        switch (rng() % 6) {
        case 0:
          gates.push_back({Kind::H, a});
          break;
        case 1:
          gates.push_back({Kind::S, a});
          break;
        case 2:
          gates.push_back({Kind::Y, a});
          break;
        case 3:
          gates.push_back({n > 1 ? Kind::CX : Kind::Sdg, a, b});
          break;
        case 4:
          gates.push_back({n > 1 ? Kind::CZ : Kind::Z, a, b});
          break;
        default:
          gates.push_back(
              Gate::rotation({uint64_t(1) << a, uint64_t(1) << a},
                             k % 2 ? Angle{1, 8} : Angle{0, 1, k + 1}));
        }
      }
      Options options;
      options.acceptOnlyImprovement = false;
      options.asymmetric = test % 2;
      options.simplify = test % 3;
      options.parallelizationCredit = (test % 3) * 0.7;
      options.frontierWeight = test % 4;
      options.frameWeight = (test % 5) * 0.2;
      options.lookahead = test % 2 ? 1 : 0;
      options.candidateNodes = test % 2 ? 1 : 0;
      auto result = synthesize(n, gates, options);
      ASSERT_TRUE(result) << "n=" << n << " test=" << test << " "
                          << result.error;
      assertUnitary(n, gates, result.gates);
    }
}
TEST(GraphSynthesis, CommutingMergeAndEntanglerReduction) {
  std::vector<Gate> input{
      {Kind::CX, 0, 1}, Gate::rotation({0, 2}, {1, 8}), {Kind::CX, 0, 1},
      {Kind::CX, 0, 1}, Gate::rotation({0, 2}, {1, 8}), {Kind::CX, 0, 1}};
  auto result = synthesize(2, input, {});
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.stats.rotationsAfter, 1u);
  EXPECT_LT(result.stats.tqes, 4u);
  assertUnitary(2, input, result.gates);
}
TEST(GraphSynthesis, ReleasePreservesJointDistribution) {
  std::vector<Gate> input{{Kind::H, 0},
                          {Kind::CX, 0, 1},
                          Gate::rotation({0, 3}, {1, 8}),
                          Gate::rotation({1, 0}, {1, 4}),
                          Gate::rotation({0, 2}, {1, 4})};
  Options options;
  options.release = true;
  auto result = synthesize(2, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_GT(result.stats.released, 0u);
  // Arbitrary complex input, not just the allocated zero state.
  State state{{0.1, 0.3}, {0.2, -0.1}, {-0.4, 0.2}, {0.6, 0.7}};
  auto lhs = simulate(input, state), rhs = simulate(result.gates, state);
  for (unsigned i = 0; i < 4; ++i)
    EXPECT_NEAR(std::norm(lhs[i]), std::norm(rhs[i]), 1e-12);
}
TEST(GraphSynthesis, BudgetFailsWithoutReturningAnAcceptedCandidate) {
  std::vector<Gate> input{Gate::rotation({0, 3}, {1, 8})};
  Options options;
  options.maxSteps = 0;
  auto result = synthesize(2, input, options);
  EXPECT_FALSE(result);
  EXPECT_NE(result.error.find("step limit"), std::string::npos);
  options.maxSteps = 1000;
  options.frameWeight = -1;
  EXPECT_FALSE(synthesize(2, input, options));
}

TEST(GraphSynthesis, RejectMalformedCertificates) {
  std::vector<Gate> identity;
  EXPECT_FALSE(equivalent(1, identity, {{Kind::H, 64}}));
  EXPECT_FALSE(equivalent(1, identity, {{Kind::Swap, 0, 0}}));
  EXPECT_FALSE(equivalent(1, identity, {Gate::phase({1, 0})}));
  EXPECT_FALSE(equivalent(1, identity, {{static_cast<Kind>(255), 0}}));
}

TEST(GraphSynthesis, AcceptanceNeverIncreasesEntanglerCost) {
  std::vector<Gate> input{{Kind::CX, 0, 1}};
  auto result = synthesize(2, input, {});
  ASSERT_TRUE(result) << result.error;
  EXPECT_FALSE(result.stats.accepted);
  ASSERT_EQ(result.gates.size(), 1u);
  EXPECT_EQ(result.gates.front().kind, Kind::CX);
}

TEST(GraphSynthesis, RotationWitnessRejectsAnticommutingReorder) {
  auto x = Gate::rotation({1, 0}, {0, 1, 1});
  auto z = Gate::rotation({0, 1}, {0, 1, 2});
  auto zOther = Gate::rotation({0, 2}, {0, 1, 2});
  EXPECT_FALSE(equivalent(1, {x, z}, {z, x}));
  EXPECT_TRUE(equivalent(2, {x, zOther}, {zOther, x}));
  EXPECT_FALSE(equivalent(1, {x}, {Gate::rotation({1, 0}, {0, 1, 1, true})}));
}

TEST(GraphSynthesis, HighestSupportedQubit) {
  std::vector<Gate> input{{Kind::H, 63},
                          {Kind::CX, 63, 0},
                          Gate::rotation({0, 1}, {0, 1, 1}),
                          {Kind::CX, 63, 0},
                          {Kind::H, 63}};
  Options options;
  options.acceptOnlyImprovement = false;
  auto result = synthesize(64, input, options);
  ASSERT_TRUE(result) << result.error;
  auto compress = [](std::vector<Gate> gates) {
    for (auto &g : gates) {
      if (g.a == 63)
        g.a = 1;
      if (g.b == 63)
        g.b = 1;
      g.pauli.x =
          uint64_t(g.pauli.x.test(0)) | (uint64_t(g.pauli.x.test(63)) << 1);
      g.pauli.z =
          uint64_t(g.pauli.z.test(0)) | (uint64_t(g.pauli.z.test(63)) << 1);
    }
    return gates;
  };
  assertUnitary(2, compress(input), compress(result.gates));
  EXPECT_TRUE(synthesize(65, input, options));
}

TEST(GraphSynthesis, DynamicWidthAndSignedReplay) {
  for (unsigned width : {63u, 64u, 65u, 118u, 128u, 433u}) {
    unsigned q = width - 1;
    std::vector<Gate> input{{Kind::H, q},     {Kind::S, q},
                            {Kind::CX, q, 0}, Gate::rotation({0, 1}, {0, 1, 1}),
                            {Kind::CX, q, 0}, {Kind::Sdg, q},
                            {Kind::H, q}};
    Options options;
    options.acceptOnlyImprovement = false;
    auto result = synthesize(width, input, options);
    ASSERT_TRUE(result) << width << ": " << result.error;
    auto compress = [&](std::vector<Gate> gates) {
      for (auto &g : gates) {
        if (g.a == q)
          g.a = 1;
        if (g.b == q)
          g.b = 1;
        g.pauli.x =
            uint64_t(g.pauli.x.test(0)) | (uint64_t(g.pauli.x.test(q)) << 1);
        g.pauli.z =
            uint64_t(g.pauli.z.test(0)) | (uint64_t(g.pauli.z.test(q)) << 1);
      }
      return gates;
    };
    assertUnitary(2, compress(input), compress(result.gates));
    EXPECT_FALSE(equivalent(width - 1, input, result.gates));
  }
}

TEST(GraphSynthesis, WideWorkEstimateCannotWrapOrAllocateBeforeRejection) {
  // Four width-squared charges used to wrap to four at this width. The bounded
  // request must fail before allocating a tableau, not run out of memory.
  constexpr unsigned width = 1u << 31;
  cudaq::pauli::SearchBudget budget(100);
  std::vector<Gate> input(4, Gate{Kind::H, 0});
  auto result = synthesize(width, input, {}, &budget);
  EXPECT_FALSE(result);
  EXPECT_TRUE(result.stats.budgetExhausted);
  EXPECT_TRUE(budget.stopped());
  EXPECT_EQ(budget.work(), 0u);
  EXPECT_TRUE(synthesize(width, {}, {}));
}

TEST(GraphSynthesis, NativeReadoutPreservesPhysicalOutcomeOrder) {
  std::vector<Pauli> words{{0, 2}, {0, 1}};
  Options options;
  auto native = synthesizeReadout(2, words, {0, 1}, options);
  ASSERT_TRUE(native) << native.error;
  ASSERT_EQ(native.gates.size(), 1u);
  EXPECT_EQ(native.gates.front().kind, Kind::Swap);
  EXPECT_EQ(nativeGateCost(native.gates, options), 1);
  for (bool cx : {false, true}) {
    options.nativeCX = cx;
    options.nativeCZ = !cx;
    options.nativeSwap = false;
    auto decomposed = synthesizeReadout(2, words, {0, 1}, options);
    ASSERT_TRUE(decomposed) << decomposed.error;
    assertUnitary(2, native.gates, decomposed.gates);
    for (auto g : decomposed.gates) {
      EXPECT_NE(g.kind, Kind::Swap);
      EXPECT_NE(g.kind, cx ? Kind::CZ : Kind::CX);
    }
    EXPECT_EQ(nativeGateCost(native.gates, options), cx ? 3 : 9);
  }
}

TEST(GraphSynthesis, MeasuredSynthesisPreservesAllJointOutcomeEffects) {
  // Carrying X/Y basis changes and reconstructing a physical permutation must
  // preserve off-diagonal POVM entries as well as computational-basis inputs.
  std::vector<Gate> input{{Kind::H, 0},
                          {Kind::CX, 0, 1},
                          Gate::rotation({2, 2}, {1, 4}),
                          {Kind::Sdg, 0},
                          {Kind::Swap, 0, 1}};
  std::vector<Pauli> outcomes{{0, 1}, {0, 2}};
  auto normalized = cudaq::pauli::pbc::normalize(2, input, outcomes);
  ASSERT_TRUE(normalized) << normalized.error;
  for (unsigned profile = 0; profile < 3; ++profile) {
    Options options;
    options.simplify = false;
    options.discrepancySteps = 3;
    options.nativeCX = profile != 2;
    options.nativeCZ = profile != 1;
    options.nativeSwap = profile == 0;
    cudaq::pauli::SearchBudget budget(10000000);
    auto result =
        synthesizeMeasured(2, normalized.rotations, normalized.measurements,
                           {0, 1}, options, &budget);
    ASSERT_TRUE(result) << result.error;
    EXPECT_EQ(result.measurements, outcomes);
    std::vector<State> before, after;
    for (unsigned col = 0; col < 4; ++col) {
      State state(4);
      state[col] = 1;
      before.push_back(simulate(input, state));
      after.push_back(simulate(result.rotations, state));
    }
    for (unsigned outcome = 0; outcome < 4; ++outcome)
      for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = 0; j < 4; ++j)
          EXPECT_LT(
              std::abs(std::conj(before[i][outcome]) * before[j][outcome] -
                       std::conj(after[i][outcome]) * after[j][outcome]),
              1e-11);
  }
}

TEST(GraphSynthesis, SignedFramesAllCliffordContexts) {
  // Independent statevector channel oracle: test both T signs in every pair
  // of Clifford contexts, including scalar phases omitted by the local DP.
  namespace local = cudaq::pauli::graph::detail;
  local::LocalCliffords group;
  for (const auto &left : group.words)
    for (const auto &right : group.words)
      for (int sign : {-1, 1}) {
        std::vector<Gate> word;
        for (Kind k : left)
          word.push_back({k, 0});
        word.push_back(Gate::rotation({0, 1}, {sign, 4}));
        for (Kind k : right)
          word.push_back({k, 0});
        std::string error;
        auto result = local::signedFrameWord(word, 0, 7, nullptr, error);
        ASSERT_TRUE(result) << error;
        EXPECT_LE(result->size(), word.size());
        std::array<State, 2> before, after;
        for (unsigned col = 0; col < 2; ++col) {
          State state(2);
          state[col] = 1;
          before[col] = simulate(word, state);
          after[col] = simulate(*result, state);
        }
        for (unsigned i = 0; i < 2; ++i)
          for (unsigned j = 0; j < 2; ++j)
            for (unsigned r = 0; r < 2; ++r)
              for (unsigned s = 0; s < 2; ++s)
                ASSERT_LT(std::abs(before[i][r] * std::conj(before[j][s]) -
                                   after[i][r] * std::conj(after[j][s])),
                          1e-11);
      }
}

TEST(GraphSynthesis, SignedFrameProofAndTerminalFreedom) {
  namespace local = cudaq::pauli::graph::detail;
  Gate t = Gate::rotation({0, 1}, {1, 4});
  Gate tdg = Gate::rotation({0, 1}, {-1, 4});
  std::string error;
  auto result = local::signedFrameWord({{Kind::H, 0}, t, {Kind::Sdg, 0}}, 0, 7,
                                       nullptr, error);
  ASSERT_TRUE(result) << error;
  EXPECT_EQ(result->size(), 2u);
  EXPECT_FALSE(local::verifyLocalFrames({t}, {tdg}, 7));
  EXPECT_TRUE(local::verifyLocalFrames({t}, {tdg}, 4));
  EXPECT_FALSE(local::verifyLocalFrames({t}, {t, {Kind::X, 0}}, 4));
  EXPECT_FALSE(local::verifyLocalFrames({t}, {}, 4)); // T-count guard
  auto measured = local::signedFrameWord({{Kind::H, 0}, {Kind::S, 0}}, 0, 4,
                                         nullptr, error);
  auto discarded = local::signedFrameWord({{Kind::H, 0}, {Kind::S, 0}}, 0, 0,
                                          nullptr, error);
  ASSERT_TRUE(measured);
  ASSERT_TRUE(discarded);
  EXPECT_EQ(measured->size(), 1u);
  EXPECT_TRUE(discarded->empty());
}

TEST(GraphSynthesis, SignedFrameBoundariesAndBudget) {
  namespace local = cudaq::pauli::graph::detail;
  using cudaq::pauli::QubitSet;
  unsigned q = 64;
  Gate t = Gate::rotation({0, QubitSet::single(q)}, {1, 4});
  Gate opaque = Gate::rotation({0, QubitSet::single(q)}, {0, 1, 7});
  std::vector<Gate> word{{Kind::H, q},     t,      {Kind::Sdg, q},
                         {Kind::CX, q, 0}, opaque, {Kind::H, 0}};
  std::string error;
  cudaq::pauli::SearchBudget empty(0);
  EXPECT_FALSE(local::optimizeSignedFrames(65, word, {0}, &empty, error));
  EXPECT_TRUE(empty.stopped());
  EXPECT_TRUE(error.empty());
  auto result = local::optimizeSignedFrames(65, word, {0}, nullptr, error);
  ASSERT_TRUE(result) << error;
  ASSERT_EQ(result->size(), 5u);
  EXPECT_EQ((*result)[2].kind, Kind::CX);
  EXPECT_EQ((*result)[2].a, q);
  EXPECT_EQ((*result)[3].angle, opaque.angle);
  // Exercise the bounded-region cut with arbitrary input coherences.
  std::vector<Gate> longWord;
  for (unsigned i = 0; i < 200; ++i) {
    longWord.push_back({Kind::H, 0});
    longWord.push_back(Gate::rotation({0, 1}, {i % 2 ? -1 : 1, 4}));
    longWord.push_back({Kind::Sdg, 0});
  }
  auto shortened =
      local::optimizeSignedFrames(1, longWord, {0}, nullptr, error);
  ASSERT_TRUE(shortened) << error;
  std::array<State, 2> before, after;
  for (unsigned i = 0; i < 2; ++i) {
    State state(2);
    state[i] = 1;
    before[i] = simulate(longWord, state);
    after[i] = simulate(*shortened, state);
  }
  for (unsigned outcome = 0; outcome < 2; ++outcome)
    for (unsigned i = 0; i < 2; ++i)
      for (unsigned j = 0; j < 2; ++j)
        EXPECT_LT(std::abs(std::conj(before[i][outcome]) * before[j][outcome] -
                           std::conj(after[i][outcome]) * after[j][outcome]),
                  1e-11);
}

TEST(GraphSynthesis, SignedFramesRandomPartialMeasurementEffects) {
  namespace local = cudaq::pauli::graph::detail;
  std::mt19937 rng(0x5349474e);
  for (unsigned trial = 0; trial < 32; ++trial) {
    std::vector<Gate> input;
    for (unsigned i = 0; i < 60; ++i) {
      unsigned q = rng() % 2;
      switch (rng() % 7) {
      case 0:
        input.push_back({Kind::H, q});
        break;
      case 1:
        input.push_back({Kind::S, q});
        break;
      case 2:
        input.push_back({Kind::Sdg, q});
        break;
      case 3:
        input.push_back({Kind::CX, q, 1 - q});
        break;
      case 4:
        input.push_back({Kind::CZ, q, 1 - q});
        break;
      default:
        input.push_back(
            Gate::rotation({0, uint64_t(1) << q}, {rng() % 2 ? -1 : 1, 4}));
        break;
      }
    }
    for (unsigned mask : {1u, 2u, 3u}) {
      std::vector<unsigned> wires;
      for (unsigned q = 0; q < 2; ++q)
        if (mask & (1u << q))
          wires.push_back(q);
      std::string error;
      auto result =
          local::optimizeSignedFrames(2, input, wires, nullptr, error);
      ASSERT_TRUE(result) << error;
      EXPECT_LE(result->size(), input.size());
      std::array<State, 4> before, after;
      for (unsigned i = 0; i < 4; ++i) {
        State state(4);
        state[i] = 1;
        before[i] = simulate(input, state);
        after[i] = simulate(*result, state);
      }
      for (unsigned outcome = 0; outcome < 4; ++outcome)
        for (unsigned i = 0; i < 4; ++i)
          for (unsigned j = 0; j < 4; ++j) {
            Complex lhs = 0, rhs = 0;
            for (unsigned r = 0; r < 4; ++r)
              if ((r & mask) == outcome) {
                lhs += std::conj(before[i][r]) * before[j][r];
                rhs += std::conj(after[i][r]) * after[j][r];
              }
            EXPECT_LT(std::abs(lhs - rhs), 1e-11);
          }
    }
  }
}

TEST(GraphSynthesis,
     MeasuredSynthesisRejectsInvalidProfilesAndInsufficientWork) {
  Options options;
  options.nativeCX = options.nativeCZ = false;
  EXPECT_FALSE(synthesizeMeasured(1, {}, {{0, 1}}, {0}, options));
  options = {};
  options.actionWeight = -1;
  EXPECT_FALSE(synthesizeMeasured(1, {}, {{0, 1}}, {0}, options));
  options = {};
  cudaq::pauli::SearchBudget budget(0);
  EXPECT_FALSE(
      synthesizeMeasured(1000000000, {}, {{0, 1}}, {0}, options, &budget));
  EXPECT_TRUE(budget.stopped());
}

TEST(GraphSynthesis, MeasuredSearchRetainsVerifiedIncumbentAtWorkLimit) {
  std::vector<Gate> rotations{Gate::rotation({0, 3}, {1, 4})};
  std::vector<Pauli> measured{{0, 1}, {0, 2}};
  Options options;
  options.discrepancySteps = 0;
  cudaq::pauli::SearchBudget first(10000000);
  auto greedy =
      synthesizeMeasured(2, rotations, measured, {0, 1}, options, &first);
  ASSERT_TRUE(greedy) << greedy.error;
  options.discrepancySteps = 22;
  cudaq::pauli::SearchBudget limited(first.work());
  auto result =
      synthesizeMeasured(2, rotations, measured, {0, 1}, options, &limited);
  ASSERT_TRUE(result) << result.error;
  EXPECT_TRUE(limited.stopped());
  assertUnitary(2, greedy.rotations, result.rotations);
}

TEST(GraphSynthesis, NativeReadoutUsesDynamicSupportStorage) {
  using cudaq::pauli::QubitSet;
  for (unsigned last : {64u, 128u}) {
    std::vector<Pauli> words{{0, QubitSet::single(last)}, {0, 1}};
    auto result = synthesizeReadout(last + 1, words, {0, last}, {});
    ASSERT_TRUE(result) << result.error;
    ASSERT_EQ(result.gates.size(), 1u);
    EXPECT_EQ(result.gates.front().kind, Kind::Swap);
    EXPECT_EQ(result.gates.front().a, last);
    EXPECT_EQ(result.gates.front().b, 0u);
    EXPECT_EQ(nativeGateCost(
                  {Gate::rotation({QubitSet::single(last), 0}, {1, 4})}, {}),
              3);
  }
}
