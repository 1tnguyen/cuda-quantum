/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisCandidates.h"
#include "gtest/gtest.h"

using namespace cudaq::pauli;
namespace synthesis = cudaq::pauli::graph;
namespace local = cudaq::pauli::graph::detail;

TEST(SynthesisCandidates, ChecksSignedOutcomesAndRetainsIncumbent) {
  const std::vector<Gate> rotations;
  const std::vector<Pauli> measured{{0, 1, true}};
  const std::vector<unsigned> wires{0};
  synthesis::Options options;
  local::MeasuredCandidateSink sink({1, rotations, measured, wires}, options,
                                    nullptr);
  EXPECT_EQ(sink.submit({}).status, local::SubmissionStatus::InvalidProof);
  EXPECT_FALSE(sink.takeBest());
  EXPECT_EQ(sink.submit({{Kind::X, 0}}).status,
            local::SubmissionStatus::Verified);
  // A failed proposal must not turn into a usable replacement.
  EXPECT_EQ(sink.submit({{Kind::H, 0}}).status,
            local::SubmissionStatus::InvalidProof);
  auto best = sink.takeBest();
  ASSERT_TRUE(best);
  ASSERT_EQ(best->gates().size(), 1u);
  EXPECT_EQ(best->gates().front().kind, Kind::X);
  EXPECT_FALSE(sink.takeBest());
}

TEST(SynthesisCandidates, PhysicalReadoutOrderIsPartOfContract) {
  const std::vector<Gate> rotations;
  const std::vector<Pauli> measured{{0, 2}, {0, 1}};
  const std::vector<unsigned> wires{0, 1};
  synthesis::Options options;
  local::MeasuredCandidateSink sink({2, rotations, measured, wires}, options,
                                    nullptr);
  EXPECT_EQ(sink.submit({}).status, local::SubmissionStatus::InvalidProof);
  EXPECT_EQ(sink.submit({{Kind::Swap, 0, 1}}).status,
            local::SubmissionStatus::Verified);
  ASSERT_TRUE(sink.takeBest());
}

TEST(SynthesisCandidates, ExactWordReplayStillRequiresTheTerminalFrame) {
  std::vector<Gate> rotations, native;
  for (unsigned i = 0; i < 128; ++i) {
    rotations.push_back(Gate::rotation({1, 0}, {1, 4}));
    rotations.push_back(Gate::rotation({0, 1}, {1, 4}));
    native.push_back({Kind::H, 0});
    native.push_back(Gate::rotation({0, 1}, {1, 4}));
    native.push_back({Kind::H, 0});
    native.push_back(Gate::rotation({0, 1}, {1, 4}));
  }
  const std::vector<Pauli> measured{{0, 1}};
  const std::vector<unsigned> wires{0};
  synthesis::Options options;
  SearchBudget goodBudget(100000000);
  local::MeasuredCandidateSink good({1, rotations, measured, wires}, options,
                                    &goodBudget);
  EXPECT_EQ(good.submit(native).status, local::SubmissionStatus::Verified);
  EXPECT_TRUE(good.takeBest());
  // The rotation word is unchanged, but this trailing Clifford changes the
  // measured observable. Equality of rotation words alone is insufficient.
  native.push_back({Kind::H, 0});
  SearchBudget badBudget(100000000);
  local::MeasuredCandidateSink bad({1, rotations, measured, wires}, options,
                                   &badBudget);
  EXPECT_EQ(bad.submit(native).status, local::SubmissionStatus::InvalidProof);
  EXPECT_FALSE(bad.takeBest());
}

TEST(SynthesisCandidates, ProofExhaustionCannotPublishPartialCandidate) {
  const std::vector<Gate> rotations;
  const std::vector<Pauli> measured{{0, 1}};
  const std::vector<unsigned> wires{0};
  synthesis::Options options;
  SearchBudget budget(0);
  local::MeasuredCandidateSink sink({1, rotations, measured, wires}, options,
                                    &budget);
  EXPECT_EQ(sink.submit({}).status,
            local::SubmissionStatus::ProofBudgetExhausted);
  EXPECT_FALSE(sink.takeBest());
}

TEST(SynthesisCandidates, CheckedRefinementComposesWithWordReplay) {
  const std::vector<Gate> rotations{Gate::rotation({0, 1}, {1, 4})};
  const std::vector<Pauli> measured{{1, 0}};
  const std::vector<unsigned> wires{0};
  synthesis::Options options;
  local::MeasuredCandidateSink sink({1, rotations, measured, wires}, options,
                                    nullptr);
  auto network = rotations;
  network.push_back({Kind::H, 0});
  EXPECT_EQ(sink.submit(network).status, local::SubmissionStatus::Verified);
  auto best = sink.takeBest();
  ASSERT_TRUE(best);
  EXPECT_EQ(best->proof(),
            local::VerifiedCandidate::Proof::MeasuredReplayAndLocalChannels);
}

TEST(SynthesisCandidates, RejectsInvalidNativeOperandsBeforeReplay) {
  const std::vector<Gate> rotations;
  const std::vector<Pauli> measured{{0, 1}};
  const std::vector<unsigned> wires{0};
  synthesis::Options options;
  local::MeasuredCandidateSink sink({1, rotations, measured, wires}, options,
                                    nullptr);
  EXPECT_EQ(sink.submit({{Kind::CX, 0, 0}}).status,
            local::SubmissionStatus::InvalidProof);
  EXPECT_EQ(sink.submit({{Kind::H, 1}}).status,
            local::SubmissionStatus::InvalidProof);
  EXPECT_FALSE(sink.takeBest());
}

TEST(SynthesisCandidates, RejectsOverLimitCandidateWithoutReplacingIncumbent) {
  const std::vector<Gate> rotations;
  const std::vector<Pauli> measured{{0, 1}};
  const std::vector<unsigned> wires{0};
  synthesis::Options options;
  options.maxGates = 1;
  local::MeasuredCandidateSink sink({1, rotations, measured, wires}, options,
                                    nullptr);
  ASSERT_TRUE(sink.error().empty()) << sink.error();
  ASSERT_EQ(sink.submit({}).status, local::SubmissionStatus::Verified);
  // This is an equivalent identity circuit. The candidate gate cap still
  // applies before any local refinement could erase its redundant gates.
  EXPECT_EQ(sink.submit({{Kind::H, 0}, {Kind::H, 0}}).status,
            local::SubmissionStatus::InvalidProof);
  auto best = sink.takeBest();
  ASSERT_TRUE(best);
  EXPECT_TRUE(best->gates().empty());
}

TEST(SynthesisCandidates, RejectsEntanglersOutsideNativeProfile) {
  const std::vector<Gate> rotations;
  const std::vector<Pauli> measured{{0, 1}, {0, 2}};
  const std::vector<unsigned> wires{0, 1};
  for (Kind kind : {Kind::CX, Kind::CZ, Kind::Swap}) {
    synthesis::Options options;
    options.nativeCX = kind != Kind::CX;
    options.nativeCZ = kind != Kind::CZ;
    options.nativeSwap = kind != Kind::Swap;
    local::MeasuredCandidateSink sink({2, rotations, measured, wires}, options,
                                      nullptr);
    ASSERT_TRUE(sink.error().empty()) << sink.error();
    // Each pair is exactly identity. Semantic equivalence alone must not
    // publish a circuit outside the target profile or miscount its metrics.
    EXPECT_EQ(sink.submit({{kind, 0, 1}, {kind, 0, 1}}).status,
              local::SubmissionStatus::InvalidProof);
    EXPECT_FALSE(sink.takeBest());
  }
}

TEST(SynthesisCandidates, ValidatesSourceBeforeComputingInputCost) {
  const std::vector<Pauli> measured{{0, 1}};
  const std::vector<unsigned> wires{0};
  synthesis::Options options;
  for (const std::vector<Gate> &rotations :
       {std::vector<Gate>{{Kind::H, 1}},
        std::vector<Gate>{Gate::rotation({0, 1}, {1, 0})}}) {
    local::MeasuredCandidateSink sink({1, rotations, measured, wires}, options,
                                      nullptr);
    EXPECT_FALSE(sink.error().empty());
    EXPECT_EQ(sink.submit({}).status, local::SubmissionStatus::InvalidProof);
    EXPECT_FALSE(sink.takeBest());
  }
  const std::vector<Gate> empty;
  local::MeasuredCandidateSink zeroWidth({0, empty, measured, wires}, options,
                                         nullptr);
  EXPECT_FALSE(zeroWidth.error().empty());
  EXPECT_EQ(zeroWidth.submit({}).status, local::SubmissionStatus::InvalidProof);
  EXPECT_FALSE(zeroWidth.takeBest());
  options.nativeCX = options.nativeCZ = false;
  local::MeasuredCandidateSink badProfile({1, empty, measured, wires}, options,
                                          nullptr);
  EXPECT_FALSE(badProfile.error().empty());
  EXPECT_EQ(badProfile.submit({}).status,
            local::SubmissionStatus::InvalidProof);
  EXPECT_FALSE(badProfile.takeBest());
}

TEST(SynthesisCandidates, RejectsMalformedReadoutContracts) {
  const std::vector<Gate> rotations;
  synthesis::Options options;
  struct Contract {
    unsigned width;
    std::vector<Pauli> measured;
    std::vector<unsigned> wires;
  };
  const Contract contracts[] = {
      {1, {{0, 1}}, {1}},            // Out-of-range physical readout.
      {1, {{0, 2}}, {0}},            // Observable support exceeds width.
      {1, {{0, 1}}, {}},             // Missing outcome wire.
      {1, {}, {}},                   // Empty measurement contract.
      {2, {{0, 1}, {0, 2}}, {0, 0}}, // Duplicate physical outcomes.
      {2, {{0, 1}, {1, 0}}, {0, 1}}, // Noncommuting observables.
      {2, {{0, 1}, {0, 1}}, {0, 1}}  // Dependent observables.
  };
  for (const auto &contract : contracts) {
    local::MeasuredCandidateSink sink(
        {contract.width, rotations, contract.measured, contract.wires}, options,
        nullptr);
    EXPECT_FALSE(sink.error().empty());
    EXPECT_EQ(sink.submit({}).status, local::SubmissionStatus::InvalidProof);
    EXPECT_FALSE(sink.takeBest());
  }
}

TEST(SynthesisCandidates, ProofExhaustionRetainsPreviouslyVerifiedCandidate) {
  const std::vector<Gate> rotations;
  const std::vector<Pauli> measured{{0, 1}};
  const std::vector<unsigned> wires{0};
  synthesis::Options options;
  constexpr uint64_t limit = 1000000;
  SearchBudget budget(limit);
  local::MeasuredCandidateSink sink({1, rotations, measured, wires}, options,
                                    &budget);
  ASSERT_TRUE(sink.error().empty()) << sink.error();
  ASSERT_EQ(sink.submit({}).status, local::SubmissionStatus::Verified);
  // Model optional exploration consuming all but one remaining work unit.
  // The next measured replay cannot complete, but the existing proof remains
  // sufficient to retrieve the incumbent without another budget charge.
  ASSERT_LT(budget.work(), limit);
  ASSERT_TRUE(budget.take(limit - budget.work() - 1));
  EXPECT_EQ(sink.submit({}).status,
            local::SubmissionStatus::ProofBudgetExhausted);
  EXPECT_TRUE(budget.stopped());
  auto work = budget.work();
  auto best = sink.takeBest();
  ASSERT_TRUE(best);
  EXPECT_TRUE(best->gates().empty());
  EXPECT_EQ(budget.work(), work);
  EXPECT_FALSE(sink.takeBest());
}
