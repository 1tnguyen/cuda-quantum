/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/
#include "Pauli/SynthesisCompletion.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
using namespace cudaq::pauli::graph::detail;

namespace {
struct Fixture {
  std::vector<Gate> rotations;
  std::vector<Pauli> observables{{{}, 1}, {{}, 2, true}};
  std::vector<unsigned> readouts{1, 0};
  Options options;
  CompletionLimits limits;
  CostVector anchor{};
  CostRank rank{100, 100, 100, 100, 100};
  Fixture() {
    options.maxGates = limits.maxGates = 128;
    options.pruneUnobserved = false;
    options.nativeCX = options.nativeCZ = options.nativeSwap = true;
    limits.enableNative = limits.enableFour = false;
    anchor.gates = anchor.t = anchor.twoQubit = anchor.depth = anchor.tDepth =
        100;
  }
  MeasuredProblem source() const {
    return {2, rotations, observables, readouts};
  }
  RegionalPlan plan() const { return {{}, observables}; }
};
// Exact signed computational-basis effect, independent of the production
// Pauli conjugation/readout routines. These fixtures emit X/CX/CZ/Swap only.
void expectOrderedEffects(const CertifiedCompletion &candidate,
                          const Fixture &fixture) {
  ASSERT_TRUE(candidate.matchesSource(fixture.source()));
  for (unsigned input = 0; input < 4; ++input) {
    unsigned output = input;
    for (const auto &gate : candidate.gates()) {
      if (gate.kind == Kind::X)
        output ^= 1u << gate.a;
      else if (gate.kind == Kind::CX)
        output ^= ((output >> gate.a) & 1u) << gate.b;
      else if (gate.kind == Kind::Swap) {
        if (((output >> gate.a) ^ (output >> gate.b)) & 1u)
          output ^= (1u << gate.a) | (1u << gate.b);
      } else
        ASSERT_TRUE(gate.kind == Kind::CZ || gate.kind == Kind::Phase ||
                    (gate.kind == Kind::Rotate && !gate.pauli.x))
            << unsigned(gate.kind);
    }
    for (unsigned i = 0; i < fixture.observables.size(); ++i) {
      const auto &row = fixture.observables[i];
      unsigned expected = row.sign;
      for (unsigned q : row.z.set_bits())
        expected ^= (input >> q) & 1u;
      EXPECT_EQ((output >> fixture.readouts[i]) & 1u, expected);
    }
  }
}
} // namespace

TEST(SynthesisCompletion, CertificatesCannotBeDefaultConstructed) {
  static_assert(!std::is_default_constructible_v<CertifiedCompletion>);
  static_assert(!std::is_default_constructible_v<CompletionRoute>);
  static_assert(!std::is_copy_constructible_v<CompletionRoute>);
}

TEST(SynthesisCompletion, EveryCheckedStageOfferedWithoutVisitorDependency) {
  Fixture f;
  SearchBudget budget(100000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  ASSERT_EQ(session.status(), CompletionStatus::Complete) << session.error();
  auto plan = f.plan();
  plan.prefix = {{Kind::X, 0}, {Kind::X, 0}};
  std::vector<CompletionStage> stages;
  auto completed = session.complete(session.originalRoute(), plan, budget,
                                    [&](const CertifiedCompletion &candidate) {
                                      stages.push_back(candidate.stage());
                                      expectOrderedEffects(candidate, f);
                                    });
  ASSERT_EQ(completed.status, CompletionStatus::Complete) << completed.error;
  ASSERT_EQ(stages.size(), 2u);
  EXPECT_EQ(stages[0], CompletionStage::RawMeasured);
  EXPECT_EQ(stages[1], CompletionStage::SinkChannels);
  ASSERT_NE(session.best(), nullptr);
  auto cached = session.complete(session.originalRoute(), plan, budget);
  EXPECT_TRUE(cached.cacheHit);
  EXPECT_EQ(cached.published, 0u);
  EXPECT_EQ(cached.rank, completed.rank);

  SearchBudget otherBudget(100000000);
  CompletionSession withoutVisitor(f.source(), f.options, f.anchor, f.rank,
                                   f.limits, otherBudget);
  auto again = withoutVisitor.complete(withoutVisitor.originalRoute(), plan,
                                       otherBudget);
  ASSERT_EQ(again.status, CompletionStatus::Complete);
  ASSERT_NE(withoutVisitor.best(), nullptr);
  EXPECT_EQ(withoutVisitor.best()->rank(), session.best()->rank());
}

TEST(SynthesisCompletion, FinalAnchorPolicyIsExplicitAndImmutable) {
  CostVector anchor{}, first{}, later{};
  anchor.gates = anchor.t = anchor.twoQubit = anchor.depth = anchor.tDepth = 10;
  first = later = anchor;
  first.gates = 8;
  first.depth = 7;
  later.gates = 7;
  later.depth = 9;
  EXPECT_TRUE(completionAdmissible(first, anchor, CompletionPolicy::AllFive));
  EXPECT_TRUE(completionAdmissible(later, anchor, CompletionPolicy::AllFive));
  EXPECT_FALSE(completionAdmissible(later, first, CompletionPolicy::AllFive));
  later.depth = 11;
  EXPECT_FALSE(completionAdmissible(later, anchor, CompletionPolicy::AllFive));
  EXPECT_TRUE(completionAdmissible(later, anchor, CompletionPolicy::GatesAndT));
  later.t = 11;
  EXPECT_FALSE(
      completionAdmissible(later, anchor, CompletionPolicy::GatesAndT));
}

TEST(SynthesisCompletion, AnchorRejectionDoesNotDiscardRawOrSinkStages) {
  Fixture f;
  f.anchor = {};
  f.rank = {};
  SearchBudget budget(100000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  unsigned observed = 0;
  auto completed = session.complete(session.originalRoute(), f.plan(), budget,
                                    [&](const auto &) { ++observed; });
  EXPECT_EQ(completed.status, CompletionStatus::Complete);
  EXPECT_EQ(observed, 2u);
  EXPECT_TRUE(completed.rank);
  EXPECT_EQ(session.best(), nullptr);
}

TEST(SynthesisCompletion, SmallProofBudgetCanRetryOutsideScorer) {
  Fixture f;
  SearchBudget parent(100000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            parent);
  SearchBudget tiny(parent, 1);
  auto missed = session.complete(session.originalRoute(), f.plan(), tiny);
  EXPECT_EQ(missed.status, CompletionStatus::Exhausted);
  EXPECT_FALSE(missed.rank);
  EXPECT_EQ(session.status(), CompletionStatus::Complete);
  EXPECT_FALSE(parent.stopped());
  auto completed = session.complete(session.originalRoute(), f.plan(), parent);
  EXPECT_EQ(completed.status, CompletionStatus::Complete) << completed.error;
  ASSERT_NE(session.best(), nullptr);
  expectOrderedEffects(*session.best(), f);
}

TEST(SynthesisCompletion, OptionalExhaustionRetainsRawAndAllowsRetry) {
  Fixture f;
  f.limits.enableNative = true;
  f.limits.nativeWork = 1;
  SearchBudget parent(100000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            parent);
  auto first = session.complete(session.originalRoute(), f.plan(), parent);
  EXPECT_EQ(first.status, CompletionStatus::Exhausted);
  ASSERT_TRUE(first.rank);
  ASSERT_NE(session.best(), nullptr);
  expectOrderedEffects(*session.best(), f);
  auto retried = session.complete(session.originalRoute(), f.plan(), parent);
  EXPECT_FALSE(retried.cacheHit);
  EXPECT_GE(retried.published, 2u);
  EXPECT_FALSE(parent.stopped());
}

TEST(SynthesisCompletion, IndexedRouteOwnsCheckedSourceAndPreservesIdentity) {
  Fixture f;
  SearchBudget budget(100000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  RepresentationAlternative proposal{
      2, f.rotations, f.observables, f.readouts, {}};
  auto indexed = session.checkIndexedRoute(proposal, budget);
  ASSERT_EQ(indexed.status, CompletionStatus::Complete) << indexed.error;
  ASSERT_TRUE(indexed.route);
  proposal.observables.clear();
  EXPECT_TRUE(indexed.route->indexed());
  EXPECT_EQ(indexed.route->problem().observables, f.observables);
  unsigned observed = 0;
  auto completed = session.complete(indexed.route, f.plan(), budget,
                                    [&](const CertifiedCompletion &candidate) {
                                      EXPECT_TRUE(candidate.indexed());
                                      expectOrderedEffects(candidate, f);
                                      ++observed;
                                    });
  EXPECT_EQ(completed.status, CompletionStatus::Complete);
  EXPECT_EQ(observed, 2u);
  // Identical plan under the original route is a separate proof/cache key.
  auto original = session.complete(session.originalRoute(), f.plan(), budget);
  EXPECT_FALSE(original.cacheHit);
}

TEST(SynthesisCompletion, CrossSessionRouteAndInvalidProofAreFatal) {
  Fixture f;
  SearchBudget budget(100000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  CompletionSession other(f.source(), f.options, f.anchor, f.rank, f.limits,
                          budget);
  ASSERT_EQ(session.complete(session.originalRoute(), f.plan(), budget).status,
            CompletionStatus::Complete);
  ASSERT_NE(session.best(), nullptr);
  auto wrong = session.complete(other.originalRoute(), f.plan(), budget);
  EXPECT_EQ(wrong.status, CompletionStatus::InvalidProof);
  EXPECT_EQ(session.best(), nullptr);
  EXPECT_EQ(session.complete(session.originalRoute(), f.plan(), budget).status,
            CompletionStatus::InvalidProof);
}

TEST(SynthesisCompletion, CheckedIndexedFlipComposesWithMeasuredCompletion) {
  Fixture f;
  f.rotations = {Gate::rotation({{}, 1}, {1, 4})};
  SearchBudget budget(100000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  RepresentationAlternative proposal{
      2, f.rotations, f.observables, f.readouts, {{0}}};
  proposal.rotations[0].angle = proposal.rotations[0].angle.negated();
  auto indexed = session.checkIndexedRoute(proposal, budget);
  ASSERT_EQ(indexed.status, CompletionStatus::Complete) << indexed.error;
  auto checked = session.complete(
      indexed.route, {proposal.rotations, proposal.observables}, budget);
  EXPECT_EQ(checked.status, CompletionStatus::Complete) << checked.error;
  ASSERT_NE(session.best(), nullptr);
  EXPECT_TRUE(session.best()->indexed());
  EXPECT_TRUE(session.best()->matchesSource(f.source()));
  expectOrderedEffects(*session.best(), f);
  // Removing the witness does not make the changed word a checked route.
  proposal.certificate.clear();
  EXPECT_EQ(session.checkIndexedRoute(proposal, budget).status,
            CompletionStatus::InvalidProof);
  EXPECT_EQ(session.best(), nullptr);
}

TEST(SynthesisCompletion, NativeContinuationSurvivesRawAnchorRejection) {
  Fixture f;
  f.rotations = {Gate::rotation({{}, 1}, {1, 4}),
                 Gate::rotation({{}, 1}, {1, 4})};
  f.limits.enableNative = true;
  f.anchor.t = 0; // Both raw T gates fail final admission before merging.
  SearchBudget budget(1000000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  bool rawRejected = false, nativeObserved = false;
  auto checked = session.complete(
      session.originalRoute(), {f.rotations, f.observables}, budget,
      [&](const CertifiedCompletion &candidate) {
        if (candidate.stage() == CompletionStage::RawMeasured)
          rawRejected = !completionAdmissible(candidate.metrics(), f.anchor,
                                              CompletionPolicy::AllFive);
        nativeObserved |= candidate.stage() == CompletionStage::Native;
      });
  ASSERT_NE(checked.status, CompletionStatus::InvalidProof) << checked.error;
  EXPECT_TRUE(rawRejected);
  EXPECT_TRUE(nativeObserved);
  ASSERT_NE(session.best(), nullptr);
  EXPECT_EQ(session.best()->metrics().t, 0u);
}

TEST(SynthesisCompletion, RankedChainUsesOneNativeSourceAndFreezesItsScore) {
  Fixture f;
  f.limits.enableNative = true;
  f.limits.rankedContinuation = true;
  SearchBudget budget(1000000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  auto plan = f.plan();
  plan.prefix = {{Kind::X, 0}, {Kind::X, 0}};
  size_t rawSize = 0, sinkSize = 0;
  auto completed =
      session.complete(session.originalRoute(), plan, budget,
                       [&](const CertifiedCompletion &candidate) {
                         expectOrderedEffects(candidate, f);
                         if (candidate.stage() == CompletionStage::RawMeasured)
                           rawSize = candidate.gates().size();
                         if (candidate.stage() == CompletionStage::SinkChannels)
                           sinkSize = candidate.gates().size();
                       });
  ASSERT_EQ(completed.status, CompletionStatus::Complete) << completed.error;
  ASSERT_LT(sinkSize, rawSize); // Distinct raw/sink continuation choices exist.
  EXPECT_EQ(completed.nativeSources, 1u);
  ASSERT_NE(session.best(), nullptr);
  auto repeated = session.complete(session.originalRoute(), plan, budget);
  EXPECT_TRUE(repeated.cacheHit);
  EXPECT_EQ(repeated.rank, completed.rank);
  EXPECT_EQ(repeated.nativeSources, 0u);
  EXPECT_EQ(repeated.published, 0u);
}

TEST(SynthesisCompletion, RankedChainContinuesDespiteGlobalAnchorRejection) {
  Fixture f;
  f.rotations = {Gate::rotation({{}, 1}, {1, 4}),
                 Gate::rotation({{}, 1}, {1, 4})};
  f.limits.enableNative = true;
  f.limits.rankedContinuation = true;
  f.anchor.t = 0;
  SearchBudget budget(1000000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  unsigned nativeCandidates = 0;
  bool rawRejected = false;
  auto completed = session.complete(
      session.originalRoute(), {f.rotations, f.observables}, budget,
      [&](const CertifiedCompletion &candidate) {
        if (candidate.stage() == CompletionStage::RawMeasured)
          rawRejected = !completionAdmissible(candidate.metrics(), f.anchor,
                                              CompletionPolicy::AllFive);
        nativeCandidates += candidate.stage() == CompletionStage::Native;
      });
  EXPECT_TRUE(rawRejected);
  EXPECT_EQ(completed.nativeSources, 1u);
  EXPECT_GT(nativeCandidates, 0u);
  ASSERT_NE(session.best(), nullptr);
  EXPECT_EQ(session.best()->metrics().t, 0u);
}

TEST(SynthesisCompletion, ScorerRankStaysFrozenAcrossRicherCompletionRetry) {
  Fixture f;
  f.rotations = {Gate::rotation({{}, 1}, {1, 4}),
                 Gate::rotation({{}, 1}, {1, 4})};
  f.limits.enableNative = true;
  f.limits.rankedContinuation = true;
  SearchBudget parent(1000000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            parent);
  RegionalPlan plan{f.rotations, f.observables};
  SearchBudget small(parent, 100000);
  auto first = session.score(session.originalRoute(), plan, small);
  ASSERT_EQ(first.status, CompletionStatus::Exhausted) << first.error;
  ASSERT_TRUE(first.rank);
  EXPECT_FALSE(parent.stopped());
  auto finished = session.complete(session.originalRoute(), plan, parent);
  ASSERT_EQ(finished.status, CompletionStatus::Complete) << finished.error;
  ASSERT_TRUE(finished.rank);
  EXPECT_LT(*finished.rank, *first.rank);
  ASSERT_NE(session.best(), nullptr);
  EXPECT_EQ(session.best()->metrics().t, 0u);
  auto feedback = session.score(session.originalRoute(), plan, parent);
  EXPECT_TRUE(feedback.cacheHit);
  EXPECT_EQ(feedback.published, 0u);
  EXPECT_EQ(feedback.rank, first.rank);
  EXPECT_TRUE(parent.take(1));
}

TEST(SynthesisCompletion, SignedReadoutMismatchCannotPublish) {
  Fixture f;
  SearchBudget budget(100000000);
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  auto wrong = f.plan();
  wrong.observables[0].sign = !wrong.observables[0].sign;
  unsigned observed = 0;
  auto checked = session.complete(session.originalRoute(), wrong, budget,
                                  [&](const auto &) { ++observed; });
  EXPECT_EQ(checked.status, CompletionStatus::InvalidProof);
  EXPECT_EQ(observed, 0u);
  EXPECT_EQ(session.best(), nullptr);
}

TEST(SynthesisCompletion, StorageAndIgnoredOperandPayloadsAreBounded) {
  Fixture f;
  SearchBudget budget(100000000);
  f.rotations = {Gate::rotation({{}, 1}, {1, 4}),
                 Gate::rotation({{}, 1}, {1, 4})};
  f.limits.maxGates = 1;
  CompletionSession sourceTooLarge(f.source(), f.options, f.anchor, f.rank,
                                   f.limits, budget);
  EXPECT_EQ(sourceTooLarge.status(), CompletionStatus::Unsupported);
  EXPECT_FALSE(sourceTooLarge.originalRoute());
  f.rotations.clear();
  f.limits.maxGates = 128;
  f.limits.maxLiveBytes = 1;
  CompletionSession tiny(f.source(), f.options, f.anchor, f.rank, f.limits,
                         budget);
  EXPECT_EQ(tiny.status(), CompletionStatus::Exhausted);
  f.limits.maxLiveBytes = 1ull << 30;
  CompletionSession session(f.source(), f.options, f.anchor, f.rank, f.limits,
                            budget);
  auto malformed = f.plan();
  Gate gate{Kind::X, 0};
  gate.pauli.x.set(4096);
  malformed.prefix.push_back(std::move(gate));
  EXPECT_EQ(session.complete(session.originalRoute(), malformed, budget).status,
            CompletionStatus::InvalidProof);
}
