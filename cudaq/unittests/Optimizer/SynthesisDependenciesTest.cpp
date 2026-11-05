/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisDependencies.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <limits>
#include <random>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph::detail;

namespace {
constexpr uint64_t unlimited = std::numeric_limits<uint64_t>::max();

bool referenceReady(const std::vector<Pauli> &word,
                    const std::vector<bool> &active, unsigned id) {
  if (!active[id])
    return false;
  for (unsigned earlier = 0; earlier < id; ++earlier)
    if (active[earlier] && word[earlier].anti(word[id]))
      return false;
  return true;
}

void checkReadySet(const DependencyCursor &cursor,
                   const std::vector<Pauli> &word,
                   const std::vector<bool> &active) {
  for (unsigned id = 0; id < word.size(); ++id) {
    auto actual = cursor.ready(id);
    ASSERT_TRUE(actual);
    EXPECT_EQ(*actual.value, referenceReady(word, active, id)) << "id=" << id;
    auto alive = cursor.isActive(id);
    ASSERT_TRUE(alive);
    EXPECT_EQ(*alive.value, active[id]);
  }
}

uint64_t referenceEdges(const std::vector<Pauli> &word) {
  uint64_t result = 0;
  for (unsigned i = 0; i < word.size(); ++i)
    for (unsigned j = i + 1; j < word.size(); ++j)
      result += word[i].anti(word[j]);
  return result;
}
} // namespace

TEST(SynthesisDependencies, ExhaustiveShortWordsAndConsumedSubsets) {
  // All six-node X/Z words and every consumed subset, including subsets that
  // cannot arise from a legal schedule. Consume in reverse order to exercise
  // non-head removal independently of the current synthesis policy.
  constexpr unsigned length = 6;
  for (unsigned axes = 0; axes < (1u << length); ++axes) {
    std::vector<Pauli> word;
    for (unsigned i = 0; i < length; ++i)
      word.push_back((axes >> i) & 1 ? Pauli{1, 0} : Pauli{0, 1});
    auto index = DependencyIndex::build(word, unlimited);
    ASSERT_TRUE(index);
    EXPECT_EQ(index.value->statistics().logicalEdges, referenceEdges(word));
    for (unsigned removed = 0; removed < (1u << length); ++removed) {
      auto cursor = DependencyCursor::create(*index.value);
      ASSERT_TRUE(cursor);
      std::vector<bool> active(length, true);
      for (unsigned i = length; i-- > 0;)
        if ((removed >> i) & 1) {
          ASSERT_TRUE(cursor.value->consume(i));
          active[i] = false;
        }
      checkReadySet(*cursor.value, word, active);
    }
  }
}

TEST(SynthesisDependencies, RandomLegalSchedulesMatchDensePredecessors) {
  std::mt19937 random(20261105);
  for (unsigned trial = 0; trial < 64; ++trial) {
    std::vector<Pauli> pool;
    for (unsigned i = 0; i < 10; ++i)
      pool.push_back({random() & 15u, random() & 15u, bool(random() & 1u)});
    std::vector<Pauli> word;
    for (unsigned i = 0; i < 24; ++i)
      word.push_back(pool[random() % pool.size()]);
    auto index = DependencyIndex::build(word, unlimited);
    ASSERT_TRUE(index);
    EXPECT_EQ(index.value->statistics().logicalEdges, referenceEdges(word));
    auto cursor = DependencyCursor::create(*index.value);
    ASSERT_TRUE(cursor);
    std::vector<bool> active(word.size(), true);
    for (unsigned left = word.size(); left; --left) {
      checkReadySet(*cursor.value, word, active);
      std::vector<unsigned> ready;
      for (unsigned i = 0; i < word.size(); ++i)
        if (referenceReady(word, active, i))
          ready.push_back(i);
      ASSERT_FALSE(ready.empty());
      unsigned chosen = ready[random() % ready.size()];
      ASSERT_TRUE(cursor.value->consume(chosen));
      active[chosen] = false;
    }
    checkReadySet(*cursor.value, word, active);
  }
}

TEST(SynthesisDependencies, SignsIdentityAndHighWireIndices) {
  QubitSet high = QubitSet::single(130);
  QubitSet regrown = high;
  regrown.set(511);
  regrown.set(511, false);
  std::vector<Pauli> word{{high, {}, false},
                          {regrown, {}, true},
                          {{}, high, false},
                          {{}, {}, true},
                          {{}, {}, false}};
  auto index = DependencyIndex::build(word, 2);
  ASSERT_TRUE(index);
  EXPECT_EQ(index.value->statistics().classes, 3u);
  EXPECT_EQ(index.value->statistics().logicalEdges, 2u);
  EXPECT_EQ(index.value->statistics().storedConflicts, 2u);
  auto cursor = DependencyCursor::create(*index.value);
  ASSERT_TRUE(cursor);
  std::vector<bool> active(word.size(), true);
  checkReadySet(*cursor.value, word, active);
  ASSERT_TRUE(cursor.value->consume(1)); // Same-axis non-head removal.
  active[1] = false;
  checkReadySet(*cursor.value, word, active);
  ASSERT_TRUE(cursor.value->consume(0)); // Skip the already consumed entry.
  active[0] = false;
  checkReadySet(*cursor.value, word, active);
  EXPECT_TRUE(*cursor.value->ready(2).value);
}

TEST(SynthesisDependencies, CommonCliffordConjugationPreservesReadiness) {
  std::mt19937 random(8137);
  std::vector<Pauli> word;
  for (unsigned i = 0; i < 16; ++i)
    word.push_back({random() & 7u, random() & 7u, bool(random() & 1u)});
  auto transformed = word;
  // Unsigned symplectic action of H(0), S(1), CX(1,2). Clifford sign changes
  // are irrelevant to dependencies; alter signs separately as an extra check.
  for (Pauli &p : transformed) {
    bool x = p.x.test(0), z = p.z.test(0);
    p.x.set(0, z);
    p.z.set(0, x);
    p.z.flip(1, p.x.test(1));
    p.x.flip(2, p.x.test(1));
    p.z.flip(1, p.z.test(2));
    p.sign = !p.sign;
  }
  auto originalIndex = DependencyIndex::build(word, unlimited);
  auto transformedIndex = DependencyIndex::build(transformed, unlimited);
  ASSERT_TRUE(originalIndex);
  ASSERT_TRUE(transformedIndex);
  EXPECT_EQ(originalIndex.value->statistics().logicalEdges,
            transformedIndex.value->statistics().logicalEdges);
  EXPECT_EQ(originalIndex.value->statistics().classes,
            transformedIndex.value->statistics().classes);
  for (unsigned trial = 0; trial < 32; ++trial) {
    auto original = DependencyCursor::create(*originalIndex.value);
    auto changed = DependencyCursor::create(*transformedIndex.value);
    ASSERT_TRUE(original);
    ASSERT_TRUE(changed);
    std::vector<bool> active(word.size(), true);
    for (unsigned i = 0; i < word.size(); ++i)
      if (random() & 1u) {
        ASSERT_TRUE(original.value->consume(i));
        ASSERT_TRUE(changed.value->consume(i));
        active[i] = false;
      }
    checkReadySet(*original.value, transformed, active);
    checkReadySet(*changed.value, word, active);
  }
}

TEST(SynthesisDependencies, LogicalCapIsIndependentOfCompressedStorage) {
  // Two class conflicts replace four million occurrence edges. Count checks
  // establish compression without allocating a quadratic reference graph.
  std::vector<Pauli> word(2000, Pauli{1, 0});
  word.insert(word.end(), 2000, Pauli{0, 1});
  auto index = DependencyIndex::build(word, 4000000);
  ASSERT_TRUE(index);
  const auto &stats = index.value->statistics();
  EXPECT_EQ(stats.occurrences, 4000u);
  EXPECT_EQ(stats.classes, 2u);
  EXPECT_EQ(stats.logicalEdges, 4000000u);
  EXPECT_EQ(stats.storedConflicts, 2u);
  auto tooSmall = DependencyIndex::build(word, 3999999);
  EXPECT_FALSE(tooSmall);
  EXPECT_FALSE(tooSmall.value.has_value());
  EXPECT_EQ(tooSmall.error, DependencyError::DependencyLimit);
  EXPECT_TRUE(tooSmall.exhausted());
  EXPECT_TRUE(DependencyIndex::build({{1, 0}, {1, 0, true}, {0, 0}}, 0));
  EXPECT_FALSE(DependencyIndex::build({{1, 0}, {0, 1}}, 0));
}

TEST(SynthesisDependencies, BudgetFailuresDoNotPublishPartialState) {
  std::vector<Pauli> word{{1, 0}, {1, 0}, {1, 0}, {0, 1}};
  SearchBudget full(1000);
  auto index = DependencyIndex::build(word, unlimited, &full);
  ASSERT_TRUE(index);
  uint64_t buildWork = full.work();
  ASSERT_GT(buildWork, 0u);
  for (uint64_t work = 0; work < buildWork; ++work) {
    SearchBudget limited(work);
    auto failed = DependencyIndex::build(word, unlimited, &limited);
    EXPECT_FALSE(failed);
    EXPECT_FALSE(failed.value.has_value());
    EXPECT_EQ(failed.error, DependencyError::WorkLimit);
    EXPECT_TRUE(failed.exhausted());
    EXPECT_TRUE(limited.stopped());
    EXPECT_LE(limited.work(), work);
  }
  SearchBudget exact(buildWork);
  EXPECT_TRUE(DependencyIndex::build(word, unlimited, &exact));
  EXPECT_EQ(exact.work(), buildWork);

  SearchBudget noWork(0);
  auto noCursor = DependencyCursor::create(*index.value, &noWork);
  EXPECT_FALSE(noCursor);
  EXPECT_EQ(noCursor.error, DependencyError::WorkLimit);
  auto cursor = DependencyCursor::create(*index.value);
  ASSERT_TRUE(cursor);
  ASSERT_TRUE(cursor.value->consume(2));
  ASSERT_TRUE(cursor.value->consume(1));
  // Removing head 0 would skip both inactive entries. Exhaust after the first
  // inspection and verify that neither active state nor class head changes.
  SearchBudget shortConsume(2);
  auto failedConsume = cursor.value->consume(0, &shortConsume);
  EXPECT_FALSE(failedConsume);
  EXPECT_EQ(failedConsume.error, DependencyError::WorkLimit);
  checkReadySet(*cursor.value, word, {true, false, false, true});
  ASSERT_TRUE(cursor.value->consume(0));
  checkReadySet(*cursor.value, word, {false, false, false, true});

  // A query can fail after its entry charge. That is not a false readiness
  // result, even when the correct answer would be false.
  auto fresh = DependencyCursor::create(*index.value);
  ASSERT_TRUE(fresh);
  SearchBudget shortQuery(1);
  auto failedQuery = fresh.value->ready(3, &shortQuery);
  EXPECT_FALSE(failedQuery);
  EXPECT_FALSE(failedQuery.value.has_value());
  EXPECT_EQ(failedQuery.error, DependencyError::WorkLimit);
  EXPECT_FALSE(*fresh.value->ready(3).value);
}

TEST(SynthesisDependencies, EmptyAndInvalidOccurrenceQueries) {
  auto empty = DependencyIndex::build({}, 0);
  ASSERT_TRUE(empty);
  EXPECT_EQ(empty.value->statistics().occurrences, 0u);
  auto cursor = DependencyCursor::create(*empty.value);
  ASSERT_TRUE(cursor);
  EXPECT_EQ(cursor.value->ready(0).error, DependencyError::InvalidOccurrence);
  EXPECT_EQ(cursor.value->isActive(0).error,
            DependencyError::InvalidOccurrence);
  EXPECT_EQ(cursor.value->consume(0).error, DependencyError::InvalidOccurrence);
  auto index = DependencyIndex::build({{1, 0}}, 0);
  ASSERT_TRUE(index);
  auto one = DependencyCursor::create(*index.value);
  ASSERT_TRUE(one);
  ASSERT_TRUE(one.value->consume(0));
  EXPECT_EQ(one.value->consume(0).error, DependencyError::InvalidOccurrence);
  EXPECT_FALSE(*one.value->ready(0).value);
}
