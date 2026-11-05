/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "gtest/gtest.h"
#include "cudaq/Optimizer/Pauli/SearchBudget.h"
#include <limits>

using cudaq::pauli::SearchBudget;

TEST(SynthesisBudget, StandaloneAccountingAndStickyFailureRemainUnchanged) {
  SearchBudget budget(7);
  EXPECT_EQ(budget.remaining(), 7u);
  EXPECT_FALSE(budget.timeRemainingMs());
  EXPECT_TRUE(budget.take(3));
  EXPECT_EQ(budget.work(), 3u);
  EXPECT_EQ(budget.remaining(), 4u);
  EXPECT_FALSE(budget.take(5));
  EXPECT_EQ(budget.work(), 3u);
  EXPECT_TRUE(budget.stopped());
  EXPECT_EQ(budget.remaining(), 0u);
  EXPECT_FALSE(budget.take(0));
}

TEST(SynthesisBudget, LocalExhaustionPreservesParentAndSiblingReserve) {
  SearchBudget parent(20);
  SearchBudget search(parent, 8);
  SearchBudget sibling(parent, 12);
  ASSERT_TRUE(search.take(6));
  EXPECT_EQ(parent.work(), 6u);
  EXPECT_EQ(search.work(), 6u);
  EXPECT_FALSE(search.take(3));
  EXPECT_EQ(parent.work(), 6u);
  EXPECT_FALSE(parent.stopped());
  EXPECT_FALSE(sibling.stopped());
  EXPECT_EQ(parent.remaining(), 14u);
  EXPECT_EQ(sibling.remaining(), 12u);
  EXPECT_TRUE(sibling.take(5));
  EXPECT_TRUE(
      parent.take(9)); // Completion/proof still uses the parent reserve.
  EXPECT_EQ(parent.work(), 20u);
  EXPECT_EQ(search.work(), 6u);
  EXPECT_EQ(sibling.work(), 5u);
}

TEST(SynthesisBudget, ParentFailurePropagatesWithoutPartialCharges) {
  SearchBudget parent(5);
  SearchBudget child(parent, 20);
  SearchBudget sibling(parent, 20);
  ASSERT_TRUE(child.take(3));
  EXPECT_EQ(child.remaining(), 2u);
  EXPECT_EQ(sibling.remaining(), 2u);
  EXPECT_FALSE(child.take(3));
  EXPECT_EQ(parent.work(), 3u);
  EXPECT_EQ(child.work(), 3u);
  EXPECT_EQ(sibling.work(), 0u);
  EXPECT_TRUE(parent.stopped());
  EXPECT_TRUE(child.stopped());
  EXPECT_TRUE(sibling.stopped());
  EXPECT_EQ(sibling.remaining(), 0u);
  EXPECT_FALSE(sibling.take());
}

TEST(SynthesisBudget, NestedChildrenChargeEveryAncestorExactlyOnce) {
  SearchBudget parent(30);
  SearchBudget child(parent, 12);
  SearchBudget grandchild(child, 8);
  ASSERT_TRUE(grandchild.take(5));
  EXPECT_EQ(parent.work(), 5u);
  EXPECT_EQ(child.work(), 5u);
  EXPECT_EQ(grandchild.work(), 5u);
  EXPECT_FALSE(grandchild.take(4));
  EXPECT_EQ(parent.work(), 5u);
  EXPECT_EQ(child.work(), 5u);
  EXPECT_FALSE(parent.stopped());
  EXPECT_FALSE(child.stopped());
  EXPECT_TRUE(child.take(4));
  EXPECT_EQ(parent.work(), 9u);
  EXPECT_EQ(child.work(), 9u);
  EXPECT_EQ(grandchild.work(), 5u);

  SearchBudget another(child, 20);
  EXPECT_EQ(another.remaining(), 3u);
  EXPECT_FALSE(another.take(4)); // Intermediate failure cannot charge the root.
  EXPECT_EQ(parent.work(), 9u);
  EXPECT_EQ(child.work(), 9u);
  EXPECT_EQ(another.work(), 0u);
  EXPECT_TRUE(child.stopped());
  EXPECT_FALSE(parent.stopped());
  EXPECT_TRUE(parent.take(21));
}

TEST(SynthesisBudget, ZeroAndMaximumAllowancesDoNotOverflow) {
  constexpr auto maximum = std::numeric_limits<uint64_t>::max();
  SearchBudget parent(maximum);
  SearchBudget zero(parent, 0);
  EXPECT_TRUE(zero.take(0));
  EXPECT_FALSE(zero.take(1));
  EXPECT_EQ(parent.work(), 0u);
  EXPECT_EQ(parent.remaining(), maximum);
  EXPECT_FALSE(parent.stopped());

  SearchBudget child(parent, maximum);
  EXPECT_TRUE(child.take(maximum));
  EXPECT_EQ(child.work(), maximum);
  EXPECT_EQ(parent.work(), maximum);
  EXPECT_EQ(child.remaining(), 0u);
  EXPECT_EQ(parent.remaining(), 0u);
  EXPECT_TRUE(child.take(0));
  EXPECT_FALSE(child.take(1));
  EXPECT_FALSE(parent.stopped()); // Failure was the child's own work cap.
  EXPECT_TRUE(parent.take(0));
  EXPECT_FALSE(parent.take(1));
  EXPECT_EQ(parent.work(), maximum);
}

TEST(SynthesisBudget, DeadlineQueriesRespectAllAncestors) {
  SearchBudget unlimited(100);
  SearchBudget noDeadline(unlimited, 50);
  EXPECT_FALSE(unlimited.timeRemainingMs());
  EXPECT_FALSE(noDeadline.timeRemainingMs());

  SearchBudget outer(100, 5000);
  SearchBudget shortChild(outer, 50, 1000);
  SearchBudget inherited(shortChild, 20);
  auto outerTime = outer.timeRemainingMs();
  auto shortTime = shortChild.timeRemainingMs();
  auto inheritedTime = inherited.timeRemainingMs();
  ASSERT_TRUE(outerTime);
  ASSERT_TRUE(shortTime);
  ASSERT_TRUE(inheritedTime);
  EXPECT_LE(*outerTime, 5000u);
  EXPECT_LE(*shortTime, 1000u);
  EXPECT_LE(*inheritedTime, *shortTime);

  SearchBudget longerChild(shortChild, 10, 10000);
  ASSERT_TRUE(longerChild.timeRemainingMs());
  EXPECT_LE(*longerChild.timeRemainingMs(), 1000u);
}

TEST(SynthesisBudget, LocalDeadlineExpiryLeavesParentUsable) {
  SearchBudget parent(100);
  SearchBudget child(parent, 100, 1);
  // One millisecond of monotonic polling avoids sleep precision assumptions or
  // worker threads. No positive-time assertion depends on CI scheduling.
  while (*child.timeRemainingMs()) {
  }
  EXPECT_FALSE(child.stopped()); // Querying the clock does not spend work.
  EXPECT_FALSE(child.take(0));
  EXPECT_TRUE(child.stopped());
  EXPECT_EQ(child.work(), 0u);
  EXPECT_EQ(parent.work(), 0u);
  EXPECT_FALSE(parent.stopped());
  EXPECT_TRUE(parent.take(100));
}

TEST(SynthesisBudget, AncestorDeadlineExpiryStopsNestedChargesAtomically) {
  SearchBudget parent(100, 1);
  SearchBudget child(parent, 80);
  SearchBudget grandchild(child, 60);
  while (*parent.timeRemainingMs()) {
  }
  EXPECT_EQ(child.timeRemainingMs(), std::optional<unsigned>(0));
  EXPECT_EQ(grandchild.timeRemainingMs(), std::optional<unsigned>(0));
  EXPECT_FALSE(grandchild.take(1));
  EXPECT_TRUE(parent.stopped());
  EXPECT_TRUE(child.stopped());
  EXPECT_TRUE(grandchild.stopped());
  EXPECT_EQ(parent.work(), 0u);
  EXPECT_EQ(child.work(), 0u);
  EXPECT_EQ(grandchild.work(), 0u);
}
