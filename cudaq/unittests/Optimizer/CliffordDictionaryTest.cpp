/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This compatibility oracle deliberately freezes the pre-extraction BFS. It
// is separate from the shared implementation so generator ordering, key
// changes, or equal-length predecessor changes cannot silently alter legacy
// GateCount output words. It is not used by production optimization.
// Pre-extraction GateCount.cpp SHA256:
// 7cc08a4271346362a946ffe916bfc1fcaf0cb69bc26ee78b958ad23e2e2cad30
#include "Pauli/CliffordDictionary.h"
#include "gtest/gtest.h"
#include <algorithm>

using namespace cudaq::pauli;
namespace frozen_gatecount {
using Key = uint32_t;
/// Encode the identity action on the local Pauli generators.
Key identity(unsigned n) {
  Key key = 0;
  for (unsigned i = 0; i < 2 * n; ++i)
    key |= (i < n ? 1u << i : 1u << (i - n + 2)) << (5 * i);
  return key;
}
/// Conjugate every generator image by one appended Clifford.
Key advance(Key key, const Gate &g, unsigned n) {
  Key next = 0;
  for (unsigned i = 0; i < 2 * n; ++i) {
    unsigned v = (key >> (5 * i)) & 31;
    Pauli p{v & 3u, (v >> 2) & 3u, bool(v & 16)};
    p = tcount::detail::conjugate(p, g);
    unsigned value = unsigned(p.sign) << 4;
    for (unsigned q : p.x.set_bits())
      value |= 1u << q;
    for (unsigned q : p.z.set_bits())
      value |= 1u << (q + 2);
    next |= value << (5 * i);
  }
  return next;
}
/// A shortest Clifford word, stored as a predecessor chain instead of copying
/// a full sequence into each database entry. Database size is fixed by width.
struct Entry {
  /// Tableau before the final generator in this shortest word.
  Key previous;
  /// Final generator appended to the predecessor word.
  Gate gate;
  /// Number of named gates, excluding scalar correction.
  unsigned length;
};
using Database = std::unordered_map<Key, Entry>;
/// Enumerate the finite local Clifford group in breadth-first word order.
Database buildDatabase(unsigned n) {
  std::vector<Gate> generators;
  for (unsigned q = 0; q < n; ++q)
    for (auto kind : {Kind::H, Kind::S, Kind::Sdg, Kind::X, Kind::Y, Kind::Z})
      generators.push_back({kind, q});
  if (n == 2)
    for (auto gate : {Gate{Kind::CX, 0, 1}, Gate{Kind::CX, 1, 0},
                      Gate{Kind::CZ, 0, 1}, Gate{Kind::Swap, 0, 1}})
      generators.push_back(gate);
  Key root = identity(n);
  Database db{{root, {root, {}, 0}}};
  std::vector<Key> queue{root};
  for (size_t i = 0; i < queue.size(); ++i) {
    auto state = queue[i];
    unsigned length = db.at(state).length;
    for (auto g : generators) {
      Key next = advance(state, g, n);
      if (db.emplace(next, Entry{state, g, length + 1}).second)
        queue.push_back(next);
    }
  }
  return db;
}

} // namespace frozen_gatecount

namespace {
template <typename Database, typename Key>
std::vector<Gate> word(const Database &database, Key key, Key identity) {
  std::vector<Gate> output;
  for (Key at = key; at != identity; at = database.at(at).previous)
    output.push_back(database.at(at).gate);
  std::reverse(output.begin(), output.end());
  return output;
}
bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}
} // namespace

TEST(CliffordDictionary, EveryLegacyWordMatchesFrozenBreadthFirstSearch) {
  for (unsigned width : {1u, 2u}) {
    const auto frozen = frozen_gatecount::buildDatabase(width);
    const auto extracted = clifford_dictionary::buildDatabase(width);
    ASSERT_EQ(frozen.size(), width == 1 ? 24u : 11520u);
    ASSERT_EQ(extracted.size(), frozen.size());
    const auto identity = frozen_gatecount::identity(width);
    ASSERT_EQ(clifford_dictionary::identity(width), identity);
    for (const auto &[key, expected] : frozen) {
      SCOPED_TRACE(::testing::Message() << "width=" << width << " key=" << key);
      auto found = extracted.find(key);
      ASSERT_NE(found, extracted.end());
      EXPECT_EQ(found->second.length, expected.length);
      // Equal minimum lengths are insufficient: different shortest words can
      // expose different cancellations in the next optimizer stage.
      ASSERT_TRUE(sameWord(word(frozen, key, identity),
                           word(extracted, key, identity)));
    }
  }
}

TEST(CliffordDictionary, CanonicalGeneratorsRetainLegacyTieOrder) {
  const std::vector<Gate> expected{
      {Kind::H, 0},     {Kind::S, 0},     {Kind::Sdg, 0},   {Kind::X, 0},
      {Kind::Y, 0},     {Kind::Z, 0},     {Kind::H, 1},     {Kind::S, 1},
      {Kind::Sdg, 1},   {Kind::X, 1},     {Kind::Y, 1},     {Kind::Z, 1},
      {Kind::CX, 0, 1}, {Kind::CX, 1, 0}, {Kind::CZ, 0, 1}, {Kind::Swap, 0, 1}};
  EXPECT_TRUE(sameWord(clifford_dictionary::orderedGenerators(2), expected));
  EXPECT_TRUE(
      sameWord(clifford_dictionary::orderedGenerators(1),
               std::vector<Gate>(expected.begin(), expected.begin() + 6)));
}
