/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// Private one-/two-wire signed Clifford dictionary mechanics shared by the
/// unweighted GateCount windows and native-profile synthesis windows. Scalar
/// phase is intentionally absent from keys and must be independently replayed.
//===----------------------------------------------------------------------===//
#pragma once
#include "TCountInternal.h"
#include <unordered_map>
#include <vector>

namespace cudaq::pauli::clifford_dictionary {
using Key = uint32_t;
/// Encode the identity action on the local Pauli generators.
inline Key identity(unsigned n) {
  Key key = 0;
  for (unsigned i = 0; i < 2 * n; ++i)
    key |= (i < n ? 1u << i : 1u << (i - n + 2)) << (5 * i);
  return key;
}
/// Conjugate every generator image by one appended Clifford.
inline Key advance(Key key, const Gate &g, unsigned n) {
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
inline std::vector<Gate> orderedGenerators(unsigned n) {
  std::vector<Gate> generators;
  for (unsigned q = 0; q < n; ++q)
    for (auto kind : {Kind::H, Kind::S, Kind::Sdg, Kind::X, Kind::Y, Kind::Z})
      generators.push_back({kind, q});
  if (n == 2)
    for (auto gate : {Gate{Kind::CX, 0, 1}, Gate{Kind::CX, 1, 0},
                      Gate{Kind::CZ, 0, 1}, Gate{Kind::Swap, 0, 1}})
      generators.push_back(gate);
  return generators;
}

/// Preserve the old unweighted BFS exactly. Its first-discovery word is a
/// compatibility contract: equal-length alternatives can behave differently in
/// later cleanup. Do not substitute weighted search, including unit-cost mode.
inline Database buildDatabase(unsigned n) {
  auto generators = orderedGenerators(n);
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

} // namespace cudaq::pauli::clifford_dictionary
