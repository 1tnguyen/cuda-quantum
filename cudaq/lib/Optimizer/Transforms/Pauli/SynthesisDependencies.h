/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Pauli/Circuit.h"
#include "cudaq/Optimizer/Pauli/SearchBudget.h"
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudaq::pauli::graph::detail {

enum class DependencyError {
  None,
  WorkLimit,
  DependencyLimit,
  StorageLimit,
  SizeOverflow,
  InvalidOccurrence
};

/// An absent value is a failure, never an answer that a node is not ready.
/// Construction and consumption publish no partial result on failure.
template <typename T>
struct DependencyResult {
  std::optional<T> value;
  DependencyError error = DependencyError::None;
  explicit operator bool() const { return value.has_value(); }
  bool exhausted() const {
    return error == DependencyError::WorkLimit ||
           error == DependencyError::DependencyLimit ||
           error == DependencyError::StorageLimit ||
           error == DependencyError::SizeOverflow;
  }
};

struct DependencyStatistics {
  uint64_t occurrences = 0, classes = 0;
  /// Number of i<j anticommuting occurrence pairs, as in the legacy DAG.
  uint64_t logicalEdges = 0;
  /// Directed axis-conflict entries actually stored (two per conflicting pair).
  uint64_t storedConflicts = 0;
  /// Conservative peak owned bytes, including classification/hash scratch.
  uint64_t storageBytes = 0;
};

class DependencyCursor;

/// Immutable ordering index for one fixed rotation word. Signs and angles do
/// not affect dependencies. Common Clifford conjugation preserves this index;
/// algebraic rewrites and suffix-only conjugation require a new index.
///
/// Storage is O(m+K+A): m occurrences, K unsigned axes, A directed conflicts.
/// Classification uses hashing; conflict discovery makes K*(K-1)/2 Pauli
/// comparisons. Work units count operations, not elapsed time or Pauli bits:
/// one build entry, one classified occurrence, one initialized conflict list,
/// one class-pair comparison, and one stored directed conflict each. Hashing
/// and Pauli comparisons can themselves depend on the support width.
class DependencyIndex {
  friend class DependencyCursor;
  struct AxisHash {
    size_t operator()(const Pauli &p) const {
      size_t x = p.x.hash();
      return x ^ (p.z.hash() + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2));
    }
  };
  std::vector<unsigned> occurrenceClass;
  std::vector<std::vector<unsigned>> occurrences, conflicts;
  DependencyStatistics counts;
  DependencyIndex() = default;

public:
  DependencyIndex(const DependencyIndex &) = delete;
  DependencyIndex &operator=(const DependencyIndex &) = delete;
  DependencyIndex(DependencyIndex &&) = default;
  DependencyIndex &operator=(DependencyIndex &&) = default;

  /// The cap retains the legacy logical-edge meaning, independently of
  /// compressed storage. A zero cap accepts exactly commuting words. Width
  /// validity is the caller's semantic contract; identity axes are supported.
  static DependencyResult<DependencyIndex>
  build(const std::vector<Pauli> &word, uint64_t maxLogicalDependencies,
        SearchBudget *budget = nullptr,
        uint64_t maxStorageBytes = std::numeric_limits<uint64_t>::max()) {
    auto take = [&](uint64_t amount = 1) {
      return !budget || budget->take(amount);
    };
    if (!take())
      return {std::nullopt, DependencyError::WorkLimit};
    // The occurrence count is also the sentinel for an exhausted class.
    if (word.size() > std::numeric_limits<unsigned>::max())
      return {std::nullopt, DependencyError::SizeOverflow};
    DependencyIndex result;
    auto reserve = [&](uint64_t bytes) {
      if (bytes > maxStorageBytes - result.counts.storageBytes)
        return false;
      result.counts.storageBytes += bytes;
      return true;
    };
    // Occurrence vectors and their growth, plus container bookkeeping. Class
    // payload and actual conflicts are admitted incrementally before copying.
    if (!reserve(1024 + 24 * uint64_t(word.size())))
      return {std::nullopt, DependencyError::StorageLimit};
    uint64_t temporaryBytes = 0;
    std::vector<Pauli> axes;
    std::unordered_map<Pauli, unsigned, AxisHash> classes;
    for (unsigned i = 0; i < word.size(); ++i) {
      if (!take())
        return {std::nullopt, DependencyError::WorkLimit};
      uint64_t payload = word[i].x.storageBytes() + word[i].z.storageBytes();
      if (3 * payload > temporaryBytes) {
        if (!reserve(3 * payload - temporaryBytes))
          return {std::nullopt, DependencyError::StorageLimit};
        temporaryBytes = 3 * payload;
      }
      Pauli axis = word[i];
      axis.sign = false;
      auto found = classes.find(axis);
      bool inserted = found == classes.end();
      if (inserted && !reserve(1024 + 4 * payload))
        return {std::nullopt, DependencyError::StorageLimit};
      auto it = inserted ? classes.emplace(axis, axes.size()).first : found;
      unsigned id = it->second;
      if (inserted) {
        axes.push_back(std::move(axis));
        result.occurrences.emplace_back();
      }
      result.occurrenceClass.push_back(id);
      result.occurrences[id].push_back(i);
    }
    if (!take(axes.size()))
      return {std::nullopt, DependencyError::WorkLimit};
    result.conflicts.resize(axes.size());
    for (unsigned a = 0; a < axes.size(); ++a)
      for (unsigned b = a + 1; b < axes.size(); ++b) {
        if (!take())
          return {std::nullopt, DependencyError::WorkLimit};
        if (!axes[a].anti(axes[b]))
          continue;
        uint64_t na = result.occurrences[a].size();
        uint64_t nb = result.occurrences[b].size();
        // Every pair of occurrences in distinct conflicting classes
        // contributes exactly one directed edge, regardless of their order.
        if (na > std::numeric_limits<uint64_t>::max() / nb)
          return {std::nullopt, DependencyError::SizeOverflow};
        uint64_t edges = na * nb;
        if (edges > maxLogicalDependencies - result.counts.logicalEdges)
          return {std::nullopt, DependencyError::DependencyLimit};
        if (result.counts.storedConflicts >
            std::numeric_limits<uint64_t>::max() - 2)
          return {std::nullopt, DependencyError::SizeOverflow};
        if (!take(2))
          return {std::nullopt, DependencyError::WorkLimit};
        if (!reserve(32))
          return {std::nullopt, DependencyError::StorageLimit};
        result.conflicts[a].push_back(b);
        result.conflicts[b].push_back(a);
        result.counts.logicalEdges += edges;
        result.counts.storedConflicts += 2;
      }
    result.counts.occurrences = word.size();
    result.counts.classes = axes.size();
    return {std::move(result), DependencyError::None};
  }

  const DependencyStatistics &statistics() const { return counts; }
};

/// Consumption state borrowing an immutable index. The index must outlive the
/// cursor and must not move while borrowed. Occurrences within an axis class
/// may be consumed in any order. The state can describe any consumed subset;
/// consume() does not certify that the removed occurrence was ready.
///
/// Readiness is exact: no conflicting axis has an active occurrence earlier
/// than the queried ID. Queries cost one entry unit plus one per conflict
/// inspected; initialization costs 1+m+K. Consumption costs one entry unit plus
/// one per occurrence inspected while advancing the class head. A failed
/// consumption leaves the active bits and head unchanged, although work already
/// inspected remains charged. Query callers must check the result before using
/// its boolean value. These costs intentionally differ from dense-DAG costs.
class DependencyCursor {
  const DependencyIndex *index;
  std::vector<bool> active;
  std::vector<unsigned> head;
  explicit DependencyCursor(const DependencyIndex &index)
      : index(&index), active(index.occurrenceClass.size(), true),
        head(index.occurrences.size(), 0) {}

public:
  DependencyCursor(const DependencyCursor &) = delete;
  DependencyCursor &operator=(const DependencyCursor &) = delete;
  DependencyCursor(DependencyCursor &&) = default;
  DependencyCursor &operator=(DependencyCursor &&) = default;

  static DependencyResult<DependencyCursor>
  create(const DependencyIndex &index, SearchBudget *budget = nullptr) {
    uint64_t work = 1 + uint64_t(index.occurrenceClass.size()) +
                    uint64_t(index.occurrences.size());
    if (budget && !budget->take(work))
      return {std::nullopt, DependencyError::WorkLimit};
    return {DependencyCursor(index), DependencyError::None};
  }

  DependencyResult<bool> isActive(unsigned id,
                                  SearchBudget *budget = nullptr) const {
    if (budget && !budget->take())
      return {std::nullopt, DependencyError::WorkLimit};
    if (id >= active.size())
      return {std::nullopt, DependencyError::InvalidOccurrence};
    return {bool(active[id]), DependencyError::None};
  }

  DependencyResult<bool> ready(unsigned id,
                               SearchBudget *budget = nullptr) const {
    if (budget && !budget->take())
      return {std::nullopt, DependencyError::WorkLimit};
    if (id >= active.size())
      return {std::nullopt, DependencyError::InvalidOccurrence};
    if (!active[id])
      return {false, DependencyError::None};
    for (unsigned other : index->conflicts[index->occurrenceClass[id]]) {
      if (budget && !budget->take())
        return {std::nullopt, DependencyError::WorkLimit};
      const auto &list = index->occurrences[other];
      if (head[other] < list.size() && list[head[other]] < id)
        return {false, DependencyError::None};
    }
    return {true, DependencyError::None};
  }

  DependencyResult<bool> consume(unsigned id, SearchBudget *budget = nullptr) {
    if (budget && !budget->take())
      return {std::nullopt, DependencyError::WorkLimit};
    if (id >= active.size() || !active[id])
      return {std::nullopt, DependencyError::InvalidOccurrence};
    unsigned axis = index->occurrenceClass[id];
    unsigned next = head[axis];
    const auto &list = index->occurrences[axis];
    if (list[next] == id) {
      ++next;
      while (next < list.size()) {
        if (budget && !budget->take())
          return {std::nullopt, DependencyError::WorkLimit};
        if (active[list[next]])
          break;
        ++next;
      }
    }
    active[id] = false;
    head[axis] = next;
    return {true, DependencyError::None};
  }
};

} // namespace cudaq::pauli::graph::detail
