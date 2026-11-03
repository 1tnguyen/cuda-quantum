/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Private interfaces shared by the T-count optimization strategies.
///
/// Declares exact algebra, Clifford frames, budget accounting, rewrite
/// strategies, and witness checks. Keeping these helpers separate from the
/// public API lets search and verification share conventions without exposing
/// implementation data.
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Pauli/TCount.h"
#include "cudaq/Optimizer/Pauli/Telemetry.h"
#include <chrono>
#include <optional>
#include <unordered_map>

namespace cudaq::pauli::tcount::detail {
using namespace cudaq::pauli;

std::optional<int> quarters(Angle);
Gate inverse(Gate);
Pauli product(const Pauli &, const Pauli &); // Commuting Hermitian inputs only.
Pauli conjugate(Pauli, const Gate &); // g P g^dagger; g must be Clifford.
bool isClifford(const Gate &);
bool valid(unsigned width, const std::vector<Gate> &);
std::vector<Gate> adjoint(const std::vector<Gate> &);
std::vector<Gate> materialize(const std::vector<Gate> &);

/// Unsigned Pauli support key; rotation coefficients carry the omitted sign.
struct Axis {
  /// Per-wire X and Z masks, with overlapping bits representing Y.
  QubitSet x, z;
  bool operator==(const Axis &) const = default;
};
/// Hash both support masks for matching-axis lookup tables.
struct AxisHash {
  size_t operator()(const Axis &a) const {
    size_t x = a.x.hash();
    return x ^ (a.z.hash() + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2));
  }
};

/// Images of X_i,Z_i under F^dagger P F; signs are authoritative.
class Frame {
  /// Number of represented wires.
  unsigned width;
  /// Signed generator images: all X images followed by all Z images.
  std::vector<Pauli> rows;

public:
  explicit Frame(unsigned);
  Pauli lookup(const Pauli &) const;
  Pauli inverseLookup(const Pauli &) const;
  void append(const Gate &);
};

/// Enforce T-call work/deadline limits together with an optional shared budget.
class Budget {
  /// Borrowed per-call limits; must outlive this budget.
  const Options &options;
  /// Borrowed counters updated by successful charges and exhaustion.
  Statistics &stats;
  /// Monotonic start of this T invocation's local deadline.
  std::chrono::steady_clock::time_point start;
  /// Sticky stop flag after either local or shared budget failure.
  bool exhausted = false;
  /// Optional enclosing budget, charged in addition to the local allowance.
  SearchBudget *shared;

public:
  /// Optional active strategy proof timer; never affects resource charging.
  uint64_t *verificationNs = nullptr;
  Budget(const Options &o, Statistics &s, SearchBudget *shared = nullptr)
      : options(o), stats(s), start(std::chrono::steady_clock::now()),
        shared(shared) {}
  bool take(uint64_t amount = 1);
  bool stopped() const { return exhausted; }
};

std::vector<Gate> cancelCliffords(const std::vector<Gate> &, Budget &);

/// Independent local certificate: transport the earlier rotation through
/// the CURRENT intervening circuit. No hash-table or cached frame is trusted.
/// Returns its signed coefficient at the later rotation, or no proof.
std::optional<int> verifyMerge(const std::vector<Gate> &, size_t first,
                               size_t last, Budget &, Statistics &);

std::vector<Gate> merge(const std::vector<Gate> &, unsigned width,
                        const Options &, Budget &, Statistics &);
std::vector<Gate> decode(const std::vector<Gate> &, unsigned width,
                         const Options &, Budget &, Statistics &);

/// Exact equality of commuting-rotation eigenphases in Z/16 for ALL
/// assignments of independent generator eigenvalues. Separate from decoding.
bool verifyPhases(const std::vector<int> &before, const std::vector<int> &after,
                  int scalar, Budget &);
} // namespace cudaq::pauli::tcount::detail
