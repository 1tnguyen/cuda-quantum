/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Shared resource accounting for Pauli optimization and verification.
///
/// Provides a deterministic work limit and an optional wall-clock deadline.
/// Strategies and their proof checks charge the same budget so nested searches
/// cannot silently exceed the caller's bound.
//===----------------------------------------------------------------------===//

#pragma once

#include <chrono>
#include <cstdint>

namespace cudaq::pauli {

/// Shared deterministic work bound and optional wall deadline for one island.
/// A failed charge leaves the previous verified incumbent available.
class SearchBudget {
  /// Maximum work allowance and successfully charged units, respectively.
  uint64_t limit, used = 0;
  /// Deadline relative to start, in milliseconds; zero disables the timer.
  unsigned timeoutMs;
  /// Sticky failure flag after a work limit or deadline rejects a charge.
  bool exhausted = false;
  /// Monotonic origin for deadline checks.
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

public:
  SearchBudget(uint64_t limit, unsigned timeoutMs = 0)
      : limit(limit), timeoutMs(timeoutMs) {}
  bool take(uint64_t amount = 1) {
    if (exhausted || amount > limit - used ||
        (timeoutMs && std::chrono::steady_clock::now() - start >=
                          std::chrono::milliseconds(timeoutMs))) {
      exhausted = true;
      return false;
    }
    used += amount;
    return true;
  }
  uint64_t work() const { return used; }
  bool stopped() const { return exhausted; }
};
} // namespace cudaq::pauli
