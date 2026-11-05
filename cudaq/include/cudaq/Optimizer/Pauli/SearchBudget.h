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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>

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
  /// Optional enclosing allowance. The parent must outlive this child.
  /// Local exhaustion never stops the parent or its other children.
  SearchBudget *parent = nullptr;
  /// Monotonic origin for deadline checks.
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

public:
  SearchBudget(uint64_t limit, unsigned timeoutMs = 0)
      : limit(limit), timeoutMs(timeoutMs) {}
  /// Bounded child slice. Successful work is charged once to every ancestor;
  /// failure at any level charges no work at any level. A local limit/deadline
  /// stops only this child, leaving the parent's proof/completion reserve
  /// usable.
  SearchBudget(SearchBudget &parent, uint64_t localLimit,
               unsigned localTimeoutMs = 0)
      : limit(localLimit), timeoutMs(localTimeoutMs), parent(&parent) {}
  bool take(uint64_t amount = 1) {
    if (exhausted || amount > limit - used ||
        (timeoutMs && std::chrono::steady_clock::now() - start >=
                          std::chrono::milliseconds(timeoutMs))) {
      exhausted = true;
      return false;
    }
    // Validate this level before asking the parent. The parent recursively
    // validates its own allowance before charging anything, so failure is
    // transactional even for nested slices. The subtraction above also ensures
    // the successful addition below cannot overflow.
    if (parent && !parent->take(amount)) {
      exhausted = true;
      return false;
    }
    used += amount;
    return true;
  }
  uint64_t work() const { return used; }
  bool stopped() const { return exhausted || (parent && parent->stopped()); }
  /// Available work, including enclosing allowances. Deadlines are reported
  /// separately; like take(), a previously stopped budget has no usable work.
  uint64_t remaining() const {
    if (stopped())
      return 0;
    uint64_t available = limit - used;
    return parent ? std::min(available, parent->remaining()) : available;
  }
  /// Minimum remaining local/ancestor deadline, or no value if all are
  /// unlimited. Round a positive remainder upward so zero means expired.
  /// Merely observing time does not stop a budget; take() records exhaustion.
  std::optional<unsigned> timeRemainingMs() const {
    std::optional<unsigned> remaining;
    if (timeoutMs) {
      auto elapsed = std::chrono::steady_clock::now() - start;
      auto allowed = std::chrono::milliseconds(timeoutMs);
      remaining = elapsed >= allowed
                      ? 0u
                      : static_cast<unsigned>(
                            std::chrono::ceil<std::chrono::milliseconds>(
                                allowed - elapsed)
                                .count());
    }
    if (parent)
      if (auto enclosing = parent->timeRemainingMs())
        remaining = remaining ? std::min(*remaining, *enclosing) : *enclosing;
    return remaining;
  }
};
} // namespace cudaq::pauli
