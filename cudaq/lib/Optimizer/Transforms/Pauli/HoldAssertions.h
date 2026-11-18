// Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "llvm/Support/ErrorHandling.h"
#include <cstdlib>
#include <cstring>

namespace cudaq::pauli::construction {
// Scoped to a HOLD invocation, including its native refinement backends.
// Other callers retain their existing independent verification policy.
inline thread_local unsigned holdDepth = 0;
inline thread_local bool holdAssertions = false;
class HoldScope {
public:
  HoldScope() {
    if (!holdDepth++) {
      const char *value = std::getenv("CUDAQ_PBC_HOLD_ASSERT");
      if (value && std::strcmp(value, "0") && std::strcmp(value, "1"))
        llvm::report_fatal_error("CUDAQ_PBC_HOLD_ASSERT must be 0 or 1");
      holdAssertions = value && !std::strcmp(value, "1");
    }
  }
  ~HoldScope() { --holdDepth; }
  HoldScope(const HoldScope &) = delete;
  HoldScope &operator=(const HoldScope &) = delete;
};
inline bool checksEnabled() { return !holdDepth || holdAssertions; }
/// Assertions remain active in release builds. They are independent of the
/// exact construction rules and cannot silently reject a buggy candidate.
template <class Check>
bool check(Check &&test) {
  if (!checksEnabled())
    return true;
  bool ok = test();
  if (!ok && holdDepth)
    llvm::report_fatal_error("PBC HOLD unitary construction assertion failed");
  return ok;
}
} // namespace cudaq::pauli::construction
