/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Network-preserving total-gate optimization over native Pauli
/// circuits.
///
/// Combines signed Pauli propagation, exact phase placement and bounded
/// Clifford windows. All transformations preserve the unitary, including scalar
/// phase; the caller chooses whether a cheaper native network survives final
/// lowering.
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Pauli/Circuit.h"
#include "cudaq/Optimizer/Pauli/SearchBudget.h"
#include "cudaq/Optimizer/Pauli/Telemetry.h"
#include <limits>
#include <vector>

namespace cudaq::pauli::gatecount {
/// Independent controls for the total-gate strategy family.
struct Options {
  /// Maximum greedy composition rounds; each round retains the original.
  unsigned rounds = 3;
  /// Maximum backward distance for exact local cancellation; zero is unbounded.
  unsigned commutationWindow = 128;
  /// Explore forward and adjoint signed Pauli-frame propagation.
  bool propagate = true;
  /// Place combined exact Z phases at first and last existing parity locations.
  bool phasePlacement = true;
  /// Minimize contiguous one-/two-wire Clifford windows in the native basis.
  bool cliffordWindows = true;
};

/// Return a verified, non-T-increasing total-gate candidate. The shared budget
/// covers scans and checks; exhaustion returns the input, never partial output.
/// Candidate size is bounded before expanding rotations into parity gadgets.
/// An absolute entangler ceiling preserves admissible alternatives when a
/// shorter Clifford window would otherwise exceed the caller's growth limit.
std::vector<Gate>
optimize(unsigned width, const std::vector<Gate> &, const Options &,
         SearchBudget &, unsigned maxGates, Telemetry * = nullptr,
         uint64_t parent = 0, uint64_t scope = 0, uint64_t *selected = nullptr,
         uint64_t *verificationNs = nullptr,
         uint64_t maxTwoQubitCount = std::numeric_limits<uint64_t>::max());
} // namespace cudaq::pauli::gatecount
