/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Lossless unitary and released terminal-measurement normalization.
///
/// Declares Clifford-frame absorption and commuting-observable validation. The
/// normalization contract preserves joint measurement probabilities for
/// arbitrary input states; the final post-measurement quantum state is not an
/// output.
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Pauli/Circuit.h"
#include "cudaq/Optimizer/Pauli/SearchBudget.h"
#include <string>
#include <vector>

namespace cudaq::pauli::pbc {
/// A basis of observable supports. Checking against the basis is sufficient
/// for pairwise commutation and avoids quadratic work on repeated measurements.
class CommutingObservables {
  /// Independent support rows, growing with the observable rank.
  std::vector<Pauli> rows;
  /// Rank of the stored basis, not the number of inserted measurements.
  unsigned count = 0;

public:
  bool insert(Pauli);
  unsigned size() const { return count; }
};

/// Normalized PBC form or failure; consume the output only on success.
struct Result {
  /// Ordered Pauli rotations with the Clifford frame absorbed.
  std::vector<Gate> rotations;
  /// Signed terminal observables, preserving input measurement order.
  std::vector<Pauli> measurements;
  /// Empty on success; otherwise describes the normalization failure.
  std::string error;
  explicit operator bool() const { return error.empty(); }
};

/// Lossless Pauli representation, in circuit order: rotations, then frame.
/// The actual Clifford/phase word fixes the scalar absent from a tableau.
/// It is a witness of the required action, not a network to preserve.
struct UnitaryForm {
  std::vector<Gate> rotations;
  std::vector<Gate> frame;
  std::string error;
  explicit operator bool() const { return error.empty(); }
};

/// Preserve the entire input unitary, including scalar phase and wire order.
/// No preparation, terminal-measurement, or initial-state assumption is used.
UnitaryForm normalizeUnitary(unsigned width, const std::vector<Gate> &,
                             SearchBudget *budget = nullptr);

/// Absorb Clifford operations into rotations and terminal joint measurements.
/// Preserves the ordered joint measurement probabilities for every input state;
/// scalar phase and the final quantum state after measurement are not outputs.
/// Arbitrary rational and opaque angles are retained without approximation.
/// Only commuting terminal observables are accepted. No state assumptions.
Result normalize(unsigned width, const std::vector<Gate> &,
                 const std::vector<Pauli> &, SearchBudget *budget = nullptr);
} // namespace cudaq::pauli::pbc
