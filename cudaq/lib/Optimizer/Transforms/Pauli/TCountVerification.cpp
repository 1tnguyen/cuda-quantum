/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Exact witness checks for Pauli rotation rewrites.
///
/// Replays the transport of a proposed merged axis and compares decoded phase
/// functions, including scalar corrections, on every generator assignment.
/// Verification shares the search budget and rejects incomplete checks.
//===----------------------------------------------------------------------===//

#include "TCountInternal.h"
#include <bit>

namespace cudaq::pauli::tcount::detail {
/// Prove the first rotation can move to the last and return its coefficient
/// in the last axis's sign convention. An empty result means no proof,
/// including budget exhaustion; it must never authorize a partial merge.
std::optional<int> verifyMerge(const std::vector<Gate> &gates, size_t first,
                               size_t last, Budget &budget, Statistics &stats) {
  TelemetryTimer timer(budget.verificationNs);
  if (first >= last || last >= gates.size() ||
      gates[first].kind != Kind::Rotate || gates[last].kind != Kind::Rotate)
    return {};
  auto k = quarters(gates[first].angle);
  if (!k || !quarters(gates[last].angle))
    return {};
  Pauli axis = gates[first].pauli;
  // Move the earlier rotation forward in circuit order. Cliffords change its
  // axis by conjugation; other rotations can be crossed only if they commute
  // with the axis at that point, regardless of their angle representation.
  for (size_t i = first + 1; i < last; ++i) {
    if (!budget.take())
      return {};
    auto g = gates[i];
    if (isClifford(g)) {
      axis = conjugate(axis, g);
    } else {
      ++stats.commutationChecks;
      if (axis.anti(g.pauli))
        return {};
    }
  }
  Pauli target = gates[last].pauli;
  if (axis.x != target.x || axis.z != target.z)
    return {};
  return axis.sign == target.sign ? *k : -*k;
}
/// Compare exact eigenphases for every assignment of independent commuting
/// generators. Coefficients are in pi/4 units; scalar is in pi/8 units.
/// This checks the reconstructed expression, independently of the coset search.
bool verifyPhases(const std::vector<int> &before, const std::vector<int> &after,
                  int scalar, Budget &budget) {
  TelemetryTimer timer(budget.verificationNs);
  if (before.size() != after.size() || !std::has_single_bit(before.size()))
    return false;
  for (unsigned assignment = 0; assignment < before.size(); ++assignment) {
    int lhs = 0, rhs = scalar;
    for (unsigned axis = 1; axis < before.size(); ++axis) {
      if (!budget.take())
        return false;
      // A subset product has eigenvalue (-1)^parity(axis & assignment).
      // R_P(k*pi/4) contributes -k*eigenvalue in pi/8 phase units.
      int sign = (std::popcount(axis & assignment) & 1) ? -1 : 1;
      lhs -= before[axis] * sign;
      rhs -= after[axis] * sign;
    }
    // Equality modulo 2*pi is modulo 16 in these units. Including scalar
    // distinguishes exact equality from equality only up to global phase.
    if ((lhs - rhs) % 16)
      return false;
  }
  return true;
}
} // namespace cudaq::pauli::tcount::detail
