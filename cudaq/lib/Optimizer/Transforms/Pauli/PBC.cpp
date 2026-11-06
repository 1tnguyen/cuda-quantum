/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Clifford-frame absorption for terminal-measurement PBC.
///
/// Validates commuting observables and conjugates rotations and measurements
/// into a common frame. Normalization preserves the ordered joint outcome
/// distribution without assuming a particular input state.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Pauli/PBC.h"
#include "TCountInternal.h"
#include <bit>
#include <limits>
#include <numeric>

using namespace cudaq::pauli;

pbc::UnitaryForm pbc::normalizeUnitary(unsigned width,
                                       const std::vector<Gate> &gates,
                                       SearchBudget *budget) {
  UnitaryForm result;
  if (!tcount::detail::valid(width, gates)) {
    result.error = "unitary PBC normalization requires valid gates";
    return result;
  }
  const uint64_t work = uint64_t(width) * width + 1;
  if (budget && !budget->take(work)) {
    result.error = "unitary PBC normalization work budget exhausted";
    return result;
  }
  tcount::detail::Frame frame(width);
  for (auto gate : gates) {
    if (budget && !budget->take(work)) {
      result.error = "unitary PBC normalization work budget exhausted";
      return result;
    }
    if (gate.kind == Kind::Rotate && !gate.pauli.weight()) {
      if (gate.angle.symbol) {
        result.error = "opaque identity rotation requires a scalar-phase angle";
        return result;
      }
      __int128 numerator = __int128(gate.angle.numerator) *
                           (gate.pauli.sign != gate.angle.negative ? 1 : -1);
      __int128 denominator = __int128(gate.angle.denominator) * 2;
      // The input denominator is positive. Reduce before narrowing the half.
      auto magnitude = numerator < 0 ? -numerator : numerator;
      while (denominator) {
        auto remainder = magnitude % denominator;
        magnitude = denominator;
        denominator = remainder;
      }
      denominator = __int128(gate.angle.denominator) * 2 / magnitude;
      numerator /= magnitude;
      if (numerator < std::numeric_limits<int64_t>::min() ||
          numerator > std::numeric_limits<int64_t>::max() ||
          denominator > std::numeric_limits<int64_t>::max()) {
        result.error = "identity rotation scalar phase exceeds exact range";
        return result;
      }
      result.frame.push_back(
          Gate::phase({int64_t(numerator), int64_t(denominator)}));
    } else if (gate.kind != Kind::Rotate) {
      // Keep exact Clifford-angle rotations in the Pauli word as well. This
      // matches the independent unitary replay's primitive-Clifford frame;
      // absorption must not exceed the checker's supported normalization.
      frame.append(gate);
      result.frame.push_back(gate);
    } else {
      gate.pauli = frame.lookup(gate.pauli);
      if (gate.pauli.sign) {
        gate.pauli.sign = false;
        gate.angle = gate.angle.negated();
      }
      result.rotations.push_back(gate);
    }
  }
  return result;
}

/// Check that word commutes with the accumulated observable span, then extend
/// its binary support basis if word is independent. Return false without
/// changing the basis on anticommutation. Dependent words (including identity)
/// succeed without increasing the rank; normalize() rejects identity inputs.
/// Signs are irrelevant to this check: rows encode supports, not measurement
/// outcomes or signed products of observables.
bool pbc::CommutingObservables::insert(Pauli word) {
  // The symplectic commutation test is linear in support bits. Commutation
  // with every basis row therefore implies commutation with every observable
  // previously inserted, including those omitted as linearly dependent.
  for (unsigned i = 0; i < count; ++i)
    if (word.anti(rows[i]))
      return false;
  word.sign = false;
  // Eliminate pivots in insertion order, treating X bits before Z bits. Each
  // stored row has zero entries at earlier rows' pivots, so an XOR cannot
  // reintroduce an already eliminated pivot. Physical Pauli-product phases
  // need not be tracked for this support-only independence calculation.
  for (unsigned i = 0; i < count; ++i) {
    const auto &row = rows[i];
    bool set =
        row.x ? word.x.test(firstBit(row.x)) : word.z.test(firstBit(row.z));
    if (set) {
      word.x ^= row.x;
      word.z ^= row.z;
    }
  }
  // A nonzero remainder contributes one independent generator. Keep this
  // compressed basis separate from the caller's ordered measurement list.
  if (word.x | word.z)
    rows.push_back(std::move(word));
  count = rows.size();
  return true;
}

/// Normalize a validated native circuit to ordered Pauli rotations followed by
/// commuting terminal observables. Accumulate Cliffords in a frame F and
/// replace each subsequent axis P by F^dagger P F. Absorb the final frame into
/// the observables instead of reconstructing it as gates.
///
/// This preserves joint outcome probabilities for arbitrary input states, not
/// the output unitary or post-measurement quantum state. Scalar phases may be
/// discarded under that contract. Rotation angles remain exact or opaque; no
/// numerical synthesis, rotation merging, or entangler search occurs here.
/// Inputs are never modified. Invalid input or exhausted work returns an error;
/// any partially populated result must be discarded by the caller.
pbc::Result pbc::normalize(unsigned width, const std::vector<Gate> &gates,
                           const std::vector<Pauli> &measurements,
                           SearchBudget *budget) {
  Result result;
  // Validate all gate kinds, supports, operands, and rational denominators
  // before the dispatch below assumes that every non-Clifford is a rotation.
  if (!tcount::detail::valid(width, gates)) {
    result.error =
        "PBC normalization requires valid gates on a nonzero wire count";
    return result;
  }
  auto charge = [&](uint64_t work) {
    if (!budget || budget->take(work))
      return true;
    result.error = "PBC normalization work budget exhausted";
    return false;
  };
  // Terminal observables must admit a joint measurement. Check this using an
  // independent support basis, but retain duplicates, signs, and input order
  // in the output: they identify distinct requested classical results.
  CommutingObservables observed;
  for (auto m : measurements) {
    if (!(m.x | m.z) || (!(m.x | m.z).fits(width))) {
      result.error = "PBC measurement must have nonempty in-range support";
      return result;
    }
    if (!charge(2 * observed.size() + 1))
      return result;
    if (!observed.insert(m)) {
      result.error = "PBC terminal measurements must commute";
      return result;
    }
  }
  if (gates.empty() && measurements.empty())
    return result;
  // Charge the first frame operation before allocating the quadratic tableau.
  // A bounded caller must be able to reject a wide island without building it.
  const uint64_t frameWork = uint64_t(width) * width + 1;
  if (!charge(frameWork))
    return result;
  bool first = true;
  // The preallocation charge also covers the first gate/observable visit;
  // subsequent visits consume the same conservative frame-work allowance.
  auto chargeFrame = [&] {
    if (first) {
      first = false;
      return true;
    }
    return charge(frameWork);
  };
  tcount::detail::Frame frame(width);
  for (auto gate : gates) {
    if (!chargeFrame())
      return result;
    // A scalar phase cancels from every measurement probability. This omission
    // is specific to this terminal-measurement contract, not unitary replay.
    if (gate.kind == Kind::Phase)
      continue;
    // Absorb both primitive Cliffords and exact Clifford-angle rotations.
    // Opaque angles cannot be classified as Clifford from their SSA identity.
    if (tcount::detail::isClifford(gate)) {
      frame.append(gate);
      continue;
    }
    // Moving the accumulated Clifford past a rotation changes its axis to
    // F^dagger P F. Preserve the order of the remaining rotations, since they
    // need not commute. The lookup includes signs introduced by conjugation.
    gate.pauli = frame.lookup(gate.pauli);
    if (!(gate.pauli.x | gate.pauli.z))
      continue; // An identity rotation contributes only scalar phase.
    if (gate.pauli.sign) {
      // R(-P, theta) = R(P, -theta); retain unsigned axes in normalized PBC.
      gate.angle = gate.angle.negated();
      gate.pauli.sign = false;
    }
    result.rotations.push_back(gate);
  }
  // Measuring M after F is equivalent in outcome probabilities to measuring
  // F^dagger M F before it. A common Clifford preserves mutual commutation.
  // Keep each transformed sign: negating an observable swaps its outcome
  // labels.
  for (auto m : measurements) {
    if (!chargeFrame())
      return result;
    result.measurements.push_back(frame.lookup(m));
  }
  return result;
}
