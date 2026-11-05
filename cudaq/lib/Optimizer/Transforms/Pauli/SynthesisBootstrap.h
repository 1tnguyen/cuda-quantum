/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SynthesisCandidates.h"
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::pauli::graph::detail {

enum class BootstrapStatus { Complete, Unsupported, Exhausted };

struct BootstrapResult {
  /// A complete full-unitary implementation of the supplied rotation word.
  /// This is a proposal, not a VerifiedCandidate; readout and independent
  /// replay remain the caller's responsibility. Empty on every failure.
  std::vector<Gate> gates;
  BootstrapStatus status = BootstrapStatus::Complete;
  uint64_t work = 0;
  std::string error;
  explicit operator bool() const { return status == BootstrapStatus::Complete; }
};

namespace bootstrap_detail {
inline bool zero(Angle angle) { return !angle.symbol && !angle.numerator; }

/// R(+/-I,theta) is exp(-/+i theta/2). Opaque half-angles have no encoding in
/// Angle. Reduce rational coefficients before halving so a large denominator
/// is rejected only when its exact reduced half cannot be represented.
inline std::optional<Angle> identityPhase(const Gate &gate) {
  if (gate.angle.symbol || gate.angle.denominator <= 0)
    return std::nullopt;
  Angle angle = gate.pauli.sign ? gate.angle : gate.angle.negated();
  uint64_t magnitude = angle.numerator < 0
                           ? uint64_t(-(angle.numerator + 1)) + 1
                           : uint64_t(angle.numerator);
  uint64_t divisor = std::gcd(magnitude, uint64_t(angle.denominator));
  angle.numerator /= static_cast<int64_t>(divisor);
  angle.denominator /= static_cast<int64_t>(divisor);
  if (angle.numerator % 2 == 0)
    angle.numerator /= 2;
  else if (angle.denominator <= std::numeric_limits<int64_t>::max() / 2)
    angle.denominator *= 2;
  else
    return std::nullopt;
  return angle;
}
} // namespace bootstrap_detail

/// Produce a first feasible prefix directly from normalized Pauli rotations,
/// without a source Clifford-network seed or a heuristic search. Each gadget
/// changes local bases, computes parity, rotates one Z axis, then restores
/// parity and bases. Thus the original terminal observables remain unchanged.
/// This function validates the word/profile, not the terminal readout contract;
/// the caller must validate that contract and independently check prefix+tail.
/// Native Clifford input is unsupported. Explicit scalar phases are preserved.
///
/// Preflight computes the exact output size before any output allocation. Work
/// units charge one entry, one width-in-limbs scan per source operation, one
/// unit per reserved output slot, two limb scans per emitted source operation,
/// and one unit per appended gate. All copying/allocation follows its charge;
/// per-gate charges also poll the shared deadline. Storage is O(output gates)
/// plus one Pauli support row; no width-sized tableau is constructed here.
inline BootstrapResult buildBootstrapPrefix(const MeasuredProblem &problem,
                                            const Options &options,
                                            SearchBudget *budget = nullptr) {
  BootstrapResult result;
  auto charge = [&](uint64_t amount) {
    if (budget && !budget->take(amount)) {
      result.status = BootstrapStatus::Exhausted;
      result.error = "bootstrap prefix work budget exhausted";
      return false;
    }
    result.work = amount > std::numeric_limits<uint64_t>::max() - result.work
                      ? std::numeric_limits<uint64_t>::max()
                      : result.work + amount;
    return true;
  };
  auto unsupported = [&](const char *message) {
    result.status = BootstrapStatus::Unsupported;
    result.error = message;
    return result;
  };
  if (!problem.width || (!options.nativeCX && !options.nativeCZ))
    return unsupported("bootstrap requires a positive width and native CX/CZ");
  if (!charge(1))
    return result;
  const uint64_t rowWork = uint64_t(problem.width) / 64 + 1;
  uint64_t outputSize = 0;
  for (const auto &gate : problem.rotations) {
    if (!charge(rowWork))
      return result;
    if (gate.kind != Kind::Rotate && gate.kind != Kind::Phase)
      return unsupported("bootstrap requires normalized rotations and phases");
    if (!gate.angle.symbol && gate.angle.denominator <= 0)
      return unsupported("bootstrap requires valid exact or opaque angles");
    if (gate.kind == Kind::Rotate && (!gate.pauli.x.fits(problem.width) ||
                                      !gate.pauli.z.fits(problem.width)))
      return unsupported("bootstrap rotation support exceeds its width");
    if (bootstrap_detail::zero(gate.angle))
      continue;
    uint64_t count = 1;
    if (gate.kind == Kind::Rotate) {
      unsigned weight = gate.pauli.weight();
      if (!weight) {
        if (!bootstrap_detail::identityPhase(gate))
          return unsupported(
              "bootstrap identity half-angle is not representable");
      } else {
        uint64_t basis =
            2 * uint64_t(gate.pauli.x.count()) +
            2 * uint64_t(gate.pauli.x.intersectionCount(gate.pauli.z));
        uint64_t parity = 2 * uint64_t(weight - 1) * (options.nativeCX ? 1 : 3);
        count += basis + parity;
      }
    }
    if (count > uint64_t(options.maxGates) - outputSize) {
      result.status = BootstrapStatus::Exhausted;
      result.error = "bootstrap prefix native gate limit exceeded";
      return result;
    }
    outputSize += count;
  }
  if (!charge(outputSize))
    return result;
  std::vector<Gate> gates;
  gates.reserve(outputSize);
  auto append = [&](Gate gate) {
    if (!charge(1))
      return false;
    gates.push_back(std::move(gate));
    return true;
  };
  auto cx = [&](unsigned control, unsigned target) {
    if (options.nativeCX)
      return append({Kind::CX, control, target});
    return append({Kind::H, target}) && append({Kind::CZ, control, target}) &&
           append({Kind::H, target});
  };
  for (const auto &gate : problem.rotations) {
    if (bootstrap_detail::zero(gate.angle))
      continue;
    // Covers support copying and construction of the one-wire Z rotation.
    if (!charge(2 * rowWork))
      return result;
    if (gate.kind == Kind::Phase) {
      if (!append(Gate::phase(gate.angle)))
        return result;
      continue;
    }
    auto support = gate.pauli.x | gate.pauli.z;
    if (!support) {
      if (!append(Gate::phase(*bootstrap_detail::identityPhase(gate))))
        return result;
      continue;
    }
    unsigned pivot = support.first();
    for (unsigned q : support.set_bits()) {
      unsigned axis = gate.pauli.axis(q);
      if (axis == 3 && !append({Kind::Sdg, q}))
        return result;
      if ((axis & 1) && !append({Kind::H, q}))
        return result;
    }
    for (unsigned q : support.set_bits())
      if (q != pivot && !cx(q, pivot))
        return result;
    Angle angle = gate.pauli.sign ? gate.angle.negated() : gate.angle;
    if (!append(Gate::rotation({{}, QubitSet::single(pivot), false}, angle)))
      return result;
    // Same-target CNOTs commute and are self-inverse, so repeating this order
    // exactly uncomputes parity. Local basis changes on distinct wires commute.
    for (unsigned q : support.set_bits())
      if (q != pivot && !cx(q, pivot))
        return result;
    for (unsigned q : support.set_bits()) {
      unsigned axis = gate.pauli.axis(q);
      if ((axis & 1) && !append({Kind::H, q}))
        return result;
      if (axis == 3 && !append({Kind::S, q}))
        return result;
    }
  }
  result.gates = std::move(gates);
  return result;
}

} // namespace cudaq::pauli::graph::detail
