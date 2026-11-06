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
  uint64_t work = 0, independentGates = 0;
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
/// without a source Clifford-network seed or a heuristic search. Neighboring
/// gadgets share a checked common Clifford prefix. Each residual basis/parity
/// change is undone before an incompatible change, and the final frame is
/// restored. Thus the full unitary and original terminal observables are
/// unchanged. This function validates the word/profile, not the terminal
/// readout contract; the caller must validate that contract and independently
/// check prefix+tail. Native Clifford input is unsupported. Explicit scalar
/// phases are preserved.
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
  // Validate the whole source first. The independent size is diagnostic, not
  // an allocation or admission bound: sharing can make a previously oversized
  // independent reconstruction fit the configured output cap.
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
        count += 2 * uint64_t(gate.pauli.x.count()) +
                 2 * uint64_t(gate.pauli.x.intersectionCount(gate.pauli.z)) +
                 2 * uint64_t(weight - 1) * (options.nativeCX ? 1 : 3);
      }
    }
    if (count > std::numeric_limits<uint64_t>::max() - result.independentGates)
      return unsupported("bootstrap independent size overflow");
    result.independentGates += count;
  }
  std::vector<Gate> gates;
  uint64_t outputSize = 0;
  auto inverse = [](Gate gate) {
    if (gate.kind == Kind::S)
      gate.kind = Kind::Sdg;
    else if (gate.kind == Kind::Sdg)
      gate.kind = Kind::S;
    return gate;
  };
  auto same = [](const Gate &a, const Gate &b) {
    return a.kind == b.kind && a.a == b.a && a.b == b.b;
  };
  // Count the actual shared word before allocating it. Both traversals retain
  // only two O(width) Clifford decompositions, never the independent expansion.
  auto reconstruct = [&](bool emit) {
    std::vector<Gate> previous;
    std::vector<unsigned> previousOrder;
    Pauli previousAxis;
    unsigned previousPivot = problem.width;
    uint64_t count = 0;
    auto append = [&](Gate gate) {
      if (!charge(1))
        return false;
      if (count == options.maxGates) {
        result.status = BootstrapStatus::Exhausted;
        result.error = "bootstrap prefix native gate limit exceeded";
        return false;
      }
      ++count;
      if (emit)
        gates.push_back(std::move(gate));
      return true;
    };
    for (size_t index = 0; index < problem.rotations.size(); ++index) {
      const auto &gate = problem.rotations[index];
      if (bootstrap_detail::zero(gate.angle))
        continue;
      if (!charge(2 * rowWork))
        return false;
      if (gate.kind == Kind::Phase) {
        if (!append(Gate::phase(gate.angle)))
          return false;
        continue;
      }
      auto support = gate.pauli.x | gate.pauli.z;
      if (!support) {
        if (!append(Gate::phase(*bootstrap_detail::identityPhase(gate))))
          return false;
        continue;
      }
      const Pauli *next = nullptr;
      for (size_t j = index + 1; j < problem.rotations.size(); ++j) {
        if (!charge(1))
          return false;
        const auto &following = problem.rotations[j];
        if (following.kind == Kind::Rotate && following.pauli.weight() &&
            !bootstrap_detail::zero(following.angle)) {
          next = &following.pauli;
          break;
        }
      }
      unsigned pivot = support.first();
      if (previousPivot < problem.width &&
          gate.pauli.axis(previousPivot) == previousAxis.axis(previousPivot))
        pivot = previousPivot;
      else if (next)
        for (unsigned q : support.set_bits())
          if (gate.pauli.axis(q) == next->axis(q)) {
            pivot = q;
            break;
          }
      std::vector<Gate> compute;
      std::vector<unsigned> order;
      auto basis = [&](unsigned q) {
        unsigned axis = gate.pauli.axis(q);
        if (axis == 3)
          compute.push_back({Kind::Sdg, q});
        if (axis & 1)
          compute.push_back({Kind::H, q});
      };
      basis(pivot);
      QubitSet used = QubitSet::single(pivot);
      auto control = [&](unsigned q) {
        basis(q);
        if (options.nativeCX)
          compute.push_back({Kind::CX, q, pivot});
        else {
          compute.push_back({Kind::H, pivot});
          compute.push_back({Kind::CZ, q, pivot});
          compute.push_back({Kind::H, pivot});
        }
        used.set(q);
        order.push_back(q);
      };
      // Preserve an already shared prefix; place future-compatible controls
      // first in the remaining suffix. Local basis changes on distinct
      // controls commute with all other same-target parity CNOTs.
      if (pivot == previousPivot)
        for (unsigned q : previousOrder) {
          if (!gate.pauli.axis(q) || gate.pauli.axis(q) != previousAxis.axis(q))
            break;
          control(q);
        }
      for (bool compatible : {true, false})
        for (unsigned q : support.set_bits()) {
          if (used.test(q))
            continue;
          bool shared = next && gate.pauli.axis(q) == next->axis(q);
          if (shared == compatible)
            control(q);
        }
      if (!charge(compute.size() + previous.size() + 1))
        return false;
      size_t common = 0;
      while (common < previous.size() && common < compute.size() &&
             same(previous[common], compute[common]))
        ++common;
      for (size_t j = previous.size(); j > common; --j)
        if (!append(inverse(previous[j - 1])))
          return false;
      for (size_t j = common; j < compute.size(); ++j)
        if (!append(compute[j]))
          return false;
      Angle angle = gate.pauli.sign ? gate.angle.negated() : gate.angle;
      if (!append(Gate::rotation({{}, QubitSet::single(pivot), false}, angle)))
        return false;
      previous = std::move(compute);
      previousOrder = std::move(order);
      previousAxis = gate.pauli;
      previousPivot = pivot;
    }
    for (auto it = previous.rbegin(); it != previous.rend(); ++it)
      if (!append(inverse(*it)))
        return false;
    if (!emit)
      outputSize = count;
    return true;
  };
  if (!reconstruct(false) || !charge(outputSize))
    return result;
  gates.reserve(outputSize);
  if (!reconstruct(true))
    return result;
  result.gates = std::move(gates);
  return result;
}

} // namespace cudaq::pauli::graph::detail
