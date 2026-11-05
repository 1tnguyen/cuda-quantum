/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SynthesisCandidates.h"
#include "TCountInternal.h"
#include "cudaq/Optimizer/Pauli/PBC.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::pauli::graph::detail {

struct AlgebraLimits {
  unsigned maxCandidates = 4;
  unsigned maxInputGates = 32768;
  unsigned maxTemporaryGates = 1048576;
  unsigned maxWidth = 4096;
  uint64_t maxWork = 100000000;
  uint64_t maxBackendWork = 1000000;
};
enum class AlgebraStatus { Complete, Unchanged, Unsupported, Exhausted };
struct AlgebraAlternatives;
AlgebraAlternatives generateAlgebraAlternatives(const MeasuredProblem &,
                                                const AlgebraLimits &,
                                                SearchBudget &);

namespace algebra_detail {
struct Source {
  unsigned width;
  std::vector<Gate> rotations;
  std::vector<Pauli> observables;
  std::vector<unsigned> readoutWires;
};
inline bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}
inline uint64_t multiply(uint64_t a, uint64_t b) {
  return llvm::SaturatingMultiply<uint64_t>(a, b);
}
inline uint64_t add(uint64_t a, uint64_t b) {
  return llvm::SaturatingAdd<uint64_t>(a, b);
}
} // namespace algebra_detail

/// Immutable composition of the checked T backend's full-unitary rewrite
/// contract with terminal-measurement normalization. This is not a portable
/// standalone witness: the existing backend checks local witnesses internally
/// and does not export them or distinguish rejected witnesses from no rewrite.
/// Only this adapter constructs a value, tied to an owned source contract.
class CertifiedAlgebraAlternative {
public:
  enum class Proof { CheckedTRewritesAndMeasuredNormalization };
  Proof proof() const {
    return Proof::CheckedTRewritesAndMeasuredNormalization;
  }
  unsigned width() const { return origin->width; }
  const std::vector<Gate> &rotations() const { return normalized.rotations; }
  const std::vector<Pauli> &observables() const {
    return normalized.measurements;
  }
  const std::vector<unsigned> &readoutWires() const {
    return origin->readoutWires;
  }
  /// Full-unitary checked intermediate, including required scalar phases.
  const std::vector<Gate> &checkedWord() const { return checked; }
  MeasuredProblem source() const {
    return {origin->width, origin->rotations, origin->observables,
            origin->readoutWires};
  }
  bool matchesSource(const MeasuredProblem &p) const {
    return p.width == origin->width && p.observables == origin->observables &&
           p.readoutWires == origin->readoutWires &&
           algebra_detail::sameWord(p.rotations, origin->rotations);
  }

private:
  friend AlgebraAlternatives
  generateAlgebraAlternatives(const MeasuredProblem &, const AlgebraLimits &,
                              SearchBudget &);
  CertifiedAlgebraAlternative(
      std::shared_ptr<const algebra_detail::Source> source,
      std::vector<Gate> word, pbc::Result normalized)
      : origin(std::move(source)), checked(std::move(word)),
        normalized(std::move(normalized)) {}
  std::shared_ptr<const algebra_detail::Source> origin;
  std::vector<Gate> checked;
  pbc::Result normalized;
};

struct AlgebraAlternatives {
  std::vector<CertifiedAlgebraAlternative> alternatives;
  AlgebraStatus status = AlgebraStatus::Unchanged;
  std::string error;
  uint64_t work = 0;
  uint64_t materializedGateBound = 0;
  tcount::Statistics backend;
};

/// One bounded generation of merge/decode alternatives from normalized PBC.
/// One portfolio round and no Clifford reuse make every strategy start from
/// the same pure rotation word, permitting preflight bounds before the
/// backend's uncharged mechanical cost/materialization work. Native synthesis
/// and final five-metric selection belong to the caller. The unchanged source
/// is always available externally and is never replaced on an incomplete
/// normalization.
inline AlgebraAlternatives
generateAlgebraAlternatives(const MeasuredProblem &source,
                            const AlgebraLimits &limits, SearchBudget &shared) {
  using algebra_detail::add;
  using algebra_detail::multiply;
  AlgebraAlternatives result;
  uint64_t startWork = shared.work();
  auto syncWork = [&] { result.work = shared.work() - startWork; };
  auto charge = [&](uint64_t amount) {
    syncWork();
    if (amount > limits.maxWork - std::min(limits.maxWork, result.work) ||
        !shared.take(amount)) {
      result.status = AlgebraStatus::Exhausted;
      result.error = "algebraic representation work budget exhausted";
      syncWork();
      return false;
    }
    syncWork();
    return true;
  };
  auto unsupported = [&](const char *message) {
    result.status = AlgebraStatus::Unsupported;
    result.error = message;
    syncWork();
    return result;
  };
  if (!limits.maxCandidates || !limits.maxInputGates ||
      !limits.maxTemporaryGates || !limits.maxWidth || !limits.maxBackendWork ||
      !source.width || source.width > limits.maxWidth ||
      source.rotations.size() > limits.maxInputGates ||
      source.observables.empty() ||
      source.observables.size() > limits.maxInputGates ||
      source.observables.size() != source.readoutWires.size())
    return unsupported("unsupported algebraic source dimensions or limits");
  uint64_t rowWork = uint64_t(source.width) / 64 + 1;
  uint64_t frameWork = uint64_t(source.width) * source.width + 1;
  if (!charge(add(frameWork, multiply(source.rotations.size() +
                                          source.observables.size() + 1,
                                      rowWork))))
    return result;
  uint64_t inputBound = 0;
  for (const auto &gate : source.rotations) {
    if (gate.kind != Kind::Rotate || !gate.pauli.weight() ||
        !gate.pauli.x.fits(source.width) || !gate.pauli.z.fits(source.width) ||
        (!gate.angle.symbol && gate.angle.denominator <= 0) ||
        tcount::detail::isClifford(gate))
      return unsupported(
          "algebraic source must be normalized non-Clifford rotations");
    inputBound = add(inputBound, 6 * uint64_t(gate.pauli.weight()) + 1);
  }
  pbc::CommutingObservables observables;
  for (size_t i = 0; i < source.observables.size(); ++i) {
    const auto &row = source.observables[i];
    if (!charge(multiply(2 * uint64_t(observables.size()) + 1, rowWork)))
      return result;
    if (!row.weight() || !(row.x | row.z).fits(source.width) ||
        source.readoutWires[i] >= source.width || !observables.insert(row))
      return unsupported("invalid signed terminal measurement contract");
  }
  if (inputBound > limits.maxTemporaryGates)
    return unsupported(
        "mechanical input expansion exceeds algebraic size limit");

  // Match the decoder's pure-rotation block boundaries in each direction.
  // A rank-r block emits at most 2^r-1 rotations and one scalar; every emitted
  // axis lies within the union of that block's physical supports. Unchanged
  // blocks retain their independently bounded mechanical input expansion.
  uint64_t outputBound = inputBound;
  for (bool reverse : {false, true}) {
    uint64_t bound = 0;
    for (size_t begin = 0; begin < source.rotations.size();) {
      pbc::CommutingObservables basis;
      QubitSet active;
      uint64_t original = 0;
      size_t end = begin;
      for (; end < source.rotations.size() && end - begin < 256; ++end) {
        size_t index = reverse ? source.rotations.size() - 1 - end : end;
        const auto &gate = source.rotations[index];
        if (!charge(multiply(2 * uint64_t(basis.size()) + 2, rowWork)))
          return result;
        if (!tcount::detail::quarters(gate.angle))
          break;
        auto extended = basis;
        if (!extended.insert(gate.pauli) || extended.size() > 5)
          break;
        basis = std::move(extended);
        active |= gate.pauli.x | gate.pauli.z;
        original = add(original, 6 * uint64_t(gate.pauli.weight()) + 1);
      }
      if (end == begin) {
        size_t index = reverse ? source.rotations.size() - 1 - begin : begin;
        bound = add(bound,
                    6 * uint64_t(source.rotations[index].pauli.weight()) + 1);
        ++begin;
      } else {
        uint64_t decoded = add(multiply((uint64_t(1) << basis.size()) - 1,
                                        6 * uint64_t(active.count()) + 1),
                               1);
        bound = add(bound, std::max(original, decoded));
        begin = end;
      }
    }
    outputBound = std::max(outputBound, bound);
  }
  result.materializedGateBound = outputBound;
  if (outputBound > limits.maxTemporaryGates)
    return unsupported("decoded reconstruction exceeds algebraic size bound");
  // Prepay conservative scans/copies for the backend's four strategies, their
  // block cost comparisons and visitor copies. This precedes tcount::cost,
  // which mechanically materializes its arguments outside its own work budget.
  unsigned candidateCap = std::min(limits.maxCandidates, 4u);
  uint64_t preflight =
      add(multiply(add(inputBound, outputBound), 48),
          multiply(multiply(outputBound, candidateCap), rowWork));
  if (!charge(preflight))
    return result;
  uint64_t remaining = limits.maxWork - result.work;
  // T Budget charges the shared budget before checking its own local cap.
  // Reserve width units for the final rejected charge, plus half the remaining
  // allowance for normalization of completed alternatives.
  if (remaining / 2 <= source.width) {
    charge(remaining + 1);
    return result;
  }
  tcount::Options options;
  options.rounds = options.beamWidth = 1;
  options.reuseCliffords = false;
  options.maxCandidates =
      5; // Four strategies; do not mislabel completing all as exhaustion.
  options.maxBlockRank = 5;
  options.maxBlockGates = 256;
  options.maxWork =
      std::min(limits.maxBackendWork, remaining / 2 - source.width);
  options.maxTwoQubitGrowth = limits.maxTemporaryGates;
  std::vector<std::vector<Gate>> collected;
  bool exceededBound = false;
  auto collect = [&](const std::vector<Gate> &gates, const tcount::Cost &,
                     uint64_t) {
    if (gates.size() > outputBound) {
      exceededBound = true;
      return;
    }
    if (collected.size() >= candidateCap ||
        algebra_detail::sameWord(gates, source.rotations))
      return;
    for (const auto &previous : collected)
      if (algebra_detail::sameWord(previous, gates))
        return;
    collected.push_back(gates);
  };
  auto optimized = tcount::optimize(source.width, source.rotations, options,
                                    &shared, nullptr, 0, 0, collect);
  syncWork();
  result.backend = optimized.stats;
  if (!optimized)
    return unsupported("T backend rejected the algebraic source/options");
  collect(optimized.gates, optimized.stats.after, 0);
  if (exceededBound)
    return unsupported("T backend proposal exceeded its preflight size bound");
  bool exhausted = optimized.stats.budgetExhaustions != 0;
  if (shared.stopped()) {
    result.status = AlgebraStatus::Exhausted;
    result.error = "shared budget exhausted before algebraic normalization";
    return result;
  }
  // Prepaid source ownership binds all returned alternatives to this exact
  // source, including signs, angles and physical output identities.
  auto owned = std::make_shared<const algebra_detail::Source>(
      algebra_detail::Source{source.width, source.rotations, source.observables,
                             source.readoutWires});
  for (auto &gates : collected) {
    // pbc::normalize uses one frame-work charge per gate/observable plus a
    // bounded observable-basis validation. Reserve it locally before invoking
    // the existing budgeted implementation; shared charges remain actual.
    uint64_t normalizeBound =
        add(multiply(gates.size() + source.observables.size() + 1, frameWork),
            multiply(source.observables.size(),
                     2 * uint64_t(source.observables.size()) + 1));
    syncWork();
    if (normalizeBound > limits.maxWork - result.work) {
      result.status = AlgebraStatus::Exhausted;
      result.error = "algebraic normalization reservation exceeds work limit";
      return result;
    }
    auto normalized =
        pbc::normalize(source.width, gates, source.observables, &shared);
    syncWork();
    if (!normalized) {
      result.status = shared.stopped() ? AlgebraStatus::Exhausted
                                       : AlgebraStatus::Unsupported;
      result.error = normalized.error;
      return result;
    }
    if (algebra_detail::sameWord(normalized.rotations, source.rotations) &&
        normalized.measurements == source.observables)
      continue;
    bool duplicate = false;
    for (const auto &existing : result.alternatives)
      duplicate |= algebra_detail::sameWord(existing.rotations(),
                                            normalized.rotations) &&
                   existing.observables() == normalized.measurements;
    if (!duplicate)
      result.alternatives.push_back(CertifiedAlgebraAlternative(
          owned, std::move(gates), std::move(normalized)));
  }
  result.status = exhausted                     ? AlgebraStatus::Exhausted
                  : result.alternatives.empty() ? AlgebraStatus::Unchanged
                                                : AlgebraStatus::Complete;
  if (exhausted)
    result.error =
        "bounded T search exhausted; completed alternatives retained";
  syncWork();
  return result;
}
} // namespace cudaq::pauli::graph::detail
