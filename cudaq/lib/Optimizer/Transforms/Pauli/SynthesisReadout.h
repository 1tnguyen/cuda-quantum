/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Bounded, certified alternatives for signed terminal readout.
///
/// Elimination order is a search choice; observable identity and physical
/// readout wire are immutable. Complete tails are checked against every
/// original signed observable, preserving joint outcomes for arbitrary input
/// states. This is a terminal measurement contract, not a live quantum output
/// contract.
//===----------------------------------------------------------------------===//

#pragma once

#include "SynthesisCost.h"
#include "cudaq/Optimizer/Pauli/PBC.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cudaq::pauli::graph::detail {

struct ReadoutLimits {
  unsigned maxCandidates = 8;
  uint64_t maxWork = 1000000;
};

enum class ReadoutStatus { Complete, Unsupported, Exhausted, InvalidProof };
struct ReadoutCandidates;
ReadoutCandidates
generateReadoutCandidates(unsigned width, const std::vector<Pauli> &observables,
                          const std::vector<unsigned> &readoutWires,
                          const Options &options, const ReadoutLimits &limits,
                          SearchBudget &budget);

/// Certified only for the original arguments of generateReadoutCandidates.
/// Consumers must compose this relation with their prefix proof; a tail cannot
/// be transferred to a different observable/readout-wire contract.
class CertifiedReadout {
public:
  const std::vector<Gate> &gates() const { return word; }

private:
  friend ReadoutCandidates
  generateReadoutCandidates(unsigned, const std::vector<Pauli> &,
                            const std::vector<unsigned> &, const Options &,
                            const ReadoutLimits &, SearchBudget &);
  explicit CertifiedReadout(std::vector<Gate> gates) : word(std::move(gates)) {}
  std::vector<Gate> word;
};

struct ReadoutCandidates {
  std::vector<CertifiedReadout> candidates;
  /// Exhaustion or unsupported generation retains already certified tails.
  /// InvalidProof must stop promotion even when earlier tails are present.
  ReadoutStatus status = ReadoutStatus::Complete;
  std::string error;
  uint64_t work = 0;
};

namespace readout_detail {
class Budget {
public:
  Budget(SearchBudget &shared, uint64_t limit, ReadoutCandidates &result)
      : shared(shared), limit(limit), result(result) {}
  bool take(uint64_t amount = 1) {
    if (result.status == ReadoutStatus::Exhausted)
      return false;
    if (amount > limit - result.work || !shared.take(amount)) {
      result.status = ReadoutStatus::Exhausted;
      result.error = "readout completion work budget exhausted";
      return false;
    }
    result.work += amount;
    return true;
  }

private:
  SearchBudget &shared;
  uint64_t limit;
  ReadoutCandidates &result;
};

inline Pauli wanted(unsigned wire) {
  return {0, QubitSet::single(wire), false};
}

/// Build one elimination step using one row only. The search state is not
/// modified until this complete native step has been chosen.
inline std::vector<Gate> step(Pauli row, unsigned target, unsigned pivot,
                              const Options &options) {
  std::vector<Gate> gates;
  auto append = [&](Gate gate) {
    auto native = nativeWord({gate}, options);
    for (const auto &g : native) {
      row = tcount::detail::conjugate(row, g);
      gates.push_back(g);
    }
  };
  auto support = row.x | row.z;
  for (unsigned wire : support.set_bits()) {
    unsigned axis = row.axis(wire);
    if (axis == 3)
      append({Kind::Sdg, wire});
    if (axis & 1)
      append({Kind::H, wire});
  }
  for (unsigned wire : support.set_bits())
    if (wire != pivot)
      append({Kind::CX, wire, pivot});
  if (pivot != target)
    append({Kind::Swap, pivot, target});
  if (row.sign)
    append({Kind::X, target});
  return gates;
}

/// Replay from immutable input rows, independently of elimination state. A
/// Clifford mapping every M_i to its specified +Z_i maps all ordered joint
/// projectors, including for partial readout and externally entangled inputs.
inline bool verify(unsigned width, const std::vector<Pauli> &observables,
                   const std::vector<unsigned> &readoutWires,
                   const std::vector<Gate> &tail, Budget &budget) {
  uint64_t rowWork = uint64_t(width) / 64 + 1;
  for (size_t index = 0; index < observables.size(); ++index) {
    Pauli row = observables[index];
    for (const auto &gate : tail) {
      if (!budget.take(rowWork))
        return false;
      row = tcount::detail::conjugate(row, gate);
    }
    if (!budget.take(rowWork) || row != wanted(readoutWires[index]))
      return false;
  }
  return true;
}
} // namespace readout_detail

/// Generate at most twelve deterministic order/pivot policies, stopping after
/// limits.maxCandidates distinct certified completions. No heuristic score is
/// a proof or lower bound. The caller retains its existing baseline and ranks
/// prefix+tail candidates with the common cost/scheduling policy.
inline ReadoutCandidates
generateReadoutCandidates(unsigned width, const std::vector<Pauli> &observables,
                          const std::vector<unsigned> &readoutWires,
                          const Options &options, const ReadoutLimits &limits,
                          SearchBudget &shared) {
  ReadoutCandidates result;
  readout_detail::Budget budget(shared, limits.maxWork, result);
  auto unsupported = [&](const char *message) {
    result.status = ReadoutStatus::Unsupported;
    result.error = message;
    return result;
  };
  if (!width || observables.empty() ||
      observables.size() != readoutWires.size() || !limits.maxCandidates ||
      !options.maxGates || (!options.nativeCX && !options.nativeCZ))
    return unsupported("invalid readout dimensions, limits, or native profile");
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost})
    if (!std::isfinite(cost) || cost < 0)
      return unsupported("invalid native readout cost");
  // Charge before any width-sized allocation or copying the observable list.
  uint64_t rowWork = uint64_t(width) / 64 + 1;
  if (!budget.take(llvm::SaturatingAdd<uint64_t>(
          width,
          llvm::SaturatingMultiply<uint64_t>(observables.size(), rowWork))))
    return result;
  pbc::CommutingObservables basis;
  std::vector<std::optional<size_t>> assigned(width);
  unsigned distinctTargets = 0;
  for (size_t i = 0; i < observables.size(); ++i) {
    const auto &row = observables[i];
    unsigned target = readoutWires[i];
    if (!row.weight() || !(row.x | row.z).fits(width) || target >= width)
      return unsupported("readout support or target is invalid");
    if (!budget.take(llvm::SaturatingMultiply<uint64_t>(
            2 * uint64_t(basis.size()) + 1, rowWork)))
      return result;
    if (!basis.insert(row))
      return unsupported("readout observables do not commute");
    if (assigned[target]) {
      if (row != observables[*assigned[target]])
        return unsupported("conflicting repeated readout wire");
    } else {
      assigned[target] = i;
      ++distinctTargets;
    }
  }
  if (basis.size() != distinctTargets)
    return unsupported("readout requires classical outcome remapping");

  // Policies 0..3: original, reverse, minimum support, native immediate cost.
  // Try minimum support first: the original-order tail is already available
  // from the caller's baseline. Do not spend the optional work allowance on
  // regenerating that tail before trying a useful alternative.
  // Pivot variants: target then first; target then last; first even when the
  // target is present. Original target identities remain fixed in every case.
  for (unsigned variant = 0; variant < 3; ++variant)
    for (unsigned policy : {2u, 0u, 3u, 1u}) {
      if (result.candidates.size() >= limits.maxCandidates)
        return result;
      if (!budget.take(llvm::SaturatingAdd<uint64_t>(
              width,
              llvm::SaturatingMultiply<uint64_t>(observables.size(), rowWork))))
        return result;
      auto words = observables;
      std::vector<bool> fixed(width), done(words.size());
      std::vector<Gate> tail;
      for (size_t completed = 0; completed < words.size(); ++completed) {
        std::optional<size_t> selected;
        std::vector<Gate> selectedStep;
        std::tuple<double, unsigned, size_t> best;
        for (size_t ordinal = 0; ordinal < words.size(); ++ordinal) {
          size_t i = policy == 1 ? words.size() - 1 - ordinal : ordinal;
          if (!budget.take())
            return result;
          if (done[i])
            continue;
          const auto &row = words[i];
          unsigned target = readoutWires[i];
          double score = policy == 0   ? double(i)
                         : policy == 1 ? double(words.size() - 1 - i)
                                       : double(row.weight());
          auto key = std::make_tuple(score, row.weight(), i);
          if (policy != 3 && selected && key >= best)
            continue;
          std::vector<Gate> candidate;
          if (fixed[target]) {
            if (row != readout_detail::wanted(target))
              return unsupported(
                  "conflicting repeated readout during completion");
          } else {
            if (!budget.take(rowWork))
              return result;
            auto support = row.x | row.z;
            std::optional<unsigned> first, last;
            for (unsigned wire : support.set_bits()) {
              if (!budget.take())
                return result;
              if (fixed[wire]) {
                if (row.axis(wire) != 2)
                  return unsupported("readout would modify a fixed observable");
              } else {
                if (!first)
                  first = wire;
                last = wire;
              }
            }
            if (!first)
              return unsupported("readout has no unfixed pivot");
            unsigned pivot = variant == 1 ? *last : *first;
            if (variant != 2 && support.test(target))
              pivot = target;
            // At most 3 basis gates and 3 native CNOT gates per support wire,
            // plus a native-expanded SWAP and sign correction. Each step also
            // updates its row and is scored before being selected.
            uint64_t gateBound = 6 * uint64_t(row.weight()) + 10;
            if (!budget.take(
                    llvm::SaturatingMultiply<uint64_t>(gateBound, rowWork + 1)))
              return result;
            candidate = readout_detail::step(row, target, pivot, options);
          }
          if (policy == 3) {
            score = evaluateNativeCost(candidate, options).value;
            key = std::make_tuple(score, row.weight(), i);
          }
          if (!selected || key < best) {
            selected = i;
            best = key;
            selectedStep = std::move(candidate);
          }
        }
        if (!selected)
          return unsupported("readout has no remaining obligation");
        if (selectedStep.size() > options.maxGates - tail.size()) {
          result.status = ReadoutStatus::Exhausted;
          result.error = "readout completion native gate limit exceeded";
          return result;
        }
        for (const auto &gate : selectedStep) {
          for (Pauli &row : words) {
            if (!budget.take(rowWork))
              return result;
            row = tcount::detail::conjugate(row, gate);
          }
          tail.push_back(gate);
        }
        if (words[*selected] !=
            readout_detail::wanted(readoutWires[*selected])) {
          result.status = ReadoutStatus::InvalidProof;
          result.error = "readout elimination failed its selected obligation";
          return result;
        }
        done[*selected] = true;
        fixed[readoutWires[*selected]] = true;
      }
      if (!readout_detail::verify(width, observables, readoutWires, tail,
                                  budget)) {
        if (result.status != ReadoutStatus::Exhausted) {
          result.status = ReadoutStatus::InvalidProof;
          result.error = "readout independent signed replay failed";
        }
        return result;
      }
      bool duplicate = false;
      for (const auto &existing : result.candidates) {
        const auto &word = existing.gates();
        if (!budget.take(word.size() == tail.size() ? tail.size() + 1 : 1))
          return result;
        if (word.size() == tail.size() &&
            std::equal(word.begin(), word.end(), tail.begin(),
                       [](const Gate &a, const Gate &b) {
                         return a.kind == b.kind && a.a == b.a && a.b == b.b &&
                                a.pauli == b.pauli && a.angle == b.angle;
                       })) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate)
        result.candidates.push_back(CertifiedReadout(std::move(tail)));
    }
  return result;
}
} // namespace cudaq::pauli::graph::detail
