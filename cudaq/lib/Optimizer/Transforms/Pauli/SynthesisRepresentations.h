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
#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_map>

namespace cudaq::pauli::graph::detail {

/// Indices refer to the current ordered word. A flip never inserts, deletes or
/// reorders rotations, so these IDs also identify the source obligations.
struct FlipTSign {
  size_t rotation;
};

/// A proposal, not a verified value. Only verifyRepresentation establishes its
/// relation to an immutable source problem. Readout identities are unchanged.
struct RepresentationAlternative {
  unsigned width;
  std::vector<Gate> rotations;
  std::vector<Pauli> observables;
  std::vector<unsigned> readoutWires;
  std::vector<FlipTSign> certificate;
};

struct RepresentationSupport {
  uint64_t rotations = 0, observables = 0;
};

enum class RepresentationSearchStatus {
  Complete,
  Unchanged,
  Unsupported,
  BudgetExhausted
};

struct RepresentationSearchResult {
  /// Contains the last fully applied proposal even on later exhaustion. Empty
  /// on unsupported input or before a useful move completes. The borrowed
  /// source remains the immutable identity alternative.
  std::optional<RepresentationAlternative> alternative;
  RepresentationSearchStatus status = RepresentationSearchStatus::Unchanged;
  RepresentationSupport before, after;
  uint64_t work = 0;
  std::string error;
};

enum class RepresentationCheckStatus {
  Verified,
  InvalidProof,
  BudgetExhausted
};
struct RepresentationCheck {
  RepresentationCheckStatus status;
  std::string error;
  explicit operator bool() const {
    return status == RepresentationCheckStatus::Verified;
  }
};

namespace representation {
/// The first provider intentionally accepts exact +/-pi/4 pivots, rather than
/// modulo-period angle classification. Widen before multiplication/negation.
inline int pivotSign(const Gate &gate) {
  if (gate.kind != Kind::Rotate || gate.angle.symbol ||
      gate.angle.denominator <= 0)
    return 0;
  __int128 coefficient = __int128(gate.angle.numerator) * 4;
  if (gate.angle.negative != gate.pauli.sign)
    coefficient = -coefficient;
  if (coefficient == gate.angle.denominator)
    return 1;
  if (coefficient == -__int128(gate.angle.denominator))
    return -1;
  return 0;
}

/// No native gates, scalar phases or internal live-output contracts enter this
/// adapter. Arbitrary exact/opaque suffix angles are preserved by conjugation.
inline bool validProblem(const MeasuredProblem &problem) {
  if (!problem.width || problem.observables.empty() ||
      problem.observables.size() != problem.readoutWires.size())
    return false;
  for (const auto &gate : problem.rotations)
    if (gate.kind != Kind::Rotate || !gate.pauli.weight() ||
        !gate.pauli.x.fits(problem.width) ||
        !gate.pauli.z.fits(problem.width) ||
        (!gate.angle.symbol && gate.angle.denominator <= 0))
      return false;
  for (size_t i = 0; i < problem.observables.size(); ++i) {
    const auto &word = problem.observables[i];
    if (!word.weight() || !word.x.fits(problem.width) ||
        !word.z.fits(problem.width) || problem.readoutWires[i] >= problem.width)
      return false;
    for (size_t j = 0; j < i; ++j)
      if (problem.readoutWires[j] == problem.readoutWires[i] ||
          word.anti(problem.observables[j]))
        return false;
  }
  return true;
}

inline RepresentationSupport support(const std::vector<Gate> &word,
                                     const std::vector<Pauli> &observables) {
  RepresentationSupport result;
  for (const auto &gate : word)
    result.rotations += gate.pauli.weight();
  for (const auto &observable : observables)
    result.observables += observable.weight();
  return result;
}

/// Account before allocating/copying or performing wide Pauli algebra. All
/// estimates are saturated so an overflow cannot turn work into a small charge.
inline uint64_t multiplyWork(uint64_t a, uint64_t b) {
  return b && a > std::numeric_limits<uint64_t>::max() / b
             ? std::numeric_limits<uint64_t>::max()
             : a * b;
}
inline uint64_t initialWork(const MeasuredProblem &p) {
  auto rows = uint64_t(p.rotations.size()) + p.observables.size() + 1;
  auto pairs = multiplyWork(p.observables.size(), p.observables.size());
  if (pairs > std::numeric_limits<uint64_t>::max() - rows)
    return std::numeric_limits<uint64_t>::max();
  return multiplyWork(rows + pairs, uint64_t(p.width) / 64 + 1);
}

/// Independent certificate algebra: signed Hermitian Pauli multiplication,
/// without calling the explorer's Clifford conjugation helper. For an
/// anticommuting Q, R_P(s*pi/2) Q R_P(-s*pi/2) = -s*i*P*Q.
inline Pauli checkedQuarterTurn(const Pauli &unsignedPivot, Pauli q,
                                int direction) {
  if (!unsignedPivot.anti(q))
    return q;
  Pauli out{unsignedPivot.x ^ q.x, unsignedPivot.z ^ q.z};
  unsigned phase = unsignedPivot.x.intersectionCount(unsignedPivot.z) +
                   q.x.intersectionCount(q.z) - out.x.intersectionCount(out.z) +
                   2 * unsignedPivot.z.intersectionCount(q.x) + 2 * q.sign;
  phase += direction > 0 ? 3 : 1;
  out.sign = (phase & 2) != 0;
  return out;
}

inline bool sameAngle(Angle a, Angle b) {
  if (a.symbol || b.symbol)
    return a == b;
  if (a.denominator <= 0 || b.denominator <= 0)
    return false;
  return __int128(a.numerator) * b.denominator * (a.negative ? -1 : 1) ==
         __int128(b.numerator) * a.denominator * (b.negative ? -1 : 1);
}
} // namespace representation

/// Verify original -> alternative independently from support ranking and the
/// explorer's conjugation routine. Each exact identity transports its Clifford
/// compensation into the suffix and all ordered terminal observables. This is
/// a terminal-measurement proof, not a full-unitary proof. Native completion
/// must subsequently be checked against the alternative's own problem.
inline RepresentationCheck
verifyRepresentation(const MeasuredProblem &source,
                     const RepresentationAlternative &alternative,
                     SearchBudget *budget = nullptr) {
  auto invalid = [](const char *message) {
    return RepresentationCheck{RepresentationCheckStatus::InvalidProof,
                               message};
  };
  auto exhausted = [] {
    return RepresentationCheck{RepresentationCheckStatus::BudgetExhausted,
                               "representation proof budget exhausted"};
  };
  if (alternative.width != source.width ||
      alternative.readoutWires != source.readoutWires ||
      alternative.rotations.size() != source.rotations.size() ||
      alternative.observables.size() != source.observables.size())
    return invalid("representation changes the problem or output identities");
  if (budget && !budget->take(representation::initialWork(source)))
    return exhausted();
  if (!representation::validProblem(source))
    return invalid(
        "representation source is not a normalized measured problem");
  auto word = source.rotations;
  auto observables = source.observables;
  const uint64_t limbs = uint64_t(source.width) / 64 + 1;
  for (const auto &step : alternative.certificate) {
    if (step.rotation >= word.size())
      return invalid("representation certificate rotation is out of range");
    int sign = representation::pivotSign(word[step.rotation]);
    if (!sign)
      return invalid("representation certificate pivot is not exact +/-pi/4");
    if (budget && !budget->take(representation::multiplyWork(
                      word.size() - step.rotation + observables.size(), limbs)))
      return exhausted();
    Pauli pivot = word[step.rotation].pauli;
    pivot.sign = false;
    word[step.rotation].angle = word[step.rotation].angle.negated();
    for (size_t j = step.rotation + 1; j < word.size(); ++j) {
      auto p = representation::checkedQuarterTurn(pivot, word[j].pauli, -sign);
      if (p.sign)
        word[j].angle = word[j].angle.negated();
      p.sign = false;
      word[j].pauli = std::move(p);
    }
    for (auto &observable : observables)
      observable = representation::checkedQuarterTurn(pivot, observable, -sign);
  }
  if (budget && !budget->take(representation::multiplyWork(
                    word.size() + observables.size() + 1, limbs)))
    return exhausted();
  for (size_t i = 0; i < word.size(); ++i) {
    const auto &actual = alternative.rotations[i];
    if (actual.kind != Kind::Rotate || actual.pauli != word[i].pauli ||
        !representation::sameAngle(actual.angle, word[i].angle))
      return invalid("representation rotations do not match its certificate");
  }
  if (observables != alternative.observables)
    return invalid("representation observables do not match its certificate");
  return {RepresentationCheckStatus::Verified, {}};
}

/// Bounded, reference-free support heuristic from the frozen architecture
/// probe. A move must reduce at least one support sum without increasing the
/// other; rank observable reduction first, rotation reduction second, then
/// original index. These counts never authorize native-cost acceptance.
///
/// The source is the identity alternative throughout. Each move is prepaid in
/// full. On exhaustion its completed predecessors remain an unverified
/// proposal; a partially transformed word never escapes.
inline RepresentationSearchResult
exploreRepresentations(const MeasuredProblem &source, unsigned maxSteps = 32,
                       SearchBudget *budget = nullptr) {
  RepresentationSearchResult result;
  auto charge = [&](uint64_t amount) {
    if (budget && !budget->take(amount)) {
      result.status = RepresentationSearchStatus::BudgetExhausted;
      result.error = "representation exploration budget exhausted";
      return false;
    }
    result.work = amount > std::numeric_limits<uint64_t>::max() - result.work
                      ? std::numeric_limits<uint64_t>::max()
                      : result.work + amount;
    return true;
  };
  if (!charge(representation::initialWork(source)))
    return result;
  if (!representation::validProblem(source)) {
    result.status = RepresentationSearchStatus::Unsupported;
    result.error = "representation explorer requires normalized measured input";
    return result;
  }
  result.before = result.after =
      representation::support(source.rotations, source.observables);
  if (!maxSteps)
    return result;
  RepresentationAlternative proposal{source.width,
                                     source.rotations,
                                     source.observables,
                                     source.readoutWires,
                                     {}};
  auto finish = [&]() {
    if (!proposal.certificate.empty()) {
      result.alternative = std::move(proposal);
      if (result.status != RepresentationSearchStatus::BudgetExhausted)
        result.status = RepresentationSearchStatus::Complete;
    }
    return std::move(result);
  };
  const uint64_t limbs = uint64_t(source.width) / 64 + 1;
  using Axis = tcount::detail::Axis;
  using AxisHash = tcount::detail::AxisHash;
  for (unsigned step = 0; step < maxSteps; ++step) {
    if (!charge(
            representation::multiplyWork(proposal.rotations.size() + 1, limbs)))
      return finish();
    std::unordered_map<Axis, uint64_t, AxisHash> suffix;
    for (const auto &gate : proposal.rotations)
      ++suffix[{gate.pauli.x, gate.pauli.z}];
    std::optional<std::tuple<int64_t, int64_t, size_t>> best;
    for (size_t i = 0; i < proposal.rotations.size(); ++i) {
      const auto &gate = proposal.rotations[i];
      --suffix[{gate.pauli.x, gate.pauli.z}];
      if (!representation::pivotSign(gate))
        continue;
      if (!charge(representation::multiplyWork(
              suffix.size() + proposal.observables.size() + 1, limbs)))
        return finish();
      int64_t rotationDelta = 0, observableDelta = 0;
      for (const auto &[axis, count] : suffix) {
        Pauli q{axis.x, axis.z};
        if (count && gate.pauli.anti(q))
          rotationDelta +=
              int64_t(count) *
              (int64_t((gate.pauli.x ^ q.x).unionCount(gate.pauli.z ^ q.z)) -
               int64_t(q.weight()));
      }
      for (const auto &q : proposal.observables)
        if (gate.pauli.anti(q))
          observableDelta +=
              int64_t((gate.pauli.x ^ q.x).unionCount(gate.pauli.z ^ q.z)) -
              int64_t(q.weight());
      if (rotationDelta > 0 || observableDelta > 0 ||
          (!rotationDelta && !observableDelta))
        continue;
      auto rank = std::tuple{observableDelta, rotationDelta, i};
      if (!best || rank < *best)
        best = rank;
    }
    if (!best)
      break;
    size_t i = std::get<2>(*best);
    if (!charge(representation::multiplyWork(proposal.rotations.size() - i +
                                                 proposal.observables.size(),
                                             limbs)))
      return finish();
    int sign = representation::pivotSign(proposal.rotations[i]);
    Pauli pivot = proposal.rotations[i].pauli;
    pivot.sign = false;
    // R(P,a)=R(P,2a)R(P,-a). Moving the first factor left through
    // the suffix conjugates axes/observables by its inverse.
    Gate inverseCompensation = Gate::rotation(pivot, {-sign, 2});
    proposal.rotations[i].angle = proposal.rotations[i].angle.negated();
    for (size_t j = i + 1; j < proposal.rotations.size(); ++j) {
      auto &gate = proposal.rotations[j];
      gate.pauli = tcount::detail::conjugate(gate.pauli, inverseCompensation);
      if (gate.pauli.sign)
        gate.angle = gate.angle.negated();
      gate.pauli.sign = false;
    }
    for (auto &observable : proposal.observables)
      observable = tcount::detail::conjugate(observable, inverseCompensation);
    proposal.certificate.push_back({i});
    // Both chosen deltas are nonpositive. Ranking already paid for these
    // support counts; update only after the entire certified move commits.
    auto decrease = [](int64_t delta) {
      return delta < 0 ? uint64_t(-(delta + 1)) + 1 : uint64_t(0);
    };
    result.after.observables -= decrease(std::get<0>(*best));
    result.after.rotations -= decrease(std::get<1>(*best));
  }
  return finish();
}

} // namespace cudaq::pauli::graph::detail
