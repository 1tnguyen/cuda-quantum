/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "SynthesisRepresentations.h"
#include "llvm/Support/MathExtras.h"

namespace cudaq::pauli::graph::detail {

struct IndexedRepresentationLimits {
  unsigned maxSteps = 64;
  /// Explicit input bounds, also capped internally at these defaults. Scores
  /// consequently fit int64_t even at maximum support on every input row.
  unsigned maxRotations = 32768, maxObservables = 4096, maxWidth = 4096;
  uint64_t maxWork = 100000000;
  /// Conservative owned-payload bound including both transactional words,
  /// wide Pauli limbs, score vectors, changed-row indices and certificate.
  uint64_t maxTemporaryBytes = 128 * 1024 * 1024;
};

namespace representation_index_detail {
struct Score {
  int64_t rotations = 0, observables = 0;
};
inline uint64_t add(uint64_t a, uint64_t b) {
  return llvm::SaturatingAdd<uint64_t>(a, b);
}
inline uint64_t multiply(uint64_t a, uint64_t b) {
  return llvm::SaturatingMultiply<uint64_t>(a, b);
}

/// Signed axes have the same support score as their unsigned counterparts.
/// An anticommuting quarter turn changes Q's unsigned bits to P xor Q.
inline int64_t contribution(const Pauli &pivot, const Pauli &row) {
  if (!pivot.anti(row))
    return 0;
  return int64_t((pivot.x ^ row.x).unionCount(pivot.z ^ row.z)) -
         int64_t(row.weight());
}
} // namespace representation_index_detail

/// The same reference-free greedy provider as exploreRepresentations, with
/// cached exact support deltas. Return the existing unverified proposal and
/// certificate contract; verifyRepresentation remains the independent checker.
///
/// Initial scoring costs O(R^2 + R*M) Pauli operations. After a flip, let J be
/// changed suffix axes and K changed observables. Recompute scores of pivots in
/// J; update every unchanged pivot only for changed targets. This costs
/// O((R+M)*|J| + R*|K|) operations plus linear copying, with linear row
/// storage. No commutation matrix or assumption of invariant cross-cut
/// commutation is used. With ample resources, every (observable delta, rotation
/// delta, index) choice matches the original provider, including duplicate-axis
/// multiplicity.
///
/// Every complete update is prepaid and constructed separately before the
/// current word, scores, certificate and support totals commit together. Local
/// exhaustion retains the last complete proposal without stopping the parent.
/// The caller must reserve completion/proof work in its enclosing slice: this
/// provider may consume all allowance supplied to it. The original source and
/// old provider are never changed.
inline RepresentationSearchResult
exploreIndexedRepresentations(const MeasuredProblem &source,
                              const IndexedRepresentationLimits &limits,
                              SearchBudget &shared) {
  namespace ri = representation_index_detail;
  RepresentationSearchResult result;
  SearchBudget budget(shared, std::min(limits.maxWork, shared.remaining()));
  auto finish = [&]() {
    result.work = budget.work();
    return std::move(result);
  };
  auto unsupported = [&](const char *message) {
    result.status = RepresentationSearchStatus::Unsupported;
    result.error = message;
  };
  auto charge = [&](uint64_t amount) {
    if (budget.take(amount))
      return true;
    result.status = RepresentationSearchStatus::BudgetExhausted;
    result.error = "indexed representation work budget exhausted";
    return false;
  };
  if (!source.width || source.width > std::min(limits.maxWidth, 4096u) ||
      source.rotations.size() > std::min(limits.maxRotations, 32768u) ||
      source.observables.size() > std::min(limits.maxObservables, 4096u)) {
    unsupported("indexed representation dimensions exceed their limits");
    return finish();
  }
  if (!charge(representation::initialWork(source)))
    return finish();
  if (!representation::validProblem(source)) {
    unsupported("indexed representation requires normalized measured input");
    return finish();
  }
  result.before = result.after =
      representation::support(source.rotations, source.observables);
  if (!limits.maxSteps || source.rotations.empty())
    return finish();
  const uint64_t rotations = source.rotations.size();
  const uint64_t observables = source.observables.size();
  const uint64_t limbs = uint64_t(source.width) / 64 + 1;
  auto work = [&](uint64_t rows) { return ri::multiply(rows, limbs); };
  // Every accepted move strictly decreases one support sum and increases
  // neither. This also bounds certificate storage when maxSteps is enormous.
  uint64_t certificateBound =
      std::min<uint64_t>(limits.maxSteps, ri::add(result.before.rotations,
                                                  result.before.observables));
  uint64_t rows = ri::add(rotations, observables);
  uint64_t rowBytes =
      ri::add(sizeof(Gate), ri::multiply(2 * sizeof(uint64_t), limbs));
  uint64_t bytes = ri::add(
      ri::multiply(2, ri::multiply(rows, rowBytes)),
      ri::add(
          ri::multiply(2 * sizeof(ri::Score), rotations),
          ri::add(ri::multiply(sizeof(size_t), rows),
                  ri::add(2 * rotations,
                          ri::add(ri::multiply(sizeof(unsigned), observables),
                                  ri::multiply(sizeof(FlipTSign),
                                               certificateBound))))));
  // Include live algebra temporaries as well as container/scalar bookkeeping.
  bytes = ri::add(bytes, ri::add(ri::multiply(8, rowBytes), 4096));
  if (bytes > limits.maxTemporaryBytes) {
    unsupported("indexed representation temporary storage limit exceeded");
    return finish();
  }
  if (!charge(work(ri::add(3 * rotations, 2 * observables + 1))))
    return finish();
  RepresentationAlternative proposal{source.width,
                                     source.rotations,
                                     source.observables,
                                     source.readoutWires,
                                     {}};
  proposal.certificate.reserve(certificateBound);
  std::vector<ri::Score> scores(rotations);
  std::vector<bool> eligible(rotations), changed(rotations);
  std::vector<size_t> changedRotations, changedObservables;
  changedRotations.reserve(rotations);
  changedObservables.reserve(observables);
  auto publish = [&]() {
    if (!proposal.certificate.empty()) {
      result.alternative = std::move(proposal);
      if (result.status != RepresentationSearchStatus::BudgetExhausted)
        result.status = RepresentationSearchStatus::Complete;
    }
    return finish();
  };
  uint64_t initialPairs = 0;
  for (size_t i = 0; i < rotations; ++i) {
    eligible[i] = representation::pivotSign(proposal.rotations[i]) != 0;
    if (eligible[i])
      initialPairs = ri::add(initialPairs, rotations - i - 1 + observables);
  }
  if (!charge(work(initialPairs)))
    return publish();
  auto recompute = [&](size_t i, const std::vector<Gate> &word,
                       const std::vector<Pauli> &measurements) {
    ri::Score score;
    const auto &pivot = word[i].pauli;
    for (size_t j = i + 1; j < word.size(); ++j)
      score.rotations += ri::contribution(pivot, word[j].pauli);
    for (const auto &observable : measurements)
      score.observables += ri::contribution(pivot, observable);
    return score;
  };
  for (size_t i = 0; i < rotations; ++i)
    if (eligible[i])
      scores[i] = recompute(i, proposal.rotations, proposal.observables);

  for (unsigned step = 0; step < limits.maxSteps; ++step) {
    if (!charge(rotations + 1))
      return publish();
    std::optional<std::tuple<int64_t, int64_t, size_t>> best;
    for (size_t i = 0; i < rotations; ++i) {
      const auto &score = scores[i];
      if (!eligible[i] || score.rotations > 0 || score.observables > 0 ||
          (!score.rotations && !score.observables))
        continue;
      auto rank = std::tuple{score.observables, score.rotations, i};
      if (!best || rank < *best)
        best = rank;
    }
    if (!best)
      break;
    size_t pivotIndex = std::get<2>(*best);
    // Finding the affected rows may stop, but never mutates the committed word.
    if (!charge(work(rotations + observables + 1)))
      return publish();
    std::fill(changed.begin(), changed.end(), false);
    changedRotations.clear();
    changedObservables.clear();
    const auto &pivot = proposal.rotations[pivotIndex].pauli;
    for (size_t j = pivotIndex + 1; j < rotations; ++j)
      if (pivot.anti(proposal.rotations[j].pauli)) {
        changed[j] = true;
        changedRotations.push_back(j);
      }
    for (size_t j = 0; j < observables; ++j)
      if (pivot.anti(proposal.observables[j]))
        changedObservables.push_back(j);

    if (!charge(rotations + 1))
      return publish();
    uint64_t pairs = 0;
    size_t laterChanged = changedRotations.size();
    for (size_t i = 0; i < rotations; ++i) {
      if (changed[i])
        --laterChanged;
      if (!eligible[i])
        continue;
      // Changed pivots require a full recomputation. Other pivots compare old
      // and new contributions only for changed suffix/observable targets.
      uint64_t count = changed[i]
                           ? rotations - i - 1 + observables
                           : 2 * (laterChanged + changedObservables.size());
      pairs = ri::add(pairs, count);
    }
    // Copy the two words and scores; transport the entire suffix to preserve
    // exact sign normalization even on commuting axes; then update scores.
    uint64_t linear = ri::add(3 * rotations, 2 * observables + 2);
    // The changed-index loops also inspect entries before the current pivot;
    // these scalar comparisons cost work even though no Pauli term changes.
    uint64_t indexWork = ri::multiply(rotations, changedRotations.size());
    if (!charge(ri::add(work(ri::add(linear, pairs)), indexWork)))
      return publish();
    auto nextRotations = proposal.rotations;
    auto nextObservables = proposal.observables;
    auto nextScores = scores;
    int sign = representation::pivotSign(nextRotations[pivotIndex]);
    Pauli unsignedPivot = nextRotations[pivotIndex].pauli;
    unsignedPivot.sign = false;
    Gate compensation = Gate::rotation(unsignedPivot, {-sign, 2});
    nextRotations[pivotIndex].angle = nextRotations[pivotIndex].angle.negated();
    for (size_t j = pivotIndex + 1; j < rotations; ++j) {
      auto &gate = nextRotations[j];
      gate.pauli = tcount::detail::conjugate(gate.pauli, compensation);
      if (gate.pauli.sign)
        gate.angle = gate.angle.negated();
      gate.pauli.sign = false;
    }
    for (auto &observable : nextObservables)
      observable = tcount::detail::conjugate(observable, compensation);
    for (size_t i = 0; i < rotations; ++i) {
      if (!eligible[i])
        continue;
      if (changed[i]) {
        nextScores[i] = recompute(i, nextRotations, nextObservables);
        continue;
      }
      const auto &unchangedPivot = proposal.rotations[i].pauli;
      for (size_t j : changedRotations)
        if (j > i)
          nextScores[i].rotations +=
              ri::contribution(unchangedPivot, nextRotations[j].pauli) -
              ri::contribution(unchangedPivot, proposal.rotations[j].pauli);
      for (size_t j : changedObservables)
        nextScores[i].observables +=
            ri::contribution(unchangedPivot, nextObservables[j]) -
            ri::contribution(unchangedPivot, proposal.observables[j]);
    }
    // The full mutation is now available and its work has already been paid.
    // Moves cannot change exact +/-pi/4 eligibility, even on signed axes.
    proposal.rotations = std::move(nextRotations);
    proposal.observables = std::move(nextObservables);
    scores = std::move(nextScores);
    proposal.certificate.push_back({pivotIndex});
    auto decrease = [](int64_t delta) {
      return delta < 0 ? uint64_t(-(delta + 1)) + 1 : uint64_t(0);
    };
    result.after.observables -= decrease(std::get<0>(*best));
    result.after.rotations -= decrease(std::get<1>(*best));
  }
  return publish();
}
} // namespace cudaq::pauli::graph::detail
