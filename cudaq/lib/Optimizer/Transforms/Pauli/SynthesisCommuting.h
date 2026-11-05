/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SynthesisReadout.h"
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace cudaq::pauli::graph::detail {

struct CommutingLimits {
  unsigned maxCandidates = 8;
  /// For rank <=4 and a diagonalized span on <=4 physical coordinates, visit
  /// every coordinate order and triangular star exit frame before shortlisting.
  /// This remains a finite star-tour family, not unrestricted optimal
  /// synthesis.
  bool exhaustiveSmallRank = false;
  /// Exact star tours use at most four controls (five active coordinates).
  /// Larger groups use a deterministic nearest-parity tour.
  unsigned maxExactRank = 5;
  unsigned maxExactTargets = 16;
  unsigned maxDiagonalizers = 4;
  uint64_t maxWork = 50000000;
  /// Each Held--Karp entry stores one uint32_t cost and one int16_t parent.
  uint64_t maxDpEntries = 1048576;
};

enum class CommutingStatus { Complete, Unsupported, Exhausted, InvalidProof };

/// An unverified proposal with the exact intended relation
///   body = exitCliffords * originalRegion.
/// Both vectors are in circuit order. The exit includes every emitted Clifford,
/// including the diagonalizer, but excludes rotations. No inverse is appended.
/// The caller transports this exit into its complete suffix and verifies the
/// resulting complete program before candidate publication.
struct CommutingImplementation {
  std::vector<Gate> body;
  std::vector<Gate> exitCliffords;
};

/// Score a complete regional implementation, optionally including a caller's
/// terminal completion. The callback must charge transport/completion/scoring
/// work to its supplied child budget. nullopt rejects an unavailable
/// completion; it is not an equivalence verdict. The caller still independently
/// verifies the final full program. The child cannot consume the solver's
/// enclosing reserve.
using CommutingScorer = std::function<std::optional<CostRank>(
    const CommutingImplementation &, SearchBudget &)>;

struct CommutingCandidates {
  std::vector<CommutingImplementation> candidates;
  CommutingStatus status = CommutingStatus::Complete;
  unsigned rank = 0;
  uint64_t completedImplementations = 0;
  unsigned wireOrders = 0;
  uint64_t work = 0;
  std::string error;
};

namespace commuting_detail {
class Budget {
public:
  Budget(SearchBudget &shared, uint64_t limit, CommutingCandidates &result)
      : shared(shared), limit(limit), result(result) {}
  bool take(uint64_t amount = 1) {
    if (result.status == CommutingStatus::Exhausted)
      return false;
    if (amount > limit - result.work || !shared.take(amount)) {
      result.status = CommutingStatus::Exhausted;
      result.error = "commuting-region work budget exhausted";
      return false;
    }
    result.work += amount;
    return true;
  }
  uint64_t remaining() const { return limit - result.work; }
  void accountExternal(uint64_t amount) { result.work += amount; }

private:
  SearchBudget &shared;
  uint64_t limit;
  CommutingCandidates &result;
};

inline uint64_t work(uint64_t count, uint64_t size) {
  return llvm::SaturatingMultiply<uint64_t>(count, size);
}

inline bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}

struct Demand {
  QubitSet controls;
  std::vector<Angle> angles;
};
struct Star {
  unsigned pivot;
  std::vector<unsigned> controls;
  std::vector<Demand> demands;
};
struct Tour {
  /// Required parities in visitation order, followed by the desired endpoint.
  /// Intervening hypercube vertices are visited by individual native CXs.
  std::vector<QubitSet> waypoints;
  uint64_t cx = 0;
};

inline QubitSet expandMask(unsigned mask,
                           const std::vector<unsigned> &controls) {
  QubitSet result;
  for (unsigned i = 0; i < controls.size(); ++i)
    if ((mask >> i) & 1)
      result |= QubitSet::single(controls[i]);
  return result;
}

/// Exact metric shortest open tour for one fixed star. With <=4 controls there
/// are <=15 nonzero required vertices. This is not unrestricted CNOT synthesis.
inline std::optional<std::vector<Tour>>
exactTours(const Star &star, const CommutingLimits &limits, Budget &budget) {
  if (star.controls.size() > 4 ||
      star.controls.size() + 1 > limits.maxExactRank)
    return std::nullopt;
  std::vector<unsigned> vertices;
  for (const auto &demand : star.demands) {
    unsigned bits = 0;
    for (unsigned i = 0; i < star.controls.size(); ++i)
      if (demand.controls.test(star.controls[i]))
        bits |= 1u << i;
    if (bits)
      vertices.push_back(bits);
  }
  std::sort(vertices.begin(), vertices.end());
  if (vertices.size() > limits.maxExactTargets)
    return std::nullopt;
  if (vertices.empty())
    return std::vector<Tour>{Tour{{QubitSet{}}, 0}};
  unsigned count = vertices.size(), subsets = 1u << count;
  uint64_t entries = uint64_t(subsets) * count;
  if (entries > limits.maxDpEntries)
    return std::nullopt;
  if (!budget.take(entries))
    return std::vector<Tour>{};
  constexpr uint32_t infinity = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> distance(entries, infinity);
  std::vector<int16_t> parent(entries, -1);
  for (unsigned i = 0; i < count; ++i)
    distance[(size_t(1) << i) * count + i] = std::popcount(vertices[i]);
  for (unsigned mask = 1; mask < subsets; ++mask)
    for (unsigned last = 0; last < count; ++last) {
      auto current = distance[size_t(mask) * count + last];
      if (current == infinity)
        continue;
      if (!budget.take(count))
        return std::vector<Tour>{};
      for (unsigned next = 0; next < count; ++next) {
        if ((mask >> next) & 1)
          continue;
        unsigned nextMask = mask | (1u << next);
        auto proposed =
            current + std::popcount(vertices[last] ^ vertices[next]);
        auto at = size_t(nextMask) * count + next;
        if (proposed < distance[at]) {
          distance[at] = proposed;
          parent[at] = last;
        }
      }
    }
  std::vector<Tour> tours;
  for (unsigned endpoint = 0; endpoint < (1u << star.controls.size());
       ++endpoint) {
    if (!budget.take(2 * count + star.controls.size() + 1))
      return std::vector<Tour>{};
    unsigned best = 0;
    uint32_t bestDistance = infinity;
    for (unsigned last = 0; last < count; ++last) {
      auto candidate = distance[size_t(subsets - 1) * count + last] +
                       std::popcount(vertices[last] ^ endpoint);
      if (candidate < bestDistance) {
        bestDistance = candidate;
        best = last;
      }
    }
    std::vector<unsigned> path;
    unsigned mask = subsets - 1;
    int last = best;
    while (last >= 0) {
      path.push_back(vertices[last]);
      int previous = parent[size_t(mask) * count + last];
      mask ^= 1u << last;
      last = previous;
    }
    std::reverse(path.begin(), path.end());
    Tour tour;
    tour.cx = bestDistance;
    for (unsigned vertex : path)
      tour.waypoints.push_back(expandMask(vertex, star.controls));
    tour.waypoints.push_back(expandMask(endpoint, star.controls));
    tours.push_back(std::move(tour));
  }
  std::stable_sort(tours.begin(), tours.end(),
                   [](const Tour &a, const Tour &b) { return a.cx < b.cx; });
  return tours;
}

/// Wider stars use the same carried-frame contract. The shortest immediate
/// Hamming move is only an ordering heuristic; there is no optimality claim.
inline std::vector<Tour> greedyTours(const Star &star,
                                     const CommutingLimits &limits,
                                     uint64_t rowWork, Budget &budget) {
  Tour open;
  QubitSet current;
  std::vector<bool> done(star.demands.size());
  for (size_t step = 0; step < star.demands.size(); ++step) {
    if (!budget.take(work(star.demands.size(), rowWork)))
      return {};
    size_t selected = star.demands.size();
    unsigned best = std::numeric_limits<unsigned>::max();
    for (size_t i = 0; i < star.demands.size(); ++i)
      if (!done[i]) {
        auto distance = (current ^ star.demands[i].controls).count();
        if (selected == star.demands.size() || distance < best) {
          selected = i;
          best = distance;
        }
      }
    done[selected] = true;
    current = star.demands[selected].controls;
    open.cx += best;
    open.waypoints.push_back(current);
  }
  std::vector<Tour> tours;
  tours.push_back(open);
  // Keeping the open end, restoring this pivot, and ending at another demand
  // expose distinct exit frames without pretending one dominates the others.
  std::vector<QubitSet> endpoints{{}};
  for (const auto &demand : star.demands)
    if (demand.controls)
      endpoints.push_back(demand.controls);
  for (const auto &endpoint : endpoints) {
    if (tours.size() >= limits.maxCandidates)
      break;
    if (endpoint == current)
      continue;
    if (!budget.take(work(open.waypoints.size() + 1, rowWork)))
      return {};
    Tour tour = open;
    tour.cx += (current ^ endpoint).count();
    tour.waypoints.push_back(endpoint);
    tours.push_back(std::move(tour));
  }
  return tours;
}

inline std::optional<std::vector<Star>>
makeStars(const std::vector<Gate> &word, const std::vector<unsigned> &wires,
          uint64_t rowWork, Budget &budget) {
  if (!budget.take(wires.size() + 1))
    return std::nullopt;
  std::unordered_map<unsigned, unsigned> position;
  std::vector<Star> stars;
  for (unsigned i = 0; i < wires.size(); ++i) {
    position[wires[i]] = i;
    stars.push_back({wires[i], {}, {}});
  }
  using Axis = tcount::detail::Axis;
  using AxisHash = tcount::detail::AxisHash;
  std::vector<std::unordered_map<Axis, size_t, AxisHash>> locations(
      wires.size());
  std::vector<QubitSet> controls(wires.size());
  for (const auto &gate : word) {
    if (!budget.take(rowWork + gate.pauli.weight() + 1))
      return std::nullopt;
    unsigned pivot = 0;
    for (unsigned wire : gate.pauli.z.set_bits())
      pivot = std::max(pivot, position.at(wire));
    QubitSet mask = gate.pauli.z ^ QubitSet::single(wires[pivot]);
    controls[pivot] |= mask;
    auto [where, inserted] = locations[pivot].try_emplace(
        Axis{{}, mask}, stars[pivot].demands.size());
    if (inserted)
      stars[pivot].demands.push_back({mask, {}});
    stars[pivot].demands[where->second].angles.push_back(
        gate.pauli.sign ? gate.angle.negated() : gate.angle);
  }
  for (unsigned i = 0; i < stars.size(); ++i) {
    if (!budget.take(controls[i].count() + 1))
      return std::nullopt;
    for (unsigned wire : controls[i].set_bits())
      stars[i].controls.push_back(wire);
    std::sort(
        stars[i].controls.begin(), stars[i].controls.end(),
        [&](unsigned a, unsigned b) { return position[a] < position[b]; });
  }
  std::reverse(stars.begin(), stars.end());
  return stars;
}

inline bool appendStar(CommutingImplementation &candidate, const Star &star,
                       const Tour &tour, const Options &options,
                       uint64_t rowWork, Budget &budget,
                       CommutingCandidates &result) {
  QubitSet current;
  std::vector<bool> emitted(star.demands.size());
  auto emitRotations = [&]() {
    if (!budget.take(work(star.demands.size() + 1, rowWork)))
      return false;
    for (size_t i = 0; i < star.demands.size(); ++i)
      if (!emitted[i] && star.demands[i].controls == current) {
        const auto &angles = star.demands[i].angles;
        if (angles.size() > options.maxGates - candidate.body.size()) {
          result.status = CommutingStatus::Exhausted;
          result.error = "commuting-region native gate limit exceeded";
          return false;
        }
        if (!budget.take(angles.size()))
          return false;
        for (Angle angle : angles)
          candidate.body.push_back(
              Gate::rotation({{}, QubitSet::single(star.pivot)}, angle));
        emitted[i] = true;
      }
    return true;
  };
  if (!emitRotations())
    return false;
  for (const auto &waypoint : tour.waypoints) {
    if (!budget.take(rowWork + star.controls.size()))
      return false;
    QubitSet changed = current ^ waypoint;
    for (unsigned wire : star.controls)
      if (changed.test(wire)) {
        auto native = nativeWord({{Kind::CX, wire, star.pivot}}, options);
        if (native.size() > options.maxGates - candidate.body.size()) {
          result.status = CommutingStatus::Exhausted;
          result.error = "commuting-region native gate limit exceeded";
          return false;
        }
        if (!budget.take(native.size()))
          return false;
        candidate.body.insert(candidate.body.end(), native.begin(),
                              native.end());
        candidate.exitCliffords.insert(candidate.exitCliffords.end(),
                                       native.begin(), native.end());
        current ^= QubitSet::single(wire);
        if (!emitRotations())
          return false;
      }
  }
  if (std::find(emitted.begin(), emitted.end(), false) != emitted.end()) {
    result.status = CommutingStatus::InvalidProof;
    result.error = "commuting parity tour left an unimplemented rotation";
    return false;
  }
  return true;
}
} // namespace commuting_detail

/// Synthesize independent commuting windows without assumed zero ancillas.
/// Named Clifford diagonalizers and parity CXs are retained as an explicit
/// quantum exit frame; only the whole-program caller may discharge that frame
/// against terminal measurements. Candidate truncation is a bounded beam, not
/// dominance between different frames or an optimality proof.
inline CommutingCandidates
solveCommutingRegion(unsigned width, const std::vector<Gate> &rotations,
                     const Options &options, const CommutingLimits &limits,
                     SearchBudget &shared, const CommutingScorer &scorer = {}) {
  using namespace commuting_detail;
  CommutingCandidates result;
  Budget budget(shared, limits.maxWork, result);
  auto unsupported = [&](const char *message) {
    result.status = CommutingStatus::Unsupported;
    result.error = message;
    return result;
  };
  if (!width || !limits.maxCandidates || !options.maxGates ||
      rotations.size() > options.maxGates ||
      (!options.nativeCX && !options.nativeCZ))
    return unsupported("invalid commuting-region dimensions or native profile");
  for (double cost : {options.singleQubitCost, options.tCost, options.cxCost,
                      options.czCost, options.swapCost})
    if (!std::isfinite(cost) || cost < 0)
      return unsupported("invalid commuting-region native costs");
  uint64_t rowWork = uint64_t(width) / 64 + 1;
  if (!budget.take(work(rotations.size() + 1, rowWork)))
    return result;
  std::vector<Pauli> basis;
  pbc::CommutingObservables independent;
  QubitSet active;
  bool diagonal = true;
  for (const auto &gate : rotations) {
    if (gate.kind != Kind::Rotate || !gate.pauli.weight() ||
        !(gate.pauli.x | gate.pauli.z).fits(width) ||
        (!gate.angle.symbol && gate.angle.denominator <= 0))
      return unsupported("commuting-region input must contain valid rotations");
    if (!budget.take(work(2 * uint64_t(independent.size()) + 1, rowWork)))
      return result;
    unsigned previous = independent.size();
    if (!independent.insert(gate.pauli))
      return unsupported("region rotations do not commute");
    if (previous != independent.size())
      basis.push_back(gate.pauli);
    active |= gate.pauli.x | gate.pauli.z;
    diagonal &= !gate.pauli.x;
  }
  result.rank = independent.size();
  if (rotations.empty()) {
    result.completedImplementations = 1;
    result.wireOrders = 1;
    if (scorer) {
      SearchBudget scoring(shared, budget.remaining());
      auto before = shared.work();
      auto score = scorer({}, scoring);
      budget.accountExternal(shared.work() - before);
      if (scoring.stopped()) {
        result.status = CommutingStatus::Exhausted;
        result.error = "commuting completion scoring budget exhausted";
        return result;
      }
      if (!score)
        return result;
      if (std::any_of(score->begin(), score->end(),
                      [](double x) { return !std::isfinite(x); }))
        return unsupported("commuting completion score must be finite");
    }
    result.candidates.push_back({});
    return result;
  }

  std::vector<std::vector<Gate>> diagonalizers;
  if (diagonal)
    diagonalizers.push_back({});
  if (limits.maxDiagonalizers) {
    std::vector<unsigned> pivots;
    for (unsigned wire : active.set_bits()) {
      pivots.push_back(wire);
      if (pivots.size() == basis.size())
        break;
    }
    ReadoutLimits readoutLimits{limits.maxDiagonalizers, budget.remaining()};
    auto before = shared.work();
    auto readouts = generateReadoutCandidates(width, basis, pivots, options,
                                              readoutLimits, shared);
    budget.accountExternal(shared.work() - before);
    for (const auto &readout : readouts.candidates)
      diagonalizers.push_back(readout.gates());
    if (readouts.status == ReadoutStatus::InvalidProof) {
      result.status = CommutingStatus::InvalidProof;
      result.error = readouts.error;
      return result;
    }
    if (readouts.status == ReadoutStatus::Exhausted) {
      result.status = CommutingStatus::Exhausted;
      result.error = readouts.error;
      return result;
    }
  }
  if (diagonalizers.empty())
    return unsupported("commuting region has no available diagonalizer");

  struct Selected {
    size_t ordinal, scheme;
    CommutingImplementation implementation;
  };
  struct Ranked {
    CostRank score;
    uint64_t ordinal;
    size_t diagonalizer;
    // Within one fixed diagonalizer, the rest of the exit is an exact CNOT
    // network. Its signed Z images identify its linear map and zero phase.
    std::vector<Pauli> exitRows;
    CommutingImplementation implementation;
  };
  std::vector<Selected> selected;
  std::vector<Ranked> shortlisted;
  const bool completedScoring = limits.exhaustiveSmallRank || bool(scorer);
  size_t scheme = 0;
  for (size_t diagonalizerId = 0; diagonalizerId < diagonalizers.size();
       ++diagonalizerId) {
    const auto &diagonalizer = diagonalizers[diagonalizerId];
    if (!budget.take(work(rotations.size() + diagonalizer.size() + 1, rowWork)))
      return result;
    auto diagonalWord = rotations;
    QubitSet wires;
    for (auto &gate : diagonalWord) {
      for (const auto &clifford : diagonalizer) {
        if (!budget.take(rowWork))
          return result;
        gate.pauli = tcount::detail::conjugate(gate.pauli, clifford);
      }
      if (gate.pauli.x) {
        result.status = CommutingStatus::InvalidProof;
        result.error = "commuting diagonalizer did not diagonalize its span";
        return result;
      }
      wires |= gate.pauli.z;
    }
    std::vector<unsigned> wireOrder;
    for (unsigned wire : wires.set_bits())
      wireOrder.push_back(wire);
    const bool exhaustive =
        limits.exhaustiveSmallRank && result.rank <= 4 && wireOrder.size() <= 4;
    for (unsigned order = 0;; ++order) {
      if (!exhaustive && order)
        std::reverse(wireOrder.begin(), wireOrder.end());
      ++result.wireOrders;
      auto stars = makeStars(diagonalWord, wireOrder, rowWork, budget);
      if (!stars)
        return result;
      std::vector<CommutingImplementation> frontier{
          {diagonalizer, diagonalizer}};
      for (const auto &star : *stars) {
        if (star.demands.empty())
          continue;
        auto exact = exactTours(star, limits, budget);
        if (exhaustive && !exact)
          return unsupported(
              "small-rank exhaustive tour exceeds its DP limits");
        auto tours = exact ? std::move(*exact)
                           : greedyTours(star, limits, rowWork, budget);
        if (result.status == CommutingStatus::Exhausted)
          return result;
        if (!exhaustive && tours.size() > limits.maxCandidates)
          tours.resize(limits.maxCandidates);
        std::vector<CommutingImplementation> next;
        for (const auto &prefix : frontier)
          for (const auto &tour : tours) {
            if (!budget.take(
                    work(prefix.body.size() + prefix.exitCliffords.size() + 1,
                         rowWork)))
              return result;
            auto candidate = prefix;
            if (!appendStar(candidate, star, tour, options, rowWork, budget,
                            result))
              return result;
            next.push_back(std::move(candidate));
          }
        if (exhaustive) {
          // At most 2^(r*(r-1)/2) <=64 exits for r<=4. Do not discard a
          // distinct exit merely because its native prefix costs more.
          frontier = std::move(next);
          continue;
        }
        // This is explicitly bounded search. Prefix rank does not establish
        // dominance across the different frames kept by the frontier.
        std::vector<std::pair<CostRank, size_t>> ranks;
        for (size_t i = 0; i < next.size(); ++i) {
          if (!budget.take(uint64_t(width) + next[i].body.size() + 1))
            return result;
          ranks.push_back({circuitScore(width, next[i].body, options), i});
        }
        std::sort(ranks.begin(), ranks.end());
        frontier.clear();
        for (size_t i = 0;
             i < std::min<size_t>(ranks.size(), limits.maxCandidates); ++i)
          frontier.push_back(std::move(next[ranks[i].second]));
      }
      if (completedScoring) {
        for (auto &candidate : frontier) {
          uint64_t ordinal = result.completedImplementations++;
          bool duplicate = false;
          for (const auto &old : shortlisted) {
            if (!budget.take(work(candidate.body.size() +
                                      candidate.exitCliffords.size() + 1,
                                  rowWork)))
              return result;
            if (sameWord(old.implementation.body, candidate.body) &&
                sameWord(old.implementation.exitCliffords,
                         candidate.exitCliffords)) {
              duplicate = true;
              break;
            }
          }
          if (duplicate)
            continue;
          std::optional<CostRank> score;
          if (scorer) {
            SearchBudget scoring(shared, budget.remaining());
            auto before = shared.work();
            score = scorer(candidate, scoring);
            budget.accountExternal(shared.work() - before);
            if (scoring.stopped()) {
              result.status = CommutingStatus::Exhausted;
              result.error = "commuting completion scoring budget exhausted";
              return result;
            }
          } else {
            if (!budget.take(uint64_t(width) + candidate.body.size() + 1))
              return result;
            score = circuitScore(width, candidate.body, options);
          }
          if (!score)
            continue;
          if (std::any_of(score->begin(), score->end(),
                          [](double x) { return !std::isfinite(x); }))
            return unsupported("commuting completion score must be finite");
          std::vector<Pauli> exitRows;
          if (exhaustive) {
            // Canonical physical-wire order, independent of the visitation
            // order. The fixed diagonalizer is part of this exact frame key.
            for (unsigned wire : wires.set_bits()) {
              Pauli row{{}, QubitSet::single(wire)};
              if (!budget.take(work(candidate.exitCliffords.size() -
                                        diagonalizer.size() + 1,
                                    rowWork)))
                return result;
              for (size_t i = diagonalizer.size();
                   i < candidate.exitCliffords.size(); ++i)
                row =
                    tcount::detail::conjugate(row, candidate.exitCliffords[i]);
              exitRows.push_back(std::move(row));
            }
          }
          auto equivalent = shortlisted.end();
          if (exhaustive)
            equivalent = std::find_if(
                shortlisted.begin(), shortlisted.end(), [&](const Ranked &old) {
                  return old.diagonalizer == diagonalizerId &&
                         old.exitRows == exitRows;
                });
          if (equivalent != shortlisted.end()) {
            if (*score >= equivalent->score)
              continue;
            *equivalent = {*score, ordinal, diagonalizerId, std::move(exitRows),
                           std::move(candidate)};
          } else {
            shortlisted.push_back({*score, ordinal, diagonalizerId,
                                   std::move(exitRows), std::move(candidate)});
          }
          std::stable_sort(shortlisted.begin(), shortlisted.end(),
                           [](const Ranked &a, const Ranked &b) {
                             return std::tie(a.score, a.ordinal) <
                                    std::tie(b.score, b.ordinal);
                           });
          if (shortlisted.size() > limits.maxCandidates)
            shortlisted.resize(limits.maxCandidates);
          // Publication is transactional: a later tour or callback failure
          // cannot invalidate the previously retained complete proposals.
          std::vector<CommutingImplementation> complete;
          for (const auto &ranked : shortlisted) {
            if (!budget.take(
                    work(ranked.implementation.body.size() +
                             ranked.implementation.exitCliffords.size() + 1,
                         rowWork)))
              return result;
            complete.push_back(ranked.implementation);
          }
          result.candidates = std::move(complete);
        }
      } else {
        result.completedImplementations += frontier.size();
        for (size_t i = 0; i < frontier.size(); ++i)
          selected.push_back({i, scheme, std::move(frontier[i])});
        // Preserve the default round-robin policy across basis/order schemes.
        // Rank within a scheme uses the common native prefix cost.
        std::stable_sort(selected.begin(), selected.end(),
                         [](const auto &a, const auto &b) {
                           return std::tie(a.ordinal, a.scheme) <
                                  std::tie(b.ordinal, b.scheme);
                         });
        if (selected.size() > limits.maxCandidates)
          selected.resize(limits.maxCandidates);
        std::vector<CommutingImplementation> complete;
        for (const auto &candidate : selected) {
          if (!budget.take(
                  work(candidate.implementation.body.size() +
                           candidate.implementation.exitCliffords.size() + 1,
                       rowWork)))
            return result;
          complete.push_back(candidate.implementation);
        }
        result.candidates = std::move(complete);
      }
      ++scheme;
      if (exhaustive) {
        if (!budget.take(wireOrder.size() + 1))
          return result;
        if (!std::next_permutation(wireOrder.begin(), wireOrder.end()))
          break;
      } else if (order == 1) {
        break;
      }
    }
  }
  return result;
}
} // namespace cudaq::pauli::graph::detail
