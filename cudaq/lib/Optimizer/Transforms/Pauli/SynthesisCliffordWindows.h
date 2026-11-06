/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CliffordDictionary.h"
#include "SynthesisCost.h"
#include <array>
#include <cmath>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>

// Complete, source-bound native Clifford rewrites across exactly commuting
// interruptions. Shares signed dictionary mechanics with GateCount; weighted
// search and bounded extraction are independent proposal policies.
namespace cudaq::pauli::graph::detail {

struct CliffordWindowLimits {
  unsigned maxLookahead = 64, maxWireSets = 8, maxSelectedGates = 64;
  unsigned maxRewrites = 1024, maxGates = 32768;
  unsigned maxStates = 11544; // 24 one-wire + 11520 two-wire states
  uint64_t maxWork = 50000000;
  uint64_t maxBytes = 512ull * 1024 * 1024;
};
enum class CliffordWindowStatus {
  Complete,
  Unchanged,
  Unsupported,
  Exhausted,
  InvalidProof
};
struct CliffordWindowResult;
CliffordWindowResult
optimizeCliffordWindows(unsigned, const std::vector<Gate> &, const Options &,
                        const CliffordWindowLimits &, SearchBudget &,
                        std::vector<Gate> *inPlace = nullptr);

namespace clifford_window_detail {
inline bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](const Gate &x, const Gate &y) {
                      return x.kind == y.kind && x.a == y.a && x.b == y.b &&
                             x.pauli == y.pauli && x.angle == y.angle;
                    });
}
struct Source {
  unsigned width;
  std::vector<Gate> gates;
};
inline bool clifford(const Gate &gate) {
  switch (gate.kind) {
  case Kind::H:
  case Kind::S:
  case Kind::Sdg:
  case Kind::X:
  case Kind::Y:
  case Kind::Z:
  case Kind::CX:
  case Kind::CZ:
  case Kind::Swap:
    return true;
  default:
    return false;
  }
}
inline QubitSet support(const Gate &gate) {
  if (gate.kind == Kind::Phase)
    return {};
  if (gate.kind == Kind::Rotate)
    return gate.pauli.x | gate.pauli.z;
  auto bits = QubitSet::single(gate.a);
  if (gate.kind == Kind::CX || gate.kind == Kind::CZ || gate.kind == Kind::Swap)
    bits.set(gate.b);
  return bits;
}
inline Gate compact(Gate gate, const std::vector<unsigned> &wires) {
  auto index = [&](unsigned q) {
    return unsigned(std::find(wires.begin(), wires.end(), q) - wires.begin());
  };
  if (gate.kind == Kind::Rotate) {
    Pauli mapped{{}, {}, gate.pauli.sign};
    for (unsigned q : gate.pauli.x.set_bits())
      mapped.x.set(index(q));
    for (unsigned q : gate.pauli.z.set_bits())
      mapped.z.set(index(q));
    gate.pauli = std::move(mapped);
  } else if (gate.kind != Kind::Phase) {
    gate.a = index(gate.a);
    if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
        gate.kind == Kind::Swap)
      gate.b = index(gate.b);
  }
  return gate;
}
inline Gate expand(Gate gate, const std::vector<unsigned> &wires) {
  if (gate.kind != Kind::Phase) {
    gate.a = wires[gate.a];
    if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
        gate.kind == Kind::Swap)
      gate.b = wires[gate.b];
  }
  return gate;
}

using clifford_dictionary::advance;
using clifford_dictionary::identity;
using clifford_dictionary::Key;
using Rank = std::tuple<double, uint64_t, uint64_t>;
inline Rank rank(const std::vector<Gate> &word, const Options &options) {
  uint64_t gates = 0, two = 0;
  for (const auto &g : word) {
    gates += g.kind != Kind::Phase;
    two += g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap;
  }
  return {evaluateNativeCost(word, options).value, gates, two};
}
inline Rank add(const Rank &a, const Rank &b) {
  return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b),
          std::get<2>(a) + std::get<2>(b)};
}
struct Action {
  Gate gate;
  Rank rank;
};
struct Entry {
  Rank rank{};
  Key previous = 0;
  unsigned action = 0;
  bool settled = false;
};
struct Table {
  std::vector<Action> actions;
  std::unordered_map<Key, Entry> entries;
};
} // namespace clifford_window_detail

/// Owns the exact synthesized source. The proof composes exact commutation
/// witnesses, signed Clifford-tableau search and independent scalar-sensitive
/// replay for each replacement. Untouched gates and physical outputs keep their
/// order.
class CertifiedCliffordWindow {
public:
  const std::vector<Gate> &gates() const { return word; }
  const std::vector<Gate> &sourceGates() const { return origin->gates; }
  bool matchesSource(unsigned width, const std::vector<Gate> &source) const {
    return origin->width == width &&
           clifford_window_detail::sameWord(origin->gates, source);
  }

private:
  friend CliffordWindowResult
  optimizeCliffordWindows(unsigned, const std::vector<Gate> &, const Options &,
                          const CliffordWindowLimits &, SearchBudget &,
                          std::vector<Gate> *);
  CertifiedCliffordWindow(
      std::shared_ptr<const clifford_window_detail::Source> origin,
      std::vector<Gate> word)
      : origin(std::move(origin)), word(std::move(word)) {}
  std::shared_ptr<const clifford_window_detail::Source> origin;
  std::vector<Gate> word;
};

struct CliffordWindowResult {
  /// At most one complete proposal. On exhaustion it contains only already
  /// verified replacements, with every other original operation retained.
  std::vector<CertifiedCliffordWindow> alternatives;
  CliffordWindowStatus status = CliffordWindowStatus::Unchanged;
  std::string error;
  uint64_t work = 0, windowsChecked = 0, commutationChecks = 0;
  uint64_t memoryBound = 0;
  unsigned costRejections = 0;
  unsigned rewrites = 0, mapStates = 0, proofChecks = 0;
};

inline CliffordWindowResult
optimizeCliffordWindows(unsigned width, const std::vector<Gate> &synthesized,
                        const Options &options,
                        const CliffordWindowLimits &limits,
                        SearchBudget &shared, std::vector<Gate> *inPlace) {
  namespace lw = clifford_window_detail;
  CliffordWindowResult result;
  uint64_t initialWork = shared.work();
  auto unsupported = [&](const char *message) {
    result.status = CliffordWindowStatus::Unsupported;
    result.error = message;
    result.work = shared.work() - initialWork;
    return result;
  };
  if (inPlace && inPlace != &synthesized)
    return unsupported("in-place Clifford source must be the retained word");
  if (!width || width > 4096 || !limits.maxLookahead || !limits.maxWireSets ||
      !limits.maxSelectedGates || !limits.maxRewrites || !limits.maxGates ||
      !limits.maxStates || synthesized.size() > limits.maxGates ||
      (!options.nativeCX && !options.nativeCZ))
    return unsupported("unsupported Clifford-window dimensions or limits");
  for (double cost : {options.singleQubitCost, options.cxCost, options.czCost,
                      options.swapCost})
    if (!std::isfinite(cost) || cost < 0)
      return unsupported("invalid Clifford-window native costs");
  auto time = shared.timeRemainingMs();
  if (shared.remaining() < 2 || (time && *time < 2)) {
    result.status = CliffordWindowStatus::Exhausted;
    return result;
  }
  SearchBudget budget(shared, std::min(limits.maxWork, shared.remaining() / 2),
                      time ? *time / 2 : 0);
  auto charge = [&](uint64_t amount = 1) {
    if (budget.take(amount))
      return true;
    result.status = CliffordWindowStatus::Exhausted;
    result.error = "Clifford-window work budget exhausted";
    return false;
  };
  std::shared_ptr<const lw::Source> origin;
  std::vector<Gate> ownedCurrent;
  auto &current = inPlace ? *inPlace : ownedCurrent;
  auto finish = [&]() {
    result.work = budget.work();
    if (result.rewrites && result.status == CliffordWindowStatus::Unchanged)
      result.status = CliffordWindowStatus::Complete;
    if (!inPlace && result.rewrites &&
        result.status != CliffordWindowStatus::InvalidProof) {
      result.alternatives.push_back(
          CertifiedCliffordWindow(origin, std::move(current)));
      if (result.status == CliffordWindowStatus::Unchanged)
        result.status = CliffordWindowStatus::Complete;
    }
    return std::move(result);
  };
  const uint64_t limbs = uint64_t(width) / 64 + 1;
  if (!charge(llvm::SaturatingMultiply<uint64_t>(
          3 * uint64_t(synthesized.size()) + 3, limbs)))
    return finish();
  // Bound ALL stored fields before any Gate/QubitSet copy, even operands
  // ignored by a named gate or Phase. Reject rather than normalize: the source
  // certificate deliberately retains exact stored-field matchesSource identity.
  for (const auto &gate : synthesized)
    if (!gate.pauli.x.fits(width) || !gate.pauli.z.fits(width))
      return unsupported("stored Pauli fields exceed the source width");
  // Include the owned source, current/next full circuits, vector growth and
  // corrected-word scratch, plus both dictionaries and bounded queue nodes.
  // Node allowance includes implementation overhead; state/queue cardinalities
  // remain capped independently. Width bounds both dynamically stored axes.
  const uint64_t gateBytes = sizeof(Gate) + 16 * limbs;
  // The borrowed mode updates an already owned, certified incumbent. It
  // retains only local patch/undo storage; the existing complete word belongs
  // to the caller and is neither copied nor used as extra scratch capacity.
  uint64_t gateSlots =
      inPlace
          ? 16 * (uint64_t(limits.maxLookahead) + limits.maxSelectedGates + 1) +
                1024
          : 8 * uint64_t(limits.maxGates) + 1024;
  result.memoryBound = gateSlots * gateBytes +
                       512 * uint64_t(std::min(limits.maxStates, 11544u));
  if (result.memoryBound > limits.maxBytes) {
    result.status = CliffordWindowStatus::Exhausted;
    result.error = "Clifford window memory cap";
    return finish();
  }
  if (!tcount::detail::valid(width, synthesized))
    return unsupported("invalid Clifford-window source circuit");
  for (const auto &gate : synthesized)
    if ((gate.kind == Kind::Rotate && gate.pauli.weight() != 1) ||
        (gate.kind == Kind::CX && !options.nativeCX) ||
        (gate.kind == Kind::CZ && !options.nativeCZ) ||
        (gate.kind == Kind::Swap && !options.nativeSwap))
      return unsupported("Clifford-window source must use its native profile");
  if (!inPlace) {
    origin = std::make_shared<const lw::Source>(lw::Source{width, synthesized});
    current = synthesized;
    result.alternatives.reserve(1);
  }
  if (inPlace) {
    // Delete exact adjacent primitive inverse pairs in one streaming pass.
    // Deletion preserves order of every surviving operation, hence cannot
    // increase any ASAP depth or count. The existing buffer is the stack.
    if (!charge(4 * uint64_t(current.size() + 1) * limbs))
      return finish();
    auto inversePair = [](const Gate &a, const Gate &b) {
      if (a.a != b.a || a.b != b.b)
        return false;
      switch (a.kind) {
      case Kind::H:
      case Kind::X:
      case Kind::Y:
      case Kind::Z:
      case Kind::CX:
      case Kind::CZ:
      case Kind::Swap:
        return a.kind == b.kind;
      case Kind::S:
        return b.kind == Kind::Sdg;
      case Kind::Sdg:
        return b.kind == Kind::S;
      default:
        return false;
      }
    };
    size_t retained = 0;
    for (size_t i = 0; i < current.size(); ++i) {
      if (retained && result.rewrites < limits.maxRewrites &&
          inversePair(current[retained - 1], current[i])) {
        --retained;
        ++result.rewrites;
        ++result.proofChecks;
      } else {
        if (retained != i)
          current[retained] = std::move(current[i]);
        ++retained;
      }
    }
    current.erase(current.begin() + retained, current.end());
  }

  // Invocation-local, profile-bound dictionaries. Each queue contains at most
  // one entry per discovered state (decrease-key via erase/insert), avoiding an
  // unbounded heap of stale paths. No cache-history-dependent work accounting.
  std::array<std::unique_ptr<lw::Table>, 2> tables;
  auto table = [&](unsigned n) -> lw::Table * {
    auto &owned = tables[n - 1];
    if (owned)
      return owned.get();
    const unsigned states = n == 1 ? 24 : 11520;
    if (states >
        limits.maxStates - std::min(limits.maxStates, result.mapStates)) {
      result.status = CliffordWindowStatus::Exhausted;
      result.error = "Clifford dictionary state cap";
      return nullptr;
    }
    if (!charge(states + 128))
      return nullptr;
    auto built = std::make_unique<lw::Table>();
    built->entries.reserve(states);
    auto action = [&](Gate g) {
      built->actions.push_back({g, lw::rank({g}, options)});
    };
    for (const auto &gate : clifford_dictionary::orderedGenerators(n)) {
      if ((gate.kind == Kind::CX && !options.nativeCX) ||
          (gate.kind == Kind::CZ && !options.nativeCZ) ||
          (gate.kind == Kind::Swap && !options.nativeSwap))
        continue;
      action(gate);
    }
    // All edges are actual native gates. Dijkstra minimizes weighted cost,
    // then gate count and entanglers. Scalar correction is free in this model.
    const auto root = lw::identity(n);
    built->entries.emplace(root, lw::Entry{{}, root, 0, false});
    std::set<std::pair<lw::Rank, lw::Key>> queue{{{}, root}};
    while (!queue.empty()) {
      if (!charge(32))
        return nullptr;
      auto [distance, key] = *queue.begin();
      queue.erase(queue.begin());
      auto &from = built->entries.at(key);
      from.settled = true;
      ++result.mapStates;
      for (unsigned ai = 0; ai < built->actions.size(); ++ai) {
        if (!charge(32 + 16 * n))
          return nullptr;
        const auto &edge = built->actions[ai];
        auto next = lw::advance(key, edge.gate, n);
        auto cost = lw::add(distance, edge.rank);
        if (!std::isfinite(std::get<0>(cost))) {
          result.status = CliffordWindowStatus::Unsupported;
          result.error = "Clifford native cost overflow";
          return nullptr;
        }
        auto found = built->entries.find(next);
        if (found != built->entries.end() && !(cost < found->second.rank))
          continue;
        if (found == built->entries.end()) {
          if (built->entries.size() == states) {
            result.status = CliffordWindowStatus::InvalidProof;
            result.error = "Clifford dictionary exceeded exact group size";
            return nullptr;
          }
          built->entries.emplace(next, lw::Entry{cost, key, ai, false});
        } else {
          if (found->second.settled) {
            result.status = CliffordWindowStatus::InvalidProof;
            result.error = "Clifford shortest-path ordering failed";
            return nullptr;
          }
          queue.erase({found->second.rank, next});
          found->second = {cost, key, ai, false};
        }
        queue.emplace(cost, next);
      }
    }
    if (built->entries.size() != states) {
      result.status = CliffordWindowStatus::InvalidProof;
      result.error = "native generators did not cover the Clifford group";
      return nullptr;
    }
    owned = std::move(built);
    return owned.get();
  };

  auto commute = [&](const Gate &a, const Gate &b) {
    ++result.commutationChecks;
    if (!charge(8 * limbs))
      return false;
    if (b.kind == Kind::Phase)
      return true;
    auto left = lw::support(a), right = lw::support(b);
    if (!(left & right))
      return true;
    // Signed equality is essential: an axis sign reversal does not commute
    // with an arbitrary/opaque rotation, even though its support is unchanged.
    if (b.kind == Kind::Rotate)
      return tcount::detail::conjugate(b.pauli, a) == b.pauli;
    if (!charge(256))
      return false;
    std::vector<unsigned> wires;
    auto both = left | right;
    for (unsigned q : both.set_bits())
      wires.push_back(q);
    auto x = lw::compact(a, wires), y = lw::compact(b, wires);
    ++result.proofChecks;
    return graph::equivalent(wires.size(), {x, y}, {y, x});
  };

  const unsigned lookahead = std::min(limits.maxLookahead, 256u);
  const unsigned setCap = std::min(limits.maxWireSets, 32u);
  const unsigned selectedCap = std::min(limits.maxSelectedGates, 128u);
  for (size_t seed = 0; seed < current.size();) {
    if (!charge())
      return finish();
    if (!lw::clifford(current[seed])) {
      ++seed;
      continue;
    }
    if (result.rewrites == limits.maxRewrites) {
      result.status = CliffordWindowStatus::Exhausted;
      result.error = "Clifford-window rewrite limit reached";
      return finish();
    }
    size_t end = seed + std::min<size_t>(lookahead, current.size() - seed);
    std::vector<std::vector<unsigned>> sets;
    auto base = lw::support(current[seed]);
    if (!charge((lookahead + 4 * setCap) * limbs))
      return finish();
    std::vector<unsigned> pair;
    for (unsigned q : base.set_bits())
      pair.push_back(q);
    sets.push_back(pair);
    if (base.count() == 1)
      for (size_t at = seed + 1; at < end && sets.size() < setCap; ++at) {
        if (!lw::clifford(current[at]))
          continue;
        auto touched = lw::support(current[at]);
        for (unsigned q : touched.set_bits()) {
          if (base.test(q))
            continue;
          auto wires = pair;
          wires.push_back(q);
          std::sort(wires.begin(), wires.end());
          if (std::find(sets.begin(), sets.end(), wires) == sets.end())
            sets.push_back(std::move(wires));
          if (sets.size() == setCap)
            break;
        }
      }
    bool rewritten = false;
    for (const auto &wires : sets) {
      ++result.windowsChecked;
      std::vector<size_t> selected;
      std::vector<Gate> local;
      for (size_t at = seed; at < end; ++at) {
        if (!charge((4 + selected.size()) * limbs))
          return finish();
        const auto &gate = current[at];
        auto touched = lw::support(gate);
        bool inside = lw::clifford(gate);
        for (unsigned q : touched.set_bits())
          inside &= std::find(wires.begin(), wires.end(), q) != wires.end();
        if (inside) {
          selected.push_back(at);
          local.push_back(lw::compact(gate, wires));
          if (selected.size() == selectedCap)
            break;
          continue;
        }
        bool crosses = true;
        for (size_t earlier : selected)
          if (!commute(current[earlier], gate)) {
            crosses = false;
            break;
          }
        if (budget.stopped())
          return finish();
        if (!crosses)
          break;
      }
      if (selected.size() < 2)
        continue;
      auto *dictionary = table(wires.size());
      if (!dictionary)
        return finish();
      if (!charge(local.size() + 1))
        return finish();
      auto target = lw::identity(wires.size());
      for (const auto &gate : local)
        target = lw::advance(target, gate, wires.size());
      auto found = dictionary->entries.find(target);
      if (found == dictionary->entries.end()) {
        result.status = CliffordWindowStatus::InvalidProof;
        result.error = "Clifford target missing from complete table";
        return finish();
      }
      uint64_t nativeSize = std::get<1>(found->second.rank);
      if (nativeSize > limits.maxGates)
        continue;
      if (!charge(nativeSize + 512))
        return finish();
      std::vector<Gate> replacement;
      for (auto key = target; key != lw::identity(wires.size());
           key = dictionary->entries.at(key).previous)
        replacement.push_back(
            dictionary->actions[dictionary->entries.at(key).action].gate);
      std::reverse(replacement.begin(), replacement.end());
      if (!(lw::rank(replacement, options) < lw::rank(local, options)))
        continue;
      bool equivalent = false;
      // The signed tableau fixes the unitary up to an eighth root of unity.
      // Recover it by independent exact replay; no arbitrary-input phase loss.
      for (int scalar = 0; scalar < 8; ++scalar) {
        const uint64_t size = local.size() + replacement.size() + 1;
        if (!charge(llvm::SaturatingAdd<uint64_t>(
                512, llvm::SaturatingMultiply<uint64_t>(size, 3 * size + 16))))
          return finish();
        auto corrected = replacement;
        if (scalar)
          corrected.push_back(Gate::phase({scalar, 4}));
        ++result.proofChecks;
        if (graph::equivalent(wires.size(), local, corrected)) {
          replacement = std::move(corrected);
          equivalent = true;
          break;
        }
      }
      if (!equivalent) {
        result.status = CliffordWindowStatus::InvalidProof;
        result.error = "Clifford native-unitary scalar proof failed";
        return finish();
      }
      size_t retained = current.size() - selected.size();
      if (replacement.size() > limits.maxGates - retained)
        continue;
      if (!charge(llvm::SaturatingMultiply<uint64_t>(
              retained + replacement.size() + 1, limbs)))
        return finish();
      size_t boundary = selected.back() + 1;
      if (inPlace) {
        // All semantic checks above finished before mutation. Keep the exact
        // bounded old window for rollback if the full-circuit depth guard
        // rejects the local improvement. Positive growth may only use capacity
        // already owned by the caller; it must never reallocate the full word.
        std::vector<Gate> patch;
        size_t selectedAt = 0;
        for (size_t at = seed; at < boundary; ++at)
          if (selectedAt < selected.size() && selected[selectedAt] == at)
            ++selectedAt;
          else
            patch.push_back(current[at]);
        for (const auto &gate : replacement)
          patch.push_back(lw::expand(gate, wires));
        size_t oldLength = boundary - seed;
        if (patch.size() > oldLength &&
            patch.size() - oldLength > current.capacity() - current.size())
          continue;
        // Prepay both complete metric traversals and every possible shift,
        // including rollback, before applying a proven local replacement.
        if (!charge(llvm::SaturatingMultiply<uint64_t>(
                32 * uint64_t(current.size() + 1), limbs)))
          return finish();
        auto before = tcount::cost(width, current);
        std::vector<Gate> undo(current.begin() + seed,
                               current.begin() + boundary);
        auto replace = [&](size_t length, const std::vector<Gate> &word) {
          if (word.size() <= length) {
            std::copy(word.begin(), word.end(), current.begin() + seed);
            if (word.size() < length) {
              auto end =
                  std::move(current.begin() + seed + length, current.end(),
                            current.begin() + seed + word.size());
              current.erase(end, current.end());
            }
          } else {
            auto split = word.begin() + length;
            std::copy(word.begin(), split, current.begin() + seed);
            current.insert(current.begin() + seed + length, split, word.end());
          }
        };
        replace(oldLength, patch);
        auto after = tcount::cost(width, current);
        if (after.gates > before.gates || after.t > before.t ||
            after.twoQubit > before.twoQubit || after.depth > before.depth ||
            after.tDepth > before.tDepth) {
          replace(patch.size(), undo);
          ++result.costRejections;
          continue;
        }
      } else {
        std::vector<Gate> next;
        next.reserve(retained + replacement.size());
        next.insert(next.end(), current.begin(), current.begin() + seed);
        size_t selectedAt = 0;
        for (size_t at = seed; at < boundary; ++at)
          if (selectedAt < selected.size() && selected[selectedAt] == at)
            ++selectedAt;
          else
            next.push_back(current[at]);
        for (const auto &gate : replacement)
          next.push_back(lw::expand(gate, wires));
        next.insert(next.end(), current.begin() + boundary, current.end());
        current = std::move(next);
      }
      ++result.rewrites;
      rewritten = true;
      break;
    }
    if (rewritten)
      seed = seed ? seed - 1 : 0;
    else
      ++seed;
  }
  return finish();
}
} // namespace cudaq::pauli::graph::detail
