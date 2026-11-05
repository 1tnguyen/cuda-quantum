/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SynthesisCost.h"
#include <array>
#include <cmath>
#include <memory>
#include <tuple>

namespace cudaq::pauli::graph::detail {

struct LinearWindowLimits {
  unsigned maxLookahead = 64, maxWireSets = 8, maxSelectedGates = 64;
  unsigned maxRewrites = 1024, maxGates = 32768;
  uint64_t maxWork = 50000000;
};
enum class LinearWindowStatus {
  Complete,
  Unchanged,
  Unsupported,
  Exhausted,
  InvalidProof
};
struct LinearWindowResult;
LinearWindowResult optimizeLinearWindows(unsigned, const std::vector<Gate> &,
                                         const Options &,
                                         const LinearWindowLimits &,
                                         SearchBudget &);

namespace linear_window_detail {
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
inline bool linear(const Gate &gate) {
  return gate.kind == Kind::CX || gate.kind == Kind::Swap;
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
  gate.a = wires[gate.a];
  if (gate.kind == Kind::CX || gate.kind == Kind::CZ || gate.kind == Kind::Swap)
    gate.b = wires[gate.b];
  return gate;
}

/// Rows of the binary basis-state map y=A*x, with three bits per row.
/// CX(a,b) adds row a to row b; SWAP exchanges two rows. These permutations
/// have no basis-dependent or scalar phase. A determines both quantum images:
/// X(v)->X(A*v), Z(w)->Z(A^-T*w), not just measured classical bits.
inline unsigned identity(unsigned width) {
  unsigned key = 0;
  for (unsigned q = 0; q < width; ++q)
    key |= (1u << q) << (3 * q);
  return key;
}
inline unsigned advance(unsigned key, const Gate &gate) {
  unsigned a = (key >> (3 * gate.a)) & 7;
  unsigned b = (key >> (3 * gate.b)) & 7;
  if (gate.kind == Kind::CX)
    return key ^ (a << (3 * gate.b));
  return key ^ ((a ^ b) << (3 * gate.a)) ^ ((a ^ b) << (3 * gate.b));
}

using Rank = std::tuple<double, uint64_t, uint64_t>;
struct Action {
  Gate logical;
  std::vector<Gate> native;
  Rank rank;
};
struct Entry {
  Rank rank{};
  unsigned previous = 0, action = 0;
  bool reached = false, settled = false;
};
struct Table {
  unsigned width = 0;
  std::vector<Action> actions;
  std::array<Entry, 512> entries;
};

inline Rank rank(const std::vector<Gate> &word, const Options &options) {
  uint64_t two = 0;
  for (const auto &gate : word)
    two += gate.kind == Kind::CX || gate.kind == Kind::CZ ||
           gate.kind == Kind::Swap;
  return {evaluateNativeCost(word, options).value, word.size(), two};
}
inline Rank add(const Rank &a, const Rank &b) {
  return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b),
          std::get<2>(a) + std::get<2>(b)};
}

/// Enumerate the finite native realization family for a logical CX/SWAP.
/// Dijkstra is exact for these generators and weighted costs, not for the
/// unrestricted three-qubit Clifford group or a depth objective.
inline std::vector<Gate> nativeAction(Gate gate, const Options &options) {
  auto cx = [&](unsigned a, unsigned b) {
    std::vector<Gate> best;
    if (options.nativeCX)
      best.push_back({Kind::CX, a, b});
    if (options.nativeCZ) {
      std::vector<Gate> alternative{
          {Kind::H, b}, {Kind::CZ, a, b}, {Kind::H, b}};
      if (best.empty() || rank(alternative, options) < rank(best, options))
        best = std::move(alternative);
    }
    return best;
  };
  if (gate.kind == Kind::CX)
    return cx(gate.a, gate.b);
  std::vector<Gate> best;
  if (options.nativeSwap)
    best.push_back(gate);
  auto alternative = cx(gate.a, gate.b);
  auto middle = cx(gate.b, gate.a);
  alternative.insert(alternative.end(), middle.begin(), middle.end());
  auto last = cx(gate.a, gate.b);
  alternative.insert(alternative.end(), last.begin(), last.end());
  if (best.empty() || rank(alternative, options) < rank(best, options))
    best = std::move(alternative);
  return best;
}
} // namespace linear_window_detail

/// Owns the exact synthesized source. The proof composes exact commutation
/// witnesses, binary linear-map equality and independent native-unitary replay
/// for each replacement. Untouched gates and physical outputs keep their order.
class CertifiedLinearWindow {
public:
  const std::vector<Gate> &gates() const { return word; }
  const std::vector<Gate> &sourceGates() const { return origin->gates; }
  bool matchesSource(unsigned width, const std::vector<Gate> &source) const {
    return origin->width == width &&
           linear_window_detail::sameWord(origin->gates, source);
  }

private:
  friend LinearWindowResult
  optimizeLinearWindows(unsigned, const std::vector<Gate> &, const Options &,
                        const LinearWindowLimits &, SearchBudget &);
  CertifiedLinearWindow(
      std::shared_ptr<const linear_window_detail::Source> origin,
      std::vector<Gate> word)
      : origin(std::move(origin)), word(std::move(word)) {}
  std::shared_ptr<const linear_window_detail::Source> origin;
  std::vector<Gate> word;
};

struct LinearWindowResult {
  /// At most one complete proposal. On exhaustion it contains only already
  /// verified replacements, with every other original operation retained.
  std::vector<CertifiedLinearWindow> alternatives;
  LinearWindowStatus status = LinearWindowStatus::Unchanged;
  std::string error;
  uint64_t work = 0, windowsChecked = 0, commutationChecks = 0;
  unsigned rewrites = 0, mapStates = 0;
};

inline LinearWindowResult
optimizeLinearWindows(unsigned width, const std::vector<Gate> &synthesized,
                      const Options &options, const LinearWindowLimits &limits,
                      SearchBudget &shared) {
  namespace lw = linear_window_detail;
  LinearWindowResult result;
  uint64_t initialWork = shared.work();
  auto unsupported = [&](const char *message) {
    result.status = LinearWindowStatus::Unsupported;
    result.error = message;
    result.work = shared.work() - initialWork;
    return result;
  };
  if (!width || width > 4096 || !limits.maxLookahead || !limits.maxWireSets ||
      !limits.maxSelectedGates || !limits.maxRewrites || !limits.maxGates ||
      synthesized.size() > limits.maxGates ||
      (!options.nativeCX && !options.nativeCZ))
    return unsupported("unsupported linear-window dimensions or limits");
  for (double cost : {options.singleQubitCost, options.cxCost, options.czCost,
                      options.swapCost})
    if (!std::isfinite(cost) || cost < 0)
      return unsupported("invalid linear-window native costs");
  auto time = shared.timeRemainingMs();
  if (shared.remaining() < 2 || (time && *time < 2)) {
    result.status = LinearWindowStatus::Exhausted;
    return result;
  }
  SearchBudget budget(shared, std::min(limits.maxWork, shared.remaining() / 2),
                      time ? *time / 2 : 0);
  auto charge = [&](uint64_t amount = 1) {
    if (budget.take(amount))
      return true;
    result.status = LinearWindowStatus::Exhausted;
    result.error = "linear-window work budget exhausted";
    return false;
  };
  std::shared_ptr<const lw::Source> origin;
  std::vector<Gate> current;
  auto finish = [&]() {
    result.work = budget.work();
    if (result.rewrites && result.status != LinearWindowStatus::InvalidProof) {
      result.alternatives.push_back(
          CertifiedLinearWindow(origin, std::move(current)));
      if (result.status == LinearWindowStatus::Unchanged)
        result.status = LinearWindowStatus::Complete;
    }
    return std::move(result);
  };
  const uint64_t limbs = uint64_t(width) / 64 + 1;
  if (!charge(llvm::SaturatingMultiply<uint64_t>(
          3 * uint64_t(synthesized.size()) + 3, limbs)))
    return finish();
  if (!tcount::detail::valid(width, synthesized))
    return unsupported("invalid linear-window source circuit");
  for (const auto &gate : synthesized)
    if ((gate.kind == Kind::Rotate && gate.pauli.weight() != 1) ||
        (gate.kind == Kind::CX && !options.nativeCX) ||
        (gate.kind == Kind::CZ && !options.nativeCZ) ||
        (gate.kind == Kind::Swap && !options.nativeSwap))
      return unsupported("linear-window source must use its native profile");
  origin = std::make_shared<const lw::Source>(lw::Source{width, synthesized});
  current = synthesized;
  result.alternatives.reserve(1);

  // Two immutable invocation-local dictionaries, with six and 168 reachable
  // states respectively. Their storage is independent of physical wire count.
  std::array<std::unique_ptr<lw::Table>, 2> tables;
  auto table = [&](unsigned n) -> lw::Table * {
    auto &owned = tables[n - 2];
    if (owned)
      return owned.get();
    if (!charge(512 + 128))
      return nullptr;
    auto built = std::make_unique<lw::Table>();
    built->width = n;
    for (unsigned a = 0; a < n; ++a)
      for (unsigned b = 0; b < n; ++b) {
        if (a == b)
          continue;
        for (Kind kind : {Kind::CX, Kind::Swap}) {
          if (kind == Kind::Swap && a > b)
            continue;
          Gate logical{kind, a, b};
          auto native = lw::nativeAction(logical, options);
          auto rank = lw::rank(native, options);
          if (!std::isfinite(std::get<0>(rank))) {
            result.status = LinearWindowStatus::Unsupported;
            result.error = "linear-window action cost overflow";
            return nullptr;
          }
          built->actions.push_back({logical, std::move(native), rank});
        }
      }
    unsigned initial = lw::identity(n);
    built->entries[initial].reached = true;
    for (;;) {
      if (!charge(512 + built->actions.size()))
        return nullptr;
      unsigned best = 512;
      for (unsigned key = 0; key < 512; ++key) {
        const auto &entry = built->entries[key];
        if (entry.reached && !entry.settled &&
            (best == 512 || entry.rank < built->entries[best].rank))
          best = key;
      }
      if (best == 512)
        break;
      auto &from = built->entries[best];
      from.settled = true;
      ++result.mapStates;
      for (unsigned a = 0; a < built->actions.size(); ++a) {
        const auto &action = built->actions[a];
        unsigned key = lw::advance(best, action.logical);
        auto &to = built->entries[key];
        auto cost = lw::add(from.rank, action.rank);
        if (!std::isfinite(std::get<0>(cost)))
          continue;
        if (!to.reached || cost < to.rank) {
          to = {cost, best, a, true, false};
        }
      }
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
    if (a.kind == Kind::CX && b.kind == Kind::CX)
      return a.a != b.b && a.b != b.a;
    // Do not infer commutation merely from a phase-blind tableau. Replay the
    // two actual gates on <=4 compact wires, retaining every scalar sign.
    if (!charge(256))
      return false;
    std::vector<unsigned> wires;
    auto both = left | right;
    for (unsigned q : both.set_bits())
      wires.push_back(q);
    auto x = lw::compact(a, wires), y = lw::compact(b, wires);
    return graph::equivalent(wires.size(), {x, y}, {y, x});
  };

  const unsigned lookahead = std::min(limits.maxLookahead, 256u);
  const unsigned setCap = std::min(limits.maxWireSets, 32u);
  const unsigned selectedCap = std::min(limits.maxSelectedGates, 128u);
  for (size_t seed = 0; seed < current.size();) {
    if (!charge())
      return finish();
    if (!lw::linear(current[seed])) {
      ++seed;
      continue;
    }
    if (result.rewrites == limits.maxRewrites) {
      result.status = LinearWindowStatus::Exhausted;
      result.error = "linear-window rewrite limit reached";
      return finish();
    }
    size_t end = seed + std::min<size_t>(lookahead, current.size() - seed);
    std::vector<std::vector<unsigned>> sets;
    auto base = lw::support(current[seed]);
    if (!charge(lookahead + 4 * setCap))
      return finish();
    std::vector<unsigned> pair;
    for (unsigned q : base.set_bits())
      pair.push_back(q);
    sets.push_back(pair);
    for (size_t at = seed + 1; at < end && sets.size() < setCap; ++at) {
      if (!lw::linear(current[at]))
        continue;
      for (unsigned q : {current[at].a, current[at].b}) {
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
        if (!charge(4 + selected.size()))
          return finish();
        const auto &gate = current[at];
        bool inside =
            lw::linear(gate) &&
            std::find(wires.begin(), wires.end(), gate.a) != wires.end() &&
            std::find(wires.begin(), wires.end(), gate.b) != wires.end();
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
      unsigned target = lw::identity(wires.size());
      for (const auto &gate : local)
        target = lw::advance(target, gate);
      if (!dictionary->entries[target].reached)
        continue;
      uint64_t nativeSize = std::get<1>(dictionary->entries[target].rank);
      if (nativeSize > limits.maxGates)
        continue;
      if (!charge(nativeSize + 512))
        return finish();
      std::vector<unsigned> path;
      for (unsigned key = target; key != lw::identity(wires.size());
           key = dictionary->entries[key].previous)
        path.push_back(dictionary->entries[key].action);
      std::reverse(path.begin(), path.end());
      std::vector<Gate> replacement, logical;
      for (unsigned action : path) {
        const auto &step = dictionary->actions[action];
        logical.push_back(step.logical);
        replacement.insert(replacement.end(), step.native.begin(),
                           step.native.end());
      }
      if (!(lw::rank(replacement, options) < lw::rank(local, options)))
        continue;
      uint64_t size = local.size() + replacement.size();
      if (!charge(llvm::SaturatingAdd<uint64_t>(
              512, llvm::SaturatingMultiply<uint64_t>(size, 3 * size + 16))))
        return finish();
      // Independent truth-table replay does not use the row-update search.
      // Since both logical words permute basis states with coefficient +1,
      // agreement on all 2^n states is full unitary equality, including phase.
      auto evaluate = [](unsigned bits, const std::vector<Gate> &word) {
        for (const auto &gate : word) {
          bool a = (bits >> gate.a) & 1, b = (bits >> gate.b) & 1;
          if (gate.kind == Kind::CX)
            bits ^= unsigned(a) << gate.b;
          else if (a != b)
            bits ^= (1u << gate.a) | (1u << gate.b);
        }
        return bits;
      };
      bool equivalent = true;
      for (unsigned bits = 0; bits < (1u << wires.size()); ++bits)
        equivalent &= evaluate(bits, local) == evaluate(bits, logical);
      equivalent &= graph::equivalent(wires.size(), local, replacement);
      if (!equivalent) {
        result.status = LinearWindowStatus::InvalidProof;
        result.error = "linear-window map or native-unitary proof failed";
        return finish();
      }
      size_t retained = current.size() - selected.size();
      if (replacement.size() > limits.maxGates - retained)
        continue;
      if (!charge(llvm::SaturatingMultiply<uint64_t>(
              retained + replacement.size() + 1, limbs)))
        return finish();
      std::vector<Gate> next;
      next.reserve(retained + replacement.size());
      next.insert(next.end(), current.begin(), current.begin() + seed);
      size_t selectedAt = 0, boundary = selected.back() + 1;
      for (size_t at = seed; at < boundary; ++at)
        if (selectedAt < selected.size() && selected[selectedAt] == at)
          ++selectedAt;
        else
          next.push_back(current[at]);
      for (const auto &gate : replacement)
        next.push_back(lw::expand(gate, wires));
      next.insert(next.end(), current.begin() + boundary, current.end());
      current = std::move(next);
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
