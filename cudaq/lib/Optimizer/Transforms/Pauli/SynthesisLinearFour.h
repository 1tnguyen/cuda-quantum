/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "SynthesisLinearWindows.h"
#include <bit>
#include <limits>

namespace cudaq::pauli::graph::detail {

struct LinearFourLimits {
  unsigned maxLookahead = 64, maxWireSets = 16, maxSelectedGates = 64;
  unsigned maxRewrites = 128, maxGates = 32768;
  uint64_t maxWork = 50000000, maxTableWork = 20000000;
  uint64_t maxTableBytes = 64 * 1024 * 1024;
  /// Hard bounded by 362881: one initial state plus one successful relaxation
  /// for each of the 20160*18 directed edges. Allocation is prepaid and fixed.
  unsigned maxHeapEntries = 362881;
};
enum class LinearFourStatus {
  Complete,
  Unchanged,
  Unsupported,
  Exhausted,
  InvalidProof
};

namespace linear_four_detail {
using Rank = linear_window_detail::Rank;
using Action = linear_window_detail::Action;
inline unsigned identity() { return 0x8421; }
inline unsigned advance(unsigned key, const Gate &gate) {
  unsigned a = (key >> (4 * gate.a)) & 15;
  unsigned b = (key >> (4 * gate.b)) & 15;
  return gate.kind == Kind::CX
             ? key ^ (a << (4 * gate.b))
             : key ^ ((a ^ b) << (4 * gate.a)) ^ ((a ^ b) << (4 * gate.b));
}
struct Entry {
  Rank rank{};
  uint16_t previous = 0;
  uint8_t action = 0;
  bool reached = false, settled = false;
};
struct Pending {
  Rank rank;
  uint16_t key;
};
inline bool later(const Pending &a, const Pending &b) {
  return std::tie(a.rank, a.key) > std::tie(b.rank, b.key);
}
inline std::optional<SearchBudget> slice(SearchBudget &parent, uint64_t limit) {
  auto time = parent.timeRemainingMs();
  if (!limit || parent.remaining() < 2 || (time && *time < 2))
    return std::nullopt;
  return SearchBudget(parent, std::min(limit, parent.remaining() / 2),
                      time ? *time / 2 : 0);
}
} // namespace linear_four_detail

struct LinearFourImplementation {
  std::vector<Gate> logical, native;
};
class LinearFourDictionary;
struct LinearFourDictionaryResult;
LinearFourDictionaryResult buildLinearFourDictionary(const Options &,
                                                     const LinearFourLimits &,
                                                     SearchBudget &);

/// Immutable, invocation-owned GL(4,2) dictionary. No global cache or hidden
/// profile state. Its 20160 reachable maps preserve the entire quantum unitary:
/// basis map A fixes X(v)->X(A*v) and Z(w)->Z(A^-T*w), with scalar phase +1.
/// Shortest means additive weighted native cost in the CX/SWAP realization
/// family, with gate/entangler-count tie breaks, not unrestricted Clifford or
/// depth optimality. Actual native realizations still receive independent proof
/// before any whole-source certificate is issued.
class LinearFourDictionary {
public:
  unsigned reachableStates() const { return states; }
  uint64_t storageBytes() const { return sizeof(*this) + actionBytes; }
  bool contains(unsigned key) const {
    return key < entries.size() && entries[key].reached;
  }
  bool matchesProfile(const Options &options) const {
    return profile.nativeCX == options.nativeCX &&
           profile.nativeCZ == options.nativeCZ &&
           profile.nativeSwap == options.nativeSwap &&
           profile.singleQubitCost == options.singleQubitCost &&
           profile.cxCost == options.cxCost &&
           profile.czCost == options.czCost &&
           profile.swapCost == options.swapCost;
  }
  std::optional<LinearFourImplementation>
  implementation(unsigned key, unsigned maxGates, SearchBudget &budget) const {
    if (!contains(key) || std::get<1>(entries[key].rank) > maxGates)
      return std::nullopt;
    uint64_t nativeSize = std::get<1>(entries[key].rank);
    // Each logical action emits at least one native gate, bounding both
    // predecessor traversal and allocation by the already known native size.
    if (!budget.take(3 * nativeSize + 1))
      return std::nullopt;
    std::vector<unsigned> path;
    path.reserve(nativeSize);
    while (key != linear_four_detail::identity()) {
      if (path.size() >= nativeSize)
        return std::nullopt;
      path.push_back(entries[key].action);
      key = entries[key].previous;
    }
    LinearFourImplementation result;
    result.logical.reserve(path.size());
    result.native.reserve(nativeSize);
    for (auto at = path.rbegin(); at != path.rend(); ++at) {
      const auto &action = actions[*at];
      result.logical.push_back(action.logical);
      result.native.insert(result.native.end(), action.native.begin(),
                           action.native.end());
    }
    return result;
  }

private:
  friend LinearFourDictionaryResult
  buildLinearFourDictionary(const Options &, const LinearFourLimits &,
                            SearchBudget &);
  explicit LinearFourDictionary(const Options &profile) : profile(profile) {}
  Options profile;
  unsigned states = 0;
  uint64_t actionBytes = 0;
  std::vector<linear_four_detail::Action> actions;
  std::array<linear_four_detail::Entry, 65536> entries;
};

struct LinearFourDictionaryResult {
  std::shared_ptr<const LinearFourDictionary> dictionary;
  LinearFourStatus status = LinearFourStatus::Unsupported;
  std::string error;
  uint64_t work = 0, memoryBound = 0;
  unsigned mapStates = 0, peakHeapEntries = 0;
};

inline LinearFourDictionaryResult
buildLinearFourDictionary(const Options &options,
                          const LinearFourLimits &limits,
                          SearchBudget &shared) {
  namespace lf = linear_four_detail;
  namespace lw = linear_window_detail;
  LinearFourDictionaryResult result;
  if ((!options.nativeCX && !options.nativeCZ) || !limits.maxHeapEntries) {
    result.error = "invalid four-wire native profile or heap limit";
    return result;
  }
  for (double cost : {options.singleQubitCost, options.cxCost, options.czCost,
                      options.swapCost})
    if (!std::isfinite(cost) || cost < 0) {
      result.error = "invalid four-wire native costs";
      return result;
    }
  unsigned heapCap = std::min(limits.maxHeapEntries, 362881u);
  // Include the fixed table, exact heap capacity and conservative storage for
  // the <=9-gate action vectors, their growth and temporary lowering words.
  // No allocation precedes this bound.
  uint64_t actionBytes = 18 * (sizeof(lf::Action) + 32 * sizeof(Gate));
  result.memoryBound = sizeof(LinearFourDictionary) + actionBytes +
                       uint64_t(heapCap) * sizeof(lf::Pending) +
                       64 * sizeof(Gate);
  if (result.memoryBound > limits.maxTableBytes) {
    result.status = LinearFourStatus::Exhausted;
    result.error = "four-wire dictionary memory limit exceeded";
    return result;
  }
  auto allowance = lf::slice(shared, limits.maxTableWork);
  if (!allowance) {
    result.status = LinearFourStatus::Exhausted;
    return result;
  }
  auto &budget = *allowance;
  auto finish = [&] {
    result.work = budget.work();
    return std::move(result);
  };
  auto take = [&](uint64_t work) {
    if (budget.take(work))
      return true;
    result.status = LinearFourStatus::Exhausted;
    result.error = "four-wire dictionary work budget exhausted";
    return false;
  };
  if (!take(65536 + uint64_t(heapCap) + 512))
    return finish();
  auto dictionary =
      std::shared_ptr<LinearFourDictionary>(new LinearFourDictionary(options));
  dictionary->actions.reserve(18);
  dictionary->actionBytes = actionBytes;
  for (unsigned a = 0; a < 4; ++a)
    for (unsigned b = 0; b < 4; ++b) {
      if (a == b)
        continue;
      for (Kind kind : {Kind::CX, Kind::Swap}) {
        if (kind == Kind::Swap && a > b)
          continue;
        Gate logical{kind, a, b};
        auto native = lw::nativeAction(logical, options);
        auto cost = lw::rank(native, options);
        if (!std::isfinite(std::get<0>(cost))) {
          result.error = "four-wire native action cost overflow";
          return finish();
        }
        dictionary->actions.push_back({logical, std::move(native), cost});
      }
    }
  std::vector<lf::Pending> pending;
  pending.reserve(heapCap);
  unsigned initial = lf::identity();
  dictionary->entries[initial].reached = true;
  pending.push_back({{}, uint16_t(initial)});
  result.peakHeapEntries = 1;
  while (!pending.empty()) {
    if (!take(2 * std::bit_width(pending.size()) + 1))
      return finish();
    std::pop_heap(pending.begin(), pending.end(), lf::later);
    auto next = pending.back();
    pending.pop_back();
    auto &from = dictionary->entries[next.key];
    if (from.settled || from.rank != next.rank)
      continue;
    from.settled = true;
    ++result.mapStates;
    if (!take(dictionary->actions.size()))
      return finish();
    for (unsigned a = 0; a < dictionary->actions.size(); ++a) {
      const auto &action = dictionary->actions[a];
      unsigned key = lf::advance(next.key, action.logical);
      auto &to = dictionary->entries[key];
      auto cost = lw::add(from.rank, action.rank);
      if (!std::isfinite(std::get<0>(cost)) ||
          (to.reached && !(cost < to.rank)))
        continue;
      if (pending.size() == heapCap) {
        result.status = LinearFourStatus::Exhausted;
        result.error = "four-wire dictionary heap-entry limit exceeded";
        return finish();
      }
      if (!take(2 * std::bit_width(pending.size() + 1) + 1))
        return finish();
      to = {cost, next.key, uint8_t(a), true, false};
      pending.push_back({cost, uint16_t(key)});
      std::push_heap(pending.begin(), pending.end(), lf::later);
      result.peakHeapEntries =
          std::max<unsigned>(result.peakHeapEntries, pending.size());
    }
  }
  if (result.mapStates != 20160) {
    result.error = "four-wire profile cannot represent all finite map costs";
    return finish();
  }
  dictionary->states = result.mapStates;
  result.dictionary = std::move(dictionary);
  result.status = LinearFourStatus::Complete;
  return finish();
}

struct LinearFourResult;
LinearFourResult
optimizeLinearFourWindows(unsigned, const std::vector<Gate> &, const Options &,
                          const LinearFourLimits &,
                          const std::shared_ptr<const LinearFourDictionary> &,
                          SearchBudget &);

class CertifiedLinearFour {
public:
  const std::vector<Gate> &gates() const { return word; }
  const std::vector<Gate> &sourceGates() const { return origin->gates; }
  bool matchesSource(unsigned width, const std::vector<Gate> &source) const {
    return origin->width == width &&
           linear_window_detail::sameWord(origin->gates, source);
  }

private:
  friend LinearFourResult
  optimizeLinearFourWindows(unsigned, const std::vector<Gate> &,
                            const Options &, const LinearFourLimits &,
                            const std::shared_ptr<const LinearFourDictionary> &,
                            SearchBudget &);
  CertifiedLinearFour(
      std::shared_ptr<const linear_window_detail::Source> origin,
      std::vector<Gate> word)
      : origin(std::move(origin)), word(std::move(word)) {}
  std::shared_ptr<const linear_window_detail::Source> origin;
  std::vector<Gate> word;
};
struct LinearFourResult {
  std::vector<CertifiedLinearFour> alternatives;
  LinearFourStatus status = LinearFourStatus::Unchanged;
  std::string error;
  uint64_t work = 0, windowsChecked = 0, commutationChecks = 0;
  unsigned rewrites = 0;
};

/// Independent extension of the existing <=3-wire strategy. The caller owns
/// and may reuse the immutable dictionary across source circuits, while every
/// resulting certificate owns its exact source. An exhausted sweep returns
/// only completed, independently verified replacements. No classical output
/// remapping, ancilla allocation or zero-input assumption is made.
inline LinearFourResult optimizeLinearFourWindows(
    unsigned width, const std::vector<Gate> &synthesized,
    const Options &options, const LinearFourLimits &limits,
    const std::shared_ptr<const LinearFourDictionary> &dictionary,
    SearchBudget &shared) {
  namespace lw = linear_window_detail;
  namespace lf = linear_four_detail;
  LinearFourResult result;
  if (width < 4 || width > 4096 || !dictionary ||
      !dictionary->matchesProfile(options) || !limits.maxLookahead ||
      !limits.maxWireSets || !limits.maxSelectedGates || !limits.maxRewrites ||
      !limits.maxGates || synthesized.size() > limits.maxGates) {
    result.status = LinearFourStatus::Unsupported;
    result.error = "invalid four-wire dimensions, limits or dictionary profile";
    return result;
  }
  auto allowance = lf::slice(shared, limits.maxWork);
  if (!allowance) {
    result.status = LinearFourStatus::Exhausted;
    return result;
  }
  auto &budget = *allowance;
  std::shared_ptr<const lw::Source> origin;
  std::vector<Gate> current;
  auto finish = [&] {
    result.work = budget.work();
    if (result.rewrites && result.status != LinearFourStatus::InvalidProof) {
      result.alternatives.push_back(
          CertifiedLinearFour(origin, std::move(current)));
      if (result.status == LinearFourStatus::Unchanged)
        result.status = LinearFourStatus::Complete;
    }
    return std::move(result);
  };
  auto take = [&](uint64_t amount = 1) {
    if (budget.take(amount))
      return true;
    result.status = LinearFourStatus::Exhausted;
    result.error = "four-wire extraction work budget exhausted";
    return false;
  };
  uint64_t limbs = uint64_t(width) / 64 + 1;
  if (!take(llvm::SaturatingMultiply<uint64_t>(
          3 * uint64_t(synthesized.size()) + 3, limbs)))
    return finish();
  bool valid = tcount::detail::valid(width, synthesized);
  for (const auto &gate : synthesized)
    valid &= !(gate.kind == Kind::Rotate && gate.pauli.weight() != 1) &&
             !(gate.kind == Kind::CX && !options.nativeCX) &&
             !(gate.kind == Kind::CZ && !options.nativeCZ) &&
             !(gate.kind == Kind::Swap && !options.nativeSwap);
  if (!valid) {
    result.status = LinearFourStatus::Unsupported;
    result.error = "invalid four-wire native source";
    return finish();
  }
  origin = std::make_shared<const lw::Source>(lw::Source{width, synthesized});
  current = synthesized;
  result.alternatives.reserve(1);
  auto commute = [&](const Gate &a, const Gate &b) {
    ++result.commutationChecks;
    if (!take(8 * limbs))
      return false;
    if (b.kind == Kind::Phase)
      return true;
    auto left = lw::support(a), right = lw::support(b);
    if (!(left & right))
      return true;
    if (a.kind == Kind::CX && b.kind == Kind::CX)
      return a.a != b.b && a.b != b.a;
    if (!take(256))
      return false;
    auto support = left | right;
    std::vector<unsigned> wires;
    for (unsigned q : support.set_bits())
      wires.push_back(q);
    auto x = lw::compact(a, wires), y = lw::compact(b, wires);
    return graph::equivalent(wires.size(), {x, y}, {y, x});
  };
  unsigned lookahead = std::min(limits.maxLookahead, 256u);
  unsigned setCap = std::min(limits.maxWireSets, 64u);
  unsigned selectedCap = std::min(limits.maxSelectedGates, 128u);
  for (size_t seed = 0; seed < current.size();) {
    if (!take())
      return finish();
    if (!lw::linear(current[seed])) {
      ++seed;
      continue;
    }
    if (result.rewrites == limits.maxRewrites) {
      result.status = LinearFourStatus::Exhausted;
      result.error = "four-wire rewrite limit reached";
      return finish();
    }
    size_t end = seed + std::min<size_t>(lookahead, current.size() - seed);
    // At most 2*lookahead distinct neighbor endpoints. Both the scan and the
    // candidate-pair loop are charged; only setCap wire sets are materialized.
    auto base = lw::support(current[seed]);
    std::vector<unsigned> extras;
    for (size_t at = seed + 1; at < end; ++at) {
      if (!take(extras.size() + 3))
        return finish();
      if (!lw::linear(current[at]))
        continue;
      for (unsigned q : {current[at].a, current[at].b})
        if (!base.test(q) &&
            std::find(extras.begin(), extras.end(), q) == extras.end())
          extras.push_back(q);
    }
    bool rewritten = false;
    unsigned tried = 0;
    for (size_t a = 0; a < extras.size() && tried < setCap && !rewritten; ++a)
      for (size_t b = a + 1; b < extras.size() && tried < setCap; ++b) {
        if (!take(8))
          return finish();
        ++tried;
        ++result.windowsChecked;
        std::vector<unsigned> wires{current[seed].a, current[seed].b, extras[a],
                                    extras[b]};
        std::sort(wires.begin(), wires.end());
        std::vector<size_t> selected;
        std::vector<Gate> local;
        for (size_t at = seed; at < end; ++at) {
          if (!take(5 + selected.size()))
            return finish();
          const auto &gate = current[at];
          bool inside =
              lw::linear(gate) &&
              std::binary_search(wires.begin(), wires.end(), gate.a) &&
              std::binary_search(wires.begin(), wires.end(), gate.b);
          if (inside) {
            selected.push_back(at);
            local.push_back(lw::compact(gate, wires));
            if (selected.size() == selectedCap)
              break;
            continue;
          }
          bool crosses = true;
          for (size_t previous : selected)
            if (!commute(current[previous], gate)) {
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
        if (!take(local.size() + 1))
          return finish();
        unsigned key = lf::identity();
        for (const auto &gate : local)
          key = lf::advance(key, gate);
        auto implementation =
            dictionary->implementation(key, limits.maxGates, budget);
        if (budget.stopped()) {
          result.status = LinearFourStatus::Exhausted;
          result.error = "four-wire reconstruction budget exhausted";
          return finish();
        }
        if (!implementation)
          continue;
        const auto &replacement = implementation->native;
        if (!take(local.size() + replacement.size() + 1))
          return finish();
        if (!(lw::rank(replacement, options) < lw::rank(local, options)))
          continue;
        uint64_t size = local.size() + replacement.size();
        if (!take(llvm::SaturatingAdd<uint64_t>(
                512, llvm::SaturatingMultiply<uint64_t>(size, 3 * size + 17))))
          return finish();
        // Truth-table replay is independent of the dictionary's GF(2) row
        // updates; every logical primitive has coefficient +1 on each basis
        // state. Native replay separately checks lowering and scalar phase.
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
        for (unsigned bits = 0; bits < 16; ++bits)
          equivalent &=
              evaluate(bits, local) == evaluate(bits, implementation->logical);
        equivalent &= graph::equivalent(4, local, replacement);
        if (!equivalent) {
          result.status = LinearFourStatus::InvalidProof;
          result.error = "four-wire map or native-unitary proof failed";
          return finish();
        }
        size_t retained = current.size() - selected.size();
        if (replacement.size() > limits.maxGates - retained)
          continue;
        if (!take(llvm::SaturatingMultiply<uint64_t>(
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
