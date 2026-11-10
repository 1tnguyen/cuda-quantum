/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/

#pragma once

#include "SynthesisNativeRefinement.h"
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include <map>

namespace cudaq::pauli::graph::detail {

/// Normalize commuting one-wire chains of an already generated word. Replay
/// each normal form with exact ring matrices, including global phase. Ranking
/// must wait until native cleanup: the normal form can temporarily use more
/// Clifford gates even when its completed realization is shorter.
inline Result normalizeHoldOneQubit(unsigned width,
                                    const std::vector<Gate> &source,
                                    const Options &options,
                                    SearchBudget &budget) {
  using Single = cudaq::synth::Gate;
  using Unitary = cudaq::synth::DOmegaUnitary;
  const size_t cap = std::min(options.maxGates, 262144u);
  Result result;
  auto fail = [&](const char *message, SearchTermination status) {
    result.error = message;
    result.termination = status;
    result.stats.budgetExhausted = status == SearchTermination::Exhausted;
    result.gates.clear();
    return false;
  };
  if (source.size() > cap) {
    fail("hold one-wire input cap", SearchTermination::Unsupported);
    return result;
  }
  if (!budget.take((source.size() + 1) * (uint64_t(width) / 64 + 1))) {
    fail("hold one-wire input budget", SearchTermination::Exhausted);
    return result;
  }
  // Sparse storage avoids allocating a chain for every physical wire.
  std::map<unsigned, std::vector<Gate>> chains;
  auto append = [&](const std::vector<Gate> &word) {
    if (word.size() > cap - result.gates.size())
      return fail("hold one-wire temporary cap",
                  SearchTermination::Unsupported);
    result.gates.insert(result.gates.end(), word.begin(), word.end());
    return true;
  };
  auto flush = [&](unsigned q) {
    auto found = chains.find(q);
    if (found == chains.end() || found->second.empty())
      return true;
    auto &chain = found->second;
    // Bound exact-integer growth and prepay conversion and matrix replay.
    const uint64_t size = chain.size() + 1;
    if (!budget.take(64 * size * size))
      return fail("hold one-wire proof budget", SearchTermination::Exhausted);
    std::vector<Single> time;
    int scalarEighths = 0;
    for (const auto &gate : chain) {
      switch (gate.kind) {
      case Kind::H:
        time.push_back(Single::H);
        break;
      case Kind::S:
        time.push_back(Single::S);
        break;
      case Kind::Sdg:
        time.insert(time.end(), 3, Single::S);
        break;
      case Kind::X:
        time.push_back(Single::X);
        break;
      case Kind::Y:
        // In time order Z then X is -iY.
        time.insert(time.end(), 2, Single::S);
        time.push_back(Single::X);
        scalarEighths += 4;
        break;
      case Kind::Z:
        time.insert(time.end(), 2, Single::S);
        break;
      case Kind::Rotate: {
        const int k =
            *tcount::detail::quarters(gate.angle) * (gate.pauli.sign ? -1 : 1);
        const bool x = bool(gate.pauli.x), y = x && bool(gate.pauli.z);
        if (y)
          time.insert(time.end(), 3, Single::S);
        if (x)
          time.push_back(Single::H);
        time.insert(time.end(), (k % 8 + 8) % 8, Single::T);
        scalarEighths -= k;
        if (x)
          time.push_back(Single::H);
        if (y)
          time.push_back(Single::S);
        break;
      }
      default:
        return fail("hold one-wire invalid chain",
                    SearchTermination::InvalidProof);
      }
    }
    // Circuit uses matrix-product order, the reverse of native time order.
    cudaq::synth::Circuit circuit;
    for (auto it = time.rbegin(); it != time.rend(); ++it)
      circuit.push_back(*it);
    const auto normalized = circuit.normalized();
    if (Unitary::from_gates(circuit) != Unitary::from_gates(normalized))
      return fail("hold one-wire exact matrix replay failed",
                  SearchTermination::InvalidProof);
    std::vector<Gate> word;
    for (auto it = normalized.rbegin(); it != normalized.rend(); ++it) {
      switch (*it) {
      case Single::H:
        word.push_back({Kind::H, q});
        break;
      case Single::S:
        word.push_back({Kind::S, q});
        break;
      case Single::X:
        word.push_back({Kind::X, q});
        break;
      case Single::T:
        word.push_back(Gate::rotation({{}, QubitSet::single(q)}, {1, 4}));
        ++scalarEighths;
        break;
      case Single::W:
        scalarEighths += 2;
        break;
      }
    }
    if (scalarEighths % 16)
      word.push_back(Gate::phase({scalarEighths % 16, 8}));
    if (!append(word))
      return false;
    chain.clear();
    return true;
  };
  for (const auto &gate : source) {
    if (gate.kind == Kind::Phase) {
      if (!append({gate}))
        return result;
      continue;
    }
    const auto support = linear_window_detail::support(gate);
    if (support.count() == 1 &&
        (gate.kind != Kind::Rotate || tcount::detail::quarters(gate.angle))) {
      const unsigned q = support.first();
      if (chains[q].size() == 512 && !flush(q))
        return result;
      chains[q].push_back(gate);
      continue;
    }
    // Entanglers and unsupported rotations are barriers on every touched wire.
    for (unsigned q : support.set_bits())
      if (!flush(q))
        return result;
    if (!append({gate}))
      return result;
  }
  for (auto &[q, chain] : chains)
    if (!flush(q))
      return result;
  return result;
}
} // namespace cudaq::pauli::graph::detail
