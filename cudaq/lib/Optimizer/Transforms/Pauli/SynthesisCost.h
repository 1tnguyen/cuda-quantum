/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Shared native emission and cost adapters for Pauli synthesis.
///
/// These adapters preserve the existing synthesis cost model and ranking.
/// They neither prove equivalence nor charge search work. Callers validate
/// circuits and target options before invoking them.
//===----------------------------------------------------------------------===//

#pragma once

#include "TCountInternal.h"
#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include "cudaq/Optimizer/Pauli/TCount.h"
#include <array>
#include <cstdlib>
#include <vector>

namespace cudaq::pauli::graph::detail {

/// Weighted cost in the existing mechanical native-lowering model. Exact
/// denotes supported Clifford+T rotations, not a minimum synthesis cost. A
/// non-quarter-turn rotation uses the legacy tCost proxy and clears exact;
/// it is never silently approximated by this adapter.
struct NativeCost {
  double value = 0;
  bool exact = true;
};

/// Five integral metrics in the existing mechanical-lowering model.
using CostVector = tcount::Cost;
struct CircuitCost {
  CostVector metrics;
  NativeCost native;
};
using CostRank = std::array<double, 5>;

/// Translate an action into the configured native entangling basis.
inline std::vector<Gate> nativeWord(const std::vector<Gate> &input,
                                    const Options &options) {
  std::vector<Gate> result;
  auto cx = [&](unsigned a, unsigned b) {
    if (options.nativeCX)
      result.push_back({Kind::CX, a, b});
    else {
      result.push_back({Kind::H, b});
      result.push_back({Kind::CZ, a, b});
      result.push_back({Kind::H, b});
    }
  };
  for (Gate gate : input) {
    if (gate.kind == Kind::CX)
      cx(gate.a, gate.b);
    else if (gate.kind == Kind::CZ && !options.nativeCZ) {
      result.push_back({Kind::H, gate.b});
      cx(gate.a, gate.b);
      result.push_back({Kind::H, gate.b});
    } else if (gate.kind == Kind::Swap && !options.nativeSwap) {
      cx(gate.a, gate.b);
      cx(gate.b, gate.a);
      cx(gate.a, gate.b);
    } else
      result.push_back(gate);
  }
  return result;
}

inline NativeCost evaluateNativeCost(const std::vector<Gate> &gates,
                                     const Options &options) {
  NativeCost result;
  for (Gate gate : nativeWord(gates, options)) {
    switch (gate.kind) {
    case Kind::Phase:
      break;
    case Kind::CX:
      result.value += options.cxCost;
      break;
    case Kind::CZ:
      result.value += options.czCost;
      break;
    case Kind::Swap:
      result.value += options.swapCost;
      break;
    case Kind::Rotate: {
      // Cost exact named Clifford+T lowering, including basis/parity gates.
      auto k = tcount::detail::quarters(gate.angle);
      if (k) {
        unsigned magnitude = std::abs(*k % 8);
        result.value += ((magnitude & 1) ? options.tCost : 0) +
                        ((magnitude & 2) ? options.singleQubitCost : 0) +
                        ((magnitude & 4) ? options.singleQubitCost : 0);
      } else {
        result.value += options.tCost; // compatibility proxy; no approximation
        result.exact = false;
      }
      auto support = gate.pauli.x | gate.pauli.z;
      for (unsigned q : support.set_bits())
        result.value += (gate.pauli.axis(q) == 3   ? 4
                         : gate.pauli.axis(q) == 1 ? 2
                                                   : 0) *
                        options.singleQubitCost;
      if (gate.pauli.weight() > 1)
        result.value +=
            2 * (gate.pauli.weight() - 1) *
            (options.nativeCX ? options.cxCost
                              : options.czCost + 2 * options.singleQubitCost);
      break;
    }
    default:
      result.value += options.singleQubitCost;
      break;
    }
  }
  return result;
}

/// Evaluate the existing emitted circuit metrics and weighted native cost.
/// Metrics intentionally use the supplied word, as the legacy score does;
/// providers must materialize their chosen native entangling basis first.
inline CircuitCost evaluateCircuitCost(unsigned width,
                                       const std::vector<Gate> &gates,
                                       const Options &options) {
  return {tcount::cost(width, gates), evaluateNativeCost(gates, options)};
}

/// Rank complete circuits in the same primitive basis used by emission.
inline CostRank circuitScore(unsigned n, const std::vector<Gate> &gates,
                             const Options &options) {
  auto evaluated = evaluateCircuitCost(n, gates, options);
  const auto &cost = evaluated.metrics;
  double total = evaluated.native.value;
  double t = cost.t, depth = cost.depth, two = cost.twoQubit, td = cost.tDepth;
  switch (options.objective) {
  case SynthesisObjective::TCount:
    return {t, total, depth, two, td};
  case SynthesisObjective::Entanglers:
    return {two, total, t, depth, td};
  case SynthesisObjective::Depth:
    return {depth, total, t, two, td};
  case SynthesisObjective::TDepth:
    return {td, t, total, depth, two};
  case SynthesisObjective::TotalGates:
    return {total, t, depth, two, td};
  }
  return {total, t, depth, two, td};
}

} // namespace cudaq::pauli::graph::detail
