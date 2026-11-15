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

/// Exact support-cost change for the symmetric two-wire candidate. Wires i/j
/// must be distinct, local axes a/b in 1..3. Frame factors require p/q to
/// anticommute, singlets ignore q. Signs do not contribute to this cost.
int candidateSupportCostDelta(const Pauli &p, const Pauli &q, bool factor,
                              unsigned i, unsigned j, unsigned a, unsigned b);

/// Score terminal reconstruction without retaining its native gates. Options,
/// dimensions and commuting, in-range observables must already be validated.
/// Unsupported reconstruction or a gate-limit failure returns infinity.
double evaluateReadoutCost(unsigned width,
                           const std::vector<Pauli> &observables,
                           const std::vector<unsigned> &readoutWires,
                           const Options &options);

/// Five integral metrics in the existing mechanical-lowering model.
using CostVector = tcount::Cost;
struct CircuitCost {
  CostVector metrics;
  NativeCost native;
};
using CostRank = std::array<double, 5>;

/// Visit one gate's native expansion in emission order, without a temporary
/// gate vector. The callback must consume or copy each gate before returning.
template <typename Emit>
inline void emitNativeGate(const Gate &gate, const Options &options,
                           Emit &&emit) {
  auto cx = [&](unsigned a, unsigned b) {
    if (options.nativeCX)
      emit(Gate{Kind::CX, a, b});
    else {
      emit(Gate{Kind::H, b});
      emit(Gate{Kind::CZ, a, b});
      emit(Gate{Kind::H, b});
    }
  };
  if (gate.kind == Kind::CX)
    cx(gate.a, gate.b);
  else if (gate.kind == Kind::CZ && !options.nativeCZ) {
    emit(Gate{Kind::H, gate.b});
    cx(gate.a, gate.b);
    emit(Gate{Kind::H, gate.b});
  } else if (gate.kind == Kind::Swap && !options.nativeSwap) {
    cx(gate.a, gate.b);
    cx(gate.b, gate.a);
    cx(gate.a, gate.b);
  } else
    emit(gate);
}

/// Translate an action into the configured native entangling basis.
inline std::vector<Gate> nativeWord(const std::vector<Gate> &input,
                                    const Options &options) {
  std::vector<Gate> result;
  for (const Gate &gate : input)
    emitNativeGate(gate, options,
                   [&](const Gate &native) { result.push_back(native); });
  return result;
}

inline NativeCost evaluateNativeCost(const std::vector<Gate> &gates,
                                     const Options &options) {
  NativeCost result;
  auto accumulate = [&](const Gate &gate) {
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
  };
  // Score the native expansion in emission order without retaining it.
  for (const Gate &gate : gates)
    emitNativeGate(gate, options, accumulate);
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
