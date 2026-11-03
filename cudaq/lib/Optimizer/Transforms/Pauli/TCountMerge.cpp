/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Clifford-aware merging of exact Pauli rotations.
///
/// Indexes rotation axes in a common frame and proposes merges across
/// intervening operations. Each merge replays Clifford transport and
/// commutation checks, retaining the scalar phase required by the exact
/// rotation convention.
//===----------------------------------------------------------------------===//

#include "TCountInternal.h"

namespace cudaq::pauli::tcount::detail {
/// Merge odd pi/4 rotations using common-frame axes to find candidates and
/// explicit transport replay to authorize each change. Completed merges remain
/// valid if the shared budget stops the scan before reaching the circuit end.
std::vector<Gate> merge(const std::vector<Gate> &input, unsigned width,
                        const Options &options, Budget &budget,
                        Statistics &stats) {
  auto gates = input;
  if (gates.empty())
    return gates;
  Frame frame(width);
  // Store only the latest candidate for each unsigned axis. This bounds the
  // matching search; an older valid partner can be missed without affecting
  // correctness. Signs are recovered by verifyMerge, not by this hash key.
  std::unordered_map<Axis, size_t, AxisHash> latest;
  for (size_t i = 0; i < gates.size(); ++i) {
    if (!budget.take(width))
      break;
    auto g = gates[i];
    if (isClifford(g)) {
      frame.append(g);
      continue;
    }
    auto k = quarters(g.angle);
    // An opaque rotation is a commutation barrier only on its support.
    if (!k || !g.pauli.weight())
      continue;
    Pauli p = frame.lookup(g.pauli);
    Axis key{p.x, p.z};
    auto it = latest.find(key);
    bool merged = false;
    if (it != latest.end() &&
        (!options.mergeWindow || i - it->second <= options.mergeWindow)) {
      size_t previous = it->second;
      auto old = quarters(gates[previous].angle);
      if (old && (*old & 1)) {
        // Matching transported axes is only a proposal: intervening rotations
        // may obstruct it. Replay the current gate sequence before mutating it.
        if (auto transported = verifyMerge(gates, previous, i, budget, stats)) {
          // Exact rotation convention retains the scalar automatically; never
          // reduce modulo 2*pi (a 2*pi rotation is -I).
          gates[previous].angle = {0, 1};
          gates[i].angle = {(*k + *transported) % 16, 4};
          // Two odd coefficients sum to a Clifford. Absorb that new Clifford
          // into subsequent lookups and retire this axis from the candidate
          // map.
          frame.append(gates[i]);
          latest.erase(it);
          ++stats.merges;
          merged = true;
        }
      }
    }
    if (!merged)
      latest[key] = i;
  }
  std::vector<Gate> out;
  // Keep indices stable throughout the scan; erase proven identities only now.
  out.reserve(gates.size());
  for (auto g : gates)
    if (!(g.kind == Kind::Rotate && quarters(g.angle) == 0))
      out.push_back(g);
  return out;
}
} // namespace cudaq::pauli::tcount::detail
