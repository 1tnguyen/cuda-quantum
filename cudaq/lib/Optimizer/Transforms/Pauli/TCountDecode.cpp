/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Exact T-parity decoding of bounded commuting Pauli blocks.
///
/// Expresses a block in independent commuting generators and searches its Reed-
/// Muller code coset. Reconstructs and verifies Clifford/scalar corrections,
/// then optionally reuses the original Clifford network to place new rotations.
//===----------------------------------------------------------------------===//

#include "TCountInternal.h"
#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>

namespace cudaq::pauli::tcount::detail {
namespace {
/// Commuting phase block expressed in an independent Pauli-generator basis.
struct Block {
  /// Clifford transport used to express incoming rotations at block entry.
  Frame frame;
  /// Independent generators and their signed products indexed by subset masks.
  std::vector<Pauli> generators, axes{{}};
  /// Rotation coefficients in pi/4 units, modulo 16, indexed like axes.
  std::vector<int> coefficients{0};
  /// Original Clifford network and scalar phases retained for reconstruction.
  std::vector<Gate> cliffords, phases;
  /// Input odd-coefficient rotation count before combining equal axes.
  unsigned t = 0;
  explicit Block(unsigned width) : frame(width) {}
  /// Find p's unsigned support in the generator span, extending that span if
  /// permitted. Reject anticommuting axes or independent axes beyond the cap.
  /// The caller accounts for any sign difference from the stored product.
  std::optional<unsigned> coordinate(Pauli p, unsigned rankLimit) {
    for (const auto &g : generators)
      if (g.anti(p))
        return {};
    unsigned index = 0;
    for (; index < axes.size(); ++index)
      if (axes[index].x == p.x && axes[index].z == p.z)
        return index;
    if (generators.size() == rankLimit)
      return {};
    // Independent commuting Pauli generators have all 2^rank eigenvalue
    // assignments. Preserve product signs, including Y-containing strings.
    p.sign = false;
    generators.push_back(p);
    unsigned oldSize = axes.size();
    // Existing subset masks keep their meaning. Setting the new high bit
    // multiplies that old subset product by the new independent generator.
    for (unsigned i = 0; i < oldSize; ++i)
      axes.push_back(product(axes[i], p));
    coefficients.resize(axes.size());
    return oldSize;
  }
};
/// Minimize odd coefficients in one bounded commuting block, recover the full
/// phase function, then compare concrete reconstructions. An incomplete coset
/// search, failed phase witness, or unacceptable cost leaves the block
/// unchanged.
std::vector<Gate> optimizeBlock(const Block &block,
                                const std::vector<Gate> &original,
                                const Options &options, Budget &budget,
                                Statistics &stats, unsigned width) {
  if (block.t < 2)
    return original;
  ++stats.decodedBlocks;
  unsigned rank = block.generators.size(), size = block.axes.size();
  // Bit v-1 represents the nonempty generator subset v. Odd coefficients need
  // one T rotation each; even coefficients are Clifford corrections. The empty
  // subset is omitted because it contributes only scalar phase.
  uint64_t odd = 0;
  for (unsigned v = 1; v < size; ++v)
    if (block.coefficients[v] & 1)
      odd |= uint64_t(1) << (v - 1);
  uint64_t best = odd;
  // Decode the punctured RM(rank-4,rank) coset exactly. At rank 5 there
  // are just 64 codewords; no search over the physical register dimension.
  // Amy and Mosca, arXiv:1601.07363, relates this coset to T parity.
  std::vector<uint64_t> codeGenerators;
  if (rank >= 4)
    for (unsigned monomial = 0; monomial < size; ++monomial) {
      if (unsigned(std::popcount(monomial)) > rank - 4)
        continue;
      uint64_t row = 0;
      // Evaluate this Boolean monomial at every nonzero subset label v. Its
      // truth table is one code generator; XOR combines generators over GF(2).
      for (unsigned v = 1; v < size; ++v)
        if ((v & monomial) == monomial)
          row |= uint64_t(1) << (v - 1);
      codeGenerators.push_back(row);
    }
  uint64_t current = odd;
  // Successive Gray-code words differ in one generator, avoiding a fresh XOR
  // sum for each candidate. Equal-weight candidates retain the earlier choice.
  for (unsigned i = 1; i < (1u << codeGenerators.size()); ++i) {
    if (!budget.take())
      return original;
    current ^= codeGenerators[std::countr_zero(i)]; // Gray-code traversal.
    if (std::popcount(current) < std::popcount(best))
      best = current;
  }
  if (unsigned(std::popcount(best)) >= block.t)
    return original;

  // Recover the complete diagonal Clifford correction. The residual Boolean
  // polynomial must have even linear terms, quadratic multiples of four,
  // and no higher-degree terms. Search success alone is never a proof.
  std::vector<int> output(size), polynomial(size);
  for (unsigned v = 1; v < size; ++v)
    output[v] = (best >> (v - 1)) & 1;
  // Build the residual phase truth table modulo 8. Here parity is a Boolean
  // 0/1 function; the constant from signed rotation eigenvalues is restored
  // separately by scalar below.
  for (unsigned x = 0; x < size; ++x)
    for (unsigned v = 1; v < size; ++v) {
      if (!budget.take())
        return original;
      if (population(x & v) & 1)
        polynomial[x] = (polynomial[x] + block.coefficients[v] - output[v]) & 7;
    }
  // Subset Mobius inversion changes truth-table values into coefficients of
  // ordinary multilinear Boolean monomials, with arithmetic modulo 8.
  for (unsigned bit = 1; bit < size; bit <<= 1)
    for (unsigned x = 0; x < size; ++x)
      if (x & bit)
        polynomial[x] = (polynomial[x] - polynomial[x ^ bit]) & 7;
  for (unsigned mask = 1; mask < size; ++mask) {
    int c = polynomial[mask];
    if (!c)
      continue;
    if (std::popcount(mask) == 1 && !(c & 1))
      output[mask] += c;
    else if (std::popcount(mask) == 2 && c == 4) {
      unsigned a = mask & -mask, b = mask ^ a;
      // Re-express a CZ-type term using 4*a*b = 2*a + 2*b - 2*(a XOR b).
      // All three added parity coefficients are even, so no T is introduced.
      output[a] += 2;
      output[b] += 2;
      output[mask] -= 2;
    } else
      return original;
  }
  for (auto &c : output)
    c &= 7;
  // Converting Boolean phases back to exp(-i*theta*P/2) changes the constant
  // phase. Restore it in pi/8 units before checking every eigenvalue
  // assignment.
  int scalar = (std::accumulate(output.begin(), output.end(), 0) -
                std::accumulate(block.coefficients.begin(),
                                block.coefficients.end(), 0)) %
               16;
  if (!verifyPhases(block.coefficients, output, scalar, budget))
    return original;
  ++stats.verifiedBlocks;
  std::vector<Gate> candidate;
  // First reconstruction: emit the decoded commuting rotations at block entry
  // and retain the original Clifford suffix. Materialization then builds each
  // parity gadget and cleanup removes any provable Clifford cancellations.
  for (unsigned v = 1; v < size; ++v)
    if (output[v])
      candidate.push_back(Gate::rotation(block.axes[v], {output[v], 4}));
  if (scalar)
    candidate.push_back(Gate::phase({scalar, 8}));
  candidate.insert(candidate.end(), block.phases.begin(), block.phases.end());
  candidate.insert(candidate.end(), block.cliffords.begin(),
                   block.cliffords.end());
  candidate = cancelCliffords(materialize(candidate), budget);
  if (options.reuseCliffords) {
    // Reuse the ORIGINAL Clifford network: a global Pauli axis may be a
    // one-qubit axis at a different point in the circuit. Place each decoded
    // rotation where its local support is smallest instead of synthesizing
    // an independent parity gadget at the block entrance.
    /// Lowest-support realization found for one decoded term in the old frame.
    struct Placement {
      /// Insert immediately before this original block operation.
      size_t position = 0;
      /// Equivalent signed axis in the Clifford frame at position.
      Pauli axis;
    };
    std::vector<Placement> placements(size);
    for (unsigned v = 1; v < size; ++v)
      placements[v].axis = block.axes[v];
    Frame frame(width);
    bool complete = true;
    for (size_t i = 0; i < original.size() && complete; ++i) {
      for (unsigned v = 1; v < size; ++v) {
        if (!output[v] || placements[v].axis.weight() == 1)
          continue;
        if (!budget.take(2 * width)) {
          complete = false;
          break;
        }
        Pauli local = frame.inverseLookup(block.axes[v]);
        // Moving through this Clifford prefix changes only the local axis.
        // Decoded terms commute in the common frame, so their placements may
        // differ without imposing a new relative order between the terms.
        if (local.weight() < placements[v].axis.weight())
          placements[v] = {i, local};
      }
      if (original[i].kind != Kind::Rotate)
        frame.append(original[i]);
    }
    // A partial placement search is still valid: untouched entries retain
    // the verified block-entrance realization. Only search quality changes.
    std::vector<std::vector<Gate>> insertions(original.size());
    for (unsigned v = 1; v < size; ++v)
      if (output[v]) {
        auto place = placements[v];
        insertions[place.position].push_back(
            Gate::rotation(place.axis, {output[v], 4}));
      }
    std::vector<Gate> placed;
    for (size_t i = 0; i < original.size(); ++i) {
      placed.insert(placed.end(), insertions[i].begin(), insertions[i].end());
      if (original[i].kind != Kind::Rotate)
        placed.push_back(original[i]);
    }
    if (scalar)
      placed.push_back(Gate::phase({scalar, 8}));
    placed = cancelCliffords(materialize(placed), budget);
    if (cost(width, placed) < cost(width, candidate))
      candidate = std::move(placed);
  }
  // A smaller parity vector is not sufficient: compare the materialized gates
  // and enforce the entangler allowance for this block's actual reconstruction.
  auto before = cost(width, original), after = cost(width, candidate);
  if (after.t >= before.t ||
      after.twoQubit > before.twoQubit + options.maxTwoQubitGrowth)
    return original;
  return candidate;
}
} // namespace
/// Partition the scan into bounded commuting blocks in their local entry
/// frames. Unsupported rotations end a block and are retained, never dropped.
/// Track one entangler allowance across all accepted block replacements.
std::vector<Gate> decode(const std::vector<Gate> &input, unsigned width,
                         const Options &options, Budget &budget,
                         Statistics &stats) {
  std::vector<Gate> out;
  uint64_t allowance = options.maxTwoQubitGrowth;
  for (size_t start = 0; start < input.size();) {
    if (budget.stopped()) {
      out.insert(out.end(), input.begin() + start, input.end());
      break;
    }
    Block block(width);
    size_t end = start;
    for (; end < input.size() && end - start < options.maxBlockGates; ++end) {
      if (!budget.take(width))
        break;
      auto g = input[end];
      if (g.kind == Kind::Phase)
        block.phases.push_back(g);
      else if (g.kind != Kind::Rotate) {
        block.frame.append(g);
        block.cliffords.push_back(g);
      } else {
        auto k = quarters(g.angle);
        if (!k || !g.pauli.weight())
          break;
        Pauli p = block.frame.lookup(g.pauli);
        auto coordinate = block.coordinate(p, options.maxBlockRank);
        if (!coordinate)
          break;
        // The stored subset product can be negative even though each generator
        // was unsigned. Express the coefficient in that product's convention.
        int signedK = p.sign == block.axes[*coordinate].sign ? *k : -*k;
        block.coefficients[*coordinate] =
            (block.coefficients[*coordinate] + signedK) % 16;
        block.t += *k & 1;
      }
    }
    if (end == start) {
      // A barrier at block entry must still advance the scan. Otherwise the
      // same opaque, identity-axis, or unsupported rotation would be retried.
      out.push_back(input[start++]);
      continue;
    }
    std::vector<Gate> original(input.begin() + start, input.begin() + end);
    auto blockOptions = options;
    blockOptions.maxTwoQubitGrowth = unsigned(
        std::min<uint64_t>(allowance, std::numeric_limits<unsigned>::max()));
    auto candidate =
        optimizeBlock(block, original, blockOptions, budget, stats, width);
    // Charge net growth, not a fresh allowance per block. Earlier savings can
    // fund later reconstruction while respecting the invocation-wide ceiling.
    allowance += cost(width, original).twoQubit;
    allowance -= cost(width, candidate).twoQubit;
    out.insert(out.end(), candidate.begin(), candidate.end());
    start = end;
  }
  return out;
}
} // namespace cudaq::pauli::tcount::detail
