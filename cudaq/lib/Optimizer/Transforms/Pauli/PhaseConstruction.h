// Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "TCountInternal.h"
#include <cassert>
#include <cmath>
#include <optional>

namespace cudaq::pauli::construction {
/// Exact Hermitian Pauli multiplication, including the scalar power of i.
inline std::pair<Pauli, unsigned> multiply(const Pauli &a, const Pauli &b) {
  Pauli c{a.x ^ b.x, a.z ^ b.z};
  int phase = a.x.intersectionCount(a.z) + b.x.intersectionCount(b.z) -
              c.x.intersectionCount(c.z) + 2 * a.z.intersectionCount(b.x) +
              2 * (a.sign + b.sign);
  return {c, unsigned(phase) & 3};
}
// Phase-bearing stabilizer arithmetic. This is maintained state, not an
// equivalence oracle: no gate sequence is replayed to recover a lost scalar.
struct PhaseState {
  /// Independent signed stabilizers determining the state's amplitude ratios.
  std::vector<Pauli> stabilizers;
  /// Computational-basis bit string with nonzero amplitude.
  QubitSet anchor;
  /// Anchor amplitude is exp(i*pi*phase/4) / sqrt(2)^exponent.
  int phase = 0, exponent = 0;
  /// Initialize |0...0>, whose stabilizers are +Z on each wire and anchor is 0.
  explicit PhaseState(unsigned n) {
    for (unsigned q = 0; q < n; ++q)
      stabilizers.push_back(Pauli{{}, QubitSet::single(q)});
  }
  /// Return k with amplitude(target)/amplitude(anchor) = i^k, or no value if
  /// target has zero amplitude. Eliminate stabilizer X supports to find a
  /// product that maps the anchor bit string to the requested target.
  std::optional<unsigned> ratio(QubitSet target) const {
    std::vector<Pauli> basis(stabilizers.size());
    for (Pauli p : stabilizers) {
      while (p.x) {
        unsigned pivot = firstBit(p.x);
        if (!basis[pivot].x) {
          basis[pivot] = p;
          break;
        }
        auto [r, k] = multiply(p, basis[pivot]);
        r.sign = k == 2;
        p = r;
      }
    }
    // The XOR difference must lie in the span of stabilizer X supports. Its
    // reducing product supplies the relative sign/Y phase at the anchor.
    auto delta = target ^ anchor;
    Pauli product;
    while (delta) {
      unsigned pivot = firstBit(delta);
      if (!basis[pivot].x)
        return std::nullopt;
      auto [p, k] = multiply(product, basis[pivot]);
      p.sign = k == 2;
      product = p;
      delta ^= basis[pivot].x;
    }
    return (product.x.intersectionCount(product.z) + 2 * product.sign +
            2 * population(product.z & anchor)) &
           3;
  }
  /// Apply a primitive Clifford while retaining an exact nonzero anchor
  /// amplitude. Stabilizer conjugation alone would lose the global phase.
  void apply(Gate g) {
    bool bit = anchor.test(g.a);
    switch (g.kind) {
    case Kind::H: {
      // H forms the sum/difference of the two amplitudes differing on this
      // wire. Choose a nonzero output as the new anchor and update its exact
      // eighth-root phase and power-of-sqrt(2) normalization.
      auto zero = anchor;
      zero.set(g.a, false);
      auto r0 = ratio(zero), r1 = ratio(zero | (QubitSet::single(g.a, 1)));
      constexpr int re[] = {1, 0, -1, 0}, im[] = {0, 1, 0, -1};
      int r = (r0 ? re[*r0] : 0) + (r1 ? re[*r1] : 0);
      int i = (r0 ? im[*r0] : 0) + (r1 ? im[*r1] : 0);
      anchor = zero;
      if (!r && !i) {
        r = (r0 ? re[*r0] : 0) - (r1 ? re[*r1] : 0);
        i = (r0 ? im[*r0] : 0) - (r1 ? im[*r1] : 0);
        anchor |= QubitSet::single(g.a, 1);
      }
      if (r && i) {
        phase += r > 0 ? (i > 0 ? 1 : 7) : (i > 0 ? 3 : 5);
      } else {
        phase += r ? (r > 0 ? 0 : 4) : (i > 0 ? 2 : 6);
        exponent += (std::abs(r) + std::abs(i) == 2) ? -1 : 1;
      }
      break;
    }
    case Kind::S:
      phase += 2 * bit;
      break;
    case Kind::Sdg:
      phase -= 2 * bit;
      break;
    case Kind::X:
      anchor ^= QubitSet::single(g.a, 1);
      break;
    case Kind::Y:
      phase += 2 + 4 * bit;
      anchor ^= QubitSet::single(g.a, 1);
      break;
    case Kind::Z:
      phase += 4 * bit;
      break;
    case Kind::CX:
      anchor ^= QubitSet::single(g.b, bit);
      break;
    case Kind::CZ:
      phase += 4 * bit * (anchor.test(g.b));
      break;
    case Kind::Swap:
      if (anchor.test(g.a) != anchor.test(g.b))
        anchor ^= (QubitSet::single(g.a, 1)) | (QubitSet::single(g.b, 1));
      break;
    default:
      break;
    }
    phase &= 7;
    for (auto &p : stabilizers)
      p = tcount::detail::conjugate(p, g);
  }
};

/// Exact scalar of a Clifford alongside its signed tableau. For two-sided
/// updates use its normalized Choi state; for append-only dictionary words
/// the image of |0> suffices. Neither allocates a dense state or unitary.
class CliffordPhase {
  unsigned width = 0;
  bool twoSided = false;
  PhaseState state;

public:
  CliffordPhase() : CliffordPhase(0) {}
  explicit CliffordPhase(unsigned n, bool bothSides = false)
      : width(n), twoSided(bothSides), state(bothSides ? 2 * n : n) {
    if (twoSided) {
      state.stabilizers.clear();
      for (unsigned q = 0; q < n; ++q) {
        auto pair = QubitSet::single(q) | QubitSet::single(n + q);
        state.stabilizers.push_back({pair, {}, false});
        state.stabilizers.push_back({{}, pair, false});
      }
      state.exponent = n;
    }
  }
  void append(Gate g) {
    if (twoSided) {
      g.a += width;
      if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
        g.b += width;
    }
    state.apply(g);
  }
  /// vec(F*g^dagger) = (conj(g) tensor I) vec(F).
  void removePrefix(Gate g) {
    assert(twoSided);
    if (g.kind == Kind::S)
      g.kind = Kind::Sdg;
    else if (g.kind == Kind::Sdg)
      g.kind = Kind::S;
    else if (g.kind == Kind::Y)
      state.phase = (state.phase + 4) & 7;
    state.apply(g);
  }
  /// Called once the separately maintained tableau has become identity.
  int identityPhase() const { return state.phase; }
  /// Same signed tableau, hence the anchor supports agree. Convert the two
  /// maintained anchors to one convention, then subtract their exact phases.
  int relativeTo(const CliffordPhase &other) const {
    auto ratio = other.state.ratio(state.anchor);
    assert(ratio && state.exponent == other.state.exponent);
    return (state.phase - other.state.phase - 2 * int(*ratio)) & 7;
  }
};

/// Exact primitive commutation rules, including signs. These are local
/// rewrite preconditions, not calls to a circuit-equivalence checker.
inline bool commuteCliffords(Gate a, Gate b) {
  auto support = [](const Gate &g) {
    auto s = QubitSet::single(g.a);
    if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
      s.set(g.b);
    return s;
  };
  if (a.kind == Kind::Phase || b.kind == Kind::Phase)
    return true;
  if (!(support(a) & support(b)))
    return true;
  auto pauli = [](Kind k) {
    return k == Kind::X || k == Kind::Y || k == Kind::Z;
  };
  if (pauli(b.kind) && !pauli(a.kind))
    std::swap(a, b);
  if (pauli(a.kind)) {
    Pauli p;
    if (a.kind != Kind::Z)
      p.x.set(a.a);
    if (a.kind != Kind::X)
      p.z.set(a.a);
    return tcount::detail::conjugate(p, b) == p;
  }
  if (b.kind == Kind::Swap)
    std::swap(a, b);
  if (a.kind == Kind::Swap) {
    auto permute = [&](unsigned q) {
      return q == a.a ? a.b : q == a.b ? a.a : q;
    };
    unsigned x = permute(b.a), y = permute(b.b);
    if (b.kind == Kind::CX)
      return x == b.a && y == b.b;
    if (b.kind == Kind::CZ || b.kind == Kind::Swap)
      return (x == b.a && y == b.b) || (x == b.b && y == b.a);
    return x == b.a;
  }
  auto diagonal = [](Kind k) {
    return k == Kind::S || k == Kind::Sdg || k == Kind::CZ;
  };
  if (diagonal(a.kind) && diagonal(b.kind))
    return true;
  if (a.kind == Kind::H || b.kind == Kind::H)
    return a.kind == Kind::H && b.kind == Kind::H; // overlap is the same wire
  if (a.kind == Kind::CX && b.kind == Kind::CX)
    return a.a != b.b && b.a != a.b;
  if (b.kind == Kind::CX)
    std::swap(a, b);
  if (a.kind == Kind::CX && diagonal(b.kind))
    return !support(b).test(a.b);
  return false;
}
} // namespace cudaq::pauli::construction
