/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Exact Pauli algebra and reconstruction costs for T optimization.
///
/// Implements signed Clifford transport, angle classification, frame lookup,
/// gate materialization, and local Clifford cancellation. The shared cost model
/// counts mechanical lowering rather than routed hardware or factory resources.
//===----------------------------------------------------------------------===//

#include "TCountInternal.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <map>
#include <tuple>

using namespace cudaq::pauli;
using namespace cudaq::pauli::tcount;
namespace cudaq::pauli::tcount::detail {
/// Return an exact pi/4 coefficient modulo 16, or no value for an unsupported
/// angle. Rotations have period 4*pi: reducing modulo 8 would lose a sign.
std::optional<int> quarters(Angle a) {
  if (a.symbol || a.denominator <= 0)
    return {};
  // Widen before scaling so classification cannot overflow an int64_t angle.
  __int128 n = __int128(a.numerator) * 4;
  if (n % a.denominator)
    return {};
  int k = int((n / a.denominator) % 16);
  return a.negative ? -k : k;
}
Gate inverse(Gate g) {
  if (g.kind == Kind::S)
    g.kind = Kind::Sdg;
  else if (g.kind == Kind::Sdg)
    g.kind = Kind::S;
  else if (g.kind == Kind::Rotate || g.kind == Kind::Phase)
    g.angle = g.angle.negated();
  return g;
}
/// Return (c, phase) with a*b = i^phase c and unsigned Hermitian c.
/// Hermitian encoding: P=(-1)^sign i^popcount(x&z) X^x Z^z.
static std::pair<Pauli, unsigned> multiply(const Pauli &a, const Pauli &b) {
  Pauli c{a.x ^ b.x, a.z ^ b.z};
  // Account for each operand's Y factors, Z/X exchanges between operands,
  // and the Y factors already encoded in c. Operand signs contribute i^2.
  unsigned phase = (a.x.intersectionCount(a.z) + b.x.intersectionCount(b.z) -
                    c.x.intersectionCount(c.z) +
                    2 * a.z.intersectionCount(b.x) + 2 * (a.sign + b.sign)) &
                   3;
  return {c, phase};
}
Pauli product(const Pauli &a, const Pauli &b) {
  auto [c, phase] = multiply(a, b);
  // Commuting Hermitian inputs have a real product, so only +/-1 remains.
  assert(!(phase & 1) && "non-Hermitian product");
  c.sign = phase & 2;
  return c;
}
/// Compute g*p*g^dagger, including its sign. Callers must supply a supported
/// Clifford; a Rotate gate must have an exact even pi/4 coefficient.
Pauli conjugate(Pauli p, const Gate &g) {
  bool x = false, z = false;
  if (g.kind != Kind::Rotate && g.kind != Kind::Phase) {
    x = p.x.test(g.a);
    z = p.z.test(g.a);
  }
  switch (g.kind) {
  case Kind::H:
    p.x.flip(g.a, x ^ z);
    p.z.flip(g.a, x ^ z);
    p.sign ^= x && z;
    break;
  case Kind::S:
  case Kind::Sdg:
    p.sign ^= x && (z ^ (g.kind == Kind::Sdg));
    p.z.flip(g.a, x);
    break;
  case Kind::X:
    p.sign ^= z;
    break;
  case Kind::Y:
    p.sign ^= x ^ z;
    break;
  case Kind::Z:
    p.sign ^= x;
    break;
  case Kind::CX: {
    bool xb = p.x.test(g.b), zb = p.z.test(g.b);
    // X propagates control-to-target and Z target-to-control. Their overlap
    // can change the Hermitian sign; updating only the masks would lose it.
    p.sign ^= x && zb && (xb ^ z ^ true);
    p.x.flip(g.b, x);
    p.z.flip(g.a, zb);
    break;
  }
  case Kind::CZ:
    return conjugate(
        conjugate(conjugate(p, {Kind::H, g.b}), {Kind::CX, g.a, g.b}),
        {Kind::H, g.b});
  case Kind::Swap: {
    auto mask = (QubitSet::single(g.a, 1)) | (QubitSet::single(g.b, 1));
    if (p.x.test(g.a) != p.x.test(g.b))
      p.x ^= mask;
    if (p.z.test(g.a) != p.z.test(g.b))
      p.z ^= mask;
    break;
  }
  case Kind::Rotate: {
    int k = *quarters(g.angle);
    assert(!(k & 1));
    if (g.pauli.anti(p)) {
      // Commuting axes are unchanged. For anticommuting axes, count quarter
      // turns after absorbing the rotation-axis sign into the angle.
      int turns = ((g.pauli.sign ? -k : k) / 2) & 3;
      if (turns & 1) {
        Pauli axis = g.pauli;
        axis.sign = false;
        auto [q, phase] = multiply(axis, p);
        // R_A(pi/2) P R_A(-pi/2) = -i A P.
        q.sign = ((phase + (turns == 1 ? 3 : 1)) & 2) != 0;
        p = q;
      } else if (turns == 2)
        p.sign = !p.sign;
    }
    break;
  }
  case Kind::Phase:
    break;
  }
  return p;
}
bool isClifford(const Gate &g) {
  if (g.kind != Kind::Rotate)
    return true;
  auto k = quarters(g.angle);
  return k && !(*k & 1);
}
bool valid(unsigned n, const std::vector<Gate> &gates) {
  if (!n)
    return false;
  for (const auto &g : gates) {
    if (unsigned(g.kind) > unsigned(Kind::Phase))
      return false;
    if (g.kind == Kind::Rotate || g.kind == Kind::Phase) {
      if (!g.angle.symbol && g.angle.denominator <= 0)
        return false;
      if (g.kind == Kind::Rotate && (!g.pauli.x.fits(n) || !g.pauli.z.fits(n)))
        return false;
    } else {
      if (g.a >= n)
        return false;
      if ((g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap) &&
          (g.b >= n || g.a == g.b))
        return false;
    }
  }
  return true;
}
Frame::Frame(unsigned n) : width(n) {
  for (unsigned i = 0; i < n; ++i)
    rows.push_back({QubitSet::single(i, 1), 0});
  for (unsigned i = 0; i < n; ++i)
    rows.push_back({0, QubitSet::single(i, 1)});
}
/// Evaluate F^dagger*p*F from the signed images of the Pauli generators.
Pauli Frame::lookup(const Pauli &p) const {
  Pauli out;
  unsigned phase = p.x.intersectionCount(p.z) + 2 * p.sign;
  // Multiply X images before Z images, matching the input's i^|X&Z| XZ
  // encoding. Intermediate products may be anti-Hermitian; keep their phase.
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned q : (half ? p.z : p.x).set_bits()) {
      auto [v, k] = multiply(out, rows[half * width + q]);
      out = std::move(v);
      phase += k;
    }
  out.sign = (phase & 2) != 0;
  return out;
}
/// Find the signed local axis whose lookup is p, without inverting a matrix.
Pauli Frame::inverseLookup(const Pauli &p) const {
  Pauli local;
  // Anticommutation with the image of Z_q extracts the local X_q bit, and
  // vice versa. A forward lookup then supplies the missing Hermitian sign.
  for (unsigned q = 0; q < width; ++q) {
    local.x |= QubitSet::single(q, p.anti(rows[width + q]));
    local.z |= QubitSet::single(q, p.anti(rows[q]));
  }
  local.sign = p.sign != lookup(local).sign;
  return local;
}
/// Append g in circuit order: F becomes g*F, hence lookup uses g^dagger*P*g
/// before applying the old frame. Global Clifford phase is not in the tableau.
void Frame::append(const Gate &g) {
  // Only touched generator images need updating for primitive Cliffords.
  if (g.kind == Kind::Phase)
    return;
  std::vector<std::pair<unsigned, Pauli>> updates;
  auto support =
      g.kind == Kind::Rotate ? g.pauli.x | g.pauli.z : QubitSet::single(g.a, 1);
  if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
    support |= QubitSet::single(g.b, 1);
  for (auto bits = support; bits; bits.resetFirst()) {
    unsigned q = firstBit(bits);
    updates.emplace_back(
        q, lookup(conjugate({QubitSet::single(q), {}}, inverse(g))));
    updates.emplace_back(
        width + q, lookup(conjugate({{}, QubitSet::single(q)}, inverse(g))));
  }
  // All new images must use the old tableau, including multi-wire updates.
  for (auto &[q, value] : updates)
    rows[q] = std::move(value);
}
std::vector<Gate> adjoint(const std::vector<Gate> &gates) {
  std::vector<Gate> out;
  out.reserve(gates.size());
  for (auto it = gates.rbegin(); it != gates.rend(); ++it)
    out.push_back(inverse(*it));
  return out;
}
/// Expand supported multi-wire rotations into exact parity gadgets. This is
/// mechanical reconstruction, with no search for shared entangling networks.
std::vector<Gate> materialize(const std::vector<Gate> &gates) {
  std::vector<Gate> out;
  for (const auto &g : gates) {
    if (g.kind != Kind::Rotate || g.pauli.weight() == 1) {
      out.push_back(g);
      continue;
    }
    if (!g.pauli.weight()) {
      // Decoding only emits nonidentity axes. Keep opaque identity rotations
      // untouched; no implicit division of an SSA angle is permitted.
      out.push_back(g);
      continue;
    }
    std::vector<Gate> basis;
    auto support = g.pauli.x | g.pauli.z;
    unsigned pivot = firstBit(support);
    // Map each local axis to Z (Y needs Sdg then H), then compute the support
    // parity onto one deterministic pivot. Its Rz realizes the full rotation.
    for (auto bits = support; bits; bits.resetFirst()) {
      unsigned q = firstBit(bits), a = g.pauli.axis(q);
      if (a == 3)
        basis.push_back({Kind::Sdg, q});
      if (a & 1)
        basis.push_back({Kind::H, q});
    }
    for (auto bits = support.withoutFirst(); bits; bits.resetFirst())
      basis.push_back({Kind::CX, unsigned(firstBit(bits)), pivot});
    out.insert(out.end(), basis.begin(), basis.end());
    out.push_back(Gate::rotation({0, QubitSet::single(pivot, 1)},
                                 g.pauli.sign ? g.angle.negated() : g.angle));
    auto undo = adjoint(basis);
    // Uncompute both parity and basis changes; retain the signed angle rather
    // than dropping the scalar phase associated with a different convention.
    out.insert(out.end(), undo.begin(), undo.end());
  }
  return out;
}
/// Cancel primitive inverse pairs only after proving every crossed gate
/// commutes. Budget exhaustion retains the unchecked remainder unchanged.
std::vector<Gate> cancelCliffords(const std::vector<Gate> &input,
                                  Budget &budget) {
  std::vector<bool> removed(input.size(), false);
  std::map<std::tuple<Kind, unsigned, unsigned>, size_t> latest;
  auto key = [](Gate g) {
    bool two = g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap;
    return std::tuple{g.kind, g.a, two ? g.b : 0};
  };
  auto support = [](Gate g) {
    auto bits = QubitSet::single(g.a, 1);
    if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
      bits |= QubitSet::single(g.b, 1);
    return bits;
  };
  auto commutes = [&](Gate g, Gate h) {
    // These are sufficient rules, not a complete commutation oracle. Axis
    // equality includes the sign: conjugation to -P does not commute with R_P.
    if (h.kind == Kind::Phase)
      return true;
    if (h.kind == Kind::Rotate)
      return conjugate(h.pauli, g) == h.pauli;
    if (!(support(g) & support(h)))
      return true;
    if (g.kind == Kind::CX && h.kind == Kind::CX)
      return g.a != h.b && g.b != h.a;
    if (g.kind == Kind::CZ && h.kind == Kind::CZ)
      return true;
    if (h.kind == Kind::X || h.kind == Kind::Y || h.kind == Kind::Z ||
        h.kind == Kind::S || h.kind == Kind::Sdg) {
      Pauli axis{h.kind == Kind::X || h.kind == Kind::Y ? QubitSet::single(h.a)
                                                        : 0,
                 h.kind == Kind::X ? 0 : QubitSet::single(h.a)};
      return conjugate(axis, g) == axis;
    }
    return false;
  };
  for (size_t i = 0; i < input.size() && !budget.stopped(); ++i) {
    auto g = input[i];
    if (g.kind == Kind::Rotate || g.kind == Kind::Phase)
      continue;
    auto it = latest.find(key(inverse(g)));
    // The map proposes a partner; it does not certify the intervening region.
    // Previously removed gates need not be crossed again.
    bool cancel = it != latest.end() && !removed[it->second];
    if (cancel)
      for (size_t j = it->second + 1; j < i; ++j) {
        if (!budget.take()) {
          cancel = false;
          break;
        }
        if (!removed[j] && !commutes(g, input[j])) {
          cancel = false;
          break;
        }
      }
    if (cancel) {
      removed[it->second] = removed[i] = true;
      latest.erase(it);
    } else
      latest[key(g)] = i;
  }
  std::vector<Gate> out;
  for (size_t i = 0; i < input.size(); ++i)
    if (!removed[i])
      out.push_back(input[i]);
  return out;
}
/// Charge local and enclosing work limits; the first failure latches the stop
/// state so callers cannot resume a partially completed proof accidentally.
bool Budget::take(uint64_t amount) {
  if (exhausted)
    return false;
  if ((shared && !shared->take(amount)) ||
      amount > options.maxWork - stats.work ||
      (options.timeoutMs && std::chrono::steady_clock::now() - start >=
                                std::chrono::milliseconds(options.timeoutMs))) {
    exhausted = true;
    ++stats.budgetExhaustions;
    return false;
  }
  stats.work += amount;
  return true;
}
} // namespace cudaq::pauli::tcount::detail

bool Cost::operator<(const Cost &other) const {
  return std::tie(t, twoQubit, depth, gates) <
         std::tie(other.t, other.twoQubit, other.depth, other.gates);
}
Cost cudaq::pauli::tcount::cost(unsigned width,
                                const std::vector<Gate> &input) {
  Cost result;
  if (input.empty())
    return result;
  std::vector<uint64_t> depths(width), tDepths(width);
  // Score concrete parity gadgets, not each multi-wire rotation as one gate.
  // Arbitrary angles remain one unresolved rotation with no estimated T cost.
  for (const auto &g : detail::materialize(input)) {
    if (g.kind == Kind::Phase)
      continue;
    auto support = g.kind == Kind::Rotate ? g.pauli.x | g.pauli.z
                                          : QubitSet::single(g.a, 1);
    unsigned count = 1, t = 0;
    if (g.kind == Kind::Rotate) {
      support = g.pauli.x | g.pauli.z;
      if (auto k = detail::quarters(g.angle)) {
        unsigned magnitude = std::abs(*k);
        if (magnitude > 8)
          magnitude = 16 - magnitude;
        // Count a Z/S/T decomposition, ignoring its scalar phase. Only an odd
        // coefficient consumes T; X/Y axes also need local basis changes.
        count = std::popcount(magnitude & 7u);
        t = (magnitude & 1) && support;
        result.t += t;
        if (count && g.pauli.x)
          count += 2 + (g.pauli.x == g.pauli.z ? 2 : 0);
      }
    } else if (g.kind == Kind::CX || g.kind == Kind::CZ ||
               g.kind == Kind::Swap) {
      support |= QubitSet::single(g.b, 1);
      ++result.twoQubit;
    }
    uint64_t depth = 0, tDepth = 0;
    // Synchronize every touched wire before advancing its schedule. Clifford
    // gates add no T layer but still propagate dependencies between T layers.
    for (auto bits = support; bits; bits.resetFirst()) {
      depth = std::max(depth, depths[firstBit(bits)]);
      tDepth = std::max(tDepth, tDepths[firstBit(bits)]);
    }
    for (auto bits = support; bits; bits.resetFirst()) {
      depths[firstBit(bits)] = depth + count;
      tDepths[firstBit(bits)] = tDepth + t;
    }
    result.tDepth = std::max(result.tDepth, tDepth + t);
    result.gates += count;
    result.depth = std::max(result.depth, depth + count);
  }
  return result;
}
