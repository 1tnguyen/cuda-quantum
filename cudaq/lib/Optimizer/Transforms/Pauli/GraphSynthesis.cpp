/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Greedy synthesis and checked replay of Pauli dependency graphs.
///
/// Imports rotations into a Clifford frame, simplifies commuting nodes, and
/// selects entangling actions using support and scheduling costs. Restricted
/// algebraic replay checks each synthesized candidate under the requested
/// unitary or terminal-measurement contract.
///
/// The workflow is: transport rotations through the input Clifford frame;
/// simplify their partially commuting word; build ordering edges between
/// anticommuting rotations; emit ready single-wire rotations while greedily
/// reducing wider supports; reconstruct the residual frame and scalar phase.
/// Synthesis heuristics propose a circuit. A separate replay must accept that
/// circuit before callers may use it.
///
/// References below use the extended v2 of Paykin et al., "PCOAST: A
/// Pauli-based Quantum Circuit Optimization Framework":
/// https://arxiv.org/pdf/2305.10966v2
/// This implements a rotation/frame subset of its workflow (Sec. V, Fig. 8).
/// Resource budgets, exact scalar-phase replay, and acceptance guards are
/// implementation-specific; the paper's proofs do not validate this code.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include "cudaq/Optimizer/Pauli/Telemetry.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;

namespace {
/// Normalize an integer exponent of i to the range [0, 3].
unsigned mod4(int v) { return unsigned(v) & 3; }
/// Construct an unsigned one-wire Pauli using axis codes 0=I, 1=X, 2=Z, 3=Y.
Pauli single(unsigned axis, unsigned q) {
  return {QubitSet::single(q, axis & 1), QubitSet::single(q, axis >> 1), false};
}
/// Return (c, k) such that a*b = i^k c, with c an unsigned Hermitian Pauli.
/// The phase accounts for Y=iXZ, operand signs, and Z/X exchanges. Keeping it
/// separately also represents products of anticommuting Hermitian operands.
std::pair<Pauli, unsigned> multiply(const Pauli &a, const Pauli &b) {
  Pauli c{a.x ^ b.x, a.z ^ b.z};
  int k = a.x.intersectionCount(a.z) + b.x.intersectionCount(b.z) -
          c.x.intersectionCount(c.z) + 2 * a.z.intersectionCount(b.x) +
          2 * (a.sign + b.sign);
  return {c, mod4(k)};
}
/// Invert a supported primitive Clifford. Only S/Sdg exchange kinds; the other
/// supported Cliffords are self-inverse. This helper is not a rotation inverse.
Gate inverse(Gate g) {
  if (g.kind == Kind::S)
    g.kind = Kind::Sdg;
  else if (g.kind == Kind::Sdg)
    g.kind = Kind::S;
  return g;
}
/// Compute g*p*g^dagger for a supported primitive Clifford, including its sign.
/// Callers validate gate kinds and wire indices before using this algebra.
Pauli conjugate(Pauli p, const Gate &g) {
  bool x = p.x.test(g.a), z = p.z.test(g.a);
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
    // X on the control spreads to the target; Z on the target spreads to the
    // control. The sign correction retains the phase of overlapping Y factors.
    p.sign ^= x && zb && (xb ^ z ^ true);
    p.x.flip(g.b, x);
    p.z.flip(g.a, zb);
    break;
  }
  case Kind::CZ:
    // Reuse the signed CX rule through CZ = H(target) CX H(target).
    p = conjugate(conjugate(conjugate(p, {Kind::H, g.b}), {Kind::CX, g.a, g.b}),
                  {Kind::H, g.b});
    break;
  case Kind::Swap: {
    auto mask = (QubitSet::single(g.a, 1)) | (QubitSet::single(g.b, 1));
    p.x ^= (p.x.test(g.a) != p.x.test(g.b)) ? mask : 0;
    p.z ^= (p.z.test(g.a) != p.z.test(g.b)) ? mask : 0;
    break;
  }
  default:
    break;
  }
  return p;
}

/// Signed Clifford tableau for the residual circuit, storing F^dagger P F.
/// See PCOAST Sec. IV-A, Definitions 11-12; our storage orders X before Z.
struct Frame {
  /// Number of wires represented by the tableau.
  unsigned n;
  /// Images of X_0..X_(n-1), followed by Z_0..Z_(n-1).
  std::vector<Pauli> rows;
  /// Start with the identity Clifford on n wires.
  explicit Frame(unsigned n) : n(n) {
    for (unsigned axis : {1u, 2u})
      for (unsigned q = 0; q < n; ++q)
        rows.push_back(single(axis, q));
  }
  /// Evaluate F^dagger*p*F by multiplying the stored generator images.
  /// Include the input's Y/sign phase so the result is a signed Hermitian
  /// Pauli.
  Pauli lookup(const Pauli &p) const {
    Pauli out;
    unsigned phase = p.x.intersectionCount(p.z) + 2 * p.sign;
    for (unsigned half = 0; half < 2; ++half)
      for (unsigned q : (half ? p.z : p.x).set_bits()) {
        auto [r, k] = multiply(out, rows[half * n + q]);
        out = std::move(r);
        phase += k;
      }
    out.sign = (phase & 2) != 0;
    return out;
  }
  /// Append an input Clifford: F <- g F, with rows representing F^dagger P F.
  /// Only generator rows on g's support change. Scalar phases live separately.
  void append(const Gate &g) {
    if (g.kind == Kind::Phase)
      return;
    std::vector<std::pair<unsigned, Pauli>> updates;
    auto update = [&](unsigned q) {
      updates.emplace_back(q, lookup(conjugate(single(1, q), inverse(g))));
      updates.emplace_back(n + q, lookup(conjugate(single(2, q), inverse(g))));
    };
    update(g.a);
    if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
      update(g.b);
    // Compute every changed row against the old tableau before committing any
    // update; a two-wire gate must not read one already-updated generator.
    for (auto &[i, value] : updates)
      rows[i] = std::move(value);
  }
  /// Remove an emitted Clifford from the start of the residual: F <- F
  /// g^dagger. Its inverse action on F conjugates every stored row by g.
  void carry(Gate g) {
    for (auto &p : rows)
      p = conjugate(p, g);
  }
};

/// Exact phase-sensitive stabilizer state for checking Clifford scalar phase.
/// An anchor amplitude fixes the global phase absent from a signed tableau.
struct StabilizerState {
  /// Independent signed stabilizers determining the state's amplitude ratios.
  std::vector<Pauli> stabilizers;
  /// Computational-basis bit string with nonzero amplitude.
  QubitSet anchor;
  /// Anchor amplitude is exp(i*pi*phase/4) / sqrt(2)^exponent.
  int phase = 0, exponent = 0;
  /// Initialize |0...0>, whose stabilizers are +Z on each wire and anchor is 0.
  explicit StabilizerState(unsigned n) {
    for (unsigned q = 0; q < n; ++q)
      stabilizers.push_back(single(2, q));
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
      p = conjugate(p, g);
  }
};

/// Synthesis item: a rotation axis or an anticommuting frame-factor pair.
struct Node {
  /// Rotation axis p, or frame images p=Z and q=X when factor is true.
  Pauli p, q;
  /// Rotation angle for a singlet; unused for a frame factor.
  Angle angle;
  /// Selects the paired frame-factor cost instead of the rotation singlet cost.
  bool factor = false;
  /// Estimate remaining entangling work from Pauli support, not T count or
  /// scheduled depth. A weight-w rotation costs w-1. For an anticommuting
  /// frame pair, add the union-support cost and the excess local
  /// anticommutation cost; zero means the pair occupies a single wire.
  /// PCOAST Sec. V-C1: Eq. (154) for singlets, Eq. (162) for factors.
  int cost() const {
    if (!factor)
      return std::max(0, int(p.weight()) - 1);
    return (population((p.x & q.z) ^ (p.z & q.x)) - 1) / 2 +
           population(p.x | p.z | q.x | q.z) - 1;
  }
};
/// Imported rotation word and Clifford frame; DAG edges are built later.
struct Graph {
  /// Ordered rotation nodes whose noncommutation determines dependencies.
  std::vector<Node> nodes;
  /// Residual Clifford action after transporting rotations to a common frame.
  Frame frame;
  /// Original Clifford sequence, retained for phase-sensitive reconstruction.
  std::vector<Gate> cliffords;
  /// Explicit scalar angles carried separately from rotation nodes.
  std::vector<Angle> phases;
  /// Begin with an empty rotation word and identity residual Clifford.
  explicit Graph(unsigned n) : frame(n) {}
};
/// Import a validated circuit as an ordered rotation word plus a final frame.
/// Move accumulated Cliffords past each rotation using R(P,a) F =
/// F R(F^dagger P F,a). Absorb a negative transformed axis into the angle, so
/// imported axes are unsigned and equal-axis merging can compare them directly.
/// See PCOAST Lemma 30 and Sec. V-A, Fig. 9 (circuit-to-graph conversion).
Graph import(unsigned n, const std::vector<Gate> &gates) {
  Graph graph(n);
  for (const Gate &g : gates) {
    if (g.kind == Kind::Phase)
      graph.phases.push_back(g.angle);
    else if (g.kind == Kind::Rotate) {
      Pauli p = graph.frame.lookup(g.pauli);
      Angle a = p.sign ? g.angle.negated() : g.angle;
      p.sign = false;
      graph.nodes.push_back({p, {}, a});
    } else {
      graph.frame.append(g);
      graph.cliffords.push_back(g);
    }
  }
  return graph;
}

/// Add exact rational angles, or cancel opposite occurrences of one opaque
/// symbol. Other symbolic sums are not representable here. Reject overflowing
/// intermediates rather than round, even if a reduced result might fit.
/// Do not reduce modulo 2*pi: that can change a rotation's scalar phase.
std::optional<Angle> add(Angle a, Angle b) {
  if (a.symbol || b.symbol) {
    if (a.symbol && a.symbol == b.symbol && a.negative != b.negative)
      return Angle{};
    return std::nullopt;
  }
  __int128 num =
      (__int128(a.numerator) * b.denominator) * (a.negative ? -1 : 1) +
      (__int128(b.numerator) * a.denominator) * (b.negative ? -1 : 1);
  __int128 den = __int128(a.denominator) * b.denominator;
  if (num < std::numeric_limits<int64_t>::min() ||
      num > std::numeric_limits<int64_t>::max() ||
      den > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  auto nn = int64_t(num), dd = int64_t(den);
  if (nn == std::numeric_limits<int64_t>::min())
    return std::nullopt;
  auto gcd = std::gcd(nn, dd);
  return Angle{nn / gcd, dd / gcd};
}
/// Simplify the rotation word with one deterministic left-to-right pass.
/// For each node, scan retained predecessors backwards through commuting axes
/// and merge the nearest equal axis using R(P,a) R(P,b) = R(P,a+b). Stop at
/// the first anticommuting axis or equal-axis match whose sum is unsupported.
/// Remove exact zero angles and turn representable identity-axis rotations into
/// scalar phases. Preserve the frame and retain unsupported arithmetic exactly.
///
/// For example, R_ZZ(a), R_ZI(b), R_ZZ(c) becomes R_ZI(b), R_ZZ(a+c).
/// Replacing the middle ZI axis with XI blocks that merge. This is local word
/// simplification, not DAG synthesis or a global T-count search. The number of
/// predecessor comparisons can be quadratic; budgeted callers charge for it.
/// See PCOAST Sec. IV-C3, Fig. 6, Eqs. (105)-(106). This implements only the
/// rotation identities, not the full merging rules of Fig. 5 / ADDNODE,
/// Fig. 10.
void simplify(Graph &graph) {
  std::vector<Node> out;
  for (Node node : graph.nodes) {
    if (!node.p.weight()) {
      // R(I,theta) = exp(-i theta/2). SSA halving is not manufactured here.
      if (!node.angle.symbol && node.angle.denominator <= INT64_MAX / 2) {
        Angle phase = node.angle.negated();
        phase.denominator *= 2;
        graph.phases.push_back(phase);
        continue;
      }
    }
    // Moving the earlier equal-axis rotation to the current position is valid
    // only if it commutes with every intervening retained node.
    for (auto it = out.rbegin(); it != out.rend(); ++it) {
      if (it->p == node.p) {
        if (auto sum = add(it->angle, node.angle)) {
          node.angle = *sum;
          out.erase(std::next(it).base());
        }
        break;
      }
      if (it->p.anti(node.p))
        break;
    }
    // Opaque symbols are never assumed zero. Rational cancellation drops the
    // rotation only when its exact numerator is zero, with no tolerance test.
    if (node.angle.symbol || node.angle.numerator)
      out.push_back(node);
  }
  graph.nodes = std::move(out);
}

/// Local Clifford sequence mapping the chosen nonidentity axis to +Z under
/// conjugation. The signed sequence is part of the synthesis/replay convention;
/// axis 2 (Z) needs no change of basis.
std::vector<Gate> basis(unsigned axis, unsigned q) {
  if (axis == 1)
    return {{Kind::H, q}, {Kind::Z, q}};
  if (axis == 3)
    return {{Kind::H, q}, {Kind::S, q}, {Kind::H, q}};
  return {};
}
/// CZ action with local basis choices and deterministic tie-break ordering.
/// PCOAST Sec. V-C, footnote 10: nine local-axis choices per wire pair.
struct Candidate {
  /// Wire pair i/j and local axis codes a/b (1=X, 2=Z, 3=Y), respectively.
  unsigned i, j, a, b;
  /// Stable ordering for deduplication and selection among equal-score actions.
  bool operator<(const Candidate &c) const {
    return std::tie(i, j, a, b) < std::tie(c.i, c.j, c.a, c.b);
  }
};
/// Materialize one candidate as local bases followed by CZ. Symmetric mode
/// appends inverse bases; asymmetric mode leaves their action in the residual
/// graph/frame so later operations can share the basis changes.
/// See PCOAST Sec. V-C1, the native-gate search discussion after Eq. (162).
std::vector<Gate> sequence(Candidate c, bool asymmetric) {
  auto gates = basis(c.a, c.i), second = basis(c.b, c.j);
  gates.insert(gates.end(), second.begin(), second.end());
  auto pre = gates;
  gates.push_back({Kind::CZ, c.i, c.j});
  if (!asymmetric)
    for (auto it = pre.rbegin(); it != pre.rend(); ++it)
      gates.push_back(inverse(*it));
  return gates;
}
/// Conjugate an axis by the complete symmetric candidate using a local table.
/// Local bases omitted by asymmetric emission do not change support weight or
/// local anticommutation, so this same transform suffices for candidate costs.
/// Actual residual updates use the emitted sequence, including its signs.
Pauli transformedPauli(Pauli p, Candidate c) {
  // The complete local Clifford action, including signs, is only 9 x 16.
  static const auto tables = [] {
    std::array<std::array<Pauli, 16>, 9> tables;
    for (unsigned a = 1; a <= 3; ++a)
      for (unsigned b = 1; b <= 3; ++b)
        for (unsigned k = 0; k < 16; ++k) {
          Pauli p{(k & 1) | ((k >> 1) & 2), ((k >> 1) & 1) | ((k >> 2) & 2)};
          for (const Gate &g : sequence({0, 1, a, b}, false))
            p = conjugate(p, g);
          tables[(a - 1) * 3 + b - 1][k] = p;
        }
    return tables;
  }();
  unsigned first = p.axis(c.i), second = p.axis(c.j);
  Pauli local = tables[(c.a - 1) * 3 + c.b - 1][first | (second << 2)];
  p.x ^= QubitSet::single(c.i, local.x.test(0) ^ bool(first & 1));
  p.x ^= QubitSet::single(c.j, local.x.test(1) ^ bool(second & 1));
  p.z ^= QubitSet::single(c.i, local.z.test(0) ^ bool(first >> 1));
  p.z ^= QubitSet::single(c.j, local.z.test(1) ^ bool(second >> 1));
  p.sign ^= local.sign;
  return p;
}
/// Transform a singlet axis, or both axes of a frame factor, for cost scoring.
Node transformed(Node v, Candidate c) {
  v.p = transformedPauli(v.p, c);
  if (v.factor)
    v.q = transformedPauli(v.q, c);
  return v;
}

/// View each logical wire's signed Z/X frame images as an anticommuting pair.
/// Their costs guide reconstruction of the residual Clifford after rotations.
std::vector<Node> factors(const Frame &f) {
  std::vector<Node> result;
  for (unsigned q = 0; q < f.n; ++q)
    result.push_back({f.rows[f.n + q], f.rows[q], {}, true});
  return result;
}

/// Bounded greedy synthesis of ready rotations followed by the residual frame.
class Synthesizer {
  /// Number of wires in the input and emitted circuit.
  unsigned n;
  /// Borrowed scoring, semantic, and resource controls for this invocation.
  const Options &options;
  /// Mutable residual rotations and Clifford frame.
  Graph graph;
  /// Accumulated output, counters, and sticky failure diagnostic.
  Result result;
  /// Emitted Clifford subsequence used to recover the exact scalar phase.
  std::vector<Gate> emittedCliffords;
  /// Current ASAP finish depth on each wire.
  std::vector<unsigned> depths;
  /// Maximum emitted depth and number of entangling synthesis actions taken.
  unsigned depth = 0, steps = 0;
  /// Optional borrowed work/deadline budget shared with search and replay.
  SearchBudget *budget;

  /// Charge estimated work and preserve the first failure as a sticky error.
  /// Without a shared budget, structural gate/edge/step limits still apply.
  bool charge(uint64_t work = 1) {
    if (!result.error.empty())
      return false;
    if (budget && !budget->take(work)) {
      result.error = "Pauli graph synthesis shared search budget exhausted";
      result.stats.budgetExhausted = true;
      return false;
    }
    return true;
  }
  /// Append a gate subject to output/work limits and update its ASAP schedule.
  /// Rotation callers must supply a single-wire axis. Scalar phases consume
  /// output storage but no depth; Cliffords also enter the phase-replay
  /// skeleton.
  bool emit(Gate g) {
    if (!charge() || result.gates.size() >= options.maxGates) {
      if (result.error.empty()) {
        result.error = "Pauli graph synthesis output gate limit exceeded";
        result.stats.budgetExhausted = true;
      }
      return false;
    }
    result.gates.push_back(g);
    if (g.kind == Kind::Rotate) {
      unsigned q = firstBit(g.pauli.x | g.pauli.z);
      depth = std::max(depth, ++depths[q]);
    } else if (g.kind != Kind::Phase) {
      emittedCliffords.push_back(g);
      bool two =
          g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap;
      unsigned d =
          two ? std::max(depths[g.a], depths[g.b]) + 1 : depths[g.a] + 1;
      depths[g.a] = d;
      if (two)
        depths[g.b] = d;
      depth = std::max(depth, d);
    }
    return true;
  }
  /// Emit a Clifford prefix and conjugate every remaining rotation and frame
  /// row to compensate. Common Clifford conjugation preserves commutation, so
  /// previously constructed DAG edges remain valid after this update.
  /// PCOAST Eq. (153), ADDGATE; Lemma 18 gives commutation preservation.
  bool carry(Gate g, const std::vector<bool> &active) {
    if (!charge(graph.nodes.size() + 2 * n) || !emit(g))
      return false;
    for (unsigned i = 0; i < graph.nodes.size(); ++i)
      if (active[i])
        graph.nodes[i].p = conjugate(graph.nodes[i].p, g);
    graph.frame.carry(g);
    return true;
  }
  /// Choose one entangling action that reduces at least one minimum-cost ready
  /// node. Enumerate supported wire pairs and local axes, then minimize the
  /// weighted support-cost change plus frame penalty minus scheduling credit.
  /// The population/lookahead influences scoring, not dependency eligibility.
  /// Return no candidate on an empty frontier, exhausted budget or no legal
  /// reducing action; no recursive or exhaustive circuit search is performed.
  /// PCOAST Fig. 12, lines 10-13; scoring follows Sec. V-C1. Candidate caps
  /// and lookahead sampling are implementation-specific resource controls.
  std::optional<Candidate> choose(const std::vector<Node> &front,
                                  const std::vector<Node> &population,
                                  bool penalizeFrame) {
    if (front.empty())
      return std::nullopt;
    int minimum =
        std::min_element(front.begin(), front.end(), [](auto a, auto b) {
          return a.cost() < b.cost();
        })->cost();
    std::set<Candidate> candidates;
    unsigned eligible = 0;
    // Restrict generation to the cheapest ready nodes, optionally capped by
    // candidateNodes. A candidate must reduce its generating node's cost;
    // other remaining nodes may become more expensive. The set deduplicates
    // actions generated by several nodes and fixes their tie-break order.
    for (Node v : front) {
      if (v.cost() != minimum)
        continue;
      if (options.candidateNodes && eligible++ >= options.candidateNodes)
        break;
      auto support = v.p.x | v.p.z | v.q.x | v.q.z;
      for (unsigned i = 0; i < n; ++i)
        if (support.test(i))
          for (unsigned j = i + 1; j < n; ++j)
            if (support.test(j))
              for (unsigned a = 1; a <= 3; ++a)
                for (unsigned b = 1; b <= 3; ++b) {
                  if (!charge())
                    return std::nullopt;
                  Candidate c{i, j, a, b};
                  if (transformed(v, c).cost() < minimum)
                    candidates.insert(c);
                }
    }
    result.stats.candidates += candidates.size();
    double best = std::numeric_limits<double>::infinity();
    std::optional<Candidate> chosen;
    auto frameFactors = factors(graph.frame);
    for (Candidate c : candidates) {
      if (!charge(std::min<size_t>(population.size(), options.lookahead
                                                          ? options.lookahead
                                                          : population.size()) +
                  front.size() + 4 * n + 1))
        return std::nullopt;
      double score = 0, norm = 0;
      auto include = [&](Node v, double weight) {
        score += weight * (transformed(v, c).cost() - v.cost());
        norm += weight;
      };
      unsigned count = 0;
      // Sample remaining nodes in graph order. lookahead bounds this scoring
      // sample; it does not explore that many future synthesis steps.
      for (Node v : population) {
        if (options.lookahead && count++ >= options.lookahead)
          break;
        include(v, 1);
      }
      // Ready nodes receive extra weight scaled by population/frontier size,
      // so a small frontier can still influence the normalized average.
      // PCOAST Sec. V-C1, footnote 12.
      for (Node v : front)
        include(v, options.frontierWeight * population.size() / front.size());
      score /= std::max(1.0, norm);
      if (penalizeFrame) {
        // Rotations can shrink while the residual Clifford becomes expensive.
        // Charge the mean frame-factor cost change when this penalty is
        // enabled.
        // PCOAST Sec. V-C1, terminating-frame penalty discussion.
        double delta = 0;
        for (Node v : frameFactors)
          delta += transformed(v, c).cost() - v.cost();
        score += options.frameWeight * delta / n;
      }
      // Explicit ASAP policy: credit the fraction of this candidate's depth
      // that fits below the current circuit depth. This is our schedule policy,
      // not a claim to reproduce Intel SDK's unpublished tie-breaking policy.
      // See PCOAST Sec. V-C1, schedule search and footnote 13; Sec. VI-C
      // evaluates fine tuning, rather than specifying this fractional credit.
      auto ds = depths;
      unsigned next = depth;
      for (const Gate &g : sequence(c, options.asymmetric)) {
        if (g.kind == Kind::CZ)
          ds[g.a] = ds[g.b] = std::max(ds[g.a], ds[g.b]) + 1;
        else
          ++ds[g.a];
        next = std::max(next, ds[g.a]);
      }
      unsigned serial = sequence(c, options.asymmetric).size();
      score -=
          options.parallelizationCredit * (1.0 - double(next - depth) / serial);
      // Strict comparison keeps the first candidate in deterministic set order
      // on a tie. A lower heuristic score is not a proof of global optimality.
      if (score < best) {
        best = score;
        chosen = c;
      }
    }
    return chosen;
  }
  /// Select and carry one entangling action, counting it against maxSteps.
  /// Failure may leave partial output, which remains unusable because result
  /// carries an error. The wrapper only accepts a complete, replay-checked run.
  bool step(const std::vector<Node> &front, const std::vector<Node> &population,
            bool withFrame, const std::vector<bool> &active) {
    if (!charge())
      return false;
    if (++steps > options.maxSteps) {
      result.stats.budgetExhausted = true;
      result.error = "Pauli graph synthesis synthesis step limit exceeded";
      return false;
    }
    auto candidate = choose(front, population, withFrame);
    if (!candidate) {
      if (result.error.empty())
        result.error =
            "Pauli graph synthesis found no monotonically reducing candidate";
      return false;
    }
    // The emitted prefix is B then CZ, so the residual is conjugated by
    // CZ B (not its inverse); this carries the omitted trailing B^dagger.
    for (const Gate &g : sequence(*candidate, options.asymmetric))
      if (!carry(g, active))
        return false;
    ++result.stats.tqes;
    return true;
  }
  /// Reconstruct the residual Clifford: reduce frame factors to single-wire
  /// pairs, restore their local axes/signs, undo the remaining wire
  /// permutation, and recover scalar phase. Even release mode retains this
  /// frame; release only authorizes the separately checked removal of invisible
  /// rotations.
  /// PCOAST Sec. V-C2 describes factor-based reconstruction for hold semantics.
  /// Its release path discards the frame and synthesizes measurements instead;
  /// our restricted release path still reconstructs the frame here.
  bool finalize(const std::vector<bool> &active) {
    // Frame factors now serve as both the ready set and scored population.
    // They already express the reconstruction objective, so step adds no
    // separate frame penalty while reducing them.
    while (true) {
      auto all = factors(graph.frame);
      std::vector<Node> remaining;
      for (Node v : all)
        if (v.cost())
          remaining.push_back(v);
      if (remaining.empty())
        break;
      if (!step(remaining, all, false, active))
        return false;
    }
    // Each factor now lives on one physical wire. Apply exact local Cliffords
    // to make its inverse images +X,+Z, then restore the physical permutation.
    for (unsigned row = 0; row < n; ++row) {
      unsigned q = firstBit(graph.frame.rows[row].x | graph.frame.rows[row].z);
      /// Breadth-first search state for restoring one signed local frame pair.
      struct Search {
        /// Current signed images of the wire's X and Z generators.
        Pauli x, z;
        /// H/S sequence taking the initial pair to these images.
        std::vector<Gate> word;
      };
      std::vector<Search> work{
          {graph.frame.rows[row], graph.frame.rows[n + row], {}}};
      std::set<std::tuple<unsigned, unsigned, bool, bool>> seen;
      bool found = false;
      // H and S generate the finite signed single-qubit Clifford actions. BFS
      // finds a shortest word in this generator set; visited keys need only
      // the two local axes and their signs, not an entire multi-wire tableau.
      for (size_t k = 0; k < work.size(); ++k) {
        auto state = work[k];
        if (state.x == single(1, q) && state.z == single(2, q)) {
          for (const Gate &g : state.word)
            if (!carry(g, active))
              return false;
          found = true;
          break;
        }
        for (Kind kind : {Kind::H, Kind::S}) {
          Gate g{kind, q};
          auto x = conjugate(state.x, g), z = conjugate(state.z, g);
          if (seen.emplace(x.axis(q), z.axis(q), x.sign, z.sign).second) {
            auto word = state.word;
            word.push_back(g);
            work.push_back({x, z, std::move(word)});
          }
        }
      }
      if (!found) {
        result.error = "Invalid local Clifford factor";
        return false;
      }
    }
    // Local normalization fixes axes on whichever physical wire holds them.
    // Explicit SWAPs then restore logical-to-physical wire identity; omitting
    // them would require an output-permutation contract this API does not have.
    for (unsigned q = 0; q < n; ++q) {
      unsigned at = firstBit(graph.frame.rows[q].x);
      if (at != q) {
        if (!carry({Kind::Swap, q, at}, active))
          return false;
        ++result.stats.swaps;
      }
    }
    Frame identity(n);
    if (graph.frame.rows != identity.rows) {
      result.error = "Terminal Clifford factorization failed";
      return false;
    }
    // F and the emitted Clifford skeleton have the same signed tableau.
    // Recover the exact eighth-root phase using polynomial stabilizer replay.
    if (!charge(llvm::SaturatingMultiply<uint64_t>(
            llvm::SaturatingAdd<uint64_t>(graph.cliffords.size(),
                                          emittedCliffords.size()),
            uint64_t(n) * n + 1)))
      return false;
    StabilizerState original(n), emitted(n);
    for (const Gate &g : graph.cliffords)
      original.apply(g);
    for (const Gate &g : emittedCliffords)
      emitted.apply(g);
    // Equal signed tableaux imply the Clifford skeletons differ only by a
    // scalar. Comparing one nonzero stabilizer-state amplitude determines that
    // scalar without constructing a dense 2^n-dimensional unitary or state.
    auto ratio = emitted.ratio(original.anchor);
    if (!ratio || original.exponent != emitted.exponent) {
      result.error = "Clifford phase witness failed";
      return false;
    }
    int correction = (original.phase - emitted.phase - 2 * int(*ratio)) & 7;
    if (correction && !emit(Gate::phase({correction, 4})))
      return false;
    return true;
  }

public:
  /// Import a validated input and start an empty output schedule. The options
  /// and optional shared budget are borrowed for the duration of run().
  Synthesizer(unsigned n, const std::vector<Gate> &input,
              const Options &options, SearchBudget *budget)
      : n(n), options(options), graph(import(n, input)), depths(n),
        budget(budget) {}
  /// Simplify/import dependencies, consume ready rotations, and finish the
  /// residual frame. This builds a candidate; synthesize() supplies the final
  /// independent equivalence replay and standalone cost-acceptance guard.
  Result run() {
    result.stats.rotationsBefore = graph.nodes.size();
    if (options.simplify) {
      if (!charge(uint64_t(graph.nodes.size()) * graph.nodes.size()))
        return result;
      simplify(graph);
    }
    if (options.release) {
      // Conservative terminal-measurement optimization: remove rotations in
      // the backward cone's complement. Keep the frame and measurement order;
      // no classical remapping is silently introduced.
      std::vector<Node> kept;
      for (auto it = graph.nodes.rbegin(); it != graph.nodes.rend(); ++it) {
        bool affects = false;
        for (unsigned q = 0; q < n; ++q)
          affects |= it->p.anti(graph.frame.rows[n + q]);
        for (Node v : kept)
          affects |= it->p.anti(v.p);
        if (affects)
          kept.push_back(*it);
        else
          ++result.stats.released;
      }
      std::reverse(kept.begin(), kept.end());
      graph.nodes = std::move(kept);
    }
    result.stats.rotationsAfter = graph.nodes.size();
    unsigned size = graph.nodes.size();
    // Original order is required only for anticommuting axes. Add an edge i->j
    // for every such pair with i<j; this is an acyclic dependency graph, not a
    // per-qubit wire graph or a transitive reduction. Never omit an edge to fit
    // the resource cap: reject the synthesis candidate instead.
    // PCOAST Sec. IV-C, Definition 26 and Lemma 27 (topological ordering).
    std::vector<std::vector<unsigned>> successors(size);
    std::vector<unsigned> degree(size);
    for (unsigned i = 0; i < size; ++i)
      for (unsigned j = i + 1; j < size; ++j) {
        if (!charge())
          return result;
        if (graph.nodes[i].p.anti(graph.nodes[j].p)) {
          if (result.stats.dependencies >= options.maxDependencies) {
            result.error = "Pauli graph synthesis dependency limit exceeded";
            result.stats.budgetExhausted = true;
            return result;
          }
          successors[i].push_back(j);
          ++degree[j];
          ++result.stats.dependencies;
        }
      }
    std::vector<bool> active(size, true);
    unsigned left = size;
    while (left) {
      if (!charge(size))
        return result;
      std::vector<Node> front, population;
      bool removed = false;
      // Degree zero means all required predecessors have been emitted. A ready
      // zero-cost singlet already fits one wire, so emit it without another
      // entangler and release its successors. Preserve any sign carried into
      // the axis by negating the rotation angle on emission.
      // PCOAST Fig. 12, lines 4-9.
      for (unsigned i = 0; i < size; ++i)
        if (active[i] && !degree[i] && !graph.nodes[i].cost()) {
          Node v = graph.nodes[i];
          if (!v.p.weight()) {
            result.error = "Identity rotation requires a rational angle";
            return result;
          }
          if (!emit(Gate::rotation({v.p.x, v.p.z, false},
                                   v.p.sign ? v.angle.negated() : v.angle)))
            return result;
          active[i] = false;
          --left;
          removed = true;
          for (unsigned j : successors[i])
            --degree[j];
        }
      if (removed)
        continue;
      // No ready rotation is directly executable. Propose an entangling action
      // from the frontier and score its effects on the remaining population.
      for (unsigned i = 0; i < size; ++i)
        if (active[i]) {
          population.push_back(graph.nodes[i]);
          if (!degree[i])
            front.push_back(graph.nodes[i]);
        }
      if (!step(front, population, !options.release, active))
        return result;
    }
    if (!finalize(active))
      return result;
    // Explicit input/identity-rotation phases commute with the entire circuit.
    // Keep them in addition to the Clifford phase correction from finalize().
    for (Angle a : graph.phases)
      if (!emit(Gate::phase(a)))
        return result;
    return result;
  }
};

/// Compare exact rational angles by cross multiplication; opaque angles must
/// match their complete symbolic encoding. No tolerance or periodicity is used.
bool equalAngles(Angle a, Angle b) {
  if (a.symbol || b.symbol)
    return a == b;
  return __int128(a.numerator) * b.denominator * (a.negative ? -1 : 1) ==
         __int128(b.numerator) * a.denominator * (b.negative ? -1 : 1);
}
/// Check native gate tags, wire ranges, distinct two-wire operands and rational
/// denominators before tableau algebra. Identity rotations are allowed here;
/// simplification or a later checked failure handles their representability.
bool validCircuit(unsigned n, const std::vector<Gate> &gates) {
  if (!n)
    return false;
  for (const Gate &g : gates) {
    if (static_cast<unsigned>(g.kind) > static_cast<unsigned>(Kind::Phase))
      return false;
    switch (g.kind) {
    case Kind::Rotate:
      if (!g.pauli.x.fits(n) || !g.pauli.z.fits(n))
        return false;
      [[fallthrough]];
    case Kind::Phase:
      if (!g.angle.symbol && g.angle.denominator <= 0)
        return false;
      break;
    case Kind::CX:
    case Kind::CZ:
    case Kind::Swap:
      if (g.b >= n || g.a == g.b)
        return false;
      [[fallthrough]];
    case Kind::H:
    case Kind::S:
    case Kind::Sdg:
    case Kind::X:
    case Kind::Y:
    case Kind::Z:
      if (g.a >= n)
        return false;
      break;
    }
  }
  return true;
}

/// Standalone acceptance estimate for circuits with single-wire rotations.
/// Count each rotation/local gate as one operation, CX/CZ as one entangler, and
/// SWAP as three CNOTs; ignore scalar phases. Return either (depth, entanglers,
/// gates) or (entanglers, depth, gates). This is not target routing, rotation
/// synthesis, or the combined portfolio's Clifford+T mechanical cost model.
std::tuple<unsigned, unsigned, unsigned>
physicalCost(unsigned n, const std::vector<Gate> &gates, bool preferDepth) {
  std::vector<unsigned> depths(n);
  unsigned twoQ = 0, count = 0;
  for (const Gate &g : gates) {
    if (g.kind == Kind::Phase)
      continue;
    ++count;
    if (g.kind == Kind::Rotate) {
      if (g.pauli.weight())
        ++depths[firstBit(g.pauli.x | g.pauli.z)];
    } else if (g.kind == Kind::CX || g.kind == Kind::CZ ||
               g.kind == Kind::Swap) {
      unsigned cost = g.kind == Kind::Swap ? 3 : 1;
      twoQ += cost;
      count += cost - 1;
      depths[g.a] = depths[g.b] = std::max(depths[g.a], depths[g.b]) + cost;
    } else
      ++depths[g.a];
  }
  unsigned depth = *std::max_element(depths.begin(), depths.end());
  return preferDepth ? std::tuple{depth, twoQ, count}
                     : std::tuple{twoQ, depth, count};
}

} // namespace

/// Prove equality within the supported Clifford/partially commuting word model.
/// Reimport both circuits independently, simplify exact rotation identities,
/// match rotations through commuting neighbors, and compare signed frames and
/// scalar phases. A false result can mean unsupported algebra or overflow, not
/// necessarily inequivalence; callers must reject either case. This is not a
/// general-purpose quantum equivalence solver.
bool cudaq::pauli::graph::equivalent(unsigned n, const std::vector<Gate> &input,
                                     const std::vector<Gate> &output) {
  if (!validCircuit(n, input) || !validCircuit(n, output))
    return false;
  auto a = import(n, input), b = import(n, output);
  // Checker normalization is independent of the synthesis-simplify option:
  // disabling a search heuristic must not disable exact verification rules.
  simplify(a);
  simplify(b);
  if (a.frame.rows != b.frame.rows || a.nodes.size() != b.nodes.size())
    return false;
  // Equality in the partially commutative word: match each input rotation to
  // the earliest equal output rotation it can reach through commuting nodes.
  for (Node v : a.nodes) {
    bool matched = false;
    for (auto it = b.nodes.begin(); it != b.nodes.end(); ++it) {
      if (v.p == it->p && equalAngles(v.angle, it->angle)) {
        b.nodes.erase(it);
        matched = true;
        break;
      }
      if (v.p.anti(it->p))
        break;
    }
    if (!matched)
      return false;
  }
  // The remaining signed tableaux agree, but tableaux omit global phase.
  // Compare phase-sensitive Clifford states, then include all explicit scalar
  // phases. Opaque phases must match; rational phase differences may be 2*pi*k.
  StabilizerState sa(n), sb(n);
  for (const Gate &g : a.cliffords)
    sa.apply(g);
  for (const Gate &g : b.cliffords)
    sb.apply(g);
  auto ratio = sb.ratio(sa.anchor);
  if (!ratio || sa.exponent != sb.exponent)
    return false;
  Angle difference{sa.phase - sb.phase - 2 * int(*ratio), 4};
  std::vector<Angle> symbolic;
  for (Angle angle : a.phases) {
    if (angle.symbol)
      symbolic.push_back(angle);
    else {
      auto sum = add(difference, angle);
      if (!sum)
        return false;
      difference = *sum;
    }
  }
  for (Angle angle : b.phases) {
    if (angle.symbol) {
      auto it = std::find(symbolic.begin(), symbolic.end(), angle);
      if (it == symbolic.end())
        return false;
      symbolic.erase(it);
    } else {
      auto sum = add(difference, angle.negated());
      if (!sum)
        return false;
      difference = *sum;
    }
  }
  return symbolic.empty() && (__int128(difference.numerator) %
                                  (__int128(2) * difference.denominator) ==
                              0);
}

/// Validate controls, synthesize a bounded candidate, and require exact replay.
/// Hold mode checks the full input unitary. Release mode first justifies
/// deleting rotations invisible to terminal joint Z readout, then checks the
/// candidate against that reduced target. The caller must establish release
/// eligibility. A nonempty error invalidates any partial output; accepted=false
/// instead denotes successful replay followed by a cost-guard fallback to the
/// input.
Result cudaq::pauli::graph::synthesize(unsigned n,
                                       const std::vector<Gate> &input,
                                       const Options &options,
                                       SearchBudget *budget,
                                       uint64_t *verificationNs) {
  Result result;
  if (!n) {
    result.error = "Pauli graph synthesis islands require a nonzero wire count";
    return result;
  }
  for (double weight : {options.parallelizationCredit, options.frontierWeight,
                        options.frameWeight})
    if (!std::isfinite(weight) || weight < 0) {
      result.error =
          "Pauli graph synthesis weights must be finite and nonnegative";
      return result;
    }
  if (!validCircuit(n, input)) {
    result.error =
        "Invalid Pauli circuit (angle, support, or Clifford indices)";
    return result;
  }
  if (!options.maxGates || !options.maxDependencies) {
    result.error = "Pauli graph synthesis size limits must be positive";
    return result;
  }
  if (input.empty())
    return result;
  // Charge input/tableau work before importing. Saturating estimates turn an
  // arithmetic overflow into an unaffordable budget charge, never cheap work.
  if (input.size() > options.maxGates ||
      (budget && !budget->take(llvm::SaturatingMultiply<uint64_t>(
                     input.size(), uint64_t(n) * n + 1)))) {
    result.error = "Pauli graph synthesis input or shared work limit exceeded";
    result.stats.budgetExhausted = true;
    return result;
  }
  result = Synthesizer(n, input, options, budget).run();
  if (!result)
    return result;
  std::vector<Gate> target = input;
  if (options.release) {
    auto graph = import(n, input);
    if (options.simplify)
      simplify(graph);
    // Check deletion witnesses against later retained nodes: each deleted node
    // commutes with every later retained rotation and every final measured
    // observable. Deleting such a node cannot change any joint Z-outcome
    // probability.
    std::vector<bool> keep(graph.nodes.size(), true);
    for (size_t i = graph.nodes.size(); i-- > 0;) {
      bool mayDelete = true;
      for (unsigned q = 0; q < n; ++q)
        mayDelete &= !graph.nodes[i].p.anti(graph.frame.rows[n + q]);
      for (size_t j = i + 1; j < graph.nodes.size(); ++j)
        if (keep[j])
          mayDelete &= !graph.nodes[i].p.anti(graph.nodes[j].p);
      keep[i] = !mayDelete;
    }
    target.clear();
    for (size_t i = 0; i < graph.nodes.size(); ++i)
      if (keep[i])
        target.push_back(
            Gate::rotation(graph.nodes[i].p, graph.nodes[i].angle));
    target.insert(target.end(), graph.cliffords.begin(), graph.cliffords.end());
    for (Angle phase : graph.phases)
      target.push_back(Gate::phase(phase));
  }
  // Bound the independent replay, including its quadratic word matching.
  uint64_t replaySize =
      llvm::SaturatingAdd<uint64_t>(target.size(), result.gates.size());
  uint64_t replayWork = llvm::SaturatingAdd<uint64_t>(
      llvm::SaturatingMultiply<uint64_t>(
          3, llvm::SaturatingMultiply(replaySize, replaySize)),
      llvm::SaturatingMultiply<uint64_t>(replaySize, uint64_t(n) * n + 1));
  if (budget && !budget->take(replayWork)) {
    result.error = "Pauli graph synthesis replay work limit exceeded";
    result.stats.budgetExhausted = true;
    return result;
  }
  bool verified;
  {
    TelemetryTimer timer(verificationNs);
    verified = equivalent(n, target, result.gates);
  }
  if (!verified) {
    result.error =
        "Pauli graph synthesis exact replay rejected the synthesized candidate";
    return result;
  }
  // The standalone guard compares actual single-wire rotation circuits. A
  // multi-wire Pauli input needs its caller's lowering-aware cost comparison;
  // the combined portfolio also supplies its own objective and growth guards.
  bool gateInput = std::all_of(input.begin(), input.end(), [](Gate g) {
    return g.kind != Kind::Rotate || g.pauli.weight() == 1;
  });
  if (options.acceptOnlyImprovement && gateInput &&
      physicalCost(n, result.gates, options.preferDepth) >=
          physicalCost(n, input, options.preferDepth)) {
    result.gates = input;
    result.stats.accepted = false;
  }
  return result;
}
