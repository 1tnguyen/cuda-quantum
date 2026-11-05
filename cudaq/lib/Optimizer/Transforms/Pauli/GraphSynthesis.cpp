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
#include "SignedCliffordFrames.h"
#include "SynthesisAlgebra.h"
#include "SynthesisBootstrap.h"
#include "SynthesisCandidates.h"
#include "SynthesisCliffordWindows.h"
#include "SynthesisCompletion.h"
#include "SynthesisCost.h"
#include "SynthesisDependencies.h"
#include "SynthesisGeneral.h"
#include "SynthesisJointPhasePlanner.h"
#include "SynthesisLinearFour.h"
#include "SynthesisNativeRefinement.h"
#include "SynthesisOpen.h"
#include "SynthesisReadout.h"
#include "SynthesisRegions.h"
#include "SynthesisRepresentationIndex.h"
#include "SynthesisRepresentations.h"
#include "TCountInternal.h"
#include "cudaq/Optimizer/Pauli/GateCount.h"
#include "cudaq/Optimizer/Pauli/PBC.h"
#include "cudaq/Optimizer/Pauli/TCount.h"
#include "cudaq/Optimizer/Pauli/Telemetry.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
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
  bool native = false;
  /// Stable ordering for deduplication and selection among equal-score actions.
  bool operator<(const Candidate &c) const {
    return std::tie(i, j, a, b, native) <
           std::tie(c.i, c.j, c.a, c.b, c.native);
  }
};
/// Materialize one candidate as local bases followed by CZ. Symmetric mode
/// appends inverse bases; asymmetric mode leaves their action in the residual
/// graph/frame so later operations can share the basis changes.
/// See PCOAST Sec. V-C1, the native-gate search discussion after Eq. (162).
std::vector<Gate> sequence(Candidate c, bool asymmetric) {
  if (c.native) {
    // Add exact native-basis words for the same nine symmetric entanglers.
    // Existing table synthesis/replay determines the word; no hand-written
    // Clifford identity or original circuit is used as a synthesis guide.
    static const auto nativeWords = [] {
      std::array<std::vector<Gate>, 9> words;
      for (unsigned a = 1; a <= 3; ++a)
        for (unsigned b = 1; b <= 3; ++b) {
          auto raw = sequence({0, 1, a, b, false}, false);
          gatecount::Options settings;
          settings.rounds = 1;
          settings.propagate = settings.phasePlacement = false;
          SearchBudget work(10000000);
          words[(a - 1) * 3 + b - 1] =
              gatecount::optimize(2, raw, settings, work, 1024);
        }
      return words;
    }();
    auto gates = nativeWords[(c.a - 1) * 3 + c.b - 1];
    for (Gate &gate : gates) {
      if (gate.kind == Kind::Phase)
        continue;
      gate.a = gate.a ? c.j : c.i;
      if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
          gate.kind == Kind::Swap)
        gate.b = gate.b ? c.j : c.i;
    }
    return gates;
  }
  auto gates = basis(c.a, c.i), second = basis(c.b, c.j);
  gates.insert(gates.end(), second.begin(), second.end());
  auto pre = gates;
  gates.push_back({Kind::CZ, c.i, c.j});
  if (!asymmetric)
    for (auto it = pre.rbegin(); it != pre.rend(); ++it)
      gates.push_back(inverse(*it));
  return gates;
}
using cudaq::pauli::graph::detail::circuitScore;
using cudaq::pauli::graph::detail::nativeWord;

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
enum class QuantumExit { Restore, Open };

class Synthesizer {
  /// Number of wires in the input and emitted circuit.
  unsigned n;
  /// Borrowed scoring, semantic, and resource controls for this invocation.
  const Options &options;
  /// Mutable residual rotations and Clifford frame.
  Graph graph;
  /// Accumulated output, counters, and sticky failure diagnostic.
  Result result;
  bool measuredMode = false;
  QuantumExit quantumExit = QuantumExit::Restore;
  bool searchUnavailable = false;
  unsigned deviationStep = 0;
  std::vector<Pauli> measurementWords;
  std::function<double(const std::vector<Pauli> &)> measurementCost;
  /// Emitted Clifford subsequence used to recover the exact scalar phase.
  std::vector<Gate> emittedCliffords;
  /// Current ASAP finish depth on each wire.
  std::vector<unsigned> depths;
  /// Maximum emitted depth and number of entangling synthesis actions taken.
  unsigned depth = 0, steps = 0;
  /// Optional borrowed work/deadline budget shared with search and replay.
  SearchBudget *budget;

  std::vector<Gate> action(Candidate c) const {
    return nativeWord(sequence(c, options.asymmetric), options);
  }

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
    for (Gate native : nativeWord({g}, options)) {
      // Dictionary scalar phases are irrelevant to conjugation. Hold-mode
      // finalization recovers the required phase from the actual emitted
      // Clifford skeleton; emitting these as well would count them twice.
      if (native.kind == Kind::Phase)
        continue;
      if (!charge(graph.nodes.size() + 2 * n) || !emit(native))
        return false;
      for (unsigned i = 0; i < graph.nodes.size(); ++i)
        if (active[i])
          graph.nodes[i].p = conjugate(graph.nodes[i].p, native);
      graph.frame.carry(native);
      for (Pauli &observable : measurementWords)
        observable = conjugate(observable, native);
    }
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
                  if (transformed(v, c).cost() < minimum) {
                    candidates.insert(c);
                    c.native = true;
                    candidates.insert(c);
                  }
                }
    }
    result.stats.candidates += candidates.size();
    double best = std::numeric_limits<double>::infinity();
    uint64_t bestTie = std::numeric_limits<uint64_t>::max();
    const uint64_t tieSeed = options.tieSeed;
    std::optional<Candidate> chosen;
    std::vector<std::pair<double, Candidate>> alternatives;
    auto frameFactors = factors(graph.frame);
    const double currentReadoutCost =
        measuredMode && measurementCost && options.readoutWeight
            ? measurementCost(measurementWords)
            : 0;
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
      if (penalizeFrame && !measuredMode) {
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
      for (const Gate &g : action(c)) {
        if (g.kind == Kind::Phase)
          continue;
        if (g.kind == Kind::CZ || g.kind == Kind::CX || g.kind == Kind::Swap)
          ds[g.a] = ds[g.b] = std::max(ds[g.a], ds[g.b]) + 1;
        else
          ++ds[g.a];
        next = std::max(next, ds[g.a]);
      }
      const auto actionGates = action(c);
      unsigned serial =
          std::count_if(actionGates.begin(), actionGates.end(),
                        [](const Gate &g) { return g.kind != Kind::Phase; });
      score -=
          options.parallelizationCredit * (1.0 - double(next - depth) / serial);
      // Charge the primitive implementation, including local basis changes.
      // Support reduction alone does not price the gates that will be emitted.
      score += options.actionWeight * nativeGateCost(actionGates, options);
      if (measuredMode && measurementCost && options.readoutWeight) {
        auto next = measurementWords;
        for (const Gate &gate : action(c))
          for (Pauli &observable : next)
            observable = conjugate(observable, gate);
        if (!charge(llvm::SaturatingAdd<uint64_t>(
                llvm::SaturatingMultiply<uint64_t>(uint64_t(n) * n,
                                                   measurementWords.size()),
                1)))
          return std::nullopt;
        score += options.readoutWeight *
                 (measurementCost(next) - currentReadoutCost);
      }
      // Strict comparison keeps the first candidate in deterministic set order
      // on a tie. A lower heuristic score is not a proof of global optimality.
      alternatives.emplace_back(score, c);
      uint64_t tie = 0;
      if (tieSeed) {
        tie = tieSeed ^ (uint64_t(steps) << 48) ^ (uint64_t(c.i) << 32) ^
              (uint64_t(c.j) << 16) ^ (c.a << 8) ^ (c.b << 4) ^ c.native;
        tie += 0x9e3779b97f4a7c15ULL;
        tie = (tie ^ (tie >> 30)) * 0xbf58476d1ce4e5b9ULL;
        tie = (tie ^ (tie >> 27)) * 0x94d049bb133111ebULL;
        tie ^= tie >> 31;
      }
      if ((!tieSeed && score < best) ||
          (tieSeed && (score < best - 1.0e-12 ||
                       (std::abs(score - best) <= 1.0e-12 && tie < bestTie)))) {
        best = score;
        bestTie = tie;
        chosen = c;
      }
    }
    // Bounded discrepancy search: explore one alternative to a greedy
    // choice, then complete the circuit with the same native scorer. Every
    // alternative retains the minimum-frontier progress condition.
    if (deviationStep && chosen && deviationStep == steps) {
      std::sort(alternatives.begin(), alternatives.end(),
                [](const auto &a, const auto &b) {
                  if (a.first != b.first)
                    return a.first < b.first;
                  return a.second < b.second;
                });
      for (const auto &entry : alternatives) {
        if (!(entry.second < *chosen) && !(*chosen < entry.second))
          continue;
        chosen = entry.second;
        break;
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
      if (result.error.empty()) {
        searchUnavailable = true;
        result.error =
            "Pauli graph synthesis found no monotonically reducing candidate";
      }
      return false;
    }
    // The emitted prefix is B then CZ, so the residual is conjugated by
    // CZ B (not its inverse); this carries the omitted trailing B^dagger.
    for (const Gate &g : action(*candidate))
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
              const Options &options, SearchBudget *budget,
              const std::vector<Pauli> &observables = {},
              std::function<double(const std::vector<Pauli> &)> cost = {},
              unsigned deviation = 0, QuantumExit exit = QuantumExit::Restore)
      : n(n), options(options), graph(import(n, input)),
        measuredMode(!observables.empty()), quantumExit(exit),
        deviationStep(deviation), measurementCost(std::move(cost)), depths(n),
        budget(budget) {
    for (const Pauli &observable : observables)
      measurementWords.push_back(graph.frame.lookup(observable));
  }
  const std::vector<Pauli> &measurements() const { return measurementWords; }
  const std::vector<Gate> &exitCliffords() const { return emittedCliffords; }
  bool unavailable() const { return searchUnavailable; }
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
    if ((!measuredMode && options.release) ||
        (measuredMode && options.pruneUnobserved)) {
      // Conservative terminal-measurement optimization: remove rotations in
      // the backward cone's complement. Keep the frame and measurement order;
      // no classical remapping is silently introduced.
      std::vector<Node> kept;
      for (auto it = graph.nodes.rbegin(); it != graph.nodes.rend(); ++it) {
        bool affects = false;
        if (measuredMode) {
          for (const Pauli &observable : measurementWords)
            affects |= it->p.anti(observable);
        } else {
          for (unsigned q = 0; q < n; ++q)
            affects |= it->p.anti(graph.frame.rows[n + q]);
        }
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
    // Preserve the exact original-order constraint for every anticommuting
    // pair, without storing an edge for every occurrence pair. A node is ready
    // precisely when no conflicting axis has an earlier active occurrence.
    // Common Clifford changes in carry() preserve this immutable relation.
    // The logical edge cap retains its legacy meaning; it is not a cap on the
    // smaller stored axis-conflict relation. No constraint is dropped to fit.
    auto dependencySucceeded = [&](const auto &operation) {
      if (operation)
        return true;
      switch (operation.error) {
      case detail::DependencyError::WorkLimit:
        result.error = "Pauli graph synthesis shared search budget exhausted";
        result.stats.budgetExhausted = true;
        break;
      case detail::DependencyError::DependencyLimit:
        result.error = "Pauli graph synthesis dependency limit exceeded";
        result.stats.budgetExhausted = true;
        break;
      case detail::DependencyError::SizeOverflow:
        result.error = "Pauli graph synthesis dependency index size exceeded";
        result.stats.budgetExhausted = true;
        break;
      case detail::DependencyError::InvalidOccurrence:
      case detail::DependencyError::None:
        result.error = "Pauli graph synthesis invalid dependency cursor state";
        break;
      }
      return false;
    };
    // Charge the temporary axis copy separately from classification/indexing.
    if (!charge(size))
      return result;
    auto dependencies = [&] {
      std::vector<Pauli> axes;
      axes.reserve(size);
      for (const Node &node : graph.nodes)
        axes.push_back(node.p);
      return detail::DependencyIndex::build(axes, options.maxDependencies,
                                            budget);
    }();
    if (!dependencySucceeded(dependencies))
      return result;
    const auto &dependencyStats = dependencies.value->statistics();
    // The checked logical cap and the input gate cap both fit unsigned.
    result.stats.dependencies =
        static_cast<unsigned>(dependencyStats.logicalEdges);
    result.stats.dependencyAxisClasses =
        static_cast<unsigned>(dependencyStats.classes);
    result.stats.dependencyConflictEntries = dependencyStats.storedConflicts;
    auto cursor = detail::DependencyCursor::create(*dependencies.value, budget);
    if (!dependencySucceeded(cursor) || !charge(size))
      return result;
    // carry() needs its own active mask; update it only after cursor removal
    // succeeds, so the two views of the residual word cannot disagree.
    std::vector<bool> active(size, true);
    unsigned left = size;
    while (left) {
      if (!charge(size))
        return result;
      std::vector<Node> front, population;
      bool removed = false;
      // A ready zero-cost singlet already fits one wire. Keep the ascending
      // occurrence scan: newly unblocked later IDs can execute in this pass,
      // as in the original DAG. Do not reorder simultaneously ready nodes.
      // Preserve a signed axis by negating the angle on emission.
      // PCOAST Fig. 12, lines 4-9.
      for (unsigned i = 0; i < size; ++i)
        if (active[i]) {
          auto ready = cursor.value->ready(i, budget);
          if (!dependencySucceeded(ready))
            return result;
          if (!*ready.value || graph.nodes[i].cost())
            continue;
          Node v = graph.nodes[i];
          if (!v.p.weight()) {
            result.error = "Identity rotation requires a rational angle";
            return result;
          }
          // Emit only the basis prefix. Carry its action into every remaining
          // rotation and terminal observable, rather than emitting its inverse
          // around this rotation during MLIR lowering. The existing independent
          // measured replay checks this transformed representation.
          {
            unsigned q = firstBit(v.p.x | v.p.z);
            unsigned axis = v.p.axis(q);
            if (axis == 3 && !carry({Kind::Sdg, q}, active))
              return result;
            if ((axis & 1) && !carry({Kind::H, q}, active))
              return result;
            v = graph.nodes[i];
          }
          if (!emit(Gate::rotation({v.p.x, v.p.z, false},
                                   v.p.sign ? v.angle.negated() : v.angle)))
            return result;
          if (!dependencySucceeded(cursor.value->consume(i, budget)))
            return result;
          active[i] = false;
          --left;
          removed = true;
        }
      if (removed)
        continue;
      // No ready rotation is directly executable. Propose an entangling action
      // from the frontier and score its effects on the remaining population.
      for (unsigned i = 0; i < size; ++i)
        if (active[i]) {
          population.push_back(graph.nodes[i]);
          auto ready = cursor.value->ready(i, budget);
          if (!dependencySucceeded(ready))
            return result;
          if (*ready.value)
            front.push_back(graph.nodes[i]);
        }
      if (!step(front, population,
                !options.release && quantumExit == QuantumExit::Restore,
                active))
        return result;
    }
    if (!measuredMode && quantumExit == QuantumExit::Restore &&
        !finalize(active))
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

std::string invalidOptions(const Options &options) {
  if (!options.nativeCX && !options.nativeCZ)
    return "Pauli synthesis requires CX or CZ in its native gate set";
  for (double value :
       {options.parallelizationCredit, options.frontierWeight,
        options.frameWeight, options.actionWeight, options.readoutWeight,
        options.singleQubitCost, options.tCost, options.cxCost, options.czCost,
        options.swapCost})
    if (!std::isfinite(value) || value < 0)
      return "Pauli synthesis costs and weights must be finite and nonnegative";
  if (!options.maxGates || !options.maxDependencies)
    return "Pauli synthesis size limits must be positive";
  if (static_cast<unsigned>(options.objective) >
      static_cast<unsigned>(SynthesisObjective::TDepth))
    return "Invalid Pauli synthesis objective";
  if (options.jointPhaseCandidates > 4 ||
      static_cast<unsigned>(options.jointPhaseAcceptance) >
          static_cast<unsigned>(JointPhaseAcceptance::GatesAndT))
    return "Invalid joint-phase candidates or acceptance policy";
  if (options.jointPhaseCandidates &&
      (!options.jointPhasePlannerWork || !options.jointPhaseMaxWork ||
       !options.jointPhaseTimeoutMs))
    return "Joint-phase work and time limits must be positive when enabled";
  return {};
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

double cudaq::pauli::graph::nativeGateCost(const std::vector<Gate> &gates,
                                           const Options &options) {
  return detail::evaluateNativeCost(gates, options).value;
}

detail::OpenRegionResult cudaq::pauli::graph::detail::synthesizeOpenRegion(
    unsigned width, const std::vector<Gate> &rotations, const Options &options,
    SearchBudget &budget, unsigned deviation) {
  OpenRegionResult result;
  auto fail = [&](OpenRegionStatus status, std::string error) {
    result.status = status;
    result.error = std::move(error);
    return result;
  };
  if (auto error = invalidOptions(options); !error.empty())
    return fail(OpenRegionStatus::Unsupported, std::move(error));
  if (!validCircuit(width, rotations) ||
      std::any_of(rotations.begin(), rotations.end(), [](const Gate &gate) {
        return (gate.kind != Kind::Rotate && gate.kind != Kind::Phase) ||
               (gate.kind == Kind::Rotate && !gate.pauli.weight());
      }))
    return fail(OpenRegionStatus::Unsupported,
                "open quantum regions require normalized rotations/phases");
  if (rotations.size() > options.maxGates ||
      !budget.take(llvm::SaturatingMultiply<uint64_t>(
          rotations.size() + 1, uint64_t(width) * width + 1)))
    return fail(OpenRegionStatus::Exhausted,
                "open quantum region input/work limit exceeded");
  Options quantum = options;
  quantum.release = false;
  quantum.pruneUnobserved = false;
  quantum.acceptOnlyImprovement = false;
  Synthesizer synthesizer(width, rotations, quantum, &budget, {}, {}, deviation,
                          QuantumExit::Open);
  auto candidate = synthesizer.run();
  result.stats = candidate.stats;
  if (!candidate)
    return fail(candidate.stats.budgetExhausted ? OpenRegionStatus::Exhausted
                : synthesizer.unavailable() ? OpenRegionStatus::Unsupported
                                            : OpenRegionStatus::InvalidProof,
                std::move(candidate.error));
  const auto &exit = synthesizer.exitCliffords();
  // The imported normalized source has no initial Clifford frame. Retaining
  // the ACTUAL emitted Clifford word makes body = exit * source exact, with
  // no phase-blind tableau replacement and no terminal-measurement assumption.
  uint64_t size = llvm::SaturatingAdd<uint64_t>(
      llvm::SaturatingAdd<uint64_t>(rotations.size(), exit.size()),
      candidate.gates.size());
  uint64_t work = llvm::SaturatingAdd<uint64_t>(
      llvm::SaturatingMultiply<uint64_t>(
          3, llvm::SaturatingMultiply<uint64_t>(size, size)),
      llvm::SaturatingMultiply<uint64_t>(size, uint64_t(width) * width + 1));
  if (!budget.take(work))
    return fail(OpenRegionStatus::Exhausted,
                "open quantum region replay work limit exceeded");
  auto target = rotations;
  target.insert(target.end(), exit.begin(), exit.end());
  if (!equivalent(width, target, candidate.gates))
    return fail(OpenRegionStatus::InvalidProof,
                "open quantum region exact replay failed");
  result.body = std::move(candidate.gates);
  result.exitCliffords = exit;
  return result;
}

/// Validate controls, synthesize a bounded candidate, and require exact replay.
/// Hold mode checks the full input unitary. Release mode first justifies
/// deleting rotations invisible to terminal joint Z readout, then checks the
/// candidate against that reduced target. The caller must establish release
/// eligibility. A nonempty error invalidates any partial output; accepted=false
/// instead denotes successful replay followed by a cost-guard fallback to the
/// input.
static Result synthesizeOne(unsigned n, const std::vector<Gate> &input,
                            const Options &options, SearchBudget *budget,
                            uint64_t *verificationNs, unsigned deviation) {
  Result result;
  if (!n) {
    result.error = "Pauli graph synthesis islands require a nonzero wire count";
    return result;
  }
  if (auto error = invalidOptions(options); !error.empty()) {
    result.error = error;
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
  result = Synthesizer(n, input, options, budget, {}, {}, deviation).run();
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
      (tcount::cost(n, result.gates).t > tcount::cost(n, input).t ||
       circuitScore(n, result.gates, options) >=
           circuitScore(n, input, options))) {
    result.gates = nativeWord(input, options);
    if (result.gates.size() > options.maxGates)
      result.error = "native input expansion exceeds the gate limit";
    result.stats.accepted = false;
  }
  return result;
}

Result cudaq::pauli::graph::synthesize(unsigned width,
                                       const std::vector<Gate> &input,
                                       const Options &options,
                                       SearchBudget *budget,
                                       uint64_t *verificationNs) {
  auto best = synthesizeOne(width, input, options, budget, verificationNs, 0);
  if (!best || input.empty())
    return best;
  unsigned alternatives = std::min(options.discrepancySteps, best.stats.tqes);
  for (unsigned step = 1; step <= alternatives; ++step) {
    auto candidate =
        synthesizeOne(width, input, options, budget, verificationNs, step);
    if (!candidate) {
      if (candidate.stats.budgetExhausted || (budget && budget->stopped())) {
        best.stats.budgetExhausted = true;
        break;
      }
      return candidate;
    }
    if (circuitScore(width, candidate.gates, options) <
        circuitScore(width, best.gates, options))
      best = std::move(candidate);
    if (step == alternatives)
      break;
  }
  return best;
}

static Result buildReadout(unsigned width,
                           const std::vector<Pauli> &observables,
                           const std::vector<unsigned> &readoutWires,
                           const Options &options, bool validate = true) {
  Result result;
  auto fail = [&](std::string error) {
    result.gates.clear();
    result.error = std::move(error);
    return result;
  };
  if (validate) {
    if (auto error = invalidOptions(options); !error.empty())
      return fail(error);
    if (!width || observables.size() != readoutWires.size())
      return fail("invalid measured readout dimensions");
    pbc::CommutingObservables basis;
    for (const Pauli &word : observables)
      if (!word.weight() || !(word.x | word.z).fits(width) ||
          !basis.insert(word))
        return fail(
            "readout requires commuting nonidentity in-range observables");
  }
  auto words = observables;
  std::vector<bool> fixed(width);
  auto append = [&](Gate gate) {
    for (Gate native : nativeWord({gate}, options)) {
      result.gates.push_back(native);
      for (Pauli &word : words)
        word = conjugate(word, native);
    }
  };
  for (unsigned index = 0; index < words.size(); ++index) {
    unsigned target = readoutWires[index];
    if (target >= width)
      return fail("readout wire is out of range");
    Pauli wanted = single(2, target);
    if (fixed[target]) {
      if (words[index] != wanted)
        return fail("conflicting repeated readout");
      continue;
    }
    std::vector<unsigned> support;
    auto wordSupport = words[index].x | words[index].z;
    for (unsigned wire : wordSupport.set_bits())
      support.push_back(wire);
    auto pivot = std::find_if(support.begin(), support.end(),
                              [&](unsigned q) { return !fixed[q]; });
    if (pivot == support.end())
      return fail("readout requires classical outcome remapping");
    unsigned q =
        std::find(support.begin(), support.end(), target) != support.end()
            ? target
            : *pivot;
    for (unsigned wire : support) {
      unsigned axis = words[index].axis(wire);
      if (fixed[wire] && axis != 2)
        return fail("readout would modify an already fixed observable");
      if (axis == 3)
        append({Kind::Sdg, wire});
      if (axis & 1)
        append({Kind::H, wire});
    }
    for (unsigned wire : support)
      if (wire != q)
        append({Kind::CX, wire, q});
    if (q != target)
      append({Kind::Swap, q, target});
    if (words[index].sign)
      append({Kind::X, target});
    if (words[index] != wanted)
      return fail("terminal observable reconstruction failed");
    fixed[target] = true;
    if (result.gates.size() > options.maxGates) {
      result.stats.budgetExhausted = true;
      return fail("terminal readout gate limit exceeded");
    }
  }
  return result;
}

Result cudaq::pauli::graph::synthesizeReadout(
    unsigned width, const std::vector<Pauli> &observables,
    const std::vector<unsigned> &readoutWires, const Options &options) {
  return buildReadout(width, observables, readoutWires, options);
}

namespace {
/// Independent measured replay. Match the partially commuting rotation words,
/// then compare each requested signed observable through the residual frames.
bool equivalentMeasured(unsigned width, const std::vector<Gate> &input,
                        const std::vector<Pauli> &observables,
                        const std::vector<Gate> &output,
                        const std::vector<Pauli> &outcomes, bool prune,
                        SearchBudget *budget) {
  uint64_t size = llvm::SaturatingAdd<uint64_t>(input.size(), output.size());
  uint64_t work = llvm::SaturatingMultiply<uint64_t>(
      size + 1,
      llvm::SaturatingAdd<uint64_t>(size, uint64_t(width) * width + 1));
  if (budget && !budget->take(work))
    return false;
  auto a = import(width, input), b = import(width, output);
  simplify(a);
  simplify(b);
  if (prune) {
    std::vector<bool> keep(a.nodes.size(), true);
    std::vector<Pauli> requested;
    for (Pauli word : observables)
      requested.push_back(a.frame.lookup(word));
    for (size_t i = a.nodes.size(); i-- > 0;) {
      bool invisible = true;
      for (const Pauli &word : requested)
        invisible &= !a.nodes[i].p.anti(word);
      for (size_t j = i + 1; j < a.nodes.size(); ++j)
        if (keep[j])
          invisible &= !a.nodes[i].p.anti(a.nodes[j].p);
      keep[i] = !invisible;
    }
    std::vector<Node> retained;
    for (size_t i = 0; i < a.nodes.size(); ++i)
      if (keep[i])
        retained.push_back(a.nodes[i]);
    a.nodes = std::move(retained);
    simplify(a);
  }
  if (a.nodes.size() != b.nodes.size() || observables.size() != outcomes.size())
    return false;
  for (const Node &node : a.nodes) {
    bool matched = false;
    for (auto it = b.nodes.begin(); it != b.nodes.end(); ++it) {
      if (node.p == it->p && equalAngles(node.angle, it->angle)) {
        b.nodes.erase(it);
        matched = true;
        break;
      }
      if (node.p.anti(it->p))
        break;
    }
    if (!matched)
      return false;
  }
  for (unsigned i = 0; i < observables.size(); ++i)
    if (a.frame.lookup(observables[i]) != b.frame.lookup(outcomes[i]))
      return false;
  return true;
}
} // namespace

detail::MeasuredCandidateSink::MeasuredCandidateSink(MeasuredProblem problem,
                                                     const Options &options,
                                                     SearchBudget *budget)
    : problem(problem), options(options), budget(budget) {
  inputError = invalidOptions(options);
  if (!inputError.empty())
    return;
  if (!validCircuit(problem.width, problem.rotations) ||
      problem.observables.empty() ||
      problem.rotations.size() > options.maxGates) {
    inputError = "invalid measured synthesis input or gate limit";
    return;
  }
  if (!problem.width ||
      (budget && !budget->take(llvm::SaturatingMultiply<uint64_t>(
                     problem.rotations.size() + problem.observables.size() + 1,
                     uint64_t(problem.width) * problem.width + 1)))) {
    inputError = "measured synthesis input work limit exceeded";
    return;
  }
  auto initialReadout = synthesizeReadout(problem.width, problem.observables,
                                          problem.readoutWires, options);
  if (!initialReadout) {
    inputError = initialReadout.error;
    return;
  }
  inputTCount = tcount::cost(problem.width, problem.rotations).t;
  for (unsigned wire : problem.readoutWires)
    outcomes.push_back(single(2, wire));
}

detail::SubmissionResult
detail::MeasuredCandidateSink::submit(std::vector<Gate> gates) {
  if (!inputError.empty())
    return {budget && budget->stopped() ? SubmissionStatus::ProofBudgetExhausted
                                        : SubmissionStatus::InvalidProof,
            inputError};
  if (gates.size() > options.maxGates ||
      std::any_of(gates.begin(), gates.end(), [&](const Gate &gate) {
        return (gate.kind == Kind::CX && !options.nativeCX) ||
               (gate.kind == Kind::CZ && !options.nativeCZ) ||
               (gate.kind == Kind::Swap && !options.nativeSwap);
      }))
    return {SubmissionStatus::InvalidProof,
            "measured synthesis candidate violates native basis or gate limit"};
  const unsigned width = problem.width;
  if (!validCircuit(width, gates))
    return {SubmissionStatus::InvalidProof,
            "measured synthesis invalid candidate operands"};
  if (!equivalentMeasured(width, problem.rotations, problem.observables, gates,
                          outcomes, options.pruneUnobserved, budget)) {
    if (budget && budget->stopped())
      return {SubmissionStatus::ProofBudgetExhausted, {}};
    return {SubmissionStatus::InvalidProof,
            "measured synthesis independent replay failed"};
  }
  if (tcount::cost(width, gates).t > inputTCount)
    return {SubmissionStatus::RejectedByCost, {}};
  VerifiedCandidate candidate(std::move(gates),
                              VerifiedCandidate::Proof::MeasuredWordReplay);
  bool rawWinner =
      !rawBest || circuitScore(width, candidate.gates(), options) <
                      circuitScore(width, rawBest->gates(), options);
  std::optional<VerifiedCandidate> rawCandidate;
  if (rawWinner)
    rawCandidate = candidate;
  std::string frameError;
  if (auto refined = detail::optimizeSignedFrames(
          width, candidate.gates(), problem.readoutWires, budget, frameError))
    candidate = VerifiedCandidate(
        std::move(*refined),
        VerifiedCandidate::Proof::MeasuredReplayAndLocalChannels);
  else if (!frameError.empty())
    return {SubmissionStatus::InvalidProof, std::move(frameError)};
  if (rawWinner) {
    rawBest = std::move(rawCandidate);
    refinedRawBest = candidate;
  }
  if (tcount::cost(width, candidate.gates()).t <= inputTCount &&
      (!best || circuitScore(width, candidate.gates(), options) <
                    circuitScore(width, best->gates(), options)))
    best = std::move(candidate);
  return {SubmissionStatus::Verified, {}};
}

std::optional<detail::VerifiedCandidate>
detail::MeasuredCandidateSink::takeBest() {
  if (!best)
    return std::nullopt;
  // Preserve the legacy five-metric guard against the best unrefined rollout.
  auto originalCost = tcount::cost(problem.width, rawBest->gates());
  auto improvedCost = tcount::cost(problem.width, best->gates());
  if (improvedCost.gates > originalCost.gates ||
      improvedCost.t > originalCost.t ||
      improvedCost.twoQubit > originalCost.twoQubit ||
      improvedCost.depth > originalCost.depth ||
      improvedCost.tDepth > originalCost.tDepth)
    best = std::move(refinedRawBest);
  auto result = std::move(best);
  best.reset();
  rawBest.reset();
  refinedRawBest.reset();
  return result;
}

namespace {
/// Native prefix and its residual signed observables from a completed rollout.
/// Retaining this boundary avoids recovering it from a refined native word.
struct ReadoutProblem {
  std::vector<Gate> prefix;
  std::vector<Pauli> observables;
};

enum class SearchTermination { Complete, Exhausted, Unsupported, InvalidProof };

pbc::Result
runMeasuredSearch(unsigned width, const std::vector<Gate> &input,
                  const std::vector<Pauli> &observables,
                  const std::vector<unsigned> &readoutWires,
                  const Options &options, SearchBudget *budget,
                  std::optional<ReadoutProblem> *completion = nullptr,
                  SearchTermination *termination = nullptr) {
  if (termination)
    *termination = SearchTermination::Complete;
  auto failure = [&](std::string error, SearchTermination kind) {
    if (termination)
      *termination = kind;
    return pbc::Result{{}, {}, std::move(error)};
  };
  std::optional<detail::CostRank> bestCompletion;
  detail::MeasuredCandidateSink sink({width, input, observables, readoutWires},
                                     options, budget);
  if (!sink.error().empty())
    return failure(sink.error(), budget && budget->stopped()
                                     ? SearchTermination::Exhausted
                                     : SearchTermination::Unsupported);
  std::vector<Pauli> outcomes;
  for (unsigned wire : readoutWires)
    outcomes.push_back(single(2, wire));
  auto readoutCost = [&](const std::vector<Pauli> &words) {
    // Clifford conjugation preserves the already validated commutation,
    // support range and dimensions. Avoid redoing Gaussian validation for
    // every heuristic candidate; final emission and replay remain checked.
    auto readout = buildReadout(width, words, readoutWires, options, false);
    return readout ? nativeGateCost(readout.gates, options)
                   : std::numeric_limits<double>::infinity();
  };
  std::string exhaustionError;
  unsigned alternatives = options.discrepancySteps;
  for (unsigned deviation = 0;; ++deviation) {
    if (budget && !budget->take(uint64_t(width) * width + input.size() + 1))
      break;
    Synthesizer search(width, input, options, budget, observables, readoutCost,
                       deviation);
    auto candidate = search.run();
    if (!deviation)
      alternatives = std::min(alternatives, candidate.stats.tqes);
    if (!candidate) {
      if (budget && budget->stopped())
        break;
      if (candidate.stats.budgetExhausted) {
        exhaustionError = candidate.error;
        break;
      }
      return failure(candidate.error, SearchTermination::InvalidProof);
    }
    auto readout =
        synthesizeReadout(width, search.measurements(), readoutWires, options);
    if (!readout) {
      if (readout.stats.budgetExhausted) {
        exhaustionError = readout.error;
        break;
      }
      return failure(readout.error, SearchTermination::InvalidProof);
    }
    if (candidate.gates.size() + readout.gates.size() > options.maxGates) {
      exhaustionError = "measured synthesis output gate limit exceeded";
      break;
    }
    const size_t prefixSize = candidate.gates.size();
    candidate.gates.insert(candidate.gates.end(), readout.gates.begin(),
                           readout.gates.end());
    std::optional<ReadoutProblem> pendingCompletion;
    detail::CostRank completionScore{};
    if (completion) {
      auto score = circuitScore(width, candidate.gates, options);
      if (!bestCompletion || score < *bestCompletion) {
        completionScore = score;
        pendingCompletion = ReadoutProblem{
            std::vector<Gate>(candidate.gates.begin(),
                              candidate.gates.begin() + prefixSize),
            search.measurements()};
      }
    }
    auto submission = sink.submit(std::move(candidate.gates));
    if (submission.status == detail::SubmissionStatus::ProofBudgetExhausted)
      break;
    if (submission.status == detail::SubmissionStatus::InvalidProof)
      return failure(std::move(submission.error),
                     SearchTermination::InvalidProof);
    if (submission.status == detail::SubmissionStatus::Verified &&
        pendingCompletion) {
      *completion = std::move(pendingCompletion);
      bestCompletion = completionScore;
    }
    if (deviation == alternatives)
      break;
  }
  auto best = sink.takeBest();
  if (!best)
    return failure(
        exhaustionError.empty()
            ? "measured synthesis budget exhausted or no T-admissible candidate"
            : std::move(exhaustionError),
        SearchTermination::Exhausted);
  return {std::move(*best).takeGates(), std::move(outcomes), {}};
}

/// The study's no-regression contract applies to complete native circuits, not
/// structural support estimates or individual readout tails.
bool improvesWithoutRegression(unsigned width,
                               const std::vector<Gate> &candidate,
                               const std::vector<Gate> &incumbent,
                               const Options &options) {
  auto a = tcount::cost(width, candidate), b = tcount::cost(width, incumbent);
  return a.gates <= b.gates && a.t <= b.t && a.twoQubit <= b.twoQubit &&
         a.depth <= b.depth && a.tDepth <= b.tDepth &&
         circuitScore(width, candidate, options) <
             circuitScore(width, incumbent, options);
}

/// Reserve at least half the caller's remaining work/deadline for completing
/// and proving candidates. Child exhaustion cannot invalidate the parent.
std::optional<SearchBudget>
explorationSlice(SearchBudget &parent,
                 uint64_t cap = std::numeric_limits<uint64_t>::max()) {
  if (parent.stopped() || parent.remaining() < 2 || !cap)
    return std::nullopt;
  unsigned timeout = 0;
  if (auto time = parent.timeRemainingMs()) {
    if (*time < 2)
      return std::nullopt;
    timeout = *time / 2;
  }
  return SearchBudget(parent, std::min(cap, parent.remaining() / 2), timeout);
}

uint64_t readoutWorkEstimate(unsigned width, size_t observables) {
  return llvm::SaturatingMultiply<uint64_t>(
      uint64_t(width) + observables + 1,
      llvm::SaturatingMultiply<uint64_t>(uint64_t(width) + 1, observables + 1));
}

pbc::Result bootstrapMeasured(const detail::MeasuredProblem &problem,
                              const Options &options, SearchBudget &budget,
                              SearchTermination &status) {
  status = SearchTermination::Exhausted;
  Options proofOptions = options;
  proofOptions.pruneUnobserved = false;
  detail::MeasuredCandidateSink sink(problem, proofOptions, &budget);
  if (!sink.error().empty()) {
    if (!budget.stopped())
      status = SearchTermination::Unsupported;
    return {{}, {}, sink.error()};
  }
  auto generation = explorationSlice(budget);
  if (!generation)
    return {{}, {}, "bootstrap completion work reserve unavailable"};
  auto prefix = detail::buildBootstrapPrefix(problem, options, &*generation);
  if (!prefix) {
    if (prefix.status == detail::BootstrapStatus::Unsupported)
      status = SearchTermination::Unsupported;
    return {{}, {}, std::move(prefix.error)};
  }
  if (!budget.take(
          readoutWorkEstimate(problem.width, problem.observables.size())))
    return {{}, {}, "bootstrap readout work reserve unavailable"};
  auto readout = synthesizeReadout(problem.width, problem.observables,
                                   problem.readoutWires, options);
  if (!readout) {
    status = readout.stats.budgetExhausted ? SearchTermination::Exhausted
                                           : SearchTermination::InvalidProof;
    return {{}, {}, std::move(readout.error)};
  }
  if (prefix.gates.size() > options.maxGates ||
      readout.gates.size() > options.maxGates - prefix.gates.size())
    return {{}, {}, "bootstrap completion native gate limit exceeded"};
  prefix.gates.insert(prefix.gates.end(), readout.gates.begin(),
                      readout.gates.end());
  auto submitted = sink.submit(std::move(prefix.gates));
  if (submitted.status == detail::SubmissionStatus::InvalidProof) {
    status = SearchTermination::InvalidProof;
    return {{}, {}, std::move(submitted.error)};
  }
  auto checked = sink.takeBest();
  if (!checked)
    return {{}, {}, "bootstrap completion proof did not finish"};
  std::vector<Pauli> outcomes;
  for (unsigned wire : problem.readoutWires)
    outcomes.push_back(single(2, wire));
  status = SearchTermination::Complete;
  return {std::move(*checked).takeGates(), std::move(outcomes), {}};
}

pbc::Result representationBaseline(const detail::MeasuredProblem &problem,
                                   const Options &options, SearchBudget &budget,
                                   std::optional<ReadoutProblem> *completion,
                                   SearchTermination *termination = nullptr) {
  if (!options.legacyWorkLimit)
    return runMeasuredSearch(problem.width, problem.rotations,
                             problem.observables, problem.readoutWires, options,
                             &budget, completion, termination);
  SearchTermination bootstrapStatus = SearchTermination::Exhausted;
  auto bootstrapBudget = explorationSlice(budget);
  auto bootstrap = bootstrapBudget
                       ? bootstrapMeasured(problem, options, *bootstrapBudget,
                                           bootstrapStatus)
                       : pbc::Result{{}, {}, "bootstrap reserve unavailable"};
  if (!bootstrap && bootstrapStatus == SearchTermination::InvalidProof) {
    if (termination)
      *termination = bootstrapStatus;
    return bootstrap;
  }
  auto search = explorationSlice(budget, options.legacyWorkLimit);
  if (!search) {
    if (termination)
      *termination = bootstrap ? SearchTermination::Complete : bootstrapStatus;
    return bootstrap;
  }
  SearchTermination legacyStatus;
  auto legacy = runMeasuredSearch(problem.width, problem.rotations,
                                  problem.observables, problem.readoutWires,
                                  options, &*search, completion, &legacyStatus);
  if (!legacy && legacyStatus == SearchTermination::InvalidProof) {
    if (termination)
      *termination = legacyStatus;
    return legacy;
  }
  if (termination)
    *termination =
        legacy || bootstrap ? SearchTermination::Complete : legacyStatus;
  if (!legacy)
    return bootstrap ? std::move(bootstrap) : std::move(legacy);
  // Bootstrap establishes feasibility. A successful legacy result defines the
  // quality anchor; arbitrary bootstrap secondary metrics must not reject it.
  if (bootstrap && improvesWithoutRegression(problem.width, bootstrap.rotations,
                                             legacy.rotations, options))
    return bootstrap;
  return legacy;
}

std::string tryReadoutAlternatives(const detail::MeasuredProblem &problem,
                                   const ReadoutProblem &completion,
                                   const Options &options, SearchBudget &budget,
                                   pbc::Result &incumbent) {
  detail::ReadoutLimits limits;
  limits.maxCandidates = options.readoutCandidates;
  auto generation = explorationSlice(budget, limits.maxWork);
  if (!generation)
    return {};
  auto tails = detail::generateReadoutCandidates(
      problem.width, completion.observables, problem.readoutWires, options,
      limits, *generation);
  if (tails.status == detail::ReadoutStatus::InvalidProof)
    return tails.error;
  for (const auto &tail : tails.candidates) {
    if (budget.stopped())
      break;
    if (completion.prefix.size() > options.maxGates ||
        tail.gates().size() > options.maxGates - completion.prefix.size())
      continue;
    // Prove the complete circuit against the original measured contract, then
    // compose the existing checked local-channel refinement. A tail's proof
    // alone does not establish the relation of an arbitrary prefix.
    auto proofBudget = explorationSlice(budget);
    if (!proofBudget)
      break;
    if (!proofBudget->take(completion.prefix.size() + tail.gates().size() + 1))
      continue;
    auto gates = completion.prefix;
    gates.insert(gates.end(), tail.gates().begin(), tail.gates().end());
    detail::MeasuredCandidateSink sink(problem, options, &*proofBudget);
    if (!sink.error().empty()) {
      if (proofBudget->stopped())
        continue;
      return sink.error();
    }
    auto submitted = sink.submit(std::move(gates));
    if (submitted.status == detail::SubmissionStatus::InvalidProof)
      return submitted.error;
    auto candidate = sink.takeBest();
    if (candidate &&
        improvesWithoutRegression(problem.width, candidate->gates(),
                                  incumbent.rotations, options))
      incumbent.rotations = std::move(*candidate).takeGates();
  }
  return {};
}

std::string tryCompletedRegion(const detail::MeasuredProblem &problem,
                               const detail::RegionalPlan &plan,
                               const Options &options, SearchBudget &budget,
                               pbc::Result &incumbent) {
  Options proofOptions = options;
  proofOptions.pruneUnobserved = false;
  auto proofBudget = explorationSlice(budget);
  if (!proofBudget || !proofBudget->take(readoutWorkEstimate(
                          problem.width, plan.observables.size())))
    return {};
  auto readout = synthesizeReadout(problem.width, plan.observables,
                                   problem.readoutWires, options);
  if (!readout)
    return readout.stats.budgetExhausted ? std::string{} : readout.error;
  if (plan.prefix.size() > options.maxGates ||
      readout.gates.size() > options.maxGates - plan.prefix.size() ||
      !proofBudget->take(plan.prefix.size() + readout.gates.size() + 1))
    return {};
  auto gates = plan.prefix;
  gates.insert(gates.end(), readout.gates.begin(), readout.gates.end());
  detail::MeasuredCandidateSink sink(problem, proofOptions, &*proofBudget);
  if (!sink.error().empty())
    return proofBudget->stopped() ? std::string{} : sink.error();
  auto submitted = sink.submit(std::move(gates));
  if (submitted.status == detail::SubmissionStatus::InvalidProof)
    return submitted.error;
  auto candidate = sink.takeBest();
  if (candidate && improvesWithoutRegression(problem.width, candidate->gates(),
                                             incumbent.rotations, options))
    incumbent.rotations = std::move(*candidate).takeGates();
  if (options.readoutCandidates && !budget.stopped())
    return tryReadoutAlternatives(problem, {plan.prefix, plan.observables},
                                  proofOptions, budget, incumbent);
  return {};
}

std::string trySmallCommuting(const detail::MeasuredProblem &problem,
                              const Options &options, SearchBudget &budget,
                              pbc::Result &incumbent) {
  detail::CommutingLimits limits;
  limits.maxCandidates = std::min(options.commutingCandidates, 16u);
  limits.exhaustiveSmallRank = true;
  auto generation = explorationSlice(budget, limits.maxWork);
  if (!generation)
    return {};
  pbc::CommutingObservables basis;
  for (const auto &gate : problem.rotations) {
    if (!generation->take(problem.width / 64 + 5))
      return {};
    if (gate.kind != Kind::Rotate || !basis.insert(gate.pauli) ||
        basis.size() > 4)
      return {};
  }
  std::string proofError;
  auto transport =
      [&](const std::vector<Gate> &exit,
          SearchBudget &work) -> std::optional<std::vector<Pauli>> {
    auto amount = llvm::SaturatingMultiply<uint64_t>(
        problem.observables.size() + 1,
        llvm::SaturatingMultiply<uint64_t>(exit.size() + 1,
                                           uint64_t(problem.width) / 64 + 1));
    if (!work.take(amount))
      return std::nullopt;
    auto observables = problem.observables;
    for (const Gate &gate : exit)
      for (Pauli &observable : observables)
        observable = tcount::detail::conjugate(observable, gate);
    return observables;
  };
  auto score = [&](const detail::CommutingImplementation &candidate,
                   SearchBudget &work) -> std::optional<detail::CostRank> {
    auto observables = transport(candidate.exitCliffords, work);
    if (!observables)
      return std::nullopt;
    auto readoutWork = llvm::SaturatingMultiply<uint64_t>(
        uint64_t(problem.width) + problem.observables.size() + 1,
        llvm::SaturatingMultiply<uint64_t>(uint64_t(problem.width) + 1,
                                           problem.observables.size() + 1));
    if (!work.take(readoutWork))
      return std::nullopt;
    auto readout = synthesizeReadout(problem.width, *observables,
                                     problem.readoutWires, options);
    if (!readout) {
      if (!readout.stats.budgetExhausted)
        proofError = readout.error;
      return std::nullopt;
    }
    if (candidate.body.size() > options.maxGates ||
        readout.gates.size() > options.maxGates - candidate.body.size())
      return std::nullopt;
    auto gates = candidate.body;
    gates.insert(gates.end(), readout.gates.begin(), readout.gates.end());
    // Rank completed implementations after the same local refinement used by
    // acceptance. This only estimates selection cost: the shortlisted raw word
    // still needs source replay followed by its own checked refinement.
    if (auto refined = detail::optimizeSignedFrames(
            problem.width, gates, problem.readoutWires, &work, proofError))
      gates = std::move(*refined);
    if (!proofError.empty() || work.stopped() ||
        !work.take(uint64_t(problem.width) + gates.size() + 1) ||
        !improvesWithoutRegression(problem.width, gates, incumbent.rotations,
                                   options))
      return std::nullopt;
    return circuitScore(problem.width, gates, options);
  };
  auto candidates = detail::solveCommutingRegion(
      problem.width, problem.rotations, options, limits, *generation, score);
  if (!proofError.empty())
    return proofError;
  if (candidates.status == detail::CommutingStatus::InvalidProof)
    return candidates.error;
  for (const auto &candidate : candidates.candidates) {
    if (budget.stopped())
      break;
    auto completionBudget = explorationSlice(budget);
    if (!completionBudget)
      break;
    auto observables = transport(candidate.exitCliffords, *completionBudget);
    if (!observables)
      continue;
    auto error = tryCompletedRegion(problem, {candidate.body, *observables},
                                    options, *completionBudget, incumbent);
    if (!error.empty())
      return error;
  }
  return {};
}

std::string tryRegionalAlternatives(const detail::MeasuredProblem &problem,
                                    const Options &options,
                                    SearchBudget &budget,
                                    pbc::Result &incumbent) {
  auto error = trySmallCommuting(problem, options, budget, incumbent);
  if (!error.empty() || budget.stopped())
    return error;
  detail::RegionLimits limits;
  limits.maxPlans = options.commutingCandidates;
  auto generation = explorationSlice(budget, limits.maxWork);
  if (!generation)
    return {};
  auto plans =
      detail::generateRegionCandidates(problem, options, limits, *generation);
  if (plans.status == detail::RegionStatus::InvalidProof)
    return plans.error;
  for (const auto &plan : plans.candidates) {
    if (budget.stopped())
      break;
    error = tryCompletedRegion(problem, plan, options, budget, incumbent);
    if (!error.empty())
      return error;
  }
  // Keep the completed sweep portfolio before exploring different carried
  // quantum boundaries. Beam ranking includes signed readout and the same
  // refinement as acceptance, but never substitutes for independent replay.
  auto beamBudget = explorationSlice(budget, limits.maxWork);
  if (!beamBudget)
    return {};
  limits.beamWidth = std::min(options.commutingCandidates, 8u);
  std::string proofError;
  auto score = [&](const detail::RegionalPlan &plan,
                   SearchBudget &work) -> std::optional<detail::CostRank> {
    auto readoutWork = llvm::SaturatingMultiply<uint64_t>(
        uint64_t(problem.width) + plan.observables.size() + 1,
        llvm::SaturatingMultiply<uint64_t>(uint64_t(problem.width) + 1,
                                           plan.observables.size() + 1));
    if (!work.take(readoutWork))
      return std::nullopt;
    auto readout = synthesizeReadout(problem.width, plan.observables,
                                     problem.readoutWires, options);
    if (!readout) {
      if (!readout.stats.budgetExhausted)
        proofError = readout.error;
      return std::nullopt;
    }
    if (plan.prefix.size() > options.maxGates ||
        readout.gates.size() > options.maxGates - plan.prefix.size() ||
        !work.take(plan.prefix.size() + readout.gates.size() + 1))
      return std::nullopt;
    auto gates = plan.prefix;
    gates.insert(gates.end(), readout.gates.begin(), readout.gates.end());
    if (auto refined = detail::optimizeSignedFrames(
            problem.width, gates, problem.readoutWires, &work, proofError))
      gates = std::move(*refined);
    if (!proofError.empty() || work.stopped() ||
        !work.take(uint64_t(problem.width) + gates.size() + 1) ||
        !improvesWithoutRegression(problem.width, gates, incumbent.rotations,
                                   options))
      return std::nullopt;
    return circuitScore(problem.width, gates, options);
  };
  auto beam = detail::generateRegionCandidates(problem, options, limits,
                                               *beamBudget, score);
  if (!proofError.empty())
    return proofError;
  if (beam.status == detail::RegionStatus::InvalidProof)
    return beam.error;
  for (const auto &plan : beam.candidates) {
    if (budget.stopped())
      break;
    error = tryCompletedRegion(problem, plan, options, budget, incumbent);
    if (!error.empty())
      return error;
  }
  detail::GeneralLimits generalLimits;
  generalLimits.maxPlans = std::min(options.commutingCandidates, 2u);
  for (bool macros : {false, true}) {
    auto generalBudget = explorationSlice(budget, generalLimits.maxWork);
    if (!generalBudget)
      break;
    generalLimits.useMacros = macros;
    auto general = detail::generateGeneralCandidates(
        problem, options, generalLimits, *generalBudget);
    if (general.status == detail::RegionStatus::InvalidProof)
      return general.error;
    for (const auto &plan : general.candidates) {
      if (budget.stopped())
        break;
      error = tryCompletedRegion(problem, plan, options, budget, incumbent);
      if (!error.empty())
        return error;
    }
  }
  detail::OpenRegionLimits openLimits;
  openLimits.maxPlans = std::min(options.commutingCandidates, 4u);
  auto openBudget = explorationSlice(budget, openLimits.maxWork);
  if (!openBudget)
    return {};
  auto open = detail::generateOpenRegionCandidates(problem, options, openLimits,
                                                   *openBudget);
  if (open.status == detail::OpenRegionStatus::InvalidProof)
    return open.error;
  for (const auto &plan : open.candidates) {
    if (budget.stopped())
      break;
    error = tryCompletedRegion(problem, plan, options, budget, incumbent);
    if (!error.empty())
      return error;
  }
  return {};
}
std::string tryNativeRefinement(unsigned width, const Options &options,
                                SearchBudget &budget, pbc::Result &incumbent) {
  detail::NativeRefinementLimits limits;
  limits.maxGates = options.maxGates;
  limits.maxTemporaryGates =
      std::max(limits.maxTemporaryGates, options.maxGates);
  limits.maxCandidates = 4 * std::min(options.algebraicCandidates, 4u);
  auto generation = explorationSlice(budget, limits.maxWork);
  if (!generation ||
      !generation->take(uint64_t(width) + incumbent.rotations.size() + 1))
    return {};
  // The source is the already proved synthesis result. Its immutable copy
  // binds every refinement even if an earlier alternative becomes incumbent.
  auto synthesized = incumbent.rotations;
  auto alternatives = detail::refineSynthesizedNative(
      width, synthesized, options, limits, *generation);
  if (alternatives.status == detail::NativeRefinementStatus::InvalidProof)
    return alternatives.error;
  for (const auto &candidate : alternatives.alternatives) {
    auto selection = explorationSlice(budget);
    if (!selection)
      break;
    if (!selection->take(uint64_t(width) + synthesized.size() +
                         candidate.gates().size() + 1))
      continue;
    if (!candidate.matchesSource(width, synthesized))
      return "native refinement belongs to a different synthesized source";
    // Compose the checked native full-unitary rewrite with the source's
    // measured proof. No original or preserve-network circuit is supplied.
    if (improvesWithoutRegression(width, candidate.gates(), incumbent.rotations,
                                  options))
      incumbent.rotations = candidate.gates();
  }
  return {};
}

/// Append a new native route only after the existing portfolio and all its
/// continuations have completed. A better intermediate network is not a safe
/// replacement for another route's input: later cleanup can reverse their
/// ranking. This extension consumes only the already certified synthesis
/// result, and every accepted replacement retains all five final metrics.
std::string tryFourWireRefinement(unsigned width, const Options &options,
                                  SearchBudget &budget,
                                  pbc::Result &incumbent) {
  if (width < 4 || incumbent.rotations.size() < 4)
    return {};
  detail::LinearFourLimits limits;
  limits.maxGates = options.maxGates;
  auto generation =
      explorationSlice(budget, limits.maxWork + limits.maxTableWork);
  if (!generation)
    return {};
  auto dictionary =
      detail::buildLinearFourDictionary(options, limits, *generation);
  if (!dictionary.dictionary)
    return dictionary.status == detail::LinearFourStatus::InvalidProof
               ? dictionary.error
               : std::string{};
  auto alternatives = detail::optimizeLinearFourWindows(
      width, incumbent.rotations, options, limits, dictionary.dictionary,
      *generation);
  if (alternatives.status == detail::LinearFourStatus::InvalidProof)
    return alternatives.error;
  // The provider publishes at most one complete source-bound rewrite. Verify
  // its binding before replacing the source, including on local exhaustion.
  for (const auto &candidate : alternatives.alternatives) {
    auto selection = explorationSlice(budget);
    if (!selection ||
        !selection->take(uint64_t(width) + incumbent.rotations.size() +
                         candidate.gates().size() + 1))
      break;
    if (!candidate.matchesSource(width, incumbent.rotations))
      return "four-wire refinement belongs to a different synthesized source";
    if (improvesWithoutRegression(width, candidate.gates(), incumbent.rotations,
                                  options))
      incumbent.rotations = candidate.gates();
  }
  return {};
}

pbc::Result runMeasuredPortfolio(unsigned width, const std::vector<Gate> &input,
                                 const std::vector<Pauli> &observables,
                                 const std::vector<unsigned> &readoutWires,
                                 const Options &options, SearchBudget *budget,
                                 SearchTermination *termination = nullptr) {
  if (termination)
    *termination = SearchTermination::Complete;
  auto invalid = [&](std::string error) {
    if (termination)
      *termination = SearchTermination::InvalidProof;
    return pbc::Result{{}, {}, std::move(error)};
  };
  SearchBudget unlimited(std::numeric_limits<uint64_t>::max());
  SearchBudget &shared = budget ? *budget : unlimited;
  detail::MeasuredProblem source{width, input, observables, readoutWires};
  std::optional<ReadoutProblem> completion;
  auto incumbent = representationBaseline(
      source, options, shared,
      options.readoutCandidates ? &completion : nullptr, termination);
  if (!incumbent || shared.stopped())
    return incumbent;
  if (!options.readoutCandidates && !options.representationSteps &&
      !options.commutingCandidates && !options.algebraicCandidates)
    return incumbent;
  if (completion) {
    auto error =
        tryReadoutAlternatives(source, *completion, options, shared, incumbent);
    if (!error.empty())
      return invalid(std::move(error));
  }
  if (options.commutingCandidates && !shared.stopped()) {
    auto error = tryRegionalAlternatives(source, options, shared, incumbent);
    if (!error.empty())
      return invalid(std::move(error));
  }
  if (options.representationSteps && !shared.stopped()) {
    auto generation = explorationSlice(shared, 100000000);
    if (!generation)
      return incumbent;
    auto proposal = detail::exploreRepresentations(
        source, options.representationSteps, &*generation);
    if (proposal.alternative) {
      auto proofBudget = explorationSlice(shared);
      if (!proofBudget)
        return incumbent;
      auto proof = detail::verifyRepresentation(source, *proposal.alternative,
                                                &*proofBudget);
      if (proof.status == detail::RepresentationCheckStatus::InvalidProof)
        return invalid(std::move(proof.error));
      if (proof) {
        const auto &word = *proposal.alternative;
        std::optional<ReadoutProblem> alternateCompletion;
        SearchTermination alternateStatus;
        detail::MeasuredProblem alternateProblem{
            width, word.rotations, word.observables, word.readoutWires};
        auto alternateBudget = explorationSlice(shared);
        if (!alternateBudget)
          return incumbent;
        auto alternative = representationBaseline(
            alternateProblem, options, *alternateBudget,
            options.readoutCandidates ? &alternateCompletion : nullptr,
            &alternateStatus);
        if (!alternative && alternateStatus == SearchTermination::InvalidProof)
          return invalid(std::move(alternative.error));
        if (alternative) {
          if (alternateCompletion && !shared.stopped()) {
            detail::MeasuredProblem alternateProblem{
                width, word.rotations, word.observables, word.readoutWires};
            auto error =
                tryReadoutAlternatives(alternateProblem, *alternateCompletion,
                                       options, shared, alternative);
            if (!error.empty())
              return invalid(std::move(error));
          }
          if (options.commutingCandidates && !shared.stopped()) {
            detail::MeasuredProblem alternateProblem{
                width, word.rotations, word.observables, word.readoutWires};
            auto error = tryRegionalAlternatives(alternateProblem, options,
                                                 shared, alternative);
            if (!error.empty())
              return invalid(std::move(error));
          }
          // The independent source-to-alternative certificate composes with the
          // alternative's word replay and checked local-channel refinement.
          if (improvesWithoutRegression(width, alternative.rotations,
                                        incumbent.rotations, options))
            incumbent = std::move(alternative);
        }
      }
    }
  }
  if (options.algebraicCandidates && !shared.stopped()) {
    // Complete the original word's native continuation while its bounded
    // work/deadline slice is still available. Algebraic reconstruction can
    // consume the remaining exploration time, so its unrefined parent is not
    // a sufficient quality anchor. Every later replacement must dominate this
    // completed result in all five metrics; optional exhaustion retains it.
    if (auto error = tryNativeRefinement(width, options, shared, incumbent);
        !error.empty())
      return invalid(std::move(error));
    detail::AlgebraLimits limits;
    limits.maxCandidates = options.algebraicCandidates;
    limits.maxInputGates = options.maxGates;
    limits.maxTemporaryGates = options.maxGates;
    auto generation = explorationSlice(shared, limits.maxWork);
    if (!generation)
      return incumbent;
    auto alternatives =
        detail::generateAlgebraAlternatives(source, limits, *generation);
    for (const auto &word : alternatives.alternatives) {
      if (shared.stopped())
        break;
      if (!word.matchesSource(source))
        return invalid("algebraic candidate belongs to a different source");
      auto solve = explorationSlice(shared);
      if (!solve)
        break;
      Options fixedWord = options;
      fixedWord.algebraicCandidates = 0;
      SearchTermination status;
      auto candidate = runMeasuredPortfolio(
          width, word.rotations(), word.observables(), word.readoutWires(),
          fixedWord, &*solve, &status);
      if (!candidate) {
        if (status == SearchTermination::InvalidProof)
          return invalid(std::move(candidate.error));
        continue;
      }
      if (auto error = tryNativeRefinement(width, options, shared, candidate);
          !error.empty())
        return invalid(std::move(error));
      // Compose checked full-unitary T rewrites, measured normalization, and
      // this alternative's complete synthesis proof. No native source network
      // was supplied to the algebraic provider.
      if (improvesWithoutRegression(width, candidate.rotations,
                                    incumbent.rotations, options))
        incumbent = std::move(candidate);
    }
    // Retain the former final cleanup as an additional checked route. Its
    // all-five guard cannot evict the completed original-word anchor or any
    // already accepted dominating algebraic continuation.
    if (auto error = tryNativeRefinement(width, options, shared, incumbent);
        !error.empty())
      return invalid(std::move(error));
  }
  return incumbent;
}

/// The source word, its certified representation, and its entire continuation
/// form one independent route. Retain the old final result before exploring:
/// support improvement alone never authorizes replacing its native circuit.
std::string tryIndexedRepresentation(const detail::MeasuredProblem &source,
                                     const Options &options,
                                     SearchBudget &budget,
                                     pbc::Result &incumbent) {
  detail::IndexedRepresentationLimits limits;
  limits.maxSteps = options.indexedRepresentationSteps;
  limits.maxRotations = std::min(limits.maxRotations, options.maxGates);
  auto generation = explorationSlice(budget, limits.maxWork);
  if (!generation)
    return {};
  auto proposal =
      detail::exploreIndexedRepresentations(source, limits, *generation);
  if (!proposal.alternative)
    return {};
  auto proofBudget = explorationSlice(budget);
  if (!proofBudget)
    return {};
  auto proof = detail::verifyRepresentation(source, *proposal.alternative,
                                            &*proofBudget);
  if (proof.status == detail::RepresentationCheckStatus::InvalidProof)
    return proof.error;
  if (!proof)
    return {};
  auto solve = explorationSlice(budget);
  if (!solve)
    return {};
  Options continuation = options;
  continuation.representationSteps = 0;
  continuation.indexedRepresentationSteps = 0;
  continuation.discrepancySteps = 0;
  // Completion is still constructed and checked. Only the repeated heuristic
  // reconstruction at every primitive decision is omitted on this route.
  continuation.readoutWeight = 0;
  // Bound each decision's support-scoring population and generating frontier.
  // These limits affect proposal quality only; readiness and final replay
  // still cover every original rotation and ordered observable.
  continuation.lookahead =
      options.lookahead ? std::min(options.lookahead, 64u) : 64u;
  continuation.candidateNodes =
      options.candidateNodes ? std::min(options.candidateNodes, 4u) : 4u;
  const auto &word = *proposal.alternative;
  SearchTermination status;
  auto candidate =
      runMeasuredPortfolio(source.width, word.rotations, word.observables,
                           word.readoutWires, continuation, &*solve, &status);
  if (!candidate)
    return status == SearchTermination::InvalidProof ? candidate.error
                                                     : std::string{};
  if (options.algebraicCandidates)
    if (auto error =
            tryFourWireRefinement(source.width, options, budget, candidate);
        !error.empty())
      return error;
  // Compose the independently checked representation certificate with the
  // alternative's own measured replay and full native-refinement proofs.
  auto selection = explorationSlice(budget);
  if (!selection || !selection->take(8 * (uint64_t(source.width) +
                                          candidate.rotations.size() +
                                          incumbent.rotations.size() + 1)))
    return {};
  if (improvesWithoutRegression(source.width, candidate.rotations,
                                incumbent.rotations, options))
    incumbent = std::move(candidate);
  return {};
}

/// Finish this independent native route before comparing it with the old
/// complete anchor. A T-count rewrite can temporarily increase T depth while
/// exposing a linear window whose subsequent refinement removes that increase.
/// Filtering the intermediate word would lose that completed improvement.
std::string tryNativeFourCompletion(unsigned width, const Options &options,
                                    SearchBudget &budget,
                                    pbc::Result &incumbent) {
  if (width < 4 || incumbent.rotations.size() < 4)
    return {};
  detail::NativeRefinementLimits limits;
  limits.maxGates = options.maxGates;
  limits.maxTemporaryGates =
      std::max(limits.maxTemporaryGates, options.maxGates);
  limits.maxCandidates = 4;
  limits.maxWork = 100000000;
  auto copyWork = [width](size_t gates) {
    return llvm::SaturatingMultiply<uint64_t>(
        uint64_t(gates) + 1, 2 * ((uint64_t(width) + 63) / 64) + 1);
  };
  auto generation = explorationSlice(budget, limits.maxWork);
  if (!generation || !generation->take(copyWork(incumbent.rotations.size())))
    return {};
  // All alternatives bind to this immutable, already proved native source,
  // even when a completed continuation later replaces the incumbent.
  const auto synthesized = incumbent.rotations;
  auto alternatives = detail::refineSynthesizedNative(
      width, synthesized, options, limits, *generation);
  if (alternatives.status == detail::NativeRefinementStatus::InvalidProof)
    return alternatives.error;
  auto consider = [&](const std::vector<Gate> &word) {
    auto selection = explorationSlice(budget);
    const auto scoringWork = llvm::SaturatingMultiply<uint64_t>(
        8, uint64_t(width) + word.size() + incumbent.rotations.size() + 1);
    if (!selection ||
        !selection->take(llvm::SaturatingAdd<uint64_t>(
            scoringWork, copyWork(word.size()))))
      return false;
    if (improvesWithoutRegression(width, word, incumbent.rotations, options))
      incumbent.rotations = word;
    return true;
  };
  for (const auto &candidate : alternatives.alternatives) {
    auto preparation = explorationSlice(budget);
    if (!preparation ||
        !preparation->take(llvm::SaturatingAdd<uint64_t>(
            copyWork(synthesized.size()), copyWork(candidate.gates().size()))))
      break;
    if (!candidate.matchesSource(width, synthesized))
      return "native completion belongs to a different synthesized source";
    detail::LinearFourLimits fourLimits;
    fourLimits.maxGates = options.maxGates;
    auto continuation = explorationSlice(
        budget, fourLimits.maxWork + fourLimits.maxTableWork);
    if (continuation) {
      auto dictionary = detail::buildLinearFourDictionary(
          options, fourLimits, *continuation);
      if (dictionary.status == detail::LinearFourStatus::InvalidProof)
        return dictionary.error;
      if (dictionary.dictionary) {
        auto completed = detail::optimizeLinearFourWindows(
            width, candidate.gates(), options, fourLimits,
            dictionary.dictionary, *continuation);
        if (completed.status == detail::LinearFourStatus::InvalidProof)
          return completed.error;
        for (const auto &word : completed.alternatives) {
          auto binding = explorationSlice(budget);
          if (!binding || !binding->take(copyWork(candidate.gates().size())))
            break;
          if (!word.matchesSource(width, candidate.gates()))
            return "four-wire completion belongs to a different native source";
          if (!consider(word.gates()))
            break;
        }
      }
    }
    // Compose the native certificate and its checked Four continuation with
    // the original measured proof. Exhaustion retains every accepted anchor;
    // only complete, dominating native circuits can replace it. The native
    // word itself is also a complete fallback, evaluated after its extension.
    // No intermediate-relative metric guard filters either complete route.
    if (!consider(candidate.gates()))
      break;
  }
  return {};
}

/// Complete commuting-Clifford rewrites and their native cleanup from one
/// immutable, already proved synthesis result. The local certificates include
/// scalar phase, so they compose directly with its measured proof.
std::string tryCliffordNativeCompletion(unsigned width, const Options &options,
                                       SearchBudget &budget,
                                       pbc::Result &incumbent) {
  detail::CliffordWindowLimits limits;
  limits.maxGates = options.maxGates;
  auto copyWork = [width](size_t gates) {
    return llvm::SaturatingMultiply<uint64_t>(
        uint64_t(gates) + 1, 2 * ((uint64_t(width) + 63) / 64) + 1);
  };
  auto generation = explorationSlice(
      budget, llvm::SaturatingMultiply<uint64_t>(2, limits.maxWork));
  if (!generation || !generation->take(copyWork(incumbent.rotations.size())))
    return {};
  const auto synthesized = incumbent.rotations;
  auto alternatives = detail::optimizeCliffordWindows(
      width, synthesized, options, limits, *generation);
  if (alternatives.status == detail::CliffordWindowStatus::InvalidProof)
    return alternatives.error;
  auto consider = [&](const std::vector<Gate> &word) {
    auto selection = explorationSlice(budget);
    const auto scoringWork = llvm::SaturatingMultiply<uint64_t>(
        8, uint64_t(width) + word.size() + incumbent.rotations.size() + 1);
    if (!selection ||
        !selection->take(llvm::SaturatingAdd<uint64_t>(
            scoringWork, copyWork(word.size()))))
      return false;
    if (improvesWithoutRegression(width, word, incumbent.rotations, options))
      incumbent.rotations = word;
    return true;
  };
  for (const auto &candidate : alternatives.alternatives) {
    auto binding = explorationSlice(budget);
    if (!binding || !binding->take(copyWork(synthesized.size())))
      break;
    if (!candidate.matchesSource(width, synthesized))
      return "Clifford completion belongs to a different synthesized source";
    detail::NativeRefinementLimits nativeLimits;
    nativeLimits.maxGates = options.maxGates;
    nativeLimits.maxTemporaryGates =
        std::max(nativeLimits.maxTemporaryGates, options.maxGates);
    auto continuation = explorationSlice(budget, nativeLimits.maxWork);
    if (continuation) {
      auto completed = detail::refineSynthesizedNative(
          width, candidate.gates(), options, nativeLimits, *continuation);
      if (completed.status == detail::NativeRefinementStatus::InvalidProof)
        return completed.error;
      for (const auto &word : completed.alternatives) {
        auto check = explorationSlice(budget);
        if (!check || !check->take(copyWork(candidate.gates().size())))
          break;
        if (!word.matchesSource(width, candidate.gates()))
          return "native completion belongs to a different Clifford source";
        if (!consider(word.gates()))
          break;
      }
    }
    // The complete local rewrite remains available if optional native cleanup
    // exhausts its slice. Compare both only against the current final result.
    if (!consider(candidate.gates()))
      break;
  }
  return {};
}
/// Append independent source-only joint-phase routes. The completion service
/// owns proof composition and immutable-anchor admission; planner scores never
/// authorize replacing the already verified native incumbent.
std::string tryJointPhaseCompletion(const detail::MeasuredProblem &source,
                                    const Options &options,
                                    SearchBudget &parent,
                                    pbc::Result &incumbent) {
  if (!options.jointPhaseCandidates || parent.remaining() < 2 ||
      !detail::open_detail::hasTime(parent))
    return {};
  SearchBudget work(parent,
                    std::min(options.jointPhaseMaxWork, parent.remaining()),
                    options.jointPhaseTimeoutMs);
  // Reserve final binding/selection/copy before any optional provider. A
  // completed certificate remains publishable after optional work exhaustion.
  const uint64_t limbs = uint64_t(source.width) / 64 + 1;
  if (!work.take(detail::compact_transport::mul(
          8 * (uint64_t(options.maxGates) * 2 + source.rotations.size() +
               source.observables.size() + source.width + 1),
          limbs)))
    return {};
  Options proofOptions = options;
  proofOptions.simplify = proofOptions.pruneUnobserved =
      proofOptions.acceptOnlyImprovement = false;
  detail::CompletionLimits completionLimits;
  completionLimits.maxGates = options.maxGates;
  completionLimits.maxWork = options.jointPhaseMaxWork;
  completionLimits.maxLiveBytes = 2ull << 30;
  completionLimits.rankedContinuation = true;
  const auto anchorCost = tcount::cost(source.width, incumbent.rotations);
  const auto anchorRank =
      detail::circuitScore(source.width, incumbent.rotations, options);
  auto bestRank = anchorRank;
  std::vector<Gate> bestWord;
  bool improved = false;
  // Each profile has an isolated completion/scoring cache. Richer results from
  // another profile cannot change this planner's heuristic decisions. Only
  // source-bound, complete results participate in common final selection.
  for (unsigned profile = 0; profile < (options.jointPhaseWideSearch ? 2u : 1u);
       ++profile) {
    if (work.remaining() < 2 || !detail::open_detail::hasTime(work))
      break;
    SearchBudget profileBudget(work, work.remaining(), 60000);
    detail::CompletionSession session(
        source, proofOptions, anchorCost, anchorRank, completionLimits,
        profileBudget,
        options.jointPhaseAcceptance == JointPhaseAcceptance::GatesAndT
            ? detail::CompletionPolicy::GatesAndT
            : detail::CompletionPolicy::AllFive);
    auto run = [&](const detail::CompletionRouteHandle &route) -> std::string {
      if (!route || profileBudget.remaining() < 2 ||
          !detail::open_detail::hasTime(profileBudget))
        return {};
      detail::JointPhasePlannerLimits limits;
      limits.maxPlans = options.jointPhaseCandidates;
      limits.maxWork = profile ? 10000000000 : options.jointPhasePlannerWork;
      limits.maxContextCalls = profile ? 16384 : 4096;
      limits.maxExpansions = profile ? 16384 : 4096;
      limits.contextScoring = true;
      limits.region.maxCoordinateLayouts = 4;
      auto plans = detail::generateJointPhasePlans(
          route->problem(), proofOptions, limits, profileBudget,
          [&](const detail::RegionalPlan &plan, SearchBudget &scoring) {
          return session.score(route, plan, scoring).rank;
          });
      if (plans.status == detail::OpenRegionStatus::InvalidProof)
        return plans.error;
      if (session.status() == detail::CompletionStatus::InvalidProof)
        return session.error();
      for (const auto &plan : plans.candidates) {
        auto result = session.complete(route, plan, profileBudget);
        if (result.status == detail::CompletionStatus::InvalidProof)
          return result.error;
      }
      return {};
    };
    if (auto error = run(session.originalRoute()); !error.empty())
      return error;
    if (options.indexedRepresentationSteps && profileBudget.remaining() > 1 &&
        detail::open_detail::hasTime(profileBudget)) {
      detail::IndexedRepresentationLimits limits;
      limits.maxSteps = options.indexedRepresentationSteps;
      limits.maxRotations = std::min(limits.maxRotations, options.maxGates);
      auto generation = explorationSlice(profileBudget, limits.maxWork);
      if (generation) {
        auto proposal =
            detail::exploreIndexedRepresentations(source, limits, *generation);
        if (proposal.alternative) {
          auto checked =
              session.checkIndexedRoute(*proposal.alternative, profileBudget);
          if (checked.status == detail::CompletionStatus::InvalidProof)
            return checked.error;
          if (auto error = run(checked.route); !error.empty())
            return error;
        }
      }
    }
    if (session.status() == detail::CompletionStatus::InvalidProof)
      return session.error();
    if (const auto *best = session.best()) {
      if (!best->matchesSource(source))
        return "joint-phase completion source mismatch";
      if (best->rank() < bestRank) {
        bestWord = best->gates();
        bestRank = best->rank();
        improved = true;
      }
    }
  }
  if (improved)
    incumbent.rotations = std::move(bestWord);
  return {};
}

} // namespace

pbc::Result cudaq::pauli::graph::synthesizeMeasured(
    unsigned width, const std::vector<Gate> &input,
    const std::vector<Pauli> &observables,
    const std::vector<unsigned> &readoutWires, const Options &options,
    SearchBudget *budget) {
  SearchBudget unlimited(std::numeric_limits<uint64_t>::max());
  SearchBudget &shared = budget ? *budget : unlimited;
  std::optional<SearchBudget> legacyBudget;
  if (options.legacyPortfolioTimeoutMs) {
    uint64_t allowance = shared.remaining();
    if (options.jointPhaseCandidates && allowance > options.jointPhaseMaxWork)
      allowance -= options.jointPhaseMaxWork;
    legacyBudget.emplace(shared, allowance, options.legacyPortfolioTimeoutMs);
  }
  SearchBudget &legacy = legacyBudget ? *legacyBudget : shared;
  auto result = runMeasuredPortfolio(width, input, observables, readoutWires,
                                     options, &legacy);
  if (result && options.algebraicCandidates && !legacy.stopped())
    if (auto error = tryFourWireRefinement(width, options, legacy, result);
        !error.empty())
      return {{}, {}, std::move(error)};
  if (result && options.indexedRepresentationSteps && !legacy.stopped())
    if (auto error = tryIndexedRepresentation(
            {width, input, observables, readoutWires}, options, legacy, result);
        !error.empty())
      return {{}, {}, std::move(error)};
  if (result && options.algebraicCandidates && !legacy.stopped())
    if (auto error = tryNativeFourCompletion(width, options, legacy, result);
        !error.empty())
      return {{}, {}, std::move(error)};
  if (result && options.algebraicCandidates && !legacy.stopped())
    if (auto error =
            tryCliffordNativeCompletion(width, options, legacy, result);
        !error.empty())
      return {{}, {}, std::move(error)};
  if (result && options.jointPhaseCandidates && !shared.stopped())
    if (auto error = tryJointPhaseCompletion(
            {width, input, observables, readoutWires}, options, shared, result);
        !error.empty())
      return {{}, {}, std::move(error)};
  return result;
}
