/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Exact Clifford and phase-network rewrites for total gate count.
///
/// Keeps the existing entangling network while moving Pauli frames and phases,
/// then minimizes small Clifford windows. Signed symplectic algebra authorizes
/// propagation, phase signatures check relocation, and independent Clifford
/// replay checks each window replacement including its scalar correction.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Pauli/GateCount.h"
#include "CliffordDictionary.h"
#include "TCountInternal.h"
#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include <algorithm>
#include <array>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>

using namespace cudaq::pauli;
using namespace cudaq::pauli::tcount::detail;
namespace {
using Circuit = std::vector<Gate>;

/// Scalar phases are free in the gate metric but retained in native semantics.
void phase(Circuit &out, int numerator, int denominator = 4) {
  if (numerator % (2 * denominator))
    out.push_back(Gate::phase({numerator, denominator}));
}

/// Obtain all touched wires, excluding scalar phase operations.
QubitSet support(const Gate &g) {
  if (g.kind == Kind::Phase)
    return {};
  if (g.kind == Kind::Rotate)
    return g.pauli.x | g.pauli.z;
  auto result = QubitSet::single(g.a);
  if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
    result.set(g.b);
  return result;
}

/// Consolidate exact eighth-root scalar phases without numerical rounding.
/// Scalar operations commute with every gate, so this also keeps scan windows
/// measured in meaningful operations instead of accumulated bookkeeping.
Circuit packPhases(Circuit input) {
  Circuit out;
  int phaseEighths = 0;
  for (auto g : input) {
    if (g.kind == Kind::Phase && !g.angle.symbol) {
      __int128 scaled = __int128(g.angle.numerator) * 8;
      if (scaled % g.angle.denominator == 0) {
        int k = int((scaled / g.angle.denominator) % 16);
        phaseEighths = (phaseEighths + (g.angle.negative ? -k : k)) % 16;
        continue;
      }
    }
    out.push_back(g);
  }
  phase(out, phaseEighths, 8);
  return out;
}

/// Expand exact rotations to the named Clifford basis and single-wire T
/// rotations. Keep their scalar correction, rather than identifying Rz with T.
/// Unknown angles stay intact and act as boundaries to specialized methods.
Circuit primitives(const Circuit &input, SearchBudget &budget, unsigned cap) {
  Circuit out;
  for (auto g : input) {
    if (!budget.take() || out.size() > cap)
      return input;
    if (g.kind != Kind::Rotate || !quarters(g.angle) || !g.pauli.weight()) {
      out.push_back(g);
      continue;
    }
    int k = *quarters(g.angle) * (g.pauli.sign ? -1 : 1);
    int r = ((k % 8 + 12) % 8) - 4;
    if (!r) {
      phase(out, -k, 8);
      continue;
    }
    Circuit basis;
    auto qs = support(g);
    unsigned pivot = qs.first();
    for (unsigned q : qs.set_bits()) {
      if (g.pauli.axis(q) == 3)
        basis.push_back({Kind::Sdg, q});
      if (g.pauli.x.test(q))
        basis.push_back({Kind::H, q});
      if (q != pivot)
        basis.push_back({Kind::CX, q, pivot});
    }
    if (out.size() + 2 * basis.size() + 4 > cap)
      return input;
    out.insert(out.end(), basis.begin(), basis.end());
    unsigned magnitude = std::abs(r);
    int odd = (magnitude & 1) ? (r < 0 ? -1 : 1) : 0;
    if (magnitude == 3) {
      // Equal-cost decompositions have different propagation opportunities:
      // S T = Z Tdg and Sdg Tdg = Z T. Expose the Pauli instead of hiding it
      // in S/Sdg, retaining the exact Rz scalar in the correction below.
      out.push_back({Kind::Z, pivot});
      odd = -odd;
    } else {
      if (magnitude & 4)
        out.push_back({Kind::Z, pivot});
      if (magnitude & 2)
        out.push_back({r < 0 ? Kind::Sdg : Kind::S, pivot});
    }
    if (odd)
      out.push_back(Gate::rotation({0, QubitSet::single(pivot)}, {odd, 4}));
    phase(out, odd - k, 8);
    auto undo = adjoint(basis);
    out.insert(out.end(), undo.begin(), undo.end());
  }
  return packPhases(std::move(out));
}

/// Carry a signed Hermitian Pauli through each operation. Invariant: the input
/// prefix equals i^scalar * pending * emitted. Multiplying anticommuting Paulis
/// contributes +/-i; Clifford conjugation supplies the remaining signs.
Circuit propagate(const Circuit &input, SearchBudget &budget) {
  Circuit out;
  Pauli pending;
  unsigned scalar = 0;
  for (auto g : input) {
    if (!budget.take(1 + support(g).count()))
      return input;
    if (g.kind == Kind::X || g.kind == Kind::Y || g.kind == Kind::Z) {
      Pauli p{g.kind != Kind::Z ? QubitSet::single(g.a) : QubitSet{},
              g.kind != Kind::X ? QubitSet::single(g.a) : QubitSet{}};
      Pauli next{p.x ^ pending.x, p.z ^ pending.z};
      scalar += p.x.intersectionCount(p.z) +
                pending.x.intersectionCount(pending.z) -
                next.x.intersectionCount(next.z) +
                2 * p.z.intersectionCount(pending.x) + 2 * pending.sign;
      scalar &= 3;
      pending = next;
    } else if (g.kind == Kind::Rotate) {
      if (pending.anti(g.pauli))
        g.angle = g.angle.negated();
      out.push_back(g);
    } else if ((g.kind == Kind::S || g.kind == Kind::Sdg) &&
               pending.x.test(g.a)) {
      // S X = i X Sdg, and Sdg X = -i X S. Choosing the inverse keeps
      // the pending support unchanged and exposes phase cancellations.
      scalar = (scalar + (g.kind == Kind::S ? 1 : 3)) & 3;
      out.push_back(inverse(g));
    } else {
      pending = conjugate(pending, g);
      out.push_back(g);
    }
  }
  auto pendingSupport = pending.x | pending.z;
  for (unsigned q : pendingSupport.set_bits())
    out.push_back(
        {pending.x.test(q) ? (pending.z.test(q) ? Kind::Y : Kind::X) : Kind::Z,
         q});
  phase(out, int(scalar + 2 * pending.sign), 2);
  return out;
}

/// Compress wire numbers for independent small-Clifford equivalence checks.
Circuit localize(const Circuit &input, const std::vector<unsigned> &qs) {
  Circuit out = input;
  auto index = [&](unsigned q) {
    return unsigned(std::find(qs.begin(), qs.end(), q) - qs.begin());
  };
  for (auto &g : out) {
    if (g.kind == Kind::Phase)
      continue;
    if (g.kind == Kind::Rotate) {
      Pauli p;
      p.sign = g.pauli.sign;
      for (unsigned q : g.pauli.x.set_bits())
        p.x.set(index(q));
      for (unsigned q : g.pauli.z.set_bits())
        p.z.set(index(q));
      g.pauli = p;
    } else {
      g.a = index(g.a);
      if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap)
        g.b = index(g.b);
    }
  }
  return out;
}

/// Prove a local commutation. Tableau equality alone is insufficient: two
/// anticommuting Paulis have the same unsigned action but opposite scalar
/// phase.
bool commute(const Gate &g, const Gate &h, SearchBudget &budget) {
  if (!budget.take())
    return false;
  if (g.kind == Kind::Phase || h.kind == Kind::Phase)
    return true;
  if (!(support(g) & support(h)))
    return true;
  if (g.kind == Kind::Rotate && h.kind == Kind::Rotate)
    return !g.pauli.anti(h.pauli);
  if (g.kind == Kind::Rotate)
    return conjugate(g.pauli, h) == g.pauli;
  if (h.kind == Kind::Rotate)
    return conjugate(h.pauli, g) == h.pauli;
  auto bits = support(g) | support(h);
  std::vector<unsigned> qs;
  for (unsigned q : bits.set_bits())
    qs.push_back(q);
  auto pair = localize({g, h}, qs);
  return graph::equivalent(qs.size(), pair, {pair[1], pair[0]});
}

/// Return the exact Rz coefficient of a supported diagonal operation.
std::optional<int> diagonal(const Gate &g) {
  if (g.kind == Kind::S)
    return 2;
  if (g.kind == Kind::Sdg)
    return -2;
  if (g.kind == Kind::Z)
    return 4;
  if (g.kind == Kind::Rotate && !g.pauli.x && g.pauli.z.count() == 1)
    if (auto k = quarters(g.angle))
      return *k * (g.pauli.sign ? -1 : 1);
  return {};
}

/// Fold exact same-axis phases and inverse pairs across a checked commuting
/// interval. This works on native gates, without invoking another MLIR pass.
Circuit reduce(const Circuit &input, SearchBudget &budget, unsigned window) {
  Circuit out;
  for (auto g : input) {
    if (!budget.take())
      return input;
    if (g.kind == Kind::Phase) {
      out.push_back(g);
      continue;
    }
    bool matched = false;
    for (size_t back = 0; back < out.size() && (!window || back < window);
         ++back) {
      size_t i = out.size() - back - 1;
      const auto &h = out[i];
      // Named S/Sdg/Z and Rz are the same axis with different scalar
      // conventions. Fold them together before the primitive decomposition
      // separates their Clifford and odd-angle parts again.
      auto gPhase = diagonal(g), hPhase = diagonal(h);
      if (gPhase && hPhase && support(g) == support(h)) {
        unsigned q = support(g).first();
        int scalar = (g.kind == Kind::Rotate ? 0 : *gPhase) +
                     (h.kind == Kind::Rotate ? 0 : *hPhase);
        out[i] = Gate::rotation({0, QubitSet::single(q)},
                                {(*gPhase + *hPhase) % 16, 4});
        phase(out, scalar, 8);
        matched = true;
        break;
      }
      if (g.kind == Kind::Rotate && h.kind == Kind::Rotate &&
          g.pauli == h.pauli && quarters(g.angle) && quarters(h.angle)) {
        out[i].angle = {(*quarters(g.angle) + *quarters(h.angle)) % 16, 4};
        matched = true;
        break;
      }
      if (g.kind != Kind::Rotate && h.kind != Kind::Rotate &&
          h.kind == inverse(g).kind && h.a == g.a &&
          ((g.kind != Kind::CX && g.kind != Kind::CZ && g.kind != Kind::Swap) ||
           h.b == g.b)) {
        out.erase(out.begin() + i);
        matched = true;
        break;
      }
      if (!commute(g, h, budget))
        break;
    }
    if (!matched)
      out.push_back(g);
  }
  return out;
}

/// A signed tableau key for at most two wires. Each generator image occupies
/// five bits: X support, Z support and sign. Scalar phase is checked
/// separately.
using clifford_dictionary::advance;
using clifford_dictionary::buildDatabase;
using clifford_dictionary::Database;
using clifford_dictionary::identity;
using clifford_dictionary::Key;

/// Replace a contiguous small Clifford window by a shortest native-basis word.
/// BFS ignores scalar phase; exact replay recovers and checks the eighth-root
/// correction before accepting a replacement. No numerical tolerance is used.
Circuit windows(const Circuit &input, SearchBudget &budget,
                uint64_t *verificationNs, bool preserveEntanglers) {
  Circuit out;
  // Charge both fixed tables once per invocation, including warm-cache runs.
  if (!budget.take(24 * 6 + 11520 * 16))
    return input;
  for (size_t i = 0; i < input.size();) {
    if (!budget.take())
      return input;
    if (input[i].kind == Kind::Rotate || input[i].kind == Kind::Phase) {
      out.push_back(input[i++]);
      continue;
    }
    QubitSet bits;
    size_t j = i;
    Circuit scalars, chunk;
    while (j < input.size() && input[j].kind != Kind::Rotate &&
           (bits | support(input[j])).count() <= 2) {
      bits |= support(input[j]);
      (input[j].kind == Kind::Phase ? scalars : chunk).push_back(input[j++]);
    }
    std::vector<unsigned> qs;
    for (unsigned q : bits.set_bits())
      qs.push_back(q);
    unsigned n = qs.size();
    // Charge fixed bounded table construction for each use, even if its cache
    // is warm, keeping search decisions independent of process history.
    if (!budget.take(chunk.size() * 2 * n + 1))
      return input;
    static const Database one = buildDatabase(1);
    static const Database two = buildDatabase(2);
    const auto &db = n == 1 ? one : two;
    auto local = localize(chunk, qs);
    Key key = identity(n);
    for (auto g : local)
      key = advance(key, g, n);
    Circuit replacement;
    for (Key k = key; k != identity(n); k = db.at(k).previous)
      replacement.push_back(db.at(k).gate);
    std::reverse(replacement.begin(), replacement.end());
    bool accepted = false;
    auto entanglers = [](const Circuit &gates) {
      return std::count_if(gates.begin(), gates.end(), [](const Gate &g) {
        return g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap;
      });
    };
    // A shortest total-gate word can introduce extra entanglers. The bounded
    // retry keeps such a window intact while still improving other windows.
    if (replacement.size() < chunk.size() &&
        (!preserveEntanglers || entanglers(replacement) <= entanglers(chunk))) {
      for (int scalar = 0; scalar < 8; ++scalar) {
        if (!budget.take(1 + local.size() + replacement.size()))
          return input;
        auto corrected = replacement;
        phase(corrected, scalar);
        bool equivalent;
        {
          TelemetryTimer proof(verificationNs);
          equivalent = graph::equivalent(n, local, corrected);
        }
        if (equivalent) {
          for (auto g : corrected) {
            if (g.kind != Kind::Phase) {
              g.a = qs[g.a];
              if (g.kind == Kind::CX || g.kind == Kind::CZ ||
                  g.kind == Kind::Swap)
                g.b = qs[g.b];
            }
            out.push_back(g);
          }
          accepted = true;
          break;
        }
      }
    }
    if (accepted)
      out.insert(out.end(), scalars.begin(), scalars.end());
    else
      out.insert(out.end(), input.begin() + i, input.begin() + j);
    i = j;
  }
  return out;
}

/// Exact phase-polynomial signature: the linear output map and coefficients
/// of R_Z(parity) in pi/4 units modulo 16. Scalars remain explicit in the
/// block.
struct Signature {
  /// Linear parity on each output wire.
  std::vector<QubitSet> rows;
  /// Rotation coefficient modulo 16 for each input parity.
  std::unordered_map<Axis, int, AxisHash> coefficients;
};
/// Bound phase-placement blocks to linear reversible gates and exact phases.
bool phaseGate(const Gate &g) {
  return g.kind == Kind::CX || g.kind == Kind::Swap || g.kind == Kind::Phase ||
         diagonal(g).has_value();
}

/// Relocate each collected phase to an existing wire carrying its parity.
/// The CX/SWAP skeleton is unchanged. First/last placements expose different
/// neighboring cancellations without constructing a fresh parity network.
Circuit relocate(unsigned width, const Circuit &input, bool last,
                 SearchBudget &budget, uint64_t *verificationNs) {
  Circuit out;
  for (size_t i = 0; i < input.size();) {
    if (!phaseGate(input[i])) {
      out.push_back(input[i++]);
      continue;
    }
    size_t j = i;
    while (j < input.size() && phaseGate(input[j]))
      ++j;
    if (!budget.take(width + j - i))
      return input;
    Signature signature;
    for (unsigned q = 0; q < width; ++q)
      signature.rows.push_back(QubitSet::single(q));
    Circuit network, scalars;
    using Location = std::pair<size_t, unsigned>;
    std::unordered_map<Axis, Location, AxisHash> locations;
    auto record = [&](unsigned q) {
      Axis key{0, signature.rows[q]};
      if (last)
        locations[key] = {network.size(), q};
      else
        locations.try_emplace(key, network.size(), q);
    };
    for (unsigned q = 0; q < width; ++q)
      record(q);
    for (size_t t = i; t < j; ++t) {
      auto g = input[t];
      if (!budget.take(1 + width / 64))
        return input;
      if (g.kind == Kind::Phase)
        scalars.push_back(g);
      else if (auto k = diagonal(g)) {
        unsigned q = g.kind == Kind::Rotate ? g.pauli.z.first() : g.a;
        auto &coefficient = signature.coefficients[{0, signature.rows[q]}];
        coefficient = (coefficient + *k) % 16;
        if (g.kind != Kind::Rotate)
          phase(scalars, *k, 8);
      } else {
        // A row remains available until just before a gate changes it, not
        // merely at the point where it was first created. Record that final
        // boundary lazily so last placement does not scan all wires per CX.
        if (last) {
          record(g.b);
          if (g.kind == Kind::Swap)
            record(g.a);
        }
        network.push_back(g);
        if (g.kind == Kind::CX)
          signature.rows[g.b] ^= signature.rows[g.a];
        else
          std::swap(signature.rows[g.a], signature.rows[g.b]);
        record(g.b);
        if (g.kind == Kind::Swap)
          record(g.a);
      }
    }
    if (last)
      for (unsigned q = 0; q < width; ++q)
        record(q);
    std::map<Location, int> placements;
    for (const auto &[axis, k] : signature.coefficients)
      if (k)
        placements[locations.at(axis)] = k;
    Circuit rebuilt;
    for (size_t t = 0; t <= network.size(); ++t) {
      auto it = placements.lower_bound({t, 0});
      for (; it != placements.end() && it->first.first == t; ++it)
        rebuilt.push_back(Gate::rotation(
            {0, QubitSet::single(it->first.second)}, {it->second, 4}));
      if (t < network.size())
        rebuilt.push_back(network[t]);
    }
    // Replay the new network and coefficients rather than trusting the chosen
    // locations. Equality modulo 16 preserves rotation scalar signs as well.
    TelemetryTimer proof(verificationNs);
    std::vector<QubitSet> rows;
    for (unsigned q = 0; q < width; ++q)
      rows.push_back(QubitSet::single(q));
    auto remaining = signature.coefficients;
    for (const auto &g : rebuilt) {
      if (!budget.take(1 + width / 64))
        return input;
      if (g.kind == Kind::Rotate)
        remaining[{0, rows[g.pauli.z.first()]}] -= *quarters(g.angle);
      else if (g.kind == Kind::CX)
        rows[g.b] ^= rows[g.a];
      else
        std::swap(rows[g.a], rows[g.b]);
    }
    bool valid =
        rows == signature.rows &&
        std::all_of(remaining.begin(), remaining.end(),
                    [](const auto &item) { return item.second % 16 == 0; });
    if (!valid)
      return input;
    out.insert(out.end(), rebuilt.begin(), rebuilt.end());
    out.insert(out.end(), scalars.begin(), scalars.end());
    i = j;
  }
  return out;
}
} // namespace

std::vector<Gate> cudaq::pauli::gatecount::optimize(
    unsigned width, const Circuit &input, const Options &options,
    SearchBudget &budget, unsigned cap, Telemetry *telemetry, uint64_t parent,
    uint64_t scope, uint64_t *selected, uint64_t *verificationNs,
    uint64_t maxTwoQubitCount) {
  if (selected)
    *selected = 0;
  if (input.empty() || !valid(width, input) || input.size() > cap)
    return input;
  Circuit best = input;
  auto initial = tcount::cost(width, input);
  auto bestCost = initial;
  auto cleanup = [&](Circuit gates, uint64_t *proofTime,
                     bool preserveEntanglers) {
    gates = primitives(gates, budget, cap);
    gates = reduce(gates, budget, options.commutationWindow);
    gates = primitives(gates, budget, cap);
    if (options.cliffordWindows)
      gates = windows(gates, budget, proofTime, preserveEntanglers);
    return primitives(reduce(gates, budget, options.commutationWindow), budget,
                      cap);
  };
  for (unsigned round = 0; round < options.rounds && !budget.stopped();
       ++round) {
    auto previous = bestCost;
    for (unsigned method = 0; method < 5 && !budget.stopped(); ++method) {
      if ((method == 1 || method == 2) && !options.propagate)
        continue;
      if (method >= 3 && !options.phasePlacement)
        continue;
      TelemetryScope observation(telemetry,
                                 method == 0   ? "gate-count.clifford-cleanup"
                                 : method == 1 ? "gate-count.pauli-forward"
                                 : method == 2 ? "gate-count.pauli-reverse"
                                 : method == 3 ? "gate-count.phase-first"
                                               : "gate-count.phase-last",
                                 parent, scope);
      auto start = budget.work();
      if (telemetry) {
        observation.event.before = bestCost;
        observation.event.unresolvedBefore = unsynthesizedRotations(best);
      }
      Circuit candidate = primitives(best, budget, cap);
      if (method == 1)
        candidate = propagate(candidate, budget);
      if (method == 2)
        candidate = adjoint(propagate(adjoint(candidate), budget));
      if (method >= 3)
        candidate = relocate(width, candidate, method == 4, budget,
                             observation.verificationTime());
      auto cleaned = cleanup(candidate, observation.verificationTime(), false);
      auto cost = tcount::cost(width, cleaned);
      // Keep the unrestricted result whenever it satisfies the global ceiling:
      // local increases may be offset by cancellations elsewhere. Otherwise
      // retry this proposal with non-growing Clifford windows, rather than
      // losing all its independent Pauli/phase/window improvements at the
      // outer portfolio's acceptance boundary. Both paths share the budget.
      if (cost.twoQubit > maxTwoQubitCount && !budget.stopped()) {
        cleaned =
            cleanup(std::move(candidate), observation.verificationTime(), true);
        cost = tcount::cost(width, cleaned);
        if (telemetry)
          observation.event.counters = {{"entangler_safe_retry", 1}};
      }
      bool accept =
          !budget.stopped() && cleaned.size() <= cap && cost.t <= initial.t &&
          cost.twoQubit <= maxTwoQubitCount &&
          std::tie(cost.gates, cost.t, cost.depth, cost.twoQubit, cost.tDepth) <
              std::tie(bestCost.gates, bestCost.t, bestCost.depth,
                       bestCost.twoQubit, bestCost.tDepth);
      if (telemetry) {
        observation.event.after = cost;
        observation.event.unresolvedAfter = unsynthesizedRotations(cleaned);
        observation.event.work = budget.work() - start;
        observation.event.decision = accept ? "selected" : "rejected";
        observation.event.reason =
            budget.stopped() ? "work limit" : "total-gate ranking";
      }
      if (verificationNs)
        *verificationNs += observation.event.verificationNs;
      if (accept) {
        best = std::move(cleaned);
        bestCost = cost;
        parent = observation.event.id;
        if (selected)
          *selected = parent;
      }
    }
    if (std::tie(previous.gates, previous.t, previous.depth, previous.twoQubit,
                 previous.tDepth) == std::tie(bestCost.gates, bestCost.t,
                                              bestCost.depth, bestCost.twoQubit,
                                              bestCost.tDepth))
      break;
  }
  if (budget.stopped() && selected)
    *selected = 0;
  return budget.stopped() ? input : best;
}
