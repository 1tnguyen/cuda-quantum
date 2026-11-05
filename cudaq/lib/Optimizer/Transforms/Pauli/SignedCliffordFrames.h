/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "TCountInternal.h"
#include "cudaq/Optimizer/Pauli/SearchBudget.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::pauli::graph::detail {

/// Signed images of X,Y,Z, modulo scalar phase. Unlike Pauli::axis, these
/// coordinates use 1=X, 2=Y, 3=Z. Keep this representation local to the DP.
using LocalFrame = std::array<int, 3>;
inline int image(LocalFrame f, int p) {
  return f[std::abs(p) - 1] * (p > 0 ? 1 : -1);
}
inline LocalFrame compose(LocalFrame a, LocalFrame b) {
  return {image(a, b[0]), image(a, b[1]), image(a, b[2])};
}
inline LocalFrame inverseFrame(LocalFrame f) {
  LocalFrame inverse{};
  for (int i = 0; i != 3; ++i)
    inverse[std::abs(f[i]) - 1] = (i + 1) * (f[i] > 0 ? 1 : -1);
  return inverse;
}
inline std::optional<LocalFrame> localClifford(Kind k) {
  switch (k) {
  case Kind::H:
    return LocalFrame{3, -2, 1};
  case Kind::S:
    return LocalFrame{2, -1, 3};
  case Kind::Sdg:
    return LocalFrame{-2, 1, 3};
  case Kind::X:
    return LocalFrame{1, -2, -3};
  case Kind::Y:
    return LocalFrame{-1, 2, -3};
  case Kind::Z:
    return LocalFrame{-1, -2, 3};
  default:
    return std::nullopt;
  }
}
inline std::optional<int> localT(const Gate &g) {
  if (g.kind != Kind::Rotate || g.pauli.x || g.pauli.z.count() != 1)
    return std::nullopt;
  auto quarters = tcount::detail::quarters(g.angle);
  if (!quarters)
    return std::nullopt;
  int k = *quarters * (g.pauli.sign ? -1 : 1);
  k = ((k % 8 + 12) % 8) - 4;
  return std::abs(k) == 1 ? std::optional<int>(k) : std::nullopt;
}

/// Exact shortest native Clifford words and their multiplication table.
/// All named one-qubit Cliffords have the same configured primitive cost.
struct LocalCliffords {
  std::vector<LocalFrame> frames{{1, 2, 3}};
  std::vector<std::vector<Kind>> words{{}};
  std::array<std::array<unsigned, 24>, 24> product{};
  std::array<unsigned, 24> inverse{};
  unsigned s, sdg;
  unsigned index(LocalFrame frame) const {
    auto it = std::find(frames.begin(), frames.end(), frame);
    assert(it != frames.end());
    return unsigned(it - frames.begin());
  }
  LocalCliffords() {
    for (unsigned i = 0; i < frames.size(); ++i)
      for (Kind g : {Kind::H, Kind::S, Kind::Sdg, Kind::X, Kind::Y, Kind::Z}) {
        auto next = compose(*localClifford(g), frames[i]);
        if (std::find(frames.begin(), frames.end(), next) != frames.end())
          continue;
        auto word = words[i];
        word.push_back(g);
        frames.push_back(next);
        words.push_back(std::move(word));
      }
    assert(frames.size() == 24);
    for (unsigned i = 0; i < 24; ++i) {
      inverse[i] = index(inverseFrame(frames[i]));
      for (unsigned j = 0; j < 24; ++j)
        product[i][j] = index(compose(frames[i], frames[j]));
    }
    s = index(*localClifford(Kind::S));
    sdg = index(*localClifford(Kind::Sdg));
  }
};

/// Independently replay the actual emitted gates, rather than the DP's frame
/// witness. The SO(3) channel has entries (a+b*sqrt(2))/2^T with exact APInts.
/// Equality of all rows proves unitary equality up to scalar phase; equality
/// of row Z proves equality of the terminal Z outcome effects. T counts must
/// agree, so denominators agree. At most 512 gates enter a region; 2*T+8 bits
/// safely bound the unnormalized integer coefficients throughout this replay.
inline bool verifyLocalFrames(const std::vector<Gate> &before,
                              const std::vector<Gate> &after, unsigned rows) {
  auto countT = [](const auto &gates) {
    return std::count_if(gates.begin(), gates.end(),
                         [](const Gate &g) { return g.kind == Kind::Rotate; });
  };
  unsigned t = countT(before);
  if (t != countT(after))
    return false;
  struct Entry {
    llvm::APInt a, b;
    bool operator==(const Entry &) const = default;
  };
  auto channel = [&](const std::vector<Gate> &gates) {
    unsigned bits = 2 * t + 8;
    std::array<Entry, 9> matrix;
    for (unsigned i = 0; i < 9; ++i)
      matrix[i] = {llvm::APInt(bits, i % 4 == 0), llvm::APInt(bits, 0)};
    for (const Gate &g : gates) {
      auto old = matrix;
      if (auto sign = localT(g)) {
        for (unsigned j = 0; j < 3; ++j) {
          // Left-multiply by Rz(+/-pi/4) and increase the common denominator
          // by two. Multiplication by sqrt(2) maps (a,b) to (2*b,a).
          auto xa =
              *sign > 0 ? old[j].a - old[3 + j].a : old[j].a + old[3 + j].a;
          auto xb =
              *sign > 0 ? old[j].b - old[3 + j].b : old[j].b + old[3 + j].b;
          auto ya =
              *sign > 0 ? old[j].a + old[3 + j].a : old[3 + j].a - old[j].a;
          auto yb =
              *sign > 0 ? old[j].b + old[3 + j].b : old[3 + j].b - old[j].b;
          matrix[j] = {xb.shl(1), xa};
          matrix[3 + j] = {yb.shl(1), ya};
          matrix[6 + j] = {old[6 + j].a.shl(1), old[6 + j].b.shl(1)};
        }
      } else {
        auto frame = localClifford(g.kind);
        assert(frame && "unsupported gate in local frame proof");
        for (unsigned i = 0; i < 3; ++i)
          for (unsigned j = 0; j < 3; ++j) {
            auto entry = old[3 * i + j];
            if ((*frame)[i] < 0)
              entry = {-entry.a, -entry.b};
            matrix[3 * (std::abs((*frame)[i]) - 1) + j] = std::move(entry);
          }
      }
    }
    return matrix;
  };
  auto a = channel(before), b = channel(after);
  for (unsigned row = 0; row < 3; ++row)
    if (rows & (1u << row))
      for (unsigned col = 0; col < 3; ++col)
        if (!(a[3 * row + col] == b[3 * row + col]))
          return false;
  return true;
}

/// Optimize a bounded one-wire region using linear-space backpointers. The
/// input is C R(P_k)...R(P_1). At each step choose a residual frame F' mapping
/// P to a*Z, a=+/-1, and an emitted T sign s=+/-1. The Clifford prefix is
/// D = S^((a-s)/2) F' F^-1, so T^s D F = F' R(P), up to global phase.
/// Allowing both signs is essential: fixing s=a misses the T=S*Tdg freedom.
/// This minimizes this representation family, not all Clifford+T circuits.
inline std::optional<std::vector<Gate>>
signedFrameWord(const std::vector<Gate> &word, unsigned q, unsigned rows,
                SearchBudget *budget, std::string &error) {
  static const LocalCliffords group;
  if (budget && !budget->take(word.size() + 1))
    return std::nullopt;
  unsigned c = 0;
  std::vector<int> axes;
  for (const Gate &g : word) {
    if (auto clifford = localClifford(g.kind))
      c = group.product[group.index(*clifford)][c];
    else
      axes.push_back(image(group.frames[group.inverse[c]], 3) * *localT(g));
  }
  constexpr unsigned infinity = std::numeric_limits<unsigned>::max() / 2;
  std::array<unsigned, 24> costs;
  costs.fill(infinity);
  costs[0] = 0;
  struct Back {
    unsigned previous = 0, prefix = 0;
    int sign = 1;
  };
  std::vector<std::array<Back, 24>> history;
  for (int axis : axes) {
    if (budget && !budget->take(24 * 24 * 2))
      return std::nullopt;
    std::array<unsigned, 24> next;
    next.fill(infinity);
    std::array<Back, 24> back;
    for (unsigned f = 0; f < 24; ++f) {
      int z = image(group.frames[f], axis);
      if (std::abs(z) != 3)
        continue;
      int a = z > 0 ? 1 : -1;
      for (int sign : {1, -1}) {
        unsigned correction = a == sign ? 0 : (a > sign ? group.s : group.sdg);
        for (unsigned prev = 0; prev < 24; ++prev) {
          unsigned prefix =
              group.product[correction][group.product[f][group.inverse[prev]]];
          unsigned cost = costs[prev] + group.words[prefix].size() + 1;
          if (cost < next[f]) {
            next[f] = cost;
            back[f] = {prev, prefix, sign};
          }
        }
      }
    }
    costs = next;
    history.push_back(back);
  }
  unsigned best = infinity, state = 0, finalPrefix = 0;
  for (unsigned target = 0; target < 24; ++target) {
    // Full channel inside the circuit; only the specified observable at a
    // terminal region. Never exchange the sign of a physical Z readout.
    if ((rows == 7 && target != c) ||
        (rows == 4 && image(group.frames[group.inverse[target]], 3) !=
                          image(group.frames[group.inverse[c]], 3)))
      continue;
    for (unsigned f = 0; f < 24; ++f) {
      unsigned prefix = group.product[target][group.inverse[f]];
      unsigned cost = costs[f] + group.words[prefix].size();
      if (cost < best) {
        best = cost;
        state = f;
        finalPrefix = prefix;
      }
    }
  }
  if (best >= word.size())
    return word;
  std::vector<Gate> out;
  auto reversePrefix = [&](unsigned prefix) {
    const auto &gates = group.words[prefix];
    for (auto it = gates.rbegin(); it != gates.rend(); ++it)
      out.push_back({*it, q});
  };
  reversePrefix(finalPrefix);
  for (auto it = history.rbegin(); it != history.rend(); ++it) {
    Back step = (*it)[state];
    out.push_back(Gate::rotation({0, QubitSet::single(q)}, {step.sign, 4}));
    reversePrefix(step.prefix);
    state = step.previous;
  }
  std::reverse(out.begin(), out.end());
  // Replay cost accounts for the bounded APInt word width, independently of
  // the DP's constant-size frame arithmetic.
  unsigned limbs = (2 * axes.size() + 8 + 63) / 64;
  if (budget && !budget->take((word.size() + out.size()) * limbs * 54 + 1))
    return std::nullopt;
  if (!verifyLocalFrames(word, out, rows)) {
    error = "signed Clifford frame independent channel replay failed";
    return std::nullopt;
  }
  return out;
}

/// Reconstruct local emission regions without changing the entangler network.
/// Internal cuts preserve the full channel; only final cuts may use terminal
/// measurement freedom. Unsupported angles remain exact region boundaries.
/// Failure to finish within budget leaves the caller's verified circuit intact.
inline std::optional<std::vector<Gate>>
optimizeLocalSignedFrames(unsigned width, const std::vector<Gate> &input,
                          const std::vector<unsigned> &readoutWires,
                          SearchBudget *budget, std::string &error) {
  if (budget && !budget->take(width + input.size() + 1))
    return std::nullopt;
  std::vector<std::vector<Gate>> pending(width);
  std::vector<bool> measured(width, false);
  for (unsigned q : readoutWires)
    measured[q] = true;
  std::vector<Gate> out;
  auto flush = [&](unsigned q, unsigned rows) {
    if (pending[q].empty())
      return true;
    auto word = signedFrameWord(pending[q], q, rows, budget, error);
    if (!word)
      return false;
    out.insert(out.end(), word->begin(), word->end());
    pending[q].clear();
    return true;
  };
  for (const Gate &g : input) {
    if (localClifford(g.kind) || localT(g)) {
      unsigned q = g.kind == Kind::Rotate ? g.pauli.z.first() : g.a;
      pending[q].push_back(g);
      if (pending[q].size() == 512 && !flush(q, 7))
        return std::nullopt;
      continue;
    }
    if (g.kind == Kind::CX || g.kind == Kind::CZ || g.kind == Kind::Swap) {
      if (!flush(g.a, 7) || !flush(g.b, 7))
        return std::nullopt;
    } else if (g.kind != Kind::Phase) {
      for (unsigned q = 0; q < width; ++q)
        if (!flush(q, 7))
          return std::nullopt;
    }
    out.push_back(g);
  }
  for (unsigned q = 0; q < width; ++q)
    if (!flush(q, measured[q] ? 4 : 0))
      return std::nullopt;
  return out;
}

/// A local Clifford moves across the interval between from/to in the current
/// word after earlier moves. SWAP crossings may change its physical wire.
struct PauliTransportMove {
  size_t from, to;
};

/// Construction uses local-frame invariants of the entangler. The independent
/// checker below replays signed X/Z generators instead of trusting these rules.
inline bool mayTransportLocalClifford(Gate &moving, const Gate &crossed) {
  auto frame = localClifford(moving.kind);
  if (!frame)
    return false;
  unsigned q = moving.a;
  if (crossed.kind == Kind::Phase)
    return true;
  if (crossed.kind == Kind::Rotate) {
    unsigned axis = crossed.pauli.axis(q);
    if (!axis)
      return true;
    int localAxis = axis == 1 ? 1 : axis == 2 ? 3 : 2;
    return image(*frame, localAxis) == localAxis;
  }
  if (crossed.kind == Kind::CX)
    return (q != crossed.a && q != crossed.b) ||
           (q == crossed.a && image(*frame, 3) == 3) ||
           (q == crossed.b && image(*frame, 1) == 1);
  if (crossed.kind == Kind::CZ)
    return (q != crossed.a && q != crossed.b) || image(*frame, 3) == 3;
  if (crossed.kind == Kind::Swap) {
    if (q == crossed.a)
      moving.a = crossed.b;
    else if (q == crossed.b)
      moving.a = crossed.a;
    return true;
  }
  return localClifford(crossed.kind) && crossed.a != q;
}

/// Check a full-channel identity for one adjacent interchange. A rotation may
/// be exact or opaque: preserving its signed Pauli axis proves commutation
/// without evaluating or approximating its angle. Primitive Clifford pairs are
/// compared on every touched signed X/Z generator, which fixes their channel
/// on arbitrary inputs, including externally entangled inputs.
inline bool verifyLocalCliffordInterchange(const Gate &moving,
                                           const Gate &crossed,
                                           const Gate &relocated,
                                           bool forward) {
  if (!localClifford(moving.kind) || relocated.kind != moving.kind)
    return false;
  if (crossed.kind == Kind::Rotate)
    return moving.a == relocated.a &&
           tcount::detail::conjugate(crossed.pauli, moving) == crossed.pauli;
  if (crossed.kind == Kind::Phase)
    return moving.a == relocated.a;
  std::vector<unsigned> wires{moving.a, relocated.a, crossed.a};
  if (crossed.kind == Kind::CX || crossed.kind == Kind::CZ ||
      crossed.kind == Kind::Swap)
    wires.push_back(crossed.b);
  std::sort(wires.begin(), wires.end());
  wires.erase(std::unique(wires.begin(), wires.end()), wires.end());
  auto through = [](Pauli axis, const Gate &first, const Gate &second) {
    return tcount::detail::conjugate(tcount::detail::conjugate(axis, first),
                                     second);
  };
  for (unsigned q : wires)
    for (bool z : {false, true}) {
      Pauli generator{QubitSet::single(q, !z), QubitSet::single(q, z)};
      auto before = forward ? through(generator, moving, crossed)
                            : through(generator, crossed, moving);
      auto after = forward ? through(generator, crossed, relocated)
                           : through(generator, relocated, crossed);
      if (before != after)
        return false;
    }
  return true;
}

/// Replay the actual complete reordered word, independently of the search's
/// compatibility rules and local-frame arithmetic. No gate is removed here;
/// later one-wire refinement has its own full/terminal channel proof.
inline bool verifyPauliTransport(unsigned width,
                                 const std::vector<Gate> &before,
                                 const std::vector<Gate> &after,
                                 const std::vector<PauliTransportMove> &moves,
                                 SearchBudget *budget, std::string &error) {
  if (budget &&
      !budget->take(uint64_t(width) + before.size() + after.size() + 1))
    return false;
  auto fail = [&] {
    error = "local Clifford transport independent full-channel replay failed";
    return false;
  };
  if (before.size() != after.size() || !tcount::detail::valid(width, before))
    return fail();
  auto replay = before;
  for (const auto &move : moves) {
    if (move.from >= replay.size() || move.to >= replay.size() ||
        move.to == move.from || !localClifford(replay[move.from].kind))
      return fail();
    bool forward = move.to > move.from;
    size_t index = move.from;
    while (index != move.to) {
      if (budget && !budget->take(16 * (uint64_t(width) / 64 + 1) + 1))
        return false;
      size_t neighbor = forward ? index + 1 : index - 1;
      Gate moving = replay[index], crossed = replay[neighbor],
           relocated = moving;
      // Infer the physical wire independently from the actual SWAP operands.
      if (crossed.kind == Kind::Swap) {
        if (relocated.a == crossed.a)
          relocated.a = crossed.b;
        else if (relocated.a == crossed.b)
          relocated.a = crossed.a;
      }
      if (!verifyLocalCliffordInterchange(moving, crossed, relocated, forward))
        return fail();
      replay[index] = crossed;
      replay[neighbor] = relocated;
      index = neighbor;
    }
  }
  if (!std::equal(replay.begin(), replay.end(), after.begin(),
                  [](const Gate &a, const Gate &b) {
                    return a.kind == b.kind && a.a == b.a && a.b == b.b &&
                           a.pauli == b.pauli && a.angle == b.angle;
                  }))
    return fail();
  return true;
}

inline std::optional<std::vector<Gate>>
transportLocalCliffords(unsigned width, const std::vector<Gate> &input,
                        bool forward, SearchBudget *budget,
                        std::string &error) {
  if (budget && !budget->take(uint64_t(width) + input.size() + 1))
    return std::nullopt;
  auto gates = input;
  std::vector<PauliTransportMove> moves;
  for (size_t ordinal = 0; ordinal < gates.size(); ++ordinal) {
    size_t from = forward ? gates.size() - 1 - ordinal : ordinal;
    if (budget && !budget->take())
      return std::nullopt;
    if (!localClifford(gates[from].kind))
      continue;
    Gate moving = gates[from];
    size_t to = from;
    bool crossedBoundary = false, joinsRegion = false;
    while (forward ? to + 1 < gates.size() : to > 0) {
      if (budget && !budget->take(uint64_t(width) / 64 + 1))
        return std::nullopt;
      size_t neighbor = forward ? to + 1 : to - 1;
      const auto &gate = gates[neighbor];
      unsigned q = moving.a;
      if ((localClifford(gate.kind) && gate.a == q) ||
          (localT(gate) && gate.pauli.z.first() == q)) {
        joinsRegion = true;
        break;
      }
      Gate relocated = moving;
      if (!mayTransportLocalClifford(relocated, gate))
        break;
      crossedBoundary |= ((gate.kind == Kind::CX || gate.kind == Kind::CZ ||
                           gate.kind == Kind::Swap) &&
                          (gate.a == q || gate.b == q)) ||
                         (gate.kind == Kind::Rotate && gate.pauli.axis(q));
      moving = relocated;
      to = neighbor;
    }
    bool terminal = forward && to + 1 == gates.size();
    if (!crossedBoundary || (!joinsRegion && !terminal) || to == from)
      continue;
    moves.push_back({from, to});
    if (forward)
      std::rotate(gates.begin() + from, gates.begin() + from + 1,
                  gates.begin() + to + 1);
    else
      std::rotate(gates.begin() + to, gates.begin() + from,
                  gates.begin() + from + 1);
    gates[to] = moving;
  }
  if (moves.empty())
    return input;
  if (!verifyPauliTransport(width, input, gates, moves, budget, error))
    return std::nullopt;
  return gates;
}

/// Original primitive-only provider, retained as an independent first
/// candidate. Its construction and proof charges intentionally match the
/// earlier policy.
inline bool isPrimitivePauli(const Gate &gate) {
  return gate.kind == Kind::X || gate.kind == Kind::Y || gate.kind == Kind::Z;
}

/// Construction rule only. Stop at a one-wire supported region so the exact
/// local solver can fuse the correction. T/opaque rotations that anticommute
/// are barriers; this provider does not change their angles to cross them.
inline bool mayTransportPauli(const Gate &pauli, const Gate &crossed) {
  unsigned q = pauli.a;
  if (crossed.kind == Kind::Phase)
    return true;
  if (crossed.kind == Kind::Rotate) {
    unsigned axis = crossed.pauli.axis(q);
    return !axis || (pauli.kind == Kind::X && axis == 1) ||
           (pauli.kind == Kind::Y && axis == 3) ||
           (pauli.kind == Kind::Z && axis == 2);
  }
  if (crossed.kind == Kind::CX)
    return (q != crossed.a && q != crossed.b) ||
           (q == crossed.a && pauli.kind == Kind::Z) ||
           (q == crossed.b && pauli.kind == Kind::X);
  if (crossed.kind == Kind::CZ)
    return (q != crossed.a && q != crossed.b) || pauli.kind == Kind::Z;
  if (crossed.kind == Kind::Swap)
    return q != crossed.a && q != crossed.b;
  return localClifford(crossed.kind) && crossed.a != q;
}

/// Independently establish the full-channel relation for the emitted reorder.
/// Each swap checks the actual signed Pauli under Clifford conjugation, or
/// exact Pauli commutation for an arbitrary-angle rotation. This does not use
/// the construction compatibility table or a claimed transformed frame.
inline bool
verifyPrimitivePauliTransport(unsigned width, const std::vector<Gate> &before,
                              const std::vector<Gate> &after,
                              const std::vector<PauliTransportMove> &moves,
                              SearchBudget *budget, std::string &error) {
  if (budget &&
      !budget->take(uint64_t(width) + before.size() + after.size() + 1))
    return false;
  auto fail = [&] {
    error = "Pauli transport independent full-channel replay failed";
    return false;
  };
  if (before.size() != after.size() || !tcount::detail::valid(width, before))
    return fail();
  auto replay = before;
  for (const auto &move : moves) {
    if (move.from >= replay.size() || move.to >= move.from ||
        !isPrimitivePauli(replay[move.from]))
      return fail();
    Gate correction = replay[move.from];
    Pauli axis{QubitSet::single(correction.a, correction.kind != Kind::Z),
               QubitSet::single(correction.a, correction.kind != Kind::X)};
    for (size_t index = move.from; index > move.to; --index) {
      if (budget && !budget->take(uint64_t(width) / 64 + 1))
        return false;
      const auto &crossed = replay[index - 1];
      if (crossed.kind == Kind::Rotate) {
        if (axis.anti(crossed.pauli))
          return fail();
      } else if (tcount::detail::conjugate(axis, crossed) != axis)
        return fail();
      std::swap(replay[index], replay[index - 1]);
    }
  }
  if (!std::equal(replay.begin(), replay.end(), after.begin(),
                  [](const Gate &a, const Gate &b) {
                    return a.kind == b.kind && a.a == b.a && a.b == b.b &&
                           a.pauli == b.pauli && a.angle == b.angle;
                  }))
    return fail();
  return true;
}

inline std::optional<std::vector<Gate>>
transportPaulis(unsigned width, const std::vector<Gate> &input,
                SearchBudget *budget, std::string &error) {
  if (budget && !budget->take(uint64_t(width) + input.size() + 1))
    return std::nullopt;
  auto gates = input;
  std::vector<PauliTransportMove> moves;
  for (size_t from = 0; from < gates.size(); ++from) {
    if (budget && !budget->take())
      return std::nullopt;
    if (!isPrimitivePauli(gates[from]))
      continue;
    unsigned q = gates[from].a;
    size_t to = from;
    bool crossedBoundary = false, joinsRegion = false;
    while (to > 0) {
      if (budget && !budget->take(uint64_t(width) / 64 + 1))
        return std::nullopt;
      const auto &gate = gates[to - 1];
      if ((localClifford(gate.kind) && gate.a == q) ||
          (localT(gate) && gate.pauli.z.first() == q)) {
        joinsRegion = true;
        break;
      }
      if (!mayTransportPauli(gates[from], gate))
        break;
      crossedBoundary |= ((gate.kind == Kind::CX || gate.kind == Kind::CZ) &&
                          (gate.a == q || gate.b == q)) ||
                         (gate.kind == Kind::Rotate && gate.pauli.axis(q));
      --to;
    }
    if (!crossedBoundary || !joinsRegion || to == from)
      continue;
    moves.push_back({from, to});
    std::rotate(gates.begin() + to, gates.begin() + from,
                gates.begin() + from + 1);
  }
  if (moves.empty())
    return input;
  if (!verifyPrimitivePauliTransport(width, input, gates, moves, budget, error))
    return std::nullopt;
  return gates;
}

/// Compare complete routes without discarding depth improvements at equal gate
/// count. Each route retains the historical strict-gate guard internally.
inline bool signedFrameRouteDominates(const tcount::Cost &candidate,
                                      const tcount::Cost &incumbent) {
  return candidate.gates <= incumbent.gates && candidate.t <= incumbent.t &&
         candidate.twoQubit <= incumbent.twoQubit &&
         candidate.depth <= incumbent.depth &&
         candidate.tDepth <= incumbent.tDepth && candidate != incumbent;
}

/// Refine an immutable local-DP seed along one historical route. A stopped
/// child returns that route's last independently checked complete candidate;
/// proof failure remains fatal. Cache only costs already charged by this route.
inline std::optional<std::vector<Gate>>
refineSignedFrameRoute(unsigned width, const std::vector<Gate> &seed,
                       const std::vector<unsigned> &readoutWires,
                       bool primitiveFirst, SearchBudget &budget,
                       std::optional<tcount::Cost> &completedCost,
                       std::string &error) {
  auto best = seed;
  completedCost.reset();
  for (unsigned stage = primitiveFirst ? 0 : 1; stage < 3; ++stage) {
    auto transported =
        stage == 0
            ? transportPaulis(width, best, &budget, error)
            : transportLocalCliffords(width, best, stage == 2, &budget, error);
    if (!transported)
      return error.empty() ? std::optional(best) : std::nullopt;
    bool changed = !std::equal(best.begin(), best.end(), transported->begin(),
                               [](const Gate &a, const Gate &b) {
                                 return a.kind == b.kind && a.a == b.a &&
                                        a.b == b.b && a.pauli == b.pauli &&
                                        a.angle == b.angle;
                               });
    if (!changed)
      continue;
    auto candidate = optimizeLocalSignedFrames(width, *transported,
                                               readoutWires, &budget, error);
    if (!candidate)
      return error.empty() ? std::optional(best) : std::nullopt;
    uint64_t costWork = uint64_t(width) + 1;
    for (const auto *word : {&best, &*candidate})
      for (const auto &gate : *word)
        costWork = llvm::SaturatingAdd<uint64_t>(
            costWork, gate.kind == Kind::Rotate
                          ? 6 * uint64_t(gate.pauli.weight()) + 1
                          : 1);
    if (!budget.take(costWork))
      return best;
    auto before = tcount::cost(width, best),
         after = tcount::cost(width, *candidate);
    if (after.gates < before.gates && after.t <= before.t &&
        after.twoQubit <= before.twoQubit && after.depth <= before.depth &&
        after.tDepth <= before.tDepth) {
      best = std::move(*candidate);
      completedCost = after;
    }
  }
  return best;
}

/// Preserve both historical generators: the primitive-first route and the
/// original generic backward/forward route start from the same immutable local
/// DP result. Their independently bounded children cannot mutate each other's
/// input or spend each other's 10M allowance when sufficient work is available.
inline std::optional<std::vector<Gate>>
optimizeSignedFrames(unsigned width, const std::vector<Gate> &input,
                     const std::vector<unsigned> &readoutWires,
                     SearchBudget *budget, std::string &error) {
  auto seed =
      optimizeLocalSignedFrames(width, input, readoutWires, budget, error);
  if (!seed)
    return std::nullopt;
  constexpr uint64_t routeWorkLimit = 10000000;
  SearchBudget standalone(2 * routeWorkLimit);
  std::optional<SearchBudget> envelope;
  SearchBudget *optionalBudget = &standalone;
  if (budget) {
    uint64_t available = std::min(2 * routeWorkLimit, budget->remaining() / 2);
    auto remainingTime = budget->timeRemainingMs();
    if (!available || (remainingTime && *remainingTime < 2))
      return seed;
    envelope.emplace(*budget, available,
                     remainingTime ? *remainingTime / 2 : 0);
    optionalBudget = &*envelope;
  }
  auto best = *seed;
  std::optional<tcount::Cost> bestCost;
  for (bool primitiveFirst : {true, false}) {
    uint64_t available = std::min(routeWorkLimit, optionalBudget->remaining());
    auto remainingTime = optionalBudget->timeRemainingMs();
    if (!available || (remainingTime && *remainingTime < 2))
      break;
    // Reserve half the optional wall time for the independent generic route.
    SearchBudget child(
        *optionalBudget, available,
        remainingTime ? (primitiveFirst ? *remainingTime / 2 : *remainingTime)
                      : 0);
    std::optional<tcount::Cost> routeCost;
    auto candidate = refineSignedFrameRoute(
        width, *seed, readoutWires, primitiveFirst, child, routeCost, error);
    if (!candidate)
      return std::nullopt;
    // A route with no accepted improvement is the original seed. Otherwise its
    // cached cost includes the complete verification/comparison charge, even
    // when a later stage exhausted its child allowance.
    if (routeCost &&
        (!bestCost || signedFrameRouteDominates(*routeCost, *bestCost))) {
      best = std::move(*candidate);
      bestCost = routeCost;
    }
  }
  return best;
}
} // namespace cudaq::pauli::graph::detail
