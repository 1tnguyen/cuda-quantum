/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/
#pragma once

#include "TCountInternal.h"
#include "cudaq/Optimizer/Pauli/Circuit.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <vector>

namespace cudaq::pauli::graph::detail::compact_transport {

// Stateless compact-wire algebra shared by source-only regional providers.
// These helpers neither establish a source certificate nor charge search work;
// callers reserve resources and validate operands before using them.
//
// exitImages requires a Clifford word valid on [0,width). transport requires a
// unique in-range physical map and its corresponding 3*physical.size() images,
// obtained from that actual exit. lift requires every used compact operand to
// index the physical map. Phase angles, signed axes, and opaque angles remain
// exact. A complete caller proof must bind the source/body/actual-exit
// relation; the image table alone does not record the exit's scalar phase.

inline uint64_t add(uint64_t a, uint64_t b) {
  return llvm::SaturatingAdd<uint64_t>(a, b);
}
inline uint64_t mul(uint64_t a, uint64_t b) {
  return llvm::SaturatingMultiply<uint64_t>(a, b);
}
inline bool sameWord(const std::vector<Gate> &a, const std::vector<Gate> &b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].kind != b[i].kind || a[i].a != b[i].a || a[i].b != b[i].b ||
        a[i].pauli != b[i].pauli || a[i].angle != b[i].angle)
      return false;
  return true;
}

/// Images of X, Z, Y on each compact input wire. Native exit phases remain in
/// the engine's quantum proof; conjugation needs its exact signed action only.
inline std::vector<Pauli> exitImages(unsigned width,
                                     const std::vector<Gate> &exit) {
  std::vector<Pauli> images;
  images.reserve(3 * width);
  for (unsigned q = 0; q < width; ++q)
    for (unsigned axis = 1; axis <= 3; ++axis) {
      Pauli image;
      image.x.set(q, axis & 1);
      image.z.set(q, axis & 2);
      for (const auto &gate : exit)
        image = tcount::detail::conjugate(image, gate);
      images.push_back(std::move(image));
    }
  return images;
}

/// Local factors from distinct input wires commute, so every product below is
/// Hermitian. Keep the original sign and all external factors independently;
/// treating the signed local restriction as unsigned would lose Y/sign data.
inline Pauli transport(Pauli input, const std::vector<unsigned> &physical,
                       const std::vector<Pauli> &images) {
  Pauli local;
  for (unsigned q = 0; q < physical.size(); ++q) {
    unsigned axis = input.axis(physical[q]);
    if (axis)
      local = tcount::detail::product(local, images[3 * q + axis - 1]);
    input.x.set(physical[q], false);
    input.z.set(physical[q], false);
  }
  input.sign ^= local.sign;
  for (unsigned q : local.x.set_bits())
    input.x.set(physical[q]);
  for (unsigned q : local.z.set_bits())
    input.z.set(physical[q]);
  return input;
}

inline Gate lift(Gate gate, const std::vector<unsigned> &physical) {
  if (gate.kind == Kind::Rotate) {
    Pauli axis{{}, {}, gate.pauli.sign};
    for (unsigned q : gate.pauli.x.set_bits())
      axis.x.set(physical[q]);
    for (unsigned q : gate.pauli.z.set_bits())
      axis.z.set(physical[q]);
    gate.pauli = std::move(axis);
  } else if (gate.kind != Kind::Phase) {
    gate.a = physical[gate.a];
    if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
        gate.kind == Kind::Swap)
      gate.b = physical[gate.b];
  }
  return gate;
}

} // namespace cudaq::pauli::graph::detail::compact_transport
