/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Basic operations on native Pauli circuit values.
///
/// Implements support queries, symplectic commutation, angle negation, and gate
/// construction. These utilities operate on the shared representation without
/// introducing dependencies on a particular optimization strategy or MLIR.
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Pauli/Circuit.h"
#include <bit>
#include <limits>

using namespace cudaq::pauli;

unsigned Pauli::axis(unsigned q) const {
  return unsigned(x.test(q)) | (unsigned(z.test(q)) << 1);
}
unsigned Pauli::weight() const { return x.unionCount(z); }
bool Pauli::anti(const Pauli &p) const {
  return (x.intersectionCount(p.z) + z.intersectionCount(p.x)) & 1;
}
Angle Angle::negated() const {
  Angle a = *this;
  if (a.symbol)
    a.negative = !a.negative;
  else if (a.numerator != std::numeric_limits<int64_t>::min())
    a.numerator = -a.numerator;
  else
    a.negative = !a.negative;
  return a;
}
Gate Gate::rotation(Pauli p, Angle a) {
  return {Kind::Rotate, 0, 0, std::move(p), a};
}
Gate Gate::phase(Angle a) { return {Kind::Phase, 0, 0, {}, a}; }
