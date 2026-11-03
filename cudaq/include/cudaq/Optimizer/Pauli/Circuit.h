/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
/// \file
/// \brief Native circuit representation for Pauli-based optimization.
///
/// Defines signed Pauli products, exact or opaque angles, and Clifford/rotation
/// operations shared by the native optimizers. The representation is
/// independent of MLIR and uses arbitrary-width LLVM integers for bit sets.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/iterator_range.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>

namespace cudaq::pauli {

/// Zero-extended qubit set backed by llvm::APInt. A single word stays inline;
/// larger supports grow on demand. Canonical trailing-zero-word removal makes
/// equality and hashing independent of allocation history. There is
/// deliberately no implicit conversion back to a machine-word mask.
class QubitSet {
  llvm::APInt bits{64, 0};
  /// Round to whole storage words without overflowing APInt's width type.
  static unsigned storageWidth(unsigned activeBits) {
    uint64_t rounded = (uint64_t(activeBits) + 63) & ~uint64_t(63);
    return std::min<uint64_t>(std::numeric_limits<unsigned>::max(),
                              std::max<uint64_t>(64, rounded));
  }
  void trim() {
    if (bits.getBitWidth() == 64)
      return;
    unsigned width = storageWidth(bits.getActiveBits());
    if (width < bits.getBitWidth())
      bits = bits.trunc(width);
  }
  unsigned next(unsigned from) const {
    unsigned width = bits.getBitWidth();
    if (from >= width)
      return width;
    unsigned word = from / 64;
    auto data = bits.getRawData();
    uint64_t pending = data[word] & (~uint64_t(0) << (from % 64));
    while (!pending && ++word < bits.getNumWords())
      pending = data[word];
    return pending ? word * 64 + std::countr_zero(pending) : width;
  }

public:
  QubitSet() = default;
  /// Convenience for literal masks and small-circuit tests, not a width bound.
  QubitSet(uint64_t mask) : bits(64, mask) {}
  static QubitSet single(unsigned q, bool value = true) {
    QubitSet result;
    result.set(q, value);
    return result;
  }
  bool test(unsigned q) const { return q < bits.getBitWidth() && bits[q]; }
  void set(unsigned q, bool value = true) {
    if (value) {
      assert(q < std::numeric_limits<unsigned>::max() &&
             "qubit index must fit in the unsigned circuit width");
      if (q >= bits.getBitWidth())
        bits = bits.zext(storageWidth(q + 1));
      bits.setBit(q);
    } else if (q < bits.getBitWidth()) {
      bits.clearBit(q);
      if (q / 64 + 1 == bits.getNumWords())
        trim();
    }
  }
  void flip(unsigned q, bool value = true) {
    if (value)
      set(q, !test(q));
  }
  explicit operator bool() const { return !bits.isZero(); }
  unsigned count() const { return bits.popcount(); }
  unsigned intersectionCount(const QubitSet &rhs) const {
    auto left = bits.getRawData(), right = rhs.bits.getRawData();
    unsigned count = 0;
    for (unsigned i = 0;
         i < std::min(bits.getNumWords(), rhs.bits.getNumWords()); ++i)
      count += std::popcount(left[i] & right[i]);
    return count;
  }
  unsigned unionCount(const QubitSet &rhs) const {
    return count() + rhs.count() - intersectionCount(rhs);
  }
  unsigned first() const {
    assert(!bits.isZero() && "first bit of an empty support");
    return bits.countr_zero();
  }
  bool fits(unsigned width) const { return bits.getActiveBits() <= width; }
  void resetFirst() { set(first(), false); }
  QubitSet withoutFirst() const {
    auto result = *this;
    if (result)
      result.resetFirst();
    return result;
  }
  /// Forward traversal skips zero words without allocating temporary masks.
  class SetBitIterator {
    const QubitSet *parent = nullptr;
    unsigned position = 0;

  public:
    using value_type = unsigned;
    using difference_type = std::ptrdiff_t;
    using reference = unsigned;
    using pointer = void;
    using iterator_category = std::forward_iterator_tag;
    SetBitIterator() = default;
    SetBitIterator(const QubitSet *parent, unsigned position)
        : parent(parent), position(position) {}
    unsigned operator*() const { return position; }
    SetBitIterator &operator++() {
      position = parent->next(position + 1);
      return *this;
    }
    SetBitIterator operator++(int) {
      auto result = *this;
      ++*this;
      return result;
    }
    bool operator==(const SetBitIterator &) const = default;
  };
  auto set_bits() const {
    return llvm::make_range(SetBitIterator(this, next(0)),
                            SetBitIterator(this, bits.getBitWidth()));
  }
  size_t hash() const {
    // Keep the old one-word key inexpensive. Canonical word count ensures that
    // growing a support and clearing its high bits restores the same hash.
    if (bits.getBitWidth() == 64)
      return static_cast<size_t>(bits.getZExtValue());
    auto words = bits.getRawData();
    return llvm::hash_combine_range(words, words + bits.getNumWords());
  }
  bool operator==(const QubitSet &rhs) const {
    return bits.getBitWidth() == rhs.bits.getBitWidth() && bits == rhs.bits;
  }
  QubitSet &operator^=(const QubitSet &rhs) {
    if (bits.getBitWidth() < rhs.bits.getBitWidth())
      bits = bits.zext(rhs.bits.getBitWidth());
    if (bits.getBitWidth() == rhs.bits.getBitWidth())
      bits ^= rhs.bits;
    else
      bits ^= rhs.bits.zext(bits.getBitWidth());
    trim();
    return *this;
  }
  QubitSet &operator|=(const QubitSet &rhs) {
    if (bits.getBitWidth() < rhs.bits.getBitWidth())
      bits = bits.zext(rhs.bits.getBitWidth());
    if (bits.getBitWidth() == rhs.bits.getBitWidth())
      bits |= rhs.bits;
    else
      bits |= rhs.bits.zext(bits.getBitWidth());
    return *this;
  }
  QubitSet &operator&=(const QubitSet &rhs) {
    if (bits.getBitWidth() > rhs.bits.getBitWidth())
      bits = bits.trunc(rhs.bits.getBitWidth());
    if (bits.getBitWidth() == rhs.bits.getBitWidth())
      bits &= rhs.bits;
    else
      bits &= rhs.bits.trunc(bits.getBitWidth());
    trim();
    return *this;
  }
  friend QubitSet operator^(QubitSet lhs, const QubitSet &rhs) {
    lhs ^= rhs;
    return lhs;
  }
  friend QubitSet operator|(QubitSet lhs, const QubitSet &rhs) {
    lhs |= rhs;
    return lhs;
  }
  friend QubitSet operator&(QubitSet lhs, const QubitSet &rhs) {
    lhs &= rhs;
    return lhs;
  }
};

inline int population(const QubitSet &bits) { return bits.count(); }
inline unsigned firstBit(const QubitSet &bits) { return bits.first(); }

/// Signed Hermitian Pauli product with dynamically sized X/Z support.
struct Pauli {
  /// Per-wire X/Z bits: (0,0)=I, (1,0)=X, (0,1)=Z, (1,1)=Y.
  QubitSet x, z;
  /// True multiplies the entire Hermitian product by -1.
  bool sign = false;
  unsigned axis(unsigned q) const;
  unsigned weight() const;
  bool anti(const Pauli &other) const;
  bool operator==(const Pauli &) const = default;
};

/// Native Clifford primitives, Pauli rotations, and explicit scalar phases.
enum class Kind { H, S, Sdg, X, Y, Z, CX, CZ, Swap, Rotate, Phase };

/// Rational multiple of pi or opaque symbol; floats are never guessed rational.
struct Angle {
  /// Rational coefficient numerator/denominator; denominator must be positive.
  int64_t numerator = 0, denominator = 1;
  /// Zero selects the rational coefficient; nonzero identifies an opaque angle.
  unsigned symbol = 0;
  /// Additional negation of either the rational coefficient or opaque symbol.
  bool negative = false;
  Angle negated() const;
  bool operator==(const Angle &) const = default;
};

/// Native operation: Rotate is exp(-i angle P/2), Phase is exp(i angle).
/// Only the operands selected by kind are meaningful.
struct Gate {
  /// Discriminator selecting a Clifford, Pauli rotation, or scalar phase.
  Kind kind;
  /// Clifford wire indices; a is the CX control, b its target/second operand.
  unsigned a = 0, b = 0;
  /// Signed rotation axis for Rotate; unused by Clifford and Phase operations.
  Pauli pauli = {};
  /// Rotation angle or scalar-phase exponent, in the Angle convention above.
  Angle angle = {};
  static Gate rotation(Pauli p, Angle angle);
  static Gate phase(Angle angle);
};

} // namespace cudaq::pauli
