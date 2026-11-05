/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code is licensed under the Apache License 2.0.                  *
 ******************************************************************************/
#include "Pauli/SynthesisCompactTransport.h"
#include "gtest/gtest.h"
#include <complex>
#include <limits>
#include <numbers>

using namespace cudaq::pauli;
namespace transport = cudaq::pauli::graph::detail::compact_transport;

namespace {
using Complex = std::complex<double>;
using Dense = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

Dense pauliAction(const Pauli &word, const Dense &state) {
  Dense output(state.size());
  for (size_t basis = 0; basis < state.size(); ++basis) {
    size_t target = basis;
    Complex phase = word.sign ? -1. : 1.;
    for (unsigned q = 0; (size_t(1) << q) < state.size(); ++q) {
      bool bit = (basis >> q) & 1;
      switch (word.axis(q)) {
      case 1:
        target ^= size_t(1) << q;
        break;
      case 2:
        phase *= bit ? -1. : 1.;
        break;
      case 3:
        target ^= size_t(1) << q;
        phase *= bit ? -imaginary : imaginary;
        break;
      }
    }
    output[target] = phase * state[basis];
  }
  return output;
}

// Direct primitive matrices, independent of production signed-Pauli algebra.
Dense apply(const Gate &gate, Dense state, bool inverse = false) {
  if (gate.kind == Kind::Rotate || gate.kind == Kind::Phase) {
    double angle = gate.angle.symbol ? .173 * gate.angle.symbol
                                     : std::numbers::pi * gate.angle.numerator /
                                           gate.angle.denominator;
    if (gate.angle.negative != inverse)
      angle = -angle;
    if (gate.kind == Kind::Phase) {
      for (auto &amplitude : state)
        amplitude *= std::exp(imaginary * angle);
    } else {
      auto product = pauliAction(gate.pauli, state);
      for (size_t i = 0; i < state.size(); ++i)
        state[i] = std::cos(angle / 2) * state[i] -
                   imaginary * std::sin(angle / 2) * product[i];
    }
  } else if (gate.kind == Kind::CX || gate.kind == Kind::CZ ||
             gate.kind == Kind::Swap) {
    auto input = state;
    for (size_t basis = 0; basis < state.size(); ++basis) {
      bool a = (basis >> gate.a) & 1, b = (basis >> gate.b) & 1;
      if (gate.kind == Kind::CX)
        state[basis ^ (size_t(a) << gate.b)] = input[basis];
      else if (gate.kind == Kind::CZ)
        state[basis] = a && b ? -input[basis] : input[basis];
      else
        state[a == b
                  ? basis
                  : basis ^ ((size_t(1) << gate.a) | (size_t(1) << gate.b))] =
            input[basis];
    }
  } else if (gate.kind == Kind::S || gate.kind == Kind::Sdg) {
    Complex phase =
        (gate.kind == Kind::Sdg) != inverse ? -imaginary : imaginary;
    for (size_t basis = 0; basis < state.size(); ++basis)
      if ((basis >> gate.a) & 1)
        state[basis] *= phase;
  } else {
    Pauli x{QubitSet::single(gate.a), {}}, z{{}, QubitSet::single(gate.a)};
    if (gate.kind == Kind::H) {
      auto a = pauliAction(x, state), b = pauliAction(z, state);
      for (size_t i = 0; i < state.size(); ++i)
        state[i] = (a[i] + b[i]) / std::sqrt(2.);
    } else if (gate.kind == Kind::X)
      state = pauliAction(x, state);
    else if (gate.kind == Kind::Z)
      state = pauliAction(z, state);
    else if (gate.kind == Kind::Y)
      state = pauliAction({x.x, z.z}, state);
    else
      ADD_FAILURE() << "unsupported native primitive";
  }
  return state;
}

} // namespace

TEST(SynthesisCompactTransport, CompactSignedTableauMatchesEverySmallPauli) {
  const std::vector<Gate> exit{
      {Kind::H, 0},   {Kind::S, 1},       {Kind::CX, 1, 0}, {Kind::CZ, 2, 0},
      {Kind::Sdg, 2}, {Kind::Swap, 0, 2}, {Kind::X, 1}};
  const auto images = transport::exitImages(3, exit);
  const std::vector<unsigned> physical{3, 0, 2};
  std::vector<Gate> lifted;
  for (const auto &gate : exit)
    lifted.push_back(transport::lift(gate, physical));
  for (unsigned packed = 0; packed < 128; ++packed) {
    Pauli input{{}, {}, bool(packed & 64)};
    for (unsigned q = 0; q < 3; ++q) {
      unsigned axis = (packed >> (2 * q)) & 3;
      input.x.set(physical[q], axis & 1);
      input.z.set(physical[q], axis & 2);
    }
    input.x.set(1); // Untouched Y factor exercises external support and phase.
    input.z.set(1);
    auto image = transport::transport(input, physical, images);
    for (unsigned column = 0; column < 16; ++column) {
      Dense state(16);
      state[column] = 1;
      auto expected = state;
      for (auto gate = lifted.rbegin(); gate != lifted.rend(); ++gate)
        expected = apply(*gate, std::move(expected), true);
      expected = pauliAction(input, expected);
      for (const auto &gate : lifted)
        expected = apply(gate, std::move(expected));
      auto actual = pauliAction(image, state);
      for (unsigned row = 0; row < 16; ++row)
        EXPECT_NEAR(std::abs(expected[row] - actual[row]), 0, 4e-12);
    }
  }
  const std::vector<unsigned> high{130, 0, 65};
  for (unsigned packed = 0; packed < 128; ++packed) {
    Pauli input{{}, {}, bool(packed & 64)};
    for (unsigned q = 0; q < 3; ++q) {
      input.x.set(high[q], (packed >> (2 * q)) & 1);
      input.z.set(high[q], (packed >> (2 * q)) & 2);
    }
    input.x.set(64);
    auto expected = input;
    for (const auto &gate : exit)
      expected =
          tcount::detail::conjugate(expected, transport::lift(gate, high));
    EXPECT_EQ(transport::transport(input, high, images), expected);
  }
}

TEST(SynthesisCompactTransport, ResourceArithmeticSaturatesWithoutWrapping) {
  constexpr auto max = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(transport::add(19, 23), 42u);
  EXPECT_EQ(transport::add(max, 0), max);
  EXPECT_EQ(transport::add(max, 1), max);
  EXPECT_EQ(transport::mul(6, 7), 42u);
  EXPECT_EQ(transport::mul(max, 0), 0u);
  EXPECT_EQ(transport::mul(max, 1), max);
  EXPECT_EQ(transport::mul(max / 2 + 1, 2), max);
}

TEST(SynthesisCompactTransport, WordIdentityIncludesSignsAnglesAndOperands) {
  const std::vector<Gate> word{
      Gate::rotation({QubitSet::single(130), QubitSet::single(65), true},
                     {0, 1, 7, true}),
      Gate::phase({3, 8}),
      {Kind::CX, 130, 65}};
  EXPECT_TRUE(transport::sameWord(word, word));
  auto changed = word;
  changed[0].pauli.sign = false;
  EXPECT_FALSE(transport::sameWord(word, changed));
  changed = word;
  changed[0].angle.negative = false;
  EXPECT_FALSE(transport::sameWord(word, changed));
  changed = word;
  changed[0].angle.symbol = 8;
  EXPECT_FALSE(transport::sameWord(word, changed));
  changed = word;
  changed[1].angle.numerator = 1;
  EXPECT_FALSE(transport::sameWord(word, changed));
  changed = word;
  changed[2].b = 64;
  EXPECT_FALSE(transport::sameWord(word, changed));
  changed = word;
  changed[2].kind = Kind::CZ;
  EXPECT_FALSE(transport::sameWord(word, changed));
  changed.pop_back();
  EXPECT_FALSE(transport::sameWord(word, changed));
}

TEST(SynthesisCompactTransport,
     EmptyAndIdentityExitsKeepExternalSignedSupport) {
  const Pauli input{QubitSet::single(4095) | QubitSet::single(65),
                    QubitSet::single(4095) | QubitSet::single(7), true};
  EXPECT_TRUE(transport::exitImages(0, {}).empty());
  EXPECT_EQ(transport::transport(input, {}, {}), input);
  const std::vector<unsigned> physical{65, 4095, 7};
  EXPECT_EQ(transport::transport(input, physical, transport::exitImages(3, {})),
            input);
  EXPECT_TRUE(input.sign);
  EXPECT_TRUE(input.x.test(4095));
}

TEST(SynthesisCompactTransport, LiftPreservesOpaqueSignedAxesAndScalarPhase) {
  const std::vector<unsigned> physical{130, 2, 65};
  const Gate rotation = Gate::rotation({5, 6, true}, {0, 1, 11, true});
  auto lifted = transport::lift(rotation, physical);
  EXPECT_EQ(lifted.pauli.x, QubitSet::single(130) | QubitSet::single(65));
  EXPECT_EQ(lifted.pauli.z, QubitSet::single(2) | QubitSet::single(65));
  EXPECT_TRUE(lifted.pauli.sign);
  EXPECT_EQ(lifted.angle, rotation.angle);
  auto phase = Gate::phase({-3, 8});
  EXPECT_TRUE(transport::sameWord({phase}, {transport::lift(phase, physical)}));
  for (Kind kind : {Kind::H, Kind::S, Kind::Sdg, Kind::X, Kind::Y, Kind::Z}) {
    auto gate = transport::lift({kind, 1}, physical);
    EXPECT_EQ(gate.kind, kind);
    EXPECT_EQ(gate.a, 2u);
  }
  for (Kind kind : {Kind::CX, Kind::CZ, Kind::Swap}) {
    auto gate = transport::lift({kind, 0, 2}, physical);
    EXPECT_EQ(gate.kind, kind);
    EXPECT_EQ(gate.a, 130u);
    EXPECT_EQ(gate.b, 65u);
  }
}

TEST(SynthesisCompactTransport, GateLiftMatchesIndependentBasisPermutation) {
  const std::vector<unsigned> physical{2, 0};
  const std::vector<Gate> compact{Gate::phase({3, 8}),
                                  {Kind::S, 0},
                                  {Kind::H, 1},
                                  {Kind::CX, 1, 0},
                                  Gate::rotation({3, 1, true}, {0, 1, 5, true}),
                                  {Kind::CZ, 0, 1},
                                  {Kind::Swap, 0, 1},
                                  {Kind::Sdg, 1}};
  std::vector<Gate> lifted;
  for (const auto &gate : compact)
    lifted.push_back(transport::lift(gate, physical));
  for (unsigned external = 0; external < 2; ++external)
    for (unsigned column = 0; column < 4; ++column) {
      Dense local(4);
      local[column] = 1;
      for (const auto &gate : compact)
        local = apply(gate, std::move(local));
      Dense global(8);
      unsigned inputIndex =
          (external << 1) | ((column & 1) << 2) | ((column >> 1) & 1);
      global[inputIndex] = 1;
      for (const auto &gate : lifted)
        global = apply(gate, std::move(global));
      for (unsigned row = 0; row < 8; ++row) {
        unsigned localRow = ((row >> 2) & 1) | ((row & 1) << 1);
        Complex expected =
            ((row >> 1) & 1) == external ? local[localRow] : Complex{};
        EXPECT_NEAR(std::abs(global[row] - expected), 0, 4e-12);
      }
    }
}
