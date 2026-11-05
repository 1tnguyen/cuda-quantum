/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisBootstrap.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Pauli/GraphSynthesis.h"
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

using namespace cudaq::pauli;
namespace synthesis = cudaq::pauli::graph;
namespace local = cudaq::pauli::graph::detail;

namespace {
using Complex = std::complex<double>;
using State = std::vector<Complex>;
constexpr Complex imaginary{0, 1};

double radians(Angle angle) {
  double value = angle.symbol ? 0.137 * angle.symbol + 0.071
                              : std::numbers::pi * double(angle.numerator) /
                                    double(angle.denominator);
  return angle.negative ? -value : value;
}

/// Independent direct Pauli action, including all signs and scalar phases.
/// Only small test widths enter this statevector oracle.
State simulate(unsigned width, const std::vector<Gate> &gates, State state) {
  for (const Gate &gate : gates) {
    if (gate.kind == Kind::Phase) {
      Complex phase = std::exp(imaginary * radians(gate.angle));
      for (auto &amplitude : state)
        amplitude *= phase;
      continue;
    }
    State next(state.size());
    if (gate.kind == Kind::Rotate) {
      double angle = radians(gate.angle);
      for (unsigned index = 0; index < state.size(); ++index) {
        unsigned target = index;
        Complex phase = gate.pauli.sign ? -1. : 1.;
        for (unsigned q = 0; q < width; ++q) {
          bool bit = (index >> q) & 1u;
          switch (gate.pauli.axis(q)) {
          case 1:
            target ^= 1u << q;
            break;
          case 2:
            phase *= bit ? -1. : 1.;
            break;
          case 3:
            target ^= 1u << q;
            phase *= bit ? -imaginary : imaginary;
            break;
          }
        }
        next[index] += std::cos(angle / 2) * state[index];
        next[target] -= imaginary * std::sin(angle / 2) * phase * state[index];
      }
    } else if (gate.kind == Kind::H) {
      unsigned mask = 1u << gate.a;
      for (unsigned index = 0; index < state.size(); ++index)
        if (!(index & mask)) {
          next[index] = (state[index] + state[index | mask]) / std::sqrt(2.);
          next[index | mask] =
              (state[index] - state[index | mask]) / std::sqrt(2.);
        }
    } else if (gate.kind == Kind::S || gate.kind == Kind::Sdg) {
      for (unsigned index = 0; index < state.size(); ++index)
        next[index] = state[index] *
                      ((index & (1u << gate.a))
                           ? (gate.kind == Kind::S ? imaginary : -imaginary)
                           : Complex{1, 0});
    } else if (gate.kind == Kind::CX) {
      for (unsigned index = 0; index < state.size(); ++index) {
        unsigned target =
            index ^ ((index & (1u << gate.a)) ? (1u << gate.b) : 0u);
        next[target] = state[index];
      }
    } else if (gate.kind == Kind::CZ) {
      for (unsigned index = 0; index < state.size(); ++index)
        next[index] =
            state[index] *
            ((index & (1u << gate.a)) && (index & (1u << gate.b)) ? -1. : 1.);
    } else {
      ADD_FAILURE() << "unsupported gate in bootstrap test simulation";
      return state;
    }
    state = std::move(next);
  }
  return state;
}

void sameUnitary(unsigned width, const std::vector<Gate> &input,
                 const std::vector<Gate> &output) {
  // Equality of every matrix column includes scalar phase and implies equality
  // for arbitrary inputs, including a system entangled with an external one.
  for (unsigned basis = 0; basis < (1u << width); ++basis) {
    State initial(1u << width);
    initial[basis] = 1;
    auto expected = simulate(width, input, initial);
    auto actual = simulate(width, output, initial);
    for (unsigned i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(actual[i].real(), expected[i].real(), 1e-12);
      EXPECT_NEAR(actual[i].imag(), expected[i].imag(), 1e-12);
    }
  }
}

local::BootstrapResult bootstrap(unsigned width, const std::vector<Gate> &input,
                                 const synthesis::Options &options,
                                 SearchBudget *budget = nullptr) {
  const std::vector<Pauli> observables{{0, 1}};
  const std::vector<unsigned> wires{0};
  return local::buildBootstrapPrefix({width, input, observables, wires},
                                     options, budget);
}
} // namespace

TEST(SynthesisBootstrap, EveryTwoWireSignedPauliPreservesTheFullUnitary) {
  for (unsigned profile = 0; profile < 3; ++profile) {
    synthesis::Options options;
    options.nativeCX = profile != 2;
    options.nativeCZ = profile != 1;
    options.nativeSwap = false;
    for (unsigned x = 0; x < 4; ++x)
      for (unsigned z = 0; z < 4; ++z) {
        if (!(x | z))
          continue;
        for (bool sign : {false, true})
          for (Angle angle :
               {Angle{1, 4}, Angle{-3, 8}, Angle{0, 1, 17, true}}) {
            const std::vector<Gate> input{Gate::rotation({x, z, sign}, angle)};
            auto result = bootstrap(2, input, options);
            ASSERT_TRUE(result) << result.error;
            sameUnitary(2, input, result.gates);
            for (const Gate &gate : result.gates) {
              if (!options.nativeCX)
                EXPECT_NE(gate.kind, Kind::CX);
              if (!options.nativeCZ)
                EXPECT_NE(gate.kind, Kind::CZ);
              EXPECT_NE(gate.kind, Kind::Swap);
              if (gate.kind == Kind::Rotate) {
                EXPECT_FALSE(bool(gate.pauli.x));
                EXPECT_EQ(gate.pauli.z.count(), 1u);
                EXPECT_FALSE(gate.pauli.sign);
              }
            }
          }
      }
  }
}

TEST(SynthesisBootstrap, NoncommutingWordAndObservablesRemainUnchanged) {
  const std::vector<Gate> input{
      Gate::rotation({1, 2, true}, {1, 4}), Gate::rotation({2, 3}, {0, 1, 31}),
      Gate::phase({1, 7}), Gate::rotation({0, 3}, {-3, 8})};
  const auto saved = input;
  const std::vector<Pauli> observables{{3, 0, true}, {0, 3}};
  const auto savedObservables = observables;
  const std::vector<unsigned> wires{1, 0};
  synthesis::Options options;
  auto result =
      local::buildBootstrapPrefix({2, input, observables, wires}, options);
  ASSERT_TRUE(result) << result.error;
  sameUnitary(2, input, result.gates);
  EXPECT_EQ(observables, savedObservables);
  for (unsigned i = 0; i < input.size(); ++i) {
    EXPECT_EQ(input[i].pauli, saved[i].pauli);
    EXPECT_EQ(input[i].angle, saved[i].angle);
    EXPECT_EQ(input[i].kind, saved[i].kind);
  }
}

TEST(SynthesisBootstrap, DynamicSupportAndOpaqueAnglesAreNotTruncated) {
  Pauli p{QubitSet::single(130), QubitSet::single(0) | QubitSet::single(130),
          true};
  Angle angle{0, 1, 71, true};
  const std::vector<Gate> input{Gate::rotation(p, angle)};
  synthesis::Options options;
  options.nativeCX = false;
  auto result = bootstrap(131, input, options);
  ASSERT_TRUE(result) << result.error;
  EXPECT_TRUE(synthesis::equivalent(131, input, result.gates));
  unsigned rotations = 0;
  bool touchesHighWire = false;
  for (const Gate &gate : result.gates) {
    touchesHighWire |= gate.a == 130 || gate.b == 130;
    if (gate.kind == Kind::Rotate) {
      ++rotations;
      EXPECT_EQ(gate.angle, angle.negated());
    }
  }
  EXPECT_EQ(rotations, 1u);
  EXPECT_TRUE(touchesHighWire);
}

TEST(SynthesisBootstrap, IdentityAndExplicitPhasesPreserveScalarPhase) {
  const std::vector<Gate> input{Gate::rotation({0, 0, false}, {1, 3}),
                                Gate::rotation({0, 0, true}, {-1, 5}),
                                Gate::phase({3, 7}),
                                Gate::phase({0, 1, 19, true}),
                                Gate::rotation({3, 2, true}, {0, 7}),
                                Gate::phase({0, 1})};
  synthesis::Options options;
  auto result = bootstrap(2, input, options);
  ASSERT_TRUE(result) << result.error;
  ASSERT_EQ(result.gates.size(), 4u);
  for (const Gate &gate : result.gates)
    EXPECT_EQ(gate.kind, Kind::Phase);
  sameUnitary(2, input, result.gates);
}

TEST(SynthesisBootstrap, IdentityHalfAnglesHandleIntegerLimitsExactly) {
  const int64_t maximum = std::numeric_limits<int64_t>::max();
  synthesis::Options options;
  for (Angle angle : {Angle{2, maximum}, Angle{3, maximum - 1},
                      Angle{std::numeric_limits<int64_t>::min(), maximum}}) {
    const std::vector<Gate> input{Gate::rotation({0, 0}, angle)};
    auto result = bootstrap(1, input, options);
    ASSERT_TRUE(result) << result.error;
    ASSERT_EQ(result.gates.size(), 1u);
    const auto phase = result.gates.front().angle;
    ASSERT_EQ(result.gates.front().kind, Kind::Phase);
    __int128 lhs = __int128(phase.numerator) * angle.denominator * 2;
    __int128 rhs = -__int128(angle.numerator) * phase.denominator;
    if (phase.negative)
      lhs = -lhs;
    EXPECT_TRUE(lhs == rhs);
  }
  for (Angle angle : {Angle{1, maximum}, Angle{0, 1, 37}}) {
    auto result = bootstrap(
        1, {Gate::phase({1, 8}), Gate::rotation({0, 0}, angle)}, options);
    EXPECT_EQ(result.status, local::BootstrapStatus::Unsupported);
    EXPECT_TRUE(result.gates.empty());
  }
}

TEST(SynthesisBootstrap, ExactGateCapsIncludeTheConfiguredNativeExpansion) {
  const std::vector<Gate> input{Gate::rotation({1, 3}, {1, 4})};
  synthesis::Options options;
  options.nativeCX = false;
  // Y basis/unbasis: four gates; two H-CZ-H parity blocks: six; Z rotation:
  // one.
  options.maxGates = 11;
  auto exact = bootstrap(2, input, options);
  ASSERT_TRUE(exact) << exact.error;
  EXPECT_EQ(exact.gates.size(), 11u);
  options.maxGates = 10;
  auto shortLimit = bootstrap(2, input, options);
  EXPECT_EQ(shortLimit.status, local::BootstrapStatus::Exhausted);
  EXPECT_TRUE(shortLimit.gates.empty());
  options.maxGates = 0;
  EXPECT_TRUE(bootstrap(2, {}, options));
  EXPECT_TRUE(bootstrap(2, {Gate::rotation({1, 3}, {0, 1})}, options));
  EXPECT_EQ(bootstrap(2, input, options).status,
            local::BootstrapStatus::Exhausted);
}

TEST(SynthesisBootstrap, EveryBudgetFailureDiscardsItsPartialPrefix) {
  const std::vector<Gate> input{Gate::phase({1, 8}),
                                Gate::rotation({1, 3}, {1, 4})};
  synthesis::Options options;
  options.nativeCX = false;
  SearchBudget complete(std::numeric_limits<uint64_t>::max());
  auto original = bootstrap(2, input, options, &complete);
  ASSERT_TRUE(original) << original.error;
  EXPECT_EQ(original.work, complete.work());
  for (uint64_t work = 0; work < complete.work(); ++work) {
    SearchBudget budget(work);
    auto failed = bootstrap(2, input, options, &budget);
    EXPECT_EQ(failed.status, local::BootstrapStatus::Exhausted);
    EXPECT_FALSE(failed.error.empty());
    EXPECT_TRUE(failed.gates.empty());
    EXPECT_TRUE(budget.stopped());
    EXPECT_LE(budget.work(), work);
  }
  SearchBudget exact(complete.work());
  auto result = bootstrap(2, input, options, &exact);
  ASSERT_TRUE(result) << result.error;
  EXPECT_EQ(result.gates.size(), original.gates.size());
  sameUnitary(2, input, result.gates);
}

TEST(SynthesisBootstrap, RejectsNonNormalizedAndMalformedInput) {
  synthesis::Options options;
  for (const std::vector<Gate> &input :
       {std::vector<Gate>{{Kind::H, 0}},
        std::vector<Gate>{Gate::rotation({4, 0}, {1, 4})},
        std::vector<Gate>{Gate::rotation({1, 0}, {1, 0})},
        std::vector<Gate>{Gate::phase({1, -1})}}) {
    auto result = bootstrap(2, input, options);
    EXPECT_EQ(result.status, local::BootstrapStatus::Unsupported);
    EXPECT_TRUE(result.gates.empty());
  }
  EXPECT_EQ(bootstrap(0, {}, options).status,
            local::BootstrapStatus::Unsupported);
  options.nativeCX = options.nativeCZ = false;
  EXPECT_EQ(bootstrap(2, {}, options).status,
            local::BootstrapStatus::Unsupported);
}
