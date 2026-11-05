/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Pauli/SynthesisCost.h"
#include "gtest/gtest.h"
#include <utility>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
namespace synthesis = cudaq::pauli::graph::detail;

TEST(SynthesisCost, NativeExpansionPreservesUnitaryAndProfileCosts) {
  const std::vector<Gate> input{{Kind::Swap, 0, 1}, {Kind::CZ, 0, 1}};
  Options options;
  options.singleQubitCost = 2;
  options.cxCost = 3;
  options.czCost = 5;
  options.swapCost = 7;
  for (unsigned profile = 0; profile < 3; ++profile) {
    options.nativeCX = profile != 2;
    options.nativeCZ = profile != 1;
    options.nativeSwap = profile == 0;
    auto native = synthesis::nativeWord(input, options);
    ASSERT_TRUE(equivalent(2, input, native));
    for (const auto &gate : native) {
      if (!options.nativeSwap)
        EXPECT_NE(gate.kind, Kind::Swap);
      if (!options.nativeCX)
        EXPECT_NE(gate.kind, Kind::CX);
      if (!options.nativeCZ)
        EXPECT_NE(gate.kind, Kind::CZ);
    }
    // Native: SWAP+CZ. CX-only: four CX and two H. CZ-only:
    // three H-CZ-H implementations of CX, followed by one native CZ.
    double expected[] = {12, 16, 32};
    auto cost = synthesis::evaluateNativeCost(input, options);
    EXPECT_TRUE(cost.exact);
    EXPECT_DOUBLE_EQ(cost.value, expected[profile]);
    EXPECT_DOUBLE_EQ(cost.value, nativeGateCost(native, options));
    EXPECT_DOUBLE_EQ(cost.value, nativeGateCost(input, options));
  }
}

TEST(SynthesisCost, MetricsAndObjectiveOrderingKeepDistinctNativeWeights) {
  const std::vector<Gate> gates{{Kind::H, 0},
                                Gate::rotation({0, 1}, {1, 4}),
                                {Kind::CX, 0, 1},
                                Gate::rotation({0, 2}, {-1, 4})};
  Options options;
  options.singleQubitCost = 2;
  options.tCost = 3;
  options.cxCost = 5;
  auto cost = synthesis::evaluateCircuitCost(2, gates, options);
  EXPECT_TRUE(cost.native.exact);
  EXPECT_DOUBLE_EQ(cost.native.value, 13);
  EXPECT_EQ(cost.metrics.gates, 4u);
  EXPECT_EQ(cost.metrics.t, 2u);
  EXPECT_EQ(cost.metrics.twoQubit, 1u);
  EXPECT_EQ(cost.metrics.depth, 4u);
  EXPECT_EQ(cost.metrics.tDepth, 2u);
  const std::pair<SynthesisObjective, synthesis::CostRank> expected[] = {
      {SynthesisObjective::TotalGates, {13, 2, 4, 1, 2}},
      {SynthesisObjective::TCount, {2, 13, 4, 1, 2}},
      {SynthesisObjective::Entanglers, {1, 13, 2, 4, 2}},
      {SynthesisObjective::Depth, {4, 13, 2, 1, 2}},
      {SynthesisObjective::TDepth, {2, 2, 13, 4, 1}}};
  for (const auto &[objective, rank] : expected) {
    options.objective = objective;
    EXPECT_EQ(synthesis::circuitScore(2, gates, options), rank);
  }
}

TEST(SynthesisCost, UnresolvedRotationProxyIsExplicitAndUnchanged) {
  Options options;
  options.tCost = 7;
  for (Angle angle : {Angle{1, 3}, Angle{0, 1, 42}}) {
    std::vector<Gate> gates{Gate::rotation({0, 1}, angle)};
    auto cost = synthesis::evaluateNativeCost(gates, options);
    EXPECT_FALSE(cost.exact);
    EXPECT_DOUBLE_EQ(cost.value, 7);
    EXPECT_DOUBLE_EQ(nativeGateCost(gates, options), cost.value);
    auto emitted = synthesis::nativeWord(gates, options);
    ASSERT_EQ(emitted.size(), 1u);
    EXPECT_EQ(emitted.front().kind, Kind::Rotate);
    EXPECT_EQ(emitted.front().angle, angle);
    EXPECT_EQ(emitted.front().pauli, gates.front().pauli);
  }
  // A scalar phase needs no native gate, including an opaque scalar phase.
  auto phase =
      synthesis::evaluateNativeCost({Gate::phase({0, 1, 42})}, options);
  EXPECT_TRUE(phase.exact);
  EXPECT_DOUBLE_EQ(phase.value, 0);
}

TEST(SynthesisCost, DynamicSupportsAndExactRotationCosts) {
  Options options;
  options.singleQubitCost = 2;
  options.tCost = 3;
  options.cxCost = 5;
  Pauli yAndZ{QubitSet::single(0), QubitSet::single(0) | QubitSet::single(128)};
  std::vector<Gate> input{Gate::rotation(yAndZ, {-3, 4})};
  // Y basis: four one-qubit gates; S+T: two gates; compute/uncompute:
  // two CX. Dynamic support must count wire 128 as a distinct wire.
  auto cost = synthesis::evaluateNativeCost(input, options);
  EXPECT_TRUE(cost.exact);
  EXPECT_DOUBLE_EQ(cost.value, 4 * 2 + 2 + 3 + 2 * 5);
  EXPECT_DOUBLE_EQ(nativeGateCost(input, options), cost.value);
}
