/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * Licensed under the Apache License, Version 2.0.                             *
 ******************************************************************************/

#include "Pauli/SynthesisJointPhasePlanner.h"
#include "gtest/gtest.h"
#include <limits>

using namespace cudaq::pauli;
using namespace cudaq::pauli::graph;
using namespace cudaq::pauli::graph::detail;
namespace jp = cudaq::pauli::graph::detail::joint_plan_detail;

namespace {
void sameCost(const Gate &gate, const Options &options, bool exact = true) {
  const double expected = evaluateNativeCost({gate}, options).value;
  const double actual = jp::rotationCost(gate, options);
  if (exact)
    EXPECT_EQ(actual, expected);
  else
    EXPECT_NEAR(actual, expected, 1e-12 * std::max(1., std::abs(expected)));
}
Options fractionalOptions(bool cx) {
  Options options;
  options.singleQubitCost = .17;
  options.tCost = 1.3;
  options.cxCost = 2.7;
  options.czCost = .91;
  options.swapCost = 3.4;
  options.nativeCX = cx;
  options.nativeCZ = true;
  return options;
}
} // namespace

TEST(SynthesisJointPhaseCost, QuarterResiduesSignedAxesAndNativeProfilesAgree) {
  for (bool nativeCX : {false, true}) {
    Options options;
    options.nativeCX = nativeCX;
    options.nativeCZ = true;
    for (unsigned x = 0; x < 8; ++x)
      for (unsigned z = 0; z < 8; ++z) {
        if (!(x | z))
          continue;
        for (int numerator = -17; numerator <= 17; ++numerator)
          for (bool sign : {false, true})
            for (bool negative : {false, true})
              sameCost(
                  Gate::rotation({x, z, sign}, {numerator, 4, 0, negative}),
                  options);
      }
  }
}

TEST(SynthesisJointPhaseCost, OpaqueAndNonQuarterAnglesKeepTheCanonicalProxy) {
  const std::vector<Angle> angles{
      {1, 3},
      {-7, 11},
      {1, 3, 0, true},
      {0, 1, 29},
      {0, 1, 29, true},
      {std::numeric_limits<int64_t>::max(), 3},
      {std::numeric_limits<int64_t>::min(), 5, 0, true}};
  for (bool nativeCX : {false, true}) {
    Options options;
    options.nativeCX = nativeCX;
    options.nativeCZ = true;
    for (Angle angle : angles)
      for (bool sign : {false, true})
        sameCost(Gate::rotation({5, 6, sign}, angle), options);
  }
}

TEST(SynthesisJointPhaseCost, FractionalWeightsDifferOnlyByRoundingOrder) {
  for (bool nativeCX : {false, true}) {
    auto options = fractionalOptions(nativeCX);
    for (unsigned x = 0; x < 16; ++x)
      for (unsigned z = 0; z < 16; ++z)
        for (Angle angle : {Angle{3, 4}, Angle{-7, 4, 0, true}, Angle{5, 9},
                            Angle{0, 1, 17, true}})
          sameCost(Gate::rotation({x, z, true}, angle), options, false);
  }
}

TEST(SynthesisJointPhaseCost, ZeroAndFullTurnRetainExistingBasisParityCost) {
  Options options;
  for (Angle angle : {Angle{0, 1}, Angle{2, 1}, Angle{-2, 1}, Angle{4, 1}}) {
    auto rotation = Gate::rotation({3, 2}, angle);
    sameCost(rotation, options);
    // This is the existing evaluator's mechanical estimate, not simplification
    // of an identity rotation: one X, one Y and a parity compute/uncompute.
    EXPECT_EQ(jp::rotationCost(rotation, options), 8.);
  }
  auto scalar = Gate::phase({0, 1, 23, true});
  scalar.pauli = {QubitSet::single(4095), QubitSet::single(2048), true};
  sameCost(scalar, options);
  EXPECT_EQ(jp::rotationCost(scalar, options), 0.);
}

TEST(SynthesisJointPhaseCost, WideSparseAndDenseSupportsCountEveryYTwice) {
  Pauli sparse{{}, {}, true};
  for (unsigned q : {0u, 63u, 64u, 130u, 4095u})
    sparse.x.set(q);
  for (unsigned q : {0u, 64u, 129u, 4095u})
    sparse.z.set(q);
  Pauli dense;
  for (unsigned q = 0; q < 192; ++q) {
    dense.x.set(q);
    dense.z.set(q);
  }
  for (bool nativeCX : {false, true}) {
    Options integer;
    integer.nativeCX = nativeCX;
    integer.nativeCZ = true;
    for (const Pauli &axis : {sparse, dense}) {
      sameCost(Gate::rotation(axis, {-3, 4}), integer);
      sameCost(Gate::rotation(axis, {0, 1, 19}), fractionalOptions(nativeCX),
               false);
    }
  }
}

TEST(SynthesisJointPhaseCost, FixedWireReadoutRecognizesSignedLocalAxes) {
  Options options;
  options.singleQubitCost = .5;
  EXPECT_EQ(jp::readoutPotential({0, 8}, 3, options), 0.);
  EXPECT_EQ(jp::readoutPotential({0, 8, true}, 3, options), .5);
  EXPECT_EQ(jp::readoutPotential({8, 0}, 3, options), .5);
  EXPECT_EQ(jp::readoutPotential({8, 8}, 3, options), 1.);
  EXPECT_EQ(jp::readoutPotential({8, 8, true}, 3, options), 1.5);
  // Changing only the required physical identity incurs the routing estimate.
  EXPECT_EQ(jp::readoutPotential({0, 8}, 4, options), options.swapCost);
}

TEST(SynthesisJointPhaseCost, ReadoutEstimateRespectsCZAndSwapLowering) {
  Options options;
  options.nativeCX = false;
  options.nativeCZ = true;
  options.singleQubitCost = .4;
  options.czCost = 1.7;
  options.swapCost = 3.1;
  const Pauli row{5, 6}; // X0, Z1, Y2.
  EXPECT_NEAR(jp::readoutPotential(row, 2, options), 6.2, 1e-12);
  EXPECT_NEAR(jp::readoutPotential(row, 4, options), 9.3, 1e-12);
  options.nativeSwap = false;
  EXPECT_NEAR(jp::readoutPotential(row, 4, options), 13.7, 1e-12);
  auto signedRow = row;
  signedRow.sign = true;
  EXPECT_NEAR(jp::readoutPotential(signedRow, 4, options) -
                  jp::readoutPotential(row, 4, options),
              options.singleQubitCost, 1e-12);
  // These assertions concern the documented independent-row heuristic, not
  // an optimality claim or a certificate for a realizable joint readout.
}

TEST(SynthesisJointPhaseCost, TouchesUsesBothMasksAndPhysicalWireIdentities) {
  const Pauli row{QubitSet::single(130),
                  QubitSet::single(63) | QubitSet::single(130), true};
  EXPECT_TRUE(jp::touches(row, {0, 63}));
  EXPECT_TRUE(jp::touches(row, {130}));
  EXPECT_FALSE(jp::touches(row, {0, 1, 64, 129}));
  EXPECT_FALSE(jp::touches(row, {}));
  EXPECT_FALSE(jp::touches(Pauli{}, {63, 130}));
  auto unsignedRow = row;
  unsignedRow.sign = false;
  EXPECT_EQ(jp::touches(row, {63}), jp::touches(unsignedRow, {63}));
}
