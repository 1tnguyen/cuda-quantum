/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#if defined(_MSC_VER)
namespace cudaq {
class quantum_platform;
}
namespace nvqir {
class CircuitSimulator;
}
extern "C" {
cudaq::quantum_platform *getQuantumPlatform();
nvqir::CircuitSimulator *getCircuitSimulator();
}
#endif

int main(int argc, char **argv) {
#if defined(_MSC_VER)
  [[maybe_unused]] auto *platform = getQuantumPlatform();
  [[maybe_unused]] auto *simulator = getCircuitSimulator();
#endif
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
