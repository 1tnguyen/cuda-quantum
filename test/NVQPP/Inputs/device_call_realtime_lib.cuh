/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "device_call_realtime_lib.h"

extern "C" CUDAQ_DEVICE_CALL_TARGET int addThem(int a, int b) { return a + b; }

extern "C" CUDAQ_DEVICE_CALL_TARGET float multiplyFloats(float a, float b) {
  return a * b;
}

extern "C" CUDAQ_DEVICE_CALL_TARGET int countTrueBits(std::uint64_t seed,
                                                      const bool *bits,
                                                      std::uint64_t count,
                                                      std::uint64_t bias) {
  int total = 0;
  for (std::uint64_t i = 0; i < count; ++i)
    total += bits[i] ? 1 : 0;
  return static_cast<int>(seed + bias + total);
}

extern "C" CUDAQ_DEVICE_CALL_TARGET int
sumIntArray(const int *values, std::uint64_t count, int bias) {
  int total = bias;
  for (std::uint64_t i = 0; i < count; ++i)
    total += values[i];
  return total;
}
