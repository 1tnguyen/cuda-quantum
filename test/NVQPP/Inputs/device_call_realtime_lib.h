/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define CUDAQ_DEVICE_CALL_TARGET __device__
#else
#define CUDAQ_DEVICE_CALL_TARGET
#endif

extern "C" CUDAQ_DEVICE_CALL_TARGET int addThem(int a, int b);
extern "C" CUDAQ_DEVICE_CALL_TARGET float multiplyFloats(float a, float b);
extern "C" CUDAQ_DEVICE_CALL_TARGET int countTrueBits(std::uint64_t seed,
                                                      const bool *bits,
                                                      std::uint64_t count,
                                                      std::uint64_t bias);
extern "C" CUDAQ_DEVICE_CALL_TARGET int
sumIntArray(const int *values, std::uint64_t count, int bias);
