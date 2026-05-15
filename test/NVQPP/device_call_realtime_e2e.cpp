/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: device-call-realtime-e2e-lib

// clang-format off
// This test intentionally mirrors device_call_realtime_bool_array_e2e.cpp.
// Both tests compile one CUDA-Q app and exercise shared-memory device_call
// dispatch through the same co-linked service library.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && nvq++ -frealtime-lowering --enable-mlir -save-temps -I%S/Inputs %s %cudaq_device_call_realtime_e2e_lib -o %t/app
// RUN: grep -R "__cudaq_device_call_acquire_realtime_frame" %t > /dev/null
// RUN: not grep -R "__cudaq_device_call_dispatch_to_device" %t
// RUN: not grep -R "define weak dso_local i32 @addThem" %t/device_call_realtime_e2e.ll
// RUN: not grep -R "define weak dso_local float @multiplyFloats" %t/device_call_realtime_e2e.ll
// RUN: LD_LIBRARY_PATH=%cudaq_lib_dir:${LD_LIBRARY_PATH} %t/app --cudaq-device-call=shared-memory | FileCheck --check-prefix=SHM %s
// clang-format on

#include <cudaq.h>

#include <cstdio>

#include "device_call_realtime_lib.h"

__qpu__ int kernel(int a, int b) {
  return cudaq::device_call(0, addThem, a, b);
}

__qpu__ float floatKernel(float a, float b) {
  return cudaq::device_call(0, multiplyFloats, a, b);
}

int main(int argc, char **argv) {
  cudaq::realtime::initialize(argc, argv);

  auto results = cudaq::run(1, kernel, 19, 23);
  int value = results.front();
  std::printf("device_call shared-memory int result = %d\n", value);

  auto floatResults = cudaq::run(1, floatKernel, 6.0f, 7.0f);
  float floatValue = floatResults.front();
  std::printf("device_call shared-memory float result = %.1f\n", floatValue);

  cudaq::realtime::finalize();
  return value == 42 && floatValue == 42.0f ? 0 : 1;
}

// SHM: device_call shared-memory int result = 42
// SHM: device_call shared-memory float result = 42.0
