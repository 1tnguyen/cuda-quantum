/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: device-call-realtime-e2e-lib

// clang-format off
// This test intentionally mirrors device_call_realtime_e2e.cpp. Both tests
// compile one CUDA-Q app and exercise shared-memory device_call dispatch
// through the same co-linked service library.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && nvq++ -frealtime-lowering --enable-mlir -save-temps -I%S/Inputs %s %cudaq_device_call_realtime_e2e_lib -o %t/app
// RUN: grep -R "__cudaq_device_call_acquire_realtime_frame" %t > /dev/null
// RUN: not grep -R "__cudaq_device_call_dispatch_to_device" %t
// RUN: not grep -R "define weak dso_local i32 @countTrueBits" %t/device_call_realtime_bool_array_e2e.ll
// RUN: not grep -R "define weak dso_local i32 @sumIntArray" %t/device_call_realtime_bool_array_e2e.ll
// RUN: LD_LIBRARY_PATH=%cudaq_lib_dir:${LD_LIBRARY_PATH} %t/app --cudaq-device-call=shared-memory | FileCheck --check-prefix=SHM %s
// clang-format on

#include <cudaq.h>

#include <cstdint>
#include <cstdio>

#include "device_call_realtime_lib.h"

__qpu__ int boolArrayKernel() {
  bool bits[6] = {true, false, true, true, false, true};
  return cudaq::device_call(0, countTrueBits, std::uint64_t{10}, bits,
                            std::uint64_t{6}, std::uint64_t{5});
}

__qpu__ int intArrayKernel() {
  int values[4] = {3, 4, 5, 6};
  return cudaq::device_call(0, sumIntArray, values, std::uint64_t{4}, 24);
}

int main(int argc, char **argv) {
  cudaq::realtime::initialize(argc, argv);

  auto boolResults = cudaq::run(1, boolArrayKernel);
  int boolValue = boolResults.front();
  std::printf("device_call shared-memory bool flat-array result = %d\n",
              boolValue);

  int intValue = intArrayKernel();
  std::printf("device_call shared-memory i32 flat-array result = %d\n",
              intValue);

  cudaq::realtime::finalize();
  return boolValue == 19 && intValue == 42 ? 0 : 1;
}

// SHM: device_call shared-memory bool flat-array result = 19
// SHM: device_call shared-memory i32 flat-array result = 42
