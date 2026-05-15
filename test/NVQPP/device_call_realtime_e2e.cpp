/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: device-call-realtime-e2e-lib
// REQUIRES: device-call-realtime-tcp-e2e-server

// clang-format off
// This test intentionally mirrors device_call_realtime_bool_array_e2e.cpp.
// Both tests compile one CUDA-Q app and exercise shared-memory, TCP external,
// and TCP auto-launch through the same co-linked service library.
// RUN: rm -rf %t && mkdir -p %t
// RUN: cd %t && nvq++ -frealtime-lowering --enable-mlir -save-temps -I%S/Inputs %s %cudaq_device_call_realtime_e2e_lib -o %t/app
// RUN: grep -R "__cudaq_device_call_acquire_realtime_frame" %t > /dev/null
// RUN: not grep -R "__cudaq_device_call_dispatch_to_device" %t
// RUN: not grep -R "define weak dso_local i32 @addThem" %t/device_call_realtime_e2e.ll
// RUN: not grep -R "define weak dso_local float @multiplyFloats" %t/device_call_realtime_e2e.ll
// RUN: LD_LIBRARY_PATH=%cudaq_lib_dir:${LD_LIBRARY_PATH} %t/app --cudaq-device-call=shared-memory | FileCheck --check-prefix=SHM %s
// RUN: bash -c 'set -e; port_file="%t/port"; ready_file="%t/ready"; server_log="%t/server.log"; app_out="%t/app-tcp-external.out"; LD_LIBRARY_PATH="%cudaq_lib_dir:${LD_LIBRARY_PATH}" %cudaq_device_call_realtime_tcp_e2e_server --host 127.0.0.1 --port 0 --gpu 0 --port-file "$port_file" --ready-file "$ready_file" --num-slots 2 --slot-size 4096 --timeout-ms 10000 > "$server_log" 2>&1 & server_pid=$!; trap "kill $server_pid 2>/dev/null || true; wait $server_pid 2>/dev/null || true" EXIT; for i in $(seq 1 100); do test -f "$ready_file" && break; sleep 0.1; done; test -f "$ready_file" || { cat "$server_log"; exit 1; }; port=$(cat "$port_file"); CUDAQ_DEVICE_CALL_PLUGIN_PATH="%cudaq_lib_dir" LD_LIBRARY_PATH="%cudaq_lib_dir:${LD_LIBRARY_PATH}" %t/app --cudaq-device-call=tcp --cudaq-device-call-tcp-endpoint=127.0.0.1:$port --cudaq-device-call-timeout-ms=10000 > "$app_out"; app_status=$?; kill $server_pid 2>/dev/null || true; wait $server_pid 2>/dev/null || true; cat "$app_out"; exit $app_status' | FileCheck --check-prefix=TCP-EXTERNAL %s
// RUN: bash -c 'set -e; app_out="%t/app-tcp-auto.out"; CUDAQ_DEVICE_CALL_PLUGIN_PATH="%cudaq_lib_dir" LD_LIBRARY_PATH="%cudaq_lib_dir:${LD_LIBRARY_PATH}" %t/app --cudaq-device-call=tcp --cudaq-device-call-tcp-launch=auto --cudaq-device-call-tcp-server=%cudaq_device_call_realtime_tcp_e2e_server --cudaq-device-call-tcp-host=127.0.0.1 --cudaq-device-call-tcp-port=0 --cudaq-device-call-tcp-gpu=0 --cudaq-device-call-slots=2 --cudaq-device-call-slot-size=4096 --cudaq-device-call-timeout-ms=10000 > "$app_out"; app_status=$?; cat "$app_out"; exit $app_status' | FileCheck --check-prefix=TCP-AUTO %s
// clang-format on

#include <cudaq.h>

#include "cudaq_internal/device_call/DeviceCallRuntime.h"

#include <cstdio>
#include <cstring>

#include "device_call_realtime_lib.h"

__qpu__ int kernel(int a, int b) {
  return cudaq::device_call(0, addThem, a, b);
}

__qpu__ float floatKernel(float a, float b) {
  return cudaq::device_call(0, multiplyFloats, a, b);
}

const char *configurationLabel(int argc, char **argv) {
  bool tcp = false;
  bool autoLaunch = false;
  for (int i = 1; i < argc; ++i) {
    if (!argv[i])
      continue;
    if (std::strcmp(argv[i], "--cudaq-device-call=tcp") == 0)
      tcp = true;
    if (std::strcmp(argv[i], "--cudaq-device-call-tcp-launch=auto") == 0)
      autoLaunch = true;
  }
  if (!tcp)
    return "shared-memory";
  return autoLaunch ? "TCP auto" : "TCP external";
}

int main(int argc, char **argv) {
  cudaq_internal::device_call::initializeDeviceCallRuntime(argc, argv);
  const char *label = configurationLabel(argc, argv);

  auto results = cudaq::run(1, kernel, 19, 23);
  int value = results.front();
  std::printf("device_call %s int result = %d\n", label, value);

  auto floatResults = cudaq::run(1, floatKernel, 6.0f, 7.0f);
  float floatValue = floatResults.front();
  std::printf("device_call %s float result = %.1f\n", label, floatValue);

  cudaq_internal::device_call::finalizeDeviceCallRuntime();
  return value == 42 && floatValue == 42.0f ? 0 : 1;
}

// SHM: device_call shared-memory int result = 42
// SHM: device_call shared-memory float result = 42.0
// TCP-EXTERNAL: device_call TCP external int result = 42
// TCP-EXTERNAL: device_call TCP external float result = 42.0
// TCP-AUTO: device_call TCP auto int result = 42
// TCP-AUTO: device_call TCP auto float result = 42.0
