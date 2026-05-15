/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

// Internal host-side service contract for realtime-backed device_call
// artifacts.
//
// This is intentionally separate from the per-call wire ABI. A service artifact
// exposes this factory once per process so CUDA-Q can create an explicit
// service handle, populate a process-local function table, and obtain the
// dispatch launch hooks that were CUDA device-linked with the handlers.

using cudaq_device_call_dispatch_synchronize_fn_t = cudaError_t (*)();

struct cudaq_realtime_device_call_service {
  int (*configure)(int argc, char **argv) = nullptr;
  int (*create)(const void *config_payload, std::size_t config_size,
                void **handle) = nullptr;
  int (*destroy)(void *handle) = nullptr;

  std::uint32_t (*get_function_count)(void *handle) = nullptr;
  int (*populate_table)(void *handle, cudaq_function_entry_t *entries,
                        std::uint32_t capacity, cudaStream_t stream) = nullptr;

  cudaq_dispatch_launch_fn_t (*get_device_dispatch_launch)(void *handle) =
      nullptr;
  cudaq_device_call_dispatch_synchronize_fn_t (
      *get_device_dispatch_synchronize)(void *handle) = nullptr;

  int (*start)(void *handle) = nullptr;
  int (*stop)(void *handle) = nullptr;
};

using cudaq_realtime_get_service_fn_t =
    int (*)(cudaq_realtime_device_call_service *out);

namespace cudaq_internal::device_call {

void registerDeviceCallServiceFactory(cudaq_realtime_get_service_fn_t factory);

void setDeviceCallFunctionTableWithLauncher(
    cudaq_function_entry_t *entries, std::uint32_t count,
    cudaq_dispatch_launch_fn_t launchFn);
void setDeviceCallFunctionTableWithLauncherForDevice(
    std::uint32_t deviceId, cudaq_function_entry_t *entries,
    std::uint32_t count, cudaq_dispatch_launch_fn_t launchFn);

} // namespace cudaq_internal::device_call
