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

// Optional service-provided synchronizer used when the dispatch loop is owned
// by the service artifact rather than by a CUDA-Q-created stream.
using cudaq_device_call_dispatch_synchronize_fn_t = cudaError_t (*)();

// CUDA-Q uses the callbacks below to create one realtime dispatch session.
struct cudaq_realtime_device_call_service {
  // Optional service-state lifecycle. create returns an opaque handle that is
  // passed to all other callbacks; destroy releases that state after stop.
  int (*create)(const void *config_payload, std::size_t config_size,
                void **handle) = nullptr;
  int (*destroy)(void *handle) = nullptr;

  // Required function table setup. CUDA-Q allocates the table storage;
  // populate_table fills up to capacity entries, optionally using stream.
  std::uint32_t (*get_function_count)(void *handle) = nullptr;
  int (*populate_table)(void *handle, cudaq_function_entry_t *entries,
                        std::uint32_t capacity, cudaStream_t stream) = nullptr;

  // Required for GPU dispatch: returns the launch function for the dispatch
  // kernel linked with the service handlers.
  cudaq_dispatch_launch_fn_t (*get_device_dispatch_launch)(void *handle) =
      nullptr;
  // Optional shutdown/completion hook for service-managed dispatch resources.
  cudaq_device_call_dispatch_synchronize_fn_t (
      *get_device_dispatch_synchronize)(void *handle) = nullptr;

  // Optional session hooks called around an active CUDA-Q device_call endpoint.
  int (*start)(void *handle) = nullptr;
  int (*stop)(void *handle) = nullptr;
};

// Factory signature for cudaq_realtime_get_service* symbols exported by
// service artifacts.
using cudaq_realtime_get_service_fn_t =
    int (*)(cudaq_realtime_device_call_service *out);
