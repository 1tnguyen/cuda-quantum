/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

/// @file opnic_bridge.cuh
/// @brief GPU OPNIC bridge interface (zero-copy direct path) -- public header.
///
/// Mirrors the CPU bridge (opnic_bridge_cpu.hpp): the bridge is compiled into
/// the `cudaq-realtime-opnic-gpu` library and consumers link that library and
/// include this header (rather than #including the .cu).  The example
/// constructs an OpnicBridgeContext directly and reaches into a couple of its
/// members (shutdown_device, opnic_streams), so the full struct is declared
/// here; its methods and the bridge vtable are defined in opnic_bridge.cu.

#include "opnic_kernels.cuh" // RING_BUFFER_PAGE_SIZE / RING_BUFFER_NUM_PAGES
#include "opnic_direct.cuh"  // OpnicDirectContext / OpnicDirectStreams
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

/// @brief Configuration for the GPU OPNIC bridge.
struct OpnicBridgeConfig {
  int gpu_id = 0;

  std::size_t slot_size = RING_BUFFER_PAGE_SIZE;
  unsigned num_slots = RING_BUFFER_NUM_PAGES;

  std::uint32_t num_blocks = 1;
  std::uint32_t threads_per_block = 1;

  std::uint16_t input_stream_id = 1;
  std::uint16_t output_stream_id = 1;

  /// When true, skip ring buffer allocation and RX/TX kernel launch.
  /// The unified dispatch kernel owns OPNIC DMA buffers directly.
  bool unified = false;
};

/// @brief Bridge context using direct OPNIC access (zero-copy path).
/// Methods are defined in opnic_bridge.cu (linked from cudaq-realtime-opnic-gpu).
struct OpnicBridgeContext {
  OpnicBridgeConfig config;

  // Dispatch ring buffers (GPU device memory).
  void *rx_data_device = nullptr;
  void *rx_flags_device = nullptr;
  void *tx_data_device = nullptr;
  void *tx_flags_device = nullptr;

  // Shutdown flag (host-mapped for GPU access).
  int *shutdown_device = nullptr;

  // Device-resident OpnicDeviceCtx for the library generic dispatch loop
  // (allocated lazily by get_device_dataplane; freed in the destructor).
  void *device_dataplane_ctx = nullptr;

  // Direct OPNIC streams (bypasses SDK IncomingStream/OutgoingStream).
  std::unique_ptr<OpnicDirectContext> opnic_direct;
  OpnicDirectStreams opnic_streams{};

  // CUDA streams for kernel execution.
  cudaStream_t rx_stream = nullptr;
  cudaStream_t tx_stream = nullptr;

  explicit OpnicBridgeContext(const OpnicBridgeConfig &cfg);
  ~OpnicBridgeContext();
};

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Get the GPU OPNIC bridge interface (static instance; do not free).
cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface(void);

#ifdef __cplusplus
}
#endif
