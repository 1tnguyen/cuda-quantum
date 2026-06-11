/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <cstddef>

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

/// @brief Configuration for the CPU OPNIC bridge.
struct OpnicBridgeCpuConfig {
  std::uint16_t input_stream_id = 1;
  std::uint16_t output_stream_id = 1;
  std::size_t buffer_count = 1024;
  /// When true, the bridge exposes only UNIFIED transport context (single-thread
  /// fused loop).  When false, it also supports RING_BUFFER and owns the RX/TX
  /// adapter threads started by `launch`.
  bool unified = false;
  /// Shutdown flag shared with the host dispatch loop (required for the
  /// 3-thread ring path so RX/TX adapter threads honor the same stop signal).
  volatile int *shutdown_flag = nullptr;
};

/// @brief OPNIC transport context.
struct opnic_cpu_transport_ctx {
  std::uint32_t *rx_buffer;          ///< OPNIC incoming DMA target (host mem)
  volatile std::uint32_t *pi_ptr;    ///< OPNIC producer index      (host mem)
  std::size_t rx_alloc_size;         ///< Per-packet stride in rx_buffer
  std::size_t buf_count;             ///< Number of packet slots in OPNIC ring

  std::uint32_t *tx_buffer;          ///< OPNIC outgoing DMA source (host mem)
  volatile std::uint64_t *doorbell_ptr; ///< mmap'd TX doorbell (PCIe BAR)
  std::size_t tx_alloc_size;         ///< Per-packet stride in tx_buffer
};

/// @brief Construct a CPU OPNIC bridge context and return an opaque handle
/// suitable for the bridge interface. Returns nullptr on failure.
///
/// Note: The caller owns the handle and must release it via the bridge
/// interface's `destroy` function pointer.
extern "C" cudaq_realtime_bridge_handle_t
opnic_bridge_cpu_create_context(const OpnicBridgeCpuConfig *cfg);

/// @brief Get the CPU OPNIC bridge interface. The returned pointer is to a
/// static instance; callers must NOT free it.
extern "C" cudaq_realtime_bridge_interface_t *
cudaq_realtime_get_opnic_cpu_bridge_interface();
