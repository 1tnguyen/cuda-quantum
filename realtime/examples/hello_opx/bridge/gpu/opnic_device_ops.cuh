/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

/// @file opnic_device_ops.cuh
/// @brief OPNIC device data-plane: the GPU equivalent of the CPU bridge's
/// rx_acquire / tx_acquire / tx_commit, for the library generic device loop.
///
/// These reproduce the per-stage logic of `opnic_unified_dispatch_kernel`
/// (poll producer index -> address slot; address tx slot; fence + doorbell),
/// but expose it as standalone __device__ ops the transport-agnostic loop can
/// drive.  The OPNIC wire slot is byte-identical to RPCHeader / RPCResponse on
/// little-endian (magic, function_id/status, arg_len/result_len, request_id,
/// ptp lo/hi, then payload words), so the loop treats an OPNIC slot directly as
/// an RPC slot with no word-by-word translation -- that is the data-plane
/// contract.  The RX/TX cursors live in the device-resident OpnicDeviceCtx
/// because the generic loop does not track transport cursors itself.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

/// @brief Device-resident OPNIC transport state for the generic GPU loop.
/// Mirrors the example's `opnic_transport_ctx` and adds the RX/TX cursors.
struct OpnicDeviceCtx {
  std::uint32_t *rx_buffer;       ///< OPNIC incoming DMA buffer (GPU mem)
  volatile std::uint32_t *pi_ptr; ///< OPNIC producer index (GPU mem)
  std::size_t rx_alloc_size;      ///< Per-packet stride in rx_buffer
  std::size_t buf_count;          ///< Number of packet slots in the OPNIC ring
  std::uint32_t *tx_buffer;       ///< OPNIC outgoing DMA buffer (GPU mem)
  std::uint64_t *doorbell_ptr;    ///< mmap'd TX doorbell (cudaHostRegistered)
  std::size_t tx_alloc_size;      ///< Per-packet stride in tx_buffer
  std::uint32_t consumer_index;   ///< RX cursor (advanced by rx_acquire)
  std::uint32_t out_index;        ///< TX cursor (advanced by tx_commit)
};

/// @brief Non-blocking RX acquire: report the next OPNIC slot if the producer
/// index has advanced past the consumer cursor, else CUDAQ_RX_EMPTY.
__device__ inline cudaq_rx_status_t
opnic_device_rx_acquire(void *ctx_v, void **out_rx_slot,
                        std::size_t *out_slot_size) {
  auto *ctx = static_cast<OpnicDeviceCtx *>(ctx_v);
  if (static_cast<std::int32_t>(*ctx->pi_ptr - ctx->consumer_index) < 1)
    return CUDAQ_RX_EMPTY;
  const std::uint32_t slot =
      ctx->consumer_index % static_cast<std::uint32_t>(ctx->buf_count);
  *out_rx_slot =
      reinterpret_cast<std::uint8_t *>(ctx->rx_buffer) + slot * ctx->rx_alloc_size;
  *out_slot_size = ctx->rx_alloc_size;
  ctx->consumer_index++;
  return CUDAQ_RX_READY;
}

/// @brief Report the current OPNIC TX slot (does not advance the cursor).
__device__ inline void opnic_device_tx_acquire(void *ctx_v, void **out_tx_slot,
                                               std::size_t *out_slot_size) {
  auto *ctx = static_cast<OpnicDeviceCtx *>(ctx_v);
  const std::uint32_t slot =
      ctx->out_index % static_cast<std::uint32_t>(ctx->buf_count);
  *out_tx_slot =
      reinterpret_cast<std::uint8_t *>(ctx->tx_buffer) + slot * ctx->tx_alloc_size;
  *out_slot_size = ctx->tx_alloc_size;
}

/// @brief Publish the current TX slot (PCIe fence + doorbell) and advance the
/// TX cursor.
__device__ inline void opnic_device_tx_commit(void *ctx_v) {
  auto *ctx = static_cast<OpnicDeviceCtx *>(ctx_v);
  __threadfence_system();
  *ctx->doorbell_ptr = 1;
  ctx->out_index++;
}

/// @brief Compile-time policy for the generic-template arm (zero indirection).
struct OpnicDeviceOps {
  static __device__ cudaq_rx_status_t rx_acquire(void *ctx, void **slot,
                                                 std::size_t *size) {
    return opnic_device_rx_acquire(ctx, slot, size);
  }
  static __device__ void tx_acquire(void *ctx, void **slot, std::size_t *size) {
    opnic_device_tx_acquire(ctx, slot, size);
  }
  static __device__ void tx_commit(void *ctx) { opnic_device_tx_commit(ctx); }
};
