/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

/// @file device_unified_generic.cuh
/// @brief Library-owned, transport-agnostic GPU dispatch loop -- the GPU analog
/// of the host `cudaq_host_unified_generic_loop`.
///
/// A single dispatch loop owns RX poll -> dispatch -> TX commit for *any*
/// transport, driving the wire through the bridge device data-plane
/// (cudaq_device_dataplane_t) instead of inlining a transport's native
/// contract.  Two arms are provided so the device-abstraction cost can be
/// measured, mirroring the CPU fused-vs-generic study:
///
///   - Arm 1 (generic-vtable): the loop calls the transport's device ops
///     through device function pointers -- true runtime polymorphism, one
///     indirect device call per op (rx_acquire / tx_acquire / tx_commit) on top
///     of the one unavoidable handler indirection inside
///     cudaq_device_dispatch_rpc.
///   - Arm 2 (generic-template): the transport supplies a compile-time policy
///     (static __device__ ops) so the loop inlines with zero indirection,
///     bounding the abstraction floor.
///
/// Header-only / device-linked (RDC): an indirect device call requires the
/// kernel and the transport's __device__ ops to live in the same device module.
/// Packaging this loop as a prebuilt `.so` symbol is a follow-up.  Single
/// thread, single block, matching the existing fused kernels.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/device_dispatch_rpc.cuh"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

//==============================================================================
// Arm 1: generic-vtable (device function pointers).
//==============================================================================

/// @brief Library-owned generic GPU dispatch kernel.  Drives the transport
/// entirely through the device data-plane function pointers in `dp`.  The
/// in-band shutdown convention (function_id == 0) and stats live here, not in
/// the transport.
static __global__ void
cudaq_device_unified_generic_kernel(cudaq_device_dataplane_t dp,
                                    cudaq_function_entry_t *function_table,
                                    std::size_t func_count,
                                    volatile int *shutdown_flag,
                                    std::uint64_t *stats) {
  std::uint64_t packet_count = 0;

  while (!*shutdown_flag) {
    // Stage 1: ask the transport for the next RX slot (non-blocking).
    void *rx_slot = nullptr;
    std::size_t rx_size = 0;
    cudaq_rx_status_t status = dp.rx_acquire(dp.device_ctx, &rx_slot, &rx_size);
    if (status == CUDAQ_RX_EMPTY)
      continue;
    if (status == CUDAQ_RX_SHUTDOWN)
      break;

    // In-band shutdown convention stays in the loop, not the transport.
    const auto *hdr = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (hdr->function_id == 0) {
      *shutdown_flag = 1;
      __threadfence_system();
      break;
    }

    // Stage 2: get the TX slot (cursor only advances on commit).
    void *tx_slot = nullptr;
    std::size_t tx_size = 0;
    dp.tx_acquire(dp.device_ctx, &tx_slot, &tx_size);

    // Stage 3: shared per-slot DEVICE_CALL dispatch (lookup + handler + frame).
    const std::size_t slot_size = rx_size < tx_size ? rx_size : tx_size;
    if (cudaq_device_dispatch_rpc(function_table, func_count, rx_slot, tx_slot,
                                  slot_size) == 0)
      continue; // unknown function / bad magic: drop, no tx_commit

    // Stage 4: publish (transport owns the fence + doorbell + cursor advance).
    dp.tx_commit(dp.device_ctx);
    ++packet_count;
  }

  atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}

/// @brief Launch wrapper matching cudaq_unified_launch_fn_t so the generic
/// kernel slots into cudaq_dispatcher_set_unified_launch with no new dispatcher
/// branch.  `transport_ctx` is a host pointer to a cudaq_device_dataplane_t the
/// bridge filled via get_device_dataplane.
inline void cudaq_launch_unified_generic_device_loop(
    void *transport_ctx, cudaq_function_entry_t *function_table,
    std::size_t func_count, volatile int *shutdown_flag, std::uint64_t *stats,
    cudaStream_t stream) {
  auto *dp = static_cast<cudaq_device_dataplane_t *>(transport_ctx);
  cudaq_device_unified_generic_kernel<<<1, 1, 0, stream>>>(
      *dp, function_table, func_count, shutdown_flag, stats);
}

//==============================================================================
// Arm 2: generic-template (compile-time policy, inlined).
//
// `Ops` is a transport-supplied struct of static __device__ methods:
//   static __device__ cudaq_rx_status_t rx_acquire(void *ctx, void **slot,
//                                                   std::size_t *size);
//   static __device__ void tx_acquire(void *ctx, void **slot,
//                                      std::size_t *size);
//   static __device__ void tx_commit(void *ctx);
//==============================================================================

template <typename Ops>
__device__ inline void cudaq_device_unified_generic_loop_templated(
    void *device_ctx, cudaq_function_entry_t *function_table,
    std::size_t func_count, volatile int *shutdown_flag, std::uint64_t *stats) {
  std::uint64_t packet_count = 0;

  while (!*shutdown_flag) {
    void *rx_slot = nullptr;
    std::size_t rx_size = 0;
    cudaq_rx_status_t status = Ops::rx_acquire(device_ctx, &rx_slot, &rx_size);
    if (status == CUDAQ_RX_EMPTY)
      continue;
    if (status == CUDAQ_RX_SHUTDOWN)
      break;

    const auto *hdr = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
    if (hdr->function_id == 0) {
      *shutdown_flag = 1;
      __threadfence_system();
      break;
    }

    void *tx_slot = nullptr;
    std::size_t tx_size = 0;
    Ops::tx_acquire(device_ctx, &tx_slot, &tx_size);

    const std::size_t slot_size = rx_size < tx_size ? rx_size : tx_size;
    if (cudaq_device_dispatch_rpc(function_table, func_count, rx_slot, tx_slot,
                                  slot_size) == 0)
      continue;

    Ops::tx_commit(device_ctx);
    ++packet_count;
  }

  atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}

template <typename Ops>
__global__ void cudaq_device_unified_generic_kernel_templated(
    void *device_ctx, cudaq_function_entry_t *function_table,
    std::size_t func_count, volatile int *shutdown_flag, std::uint64_t *stats) {
  cudaq_device_unified_generic_loop_templated<Ops>(
      device_ctx, function_table, func_count, shutdown_flag, stats);
}
