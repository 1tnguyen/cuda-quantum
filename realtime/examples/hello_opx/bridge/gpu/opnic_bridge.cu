/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file opnic_bridge.cu
/// @brief Reference GPU bridge -- a worked example of implementing the realtime
///        transport contract `cudaq_realtime_bridge_interface_t` for a GPU data
///        plane (zero-copy direct path).
///
/// NVIDIA prototype over the OPNIC transport; a transport vendor (e.g. Quantum
/// Machines) implements their own against the same contract.  Read the CPU
/// reference bridge `opnic_bridge_cpu.cpp` first for the vtable walkthrough --
/// the lifecycle (create/destroy, get_transport_context, connect/launch/
/// disconnect) is identical in spirit.  This file adds the GPU specifics:
///
///   - get_device_dataplane: the GPU analog of get_host_dataplane.  Device code
///     cannot call host function pointers, so the transport's RX/TX ops are
///     `__device__` functions whose addresses are captured with
///     cudaMemcpyFromSymbol and handed to the library generic dispatch *kernel*.
///   - RDC requirement: that kernel indirect-calls those device ops, so the
///     kernel and the ops must be device-linked into the SAME device module.
///     This bridge is compiled with relocatable device code (-rdc=true) and
///     linked into the consumer.  See docs/device_loop_launch_ownership.md.
///   - ring vs unified buffers: the 3-kernel ring path allocates intermediate
///     device ring buffers; unified mode lets the single dispatch kernel work
///     on the OPNIC DMA buffers directly.
///
/// Comments tagged "OPNIC-specific" mark what you replace for your transport.
/// The vtable shapes, the opaque-handle pattern, and the device-op capture
/// mechanism are the parts to keep.

#include "opnic_bridge.cuh"
#include "opnic_device_ops.cuh"
#include "opnic_type.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <memory>

//==============================================================================
// Device data-plane function-pointer symbols (consumed by get_device_dataplane
// in section 2).  The library generic dispatch kernel calls the transport's
// RX/TX ops through device function pointers; we capture these __device__
// symbol addresses on the host with cudaMemcpyFromSymbol (the same mechanism
// CUDA uses for a handler's device_fn_ptr) and hand them to the kernel inside
// cudaq_device_dataplane_t.  Defined in this single TU to avoid duplicate
// device symbols.  A different transport defines its own __device__ ops here.
//==============================================================================
__device__ cudaq_device_rx_acquire_fn_t d_opnic_rx_acquire =
    opnic_device_rx_acquire;
__device__ cudaq_device_tx_acquire_fn_t d_opnic_tx_acquire =
    opnic_device_tx_acquire;
__device__ cudaq_device_tx_commit_fn_t d_opnic_tx_commit =
    opnic_device_tx_commit;

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << cudaGetErrorString(err) << std::endl;                       \
    }                                                                          \
  } while (0)

//==============================================================================
// 1. Your transport state, behind the opaque handle
//==============================================================================
// OpnicBridgeContext (declared in opnic_bridge.cuh) is the GPU analog of the CPU
// bridge's context: everything the transport needs, handed back to the runtime
// as an opaque `cudaq_realtime_bridge_handle_t`.  For the GPU that includes the
// device-side ring buffers (3-kernel path only), the host-mapped shutdown flag
// the persistent kernels poll, the OPNIC DMA streams, and the lazily-allocated
// device data-plane context.  Construction allocates them; the destructor frees
// them.
//==============================================================================

OpnicBridgeContext::OpnicBridgeContext(const OpnicBridgeConfig &cfg)
    : config(cfg) {
  // In unified mode the dispatch kernel operates directly on OPNIC DMA
  // buffers, so the intermediate ring buffers are not needed.
  if (!cfg.unified) {
    CUDA_CHECK(cudaMalloc(&rx_data_device, cfg.num_slots * cfg.slot_size));
    CUDA_CHECK(cudaMalloc(&rx_flags_device, cfg.num_slots * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemset(rx_data_device, 0, cfg.num_slots * cfg.slot_size));
    CUDA_CHECK(cudaMemset(rx_flags_device, 0, cfg.num_slots * sizeof(std::uint64_t)));

    CUDA_CHECK(cudaMalloc(&tx_data_device, cfg.num_slots * cfg.slot_size));
    CUDA_CHECK(cudaMalloc(&tx_flags_device, cfg.num_slots * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemset(tx_data_device, 0, cfg.num_slots * cfg.slot_size));
    CUDA_CHECK(cudaMemset(tx_flags_device, 0, cfg.num_slots * sizeof(std::uint64_t)));
  }

  // Allocate host-mapped shutdown flag
  CUDA_CHECK(cudaHostAlloc(&shutdown_device, sizeof(int), cudaHostAllocMapped));
  *shutdown_device = 0;
  CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&shutdown_device),
                                      shutdown_device, 0));

  // Initialize direct OPNIC streams
  opnic_direct = std::make_unique<OpnicDirectContext>(
      cfg.input_stream_id, cfg.output_stream_id, cfg.num_slots);
  opnic_streams = opnic_direct->get_streams();
}

OpnicBridgeContext::~OpnicBridgeContext() {
  if (rx_data_device)
    cudaFree(rx_data_device);
  if (rx_flags_device)
    cudaFree(rx_flags_device);
  if (tx_data_device)
    cudaFree(tx_data_device);
  if (tx_flags_device)
    cudaFree(tx_flags_device);
  if (shutdown_device)
    cudaFreeHost(shutdown_device);
  if (device_dataplane_ctx)
    cudaFree(device_dataplane_ctx);

  if (rx_stream)
    cudaStreamDestroy(rx_stream);
  if (tx_stream)
    cudaStreamDestroy(tx_stream);
}

extern "C" {

//==============================================================================
// 2. The vtable functions
//==============================================================================
// Each function fills one slot of cudaq_realtime_bridge_interface_t (registered
// in section 3).  Same contract as the CPU bridge: take the opaque handle,
// return cudaq_status_t.

// create(): the bridge's construction entry point.  Read any transport options
// from argv, allocate your transport state, and hand it back via *handle as an
// opaque cudaq_realtime_bridge_handle_t.  The runtime owns the handle until it
// calls destroy().  (The CPU reference bridge shows the alternative -- a typed
// create_context() the consumer fills directly; pick whichever your integration
// prefers.)
static cudaq_status_t
opnic_bridge_create(cudaq_realtime_bridge_handle_t *handle, int argc,
                    char **argv) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;

  // OPNIC-specific: build the config from argv plus sensible defaults.  Unified
  // mode (one dispatch kernel owns transport) skips the intermediate device
  // ring buffers, so the *transport shape* must be known at construction; the
  // function table and dispatch wiring remain the consumer's concern.
  OpnicBridgeConfig config; // field defaults match the 3-kernel ring path
  for (int i = 1; i < argc; ++i) {
    if (argv[i] && std::strcmp(argv[i], "--unified") == 0)
      config.unified = true;
  }

  try {
    *handle = new OpnicBridgeContext(config);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Failed to create OpnicBridgeContext: " << e.what()
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  return CUDAQ_OK;
}

// destroy(): free everything create() allocated.  Called once at teardown; the
// handle must not be used afterwards.
static cudaq_status_t
opnic_bridge_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  OpnicBridgeContext *ctx = reinterpret_cast<OpnicBridgeContext *>(handle);

  delete ctx;
  return CUDAQ_OK;
}


// get_transport_context(): hand the dispatcher the ring buffer for the 3-kernel
// path.  Here the cudaq_ringbuffer_t points at this bridge's device-resident
// ring buffers; the library dispatch kernel and the bridge's RX/TX kernels
// rendezvous on them.  UNIFIED has no ring buffer -- the single dispatch kernel
// works the OPNIC DMA buffers directly and obtains its device-callable ops from
// get_device_dataplane (section 2, below), so UNIFIED returns an error here.
static cudaq_status_t opnic_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t handle,
    cudaq_realtime_transport_context_t context_type, void *out_context) {

  if (!handle || !out_context)
    return CUDAQ_ERR_INVALID_ARG;
  OpnicBridgeContext *ctx = reinterpret_cast<OpnicBridgeContext *>(handle);

  if (context_type == RING_BUFFER) {
    // The dispatch ring buffers only exist in the (non-unified) 3-kernel path.
    if (!ctx->rx_data_device || !ctx->rx_flags_device || !ctx->tx_data_device ||
        !ctx->tx_flags_device) {
      std::cerr << "ERROR: Failed to get ring buffer pointers" << std::endl;
      return CUDAQ_ERR_INTERNAL;
    }
    cudaq_ringbuffer_t *ringbuffer =
        reinterpret_cast<cudaq_ringbuffer_t *>(out_context);
    ringbuffer->rx_flags = static_cast<volatile std::uint64_t *>(ctx->rx_flags_device);
    ringbuffer->tx_flags = static_cast<volatile std::uint64_t *>(ctx->tx_flags_device);
    ringbuffer->rx_data = static_cast<std::uint8_t *>(ctx->rx_data_device);
    ringbuffer->tx_data = static_cast<std::uint8_t *>(ctx->tx_data_device);
    ringbuffer->rx_stride_sz = ctx->config.slot_size;
    ringbuffer->tx_stride_sz = ctx->config.slot_size;
    return CUDAQ_OK;
  }

  // UNIFIED: the dispatch loop owns the OPNIC DMA buffers directly.  The fused
  // path reads them from OpnicBridgeContext::opnic_streams; the library generic
  // device loop obtains its device-callable ops via get_device_dataplane.
  std::cerr << "ERROR: UNIFIED transport context is delivered via "
               "get_device_dataplane (generic loop) or opnic_streams (fused)"
            << std::endl;
  return CUDAQ_ERR_INVALID_ARG;
}

// connect(): bring the link up and block until the OPX is ready, then create
// the CUDA streams the transport kernels run on (ring path only -- unified runs
// everything in the one dispatch kernel).  Must precede launch() and
// get_device_dataplane().
static cudaq_status_t
opnic_bridge_connect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  OpnicBridgeContext *ctx = reinterpret_cast<OpnicBridgeContext *>(handle);

  // Mandatory OPX handshake via direct context
  printf("[HOST] Waiting for OPX handshake, start the qua program\n");
  ctx->opnic_direct->sync();
  printf("[HOST] OPX<->OPNIC sync is complete (direct path). Packets can now be "
         "sent/received\n");

  // In unified mode the single dispatch kernel handles both RX and TX, so
  // dedicated CUDA streams for the separate transport kernels are not needed.
  if (!ctx->config.unified) {
    CUDA_CHECK(cudaStreamCreate(&ctx->rx_stream));
    CUDA_CHECK(cudaStreamCreate(&ctx->tx_stream));
  }
  return CUDAQ_OK;
}

// Helper to force eager GPU module loading for direct RX kernel
__host__ inline int query_rx_kernel_direct_occupancy() {
  int num_blocks = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, opnic_rx_kernel_direct, 1, 0);
  if (err != cudaSuccess) {
    std::cerr << "ERROR: RX direct kernel occupancy query failed\n";
    return -1;
  }
  return num_blocks;
}

// Helper to force eager GPU module loading for direct TX kernel
__host__ inline int query_tx_kernel_direct_occupancy() {
  int num_blocks = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, opnic_tx_kernel_direct, 1, 0);
  if (err != cudaSuccess) {
    std::cerr << "ERROR: TX direct kernel occupancy query failed\n";
    return -1;
  }
  return num_blocks;
}

// launch(): start the transport's persistent RX/TX kernels that move bytes
// between OPNIC and the device ring buffers the dispatch kernel consumes.  Ring
// path only; in unified mode the single dispatch kernel IS the transport, so
// the bridge launches nothing.  (The occupancy queries in the body force eager
// CUDA module loading -- OPNIC-specific hardening against lazy-load deadlocks
// with persistent kernels.)
static cudaq_status_t
opnic_bridge_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  OpnicBridgeContext *ctx = reinterpret_cast<OpnicBridgeContext *>(handle);

  if (ctx->config.unified)
    return CUDAQ_OK;

  if (!ctx->rx_stream || !ctx->tx_stream)
    return CUDAQ_ERR_INVALID_ARG;

  // CRITICAL: Force eager CUDA module loading via occupancy queries
  // This prevents lazy-loading deadlocks with persistent kernels and dispatcher
  int rx_occ = query_rx_kernel_direct_occupancy();
  int tx_occ = query_tx_kernel_direct_occupancy();
  if (rx_occ == -1 || tx_occ == -1) {
    std::cerr << "ERROR: Failed to query kernel occupancy" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  const auto& opnic = ctx->opnic_streams;

  // Launch direct TX kernel (writes directly to OPNIC buffer + doorbell)
  opnic_tx_kernel_direct<<<1, 1, 0, ctx->tx_stream>>>(
      opnic.tx_buffer,
      opnic.tx_doorbell_ptr,
      opnic.tx_allocation_size,
      opnic.buffer_count,
      static_cast<volatile std::uint64_t *>(ctx->tx_flags_device),
      static_cast<std::uint8_t *>(ctx->tx_data_device),
      static_cast<volatile int *>(ctx->shutdown_device));
  CUDA_CHECK(cudaGetLastError());

  // Launch direct RX kernel (reads directly from OPNIC buffer + pi_ptr)
  opnic_rx_kernel_direct<<<1, 1, 0, ctx->rx_stream>>>(
      opnic.rx_buffer,
      opnic.rx_pi_ptr,
      opnic.rx_allocation_size,
      opnic.buffer_count,
      static_cast<volatile std::uint64_t *>(ctx->rx_flags_device),
      static_cast<std::uint8_t *>(ctx->rx_data_device),
      static_cast<volatile int *>(ctx->shutdown_device));
  CUDA_CHECK(cudaGetLastError());

  return CUDAQ_OK;
}

// disconnect(): quiesce the transport kernels by synchronizing their streams.
// Mirror of launch(); a no-op in unified mode (no separate streams created).
static cudaq_status_t
opnic_bridge_disconnect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  OpnicBridgeContext *ctx = reinterpret_cast<OpnicBridgeContext *>(handle);

  if (ctx->config.unified)
    return CUDAQ_OK;

  if (!ctx->rx_stream || !ctx->tx_stream)
    return CUDAQ_ERR_INVALID_ARG;

  CUDA_CHECK(cudaStreamSynchronize(ctx->rx_stream));
  CUDA_CHECK(cudaStreamSynchronize(ctx->tx_stream));
  return CUDAQ_OK;
}

// get_device_dataplane(): OPTIONAL GPU analog of get_host_dataplane.  Builds a
// device-resident OpnicDeviceCtx from the live OPNIC streams and captures the
// transport's __device__ RX/TX op addresses (the d_opnic_* symbols at the top
// of this file) via cudaMemcpyFromSymbol, so the library generic dispatch
// *kernel* can drive OPNIC without knowing its native contract.  Two rules a
// vendor must respect:
//   1. Call only after the streams exist (after construction / connect).
//   2. The captured device function pointers are valid ONLY inside the device
//      module they were defined in; the dispatch kernel that calls them must be
//      device-linked (-rdc=true) into that same module (the consumer).  This is
//      why the launch lives in the consumer, not in libcudaq-realtime.so -- see
//      docs/device_loop_launch_ownership.md.
// Leave the vtable slot NULL if you do not implement a GPU generic path.
static cudaq_status_t
opnic_bridge_get_device_dataplane(cudaq_realtime_bridge_handle_t handle,
                                  cudaq_device_dataplane_t *out) {
  if (!handle || !out)
    return CUDAQ_ERR_INVALID_ARG;
  OpnicBridgeContext *ctx = reinterpret_cast<OpnicBridgeContext *>(handle);
  const auto &s = ctx->opnic_streams;
  if (!s.rx_buffer || !s.rx_pi_ptr || !s.tx_buffer || !s.tx_doorbell_ptr) {
    std::cerr << "ERROR: OPNIC streams not initialized for device data-plane"
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  // Allocate (once) and populate the device-resident transport context.
  if (!ctx->device_dataplane_ctx) {
    if (cudaMalloc(&ctx->device_dataplane_ctx, sizeof(OpnicDeviceCtx)) !=
        cudaSuccess) {
      std::cerr << "ERROR: cudaMalloc(OpnicDeviceCtx) failed" << std::endl;
      return CUDAQ_ERR_INTERNAL;
    }
  }

  OpnicDeviceCtx host_ctx{};
  host_ctx.rx_buffer = s.rx_buffer;
  host_ctx.pi_ptr = s.rx_pi_ptr;
  host_ctx.rx_alloc_size = s.rx_allocation_size;
  host_ctx.buf_count = s.buffer_count;
  host_ctx.tx_buffer = s.tx_buffer;
  host_ctx.doorbell_ptr = s.tx_doorbell_ptr;
  host_ctx.tx_alloc_size = s.tx_allocation_size;
  host_ctx.consumer_index = 0;
  host_ctx.out_index = 0;
  if (cudaMemcpy(ctx->device_dataplane_ctx, &host_ctx, sizeof(OpnicDeviceCtx),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "ERROR: cudaMemcpy(OpnicDeviceCtx) failed" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  // Capture the transport's device op addresses (valid as device function
  // pointers once this lib is device-linked into the consuming executable).
  out->device_ctx = ctx->device_dataplane_ctx;
  if (cudaMemcpyFromSymbol(&out->rx_acquire, d_opnic_rx_acquire,
                           sizeof(out->rx_acquire)) != cudaSuccess ||
      cudaMemcpyFromSymbol(&out->tx_acquire, d_opnic_tx_acquire,
                           sizeof(out->tx_acquire)) != cudaSuccess ||
      cudaMemcpyFromSymbol(&out->tx_commit, d_opnic_tx_commit,
                           sizeof(out->tx_commit)) != cudaSuccess) {
    std::cerr << "ERROR: cudaMemcpyFromSymbol(device ops) failed" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  return CUDAQ_OK;
}

//==============================================================================
// 3. Publish the vtable
//==============================================================================
// The runtime obtains every entry point through this single getter.  Field
// order matches cudaq_realtime_bridge_interface_t exactly; optional getters you
// do not support are NULL.  This GPU bridge mirrors the CPU one but swaps the
// data-plane: no host data-plane, yes device data-plane.
cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t cudaq_opnic_bridge_interface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      opnic_bridge_create,
      opnic_bridge_destroy,
      opnic_bridge_get_transport_context,
      opnic_bridge_connect,
      opnic_bridge_launch,
      opnic_bridge_disconnect,
      nullptr, // get_host_dataplane (GPU bridge has none)
      opnic_bridge_get_device_dataplane,
  };
  return &cudaq_opnic_bridge_interface;
}

}
