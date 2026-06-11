/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file opnic_bridge_cpu.cpp
/// @brief Reference CPU bridge -- a worked example of implementing the realtime
///        transport contract `cudaq_realtime_bridge_interface_t`.
///
/// This is an NVIDIA prototype over the OPNIC transport.  A transport vendor
/// (e.g. Quantum Machines) implements their *own* bridge against the same
/// contract; this file exists to refer to while doing so.  The realtime library
/// never learns OPNIC's native contract -- it only calls the vtable returned by
/// `cudaq_realtime_get_opnic_cpu_bridge_interface()` at the bottom of this file.
///
/// What you implement (the slots of `cudaq_realtime_bridge_interface_t`):
///   create / destroy            allocate / free your transport state.
///   get_transport_context       hand the dispatcher either a ring buffer
///                               (portable 3-thread ring path) or your unified
///                               transport context.
///   connect / launch / disconnect  bring the link up, start/stop your RX/TX
///                               workers, bring it down.
///   get_host_dataplane          OPTIONAL: expose per-slot rx/tx ops so the
///                               library's generic unified loop can drive your
///                               transport without knowing its wire format.
///   get_device_dataplane        OPTIONAL: the GPU analog -- NULL here; see the
///                               GPU reference bridge `opnic_bridge.cu`.
///
/// Lifecycle the runtime drives (see `incrementer_cpu.cpp`):
///   construct -> connect -> {get_transport_context | get_host_dataplane}
///             -> launch (ring only) -> [dispatch runs] -> disconnect -> destroy
///
/// Comments tagged "OPNIC-specific" mark what you replace for your transport.
/// The function shapes, the cudaq_status_t return codes, and the opaque-handle
/// pattern are the parts to keep.

#include "opnic_bridge_cpu.hpp"
#include "opnic_direct_cpu.hpp"
#include "opnic_ring_shim.hpp"
#include "opnic_type.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>

namespace {

//=============================================================================
// 1. Your transport state, behind the opaque handle
//=============================================================================
// The runtime treats the bridge as an opaque `cudaq_realtime_bridge_handle_t`
// (a void*).  Put everything your transport needs in one struct, hand its
// address back as the handle, and every vtable call casts it back.  For OPNIC
// that is the DMA stream pointers (rx/tx buffers, producer index, doorbell),
// the optional ring adapter, and the slot cursors used by the generic host
// loop.  A different transport would hold its own descriptors here instead.

struct OpnicBridgeCpuContext {
  using OpnicDirectCpu =
      OpnicDirectCpuContext<RPCInputPacket, RPCOutputPacket>;

  OpnicBridgeCpuConfig config;
  std::unique_ptr<OpnicDirectCpu> opnic_direct;
  OpnicDirectCpuStreams opnic_streams{};

  /// Snapshot of the streams packed into the form the unified host loop
  /// consumes. Held by-value so the bridge can hand out a stable pointer via
  /// `get_transport_context`.
  opnic_cpu_transport_ctx transport_ctx{};

  /// RX/TX adapter for the 3-thread ring path (OPNIC pi/doorbell -> flag-ring).
  std::unique_ptr<OpnicRingShim> ring_shim;

  /// Slot cursors for the host data-plane interface (the library generic
  /// unified loop).  The transport owns these; the fused escape-hatch loop
  /// keeps its own locals.
  std::uint32_t consumer_index = 0;
  std::uint32_t out_index = 0;

  explicit OpnicBridgeCpuContext(const OpnicBridgeCpuConfig &cfg)
      : config(cfg) {
    opnic_direct = std::make_unique<OpnicDirectCpu>(
        cfg.input_stream_id, cfg.output_stream_id, cfg.buffer_count);
    opnic_streams = opnic_direct->get_streams();

    transport_ctx.rx_buffer = opnic_streams.rx_buffer;
    transport_ctx.pi_ptr = opnic_streams.rx_pi_ptr;
    transport_ctx.rx_alloc_size = opnic_streams.rx_allocation_size;
    transport_ctx.buf_count = opnic_streams.buffer_count;
    transport_ctx.tx_buffer = opnic_streams.tx_buffer;
    transport_ctx.doorbell_ptr = opnic_streams.tx_doorbell_ptr;
    transport_ctx.tx_alloc_size = opnic_streams.tx_allocation_size;

    if (!cfg.unified) {
      if (!cfg.shutdown_flag)
        throw std::invalid_argument(
            "OpnicBridgeCpuConfig::shutdown_flag required for ring path");
      ring_shim =
          std::make_unique<OpnicRingShim>(transport_ctx, cfg.shutdown_flag);
    }
  }

  ~OpnicBridgeCpuContext() {
    if (ring_shim)
      ring_shim->join();
  }
};

//=============================================================================
// 2. The vtable functions
//=============================================================================
// Each function below fills one slot of cudaq_realtime_bridge_interface_t
// (registered in section 4).  Every one receives the opaque handle and returns
// cudaq_status_t (CUDAQ_OK on success, a CUDAQ_ERR_* code otherwise).

// create(): the plugin entry point for the dlopen path -- parse argv, allocate
// your context, and write it to *handle.  This example instead constructs the
// context explicitly via opnic_bridge_cpu_create_context() (see section 3) and
// passes the handle into the interface, so create() is a no-op here.  Implement
// whichever construction model your integration uses; you do not need both.
cudaq_status_t opnic_bridge_cpu_create(cudaq_realtime_bridge_handle_t *,
                                       int /*argc*/, char ** /*argv*/) {
  return CUDAQ_OK;
}

// destroy(): free everything your constructor / create() allocated.  Called
// once at teardown; the handle must not be used afterwards.
cudaq_status_t opnic_bridge_cpu_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  delete reinterpret_cast<OpnicBridgeCpuContext *>(handle);
  return CUDAQ_OK;
}

// get_transport_context(): hand the dispatcher the input it dispatches against,
// selected by `context_type`:
//   RING_BUFFER -> fill a cudaq_ringbuffer_t.  This is the portable path: an
//                  RX/TX adapter (OpnicRingShim) moves bytes between OPNIC and
//                  a flag-ring the library dispatch loop consumes, so no
//                  data-plane vtable is needed.
//   UNIFIED     -> hand back your transport-native context (here a by-value
//                  snapshot of the OPNIC stream pointers).
// Return CUDAQ_ERR_INVALID_ARG for context types you do not support.
cudaq_status_t opnic_bridge_cpu_get_transport_context(
    cudaq_realtime_bridge_handle_t handle,
    cudaq_realtime_transport_context_t context_type, void *out_context) {
  if (!handle || !out_context)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<OpnicBridgeCpuContext *>(handle);

  if (context_type == UNIFIED) {
    *static_cast<opnic_cpu_transport_ctx *>(out_context) = ctx->transport_ctx;
    return CUDAQ_OK;
  }
  if (context_type == RING_BUFFER) {
    if (ctx->config.unified) {
      std::cerr << "ERROR: CPU OPNIC bridge RING_BUFFER context unavailable in "
                   "unified mode"
                << std::endl;
      return CUDAQ_ERR_INVALID_ARG;
    }
    if (!ctx->ring_shim) {
      std::cerr << "ERROR: CPU OPNIC bridge ring shim not initialized"
                << std::endl;
      return CUDAQ_ERR_INTERNAL;
    }
    *static_cast<cudaq_ringbuffer_t *>(out_context) = ctx->ring_shim->ring();
    return CUDAQ_OK;
  }
  std::cerr << "ERROR: Unsupported CPU OPNIC bridge transport context type"
            << std::endl;
  return CUDAQ_ERR_INVALID_ARG;
}

// connect(): bring the link up and block until the peer (the OPX) is ready.
// Must run before launch() and before any data-plane getter that reads live
// stream pointers.  OPNIC-specific: performs the OPX<->OPNIC handshake.
cudaq_status_t opnic_bridge_cpu_connect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<OpnicBridgeCpuContext *>(handle);
  printf("[HOST] Waiting for OPX handshake, start the qua program\n");
  ctx->opnic_direct->sync();
  printf("[HOST] OPX<->OPNIC sync is complete (CPU path). Packets can now be "
         "sent/received\n");
  return CUDAQ_OK;
}

// launch(): start the transport's own RX/TX workers that move bytes between the
// wire and the ring buffer handed out by get_transport_context(RING_BUFFER).
// Only the 3-thread ring path needs them; in unified mode the dispatch loop
// owns poll -> dispatch -> doorbell itself, so launch() is a no-op.
cudaq_status_t opnic_bridge_cpu_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<OpnicBridgeCpuContext *>(handle);

  if (ctx->config.unified)
    return CUDAQ_OK;

  if (!ctx->ring_shim)
    return CUDAQ_ERR_INTERNAL;

  ctx->ring_shim->start();
  return CUDAQ_OK;
}

// disconnect(): stop and join whatever launch() started.  Mirror of launch();
// a no-op in unified mode because nothing was launched.
cudaq_status_t opnic_bridge_cpu_disconnect(
    cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<OpnicBridgeCpuContext *>(handle);

  if (ctx->config.unified)
    return CUDAQ_OK;

  if (ctx->ring_shim)
    ctx->ring_shim->join();
  return CUDAQ_OK;
}

//=============================================================================
// 3. OPTIONAL host data-plane -- let the library's generic loop drive you
//=============================================================================
// Implement these three ops (and get_host_dataplane below) only if you want the
// library-owned generic unified host loop to run your transport.  The division
// of labor: the LIBRARY owns the protocol (per-slot dispatch, the
// function_id == 0 in-band shutdown convention, stats); YOU own only moving
// bytes for one slot at a time.  Per-slot cursors live in your context, so the
// ops are valid only in unified mode.
//
//   rx_acquire: NON-BLOCKING.  If a packet is ready, point *out_rx_slot at it,
//               set *out_slot_size, advance the RX cursor, return
//               CUDAQ_RX_READY.  If nothing is ready return CUDAQ_RX_EMPTY (the
//               loop relaxes and retries).  Return CUDAQ_RX_SHUTDOWN to end it.
//   tx_acquire: report the current TX slot to write into.  Does NOT advance the
//               TX cursor (tx_commit does), so a dropped request can simply skip
//               tx_commit.
//   tx_commit:  publish the current TX slot -- ordering fence + doorbell -- and
//               advance the TX cursor.  Slots use the RPCHeader/RPCResponse
//               layout the dispatcher expects.

cudaq_rx_status_t opnic_host_rx_acquire(void *host_ctx, void **out_rx_slot,
                                        size_t *out_slot_size) {
  auto *ctx = static_cast<OpnicBridgeCpuContext *>(host_ctx);
  const opnic_cpu_transport_ctx &t = ctx->transport_ctx;

  if (static_cast<std::int32_t>(*t.pi_ptr - ctx->consumer_index) < 1)
    return CUDAQ_RX_EMPTY;
  const std::uint32_t slot =
      ctx->consumer_index % static_cast<std::uint32_t>(t.buf_count);
  *out_rx_slot = reinterpret_cast<std::uint8_t *>(t.rx_buffer) +
                 slot * t.rx_alloc_size;
  *out_slot_size = t.rx_alloc_size;
  ctx->consumer_index++;
  return CUDAQ_RX_READY;
}

cudaq_status_t opnic_host_tx_acquire(void *host_ctx, void **out_tx_slot,
                                     size_t *out_slot_size) {
  auto *ctx = static_cast<OpnicBridgeCpuContext *>(host_ctx);
  const opnic_cpu_transport_ctx &t = ctx->transport_ctx;

  const std::uint32_t slot =
      ctx->out_index % static_cast<std::uint32_t>(t.buf_count);
  *out_tx_slot = reinterpret_cast<std::uint8_t *>(t.tx_buffer) +
                 slot * t.tx_alloc_size;
  *out_slot_size = t.tx_alloc_size;
  return CUDAQ_OK;
}

cudaq_status_t opnic_host_tx_commit(void *host_ctx) {
  auto *ctx = static_cast<OpnicBridgeCpuContext *>(host_ctx);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  *ctx->transport_ctx.doorbell_ptr = 1;
  ctx->out_index++;
  return CUDAQ_OK;
}

// get_host_dataplane(): bundle the three ops above plus your context pointer
// for the dispatcher (which calls cudaq_dispatcher_set_host_dataplane).  Leave
// this vtable slot NULL if you do not implement a host generic path.
cudaq_status_t
opnic_bridge_cpu_get_host_dataplane(cudaq_realtime_bridge_handle_t handle,
                                    cudaq_host_dataplane_t *out) {
  if (!handle || !out)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<OpnicBridgeCpuContext *>(handle);
  if (!ctx->config.unified)
    return CUDAQ_ERR_INVALID_ARG;
  out->host_ctx = ctx;
  out->rx_acquire = opnic_host_rx_acquire;
  out->tx_acquire = opnic_host_tx_acquire;
  out->tx_commit = opnic_host_tx_commit;
  return CUDAQ_OK;
}

} // namespace

//=============================================================================
// 3b. Explicit construction helper (this example's construction model)
//=============================================================================
// The example builds the context directly (full config in hand) instead of
// going through create()/argv.  Returns the opaque handle; the caller releases
// it via the vtable's destroy().
extern "C" cudaq_realtime_bridge_handle_t
opnic_bridge_cpu_create_context(const OpnicBridgeCpuConfig *cfg) {
  if (!cfg)
    return nullptr;
  try {
    return new OpnicBridgeCpuContext(*cfg);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: Failed to create OpnicBridgeCpuContext: " << e.what()
              << std::endl;
    return nullptr;
  }
}

//=============================================================================
// 4. Publish the vtable
//=============================================================================
// The runtime obtains every entry point through this single getter.  Field
// order matches cudaq_realtime_bridge_interface_t exactly; optional getters you
// do not support are NULL (here: no device data-plane on the CPU bridge).  The
// returned pointer is to a static instance -- callers must not free it.
extern "C" cudaq_realtime_bridge_interface_t *
cudaq_realtime_get_opnic_cpu_bridge_interface() {
  static cudaq_realtime_bridge_interface_t iface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      opnic_bridge_cpu_create,
      opnic_bridge_cpu_destroy,
      opnic_bridge_cpu_get_transport_context,
      opnic_bridge_cpu_connect,
      opnic_bridge_cpu_launch,
      opnic_bridge_cpu_disconnect,
      opnic_bridge_cpu_get_host_dataplane,
      nullptr, // get_device_dataplane (CPU bridge has none)
  };
  return &iface;
}
