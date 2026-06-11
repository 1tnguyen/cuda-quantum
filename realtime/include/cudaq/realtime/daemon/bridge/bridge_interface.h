/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file bridge_interface.h
/// @brief Runtime plugin contract for transport layer providers (e.g. Hololink).
///
/// Providers are loaded via `dlopen` and expose a static
/// `cudaq_realtime_bridge_interface_t` vtable through
/// `cudaq_realtime_get_bridge_interface`.  Set `CUDAQ_REALTIME_BRIDGE_LIB` to
/// the shared library path when using `CUDAQ_PROVIDER_EXTERNAL`.
///
/// Required vtable entries cover bridge lifecycle and context export.  Optional
/// host and device data-plane getters (left NULL when unsupported) let the
/// library generic dispatch loops drive a transport without knowing its native
/// contract (producer index, doorbell, CQ, DOCA verbs, ...).

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#ifdef __cplusplus
extern "C" {
#endif

//===------------------------------------------------------------------------===
// Core identifiers
//===------------------------------------------------------------------------===

/// Opaque transport connection state.
typedef void *cudaq_realtime_bridge_handle_t;

typedef enum {
  CUDAQ_PROVIDER_HOLOLINK = 0, ///< Built-in Hololink provider.
  CUDAQ_PROVIDER_EXTERNAL = 1, ///< Externally managed transport (`dlopen`).
} cudaq_realtime_transport_provider_t;

typedef enum {
  RING_BUFFER = 0, ///< Ring-buffer context.
  UNIFIED = 1,     ///< Unified dispatch context.
} cudaq_realtime_transport_context_t;

/// Must match the `version` field providers set in their vtable instance.
#define CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION 1

//===------------------------------------------------------------------------===
// Shared data-plane protocol
//===------------------------------------------------------------------------===

/// Result of a non-blocking RX acquire on either host or device data-plane.
typedef enum {
  CUDAQ_RX_READY = 0,    ///< Packet available; *out_rx_slot is valid.
  CUDAQ_RX_EMPTY = 1,    ///< Nothing ready; caller should retry/relax.
  CUDAQ_RX_SHUTDOWN = 2, ///< Transport signalled shutdown; stop the loop.
} cudaq_rx_status_t;

//===------------------------------------------------------------------------===
// Host (CPU) data-plane
//===------------------------------------------------------------------------===

// Per-slot RX/TX ops for the library generic host loop
// (`cudaq_host_unified_generic_loop`).  Providers implement get_host_dataplane
// to fill cudaq_host_dataplane_t; callers resolve it once and pass the struct
// to the loop.  The transport owns host-resident RX/TX cursors in host_ctx;
// protocol concerns (e.g. the function_id == 0 shutdown convention) stay in the
// loop.  Providers without a host unified path leave get_host_dataplane NULL.
//
// rx_acquire: non-blocking.  On CUDAQ_RX_READY, *out_rx_slot / *out_slot_size
// describe a received packet and the RX cursor advances.  On CUDAQ_RX_EMPTY
// nothing is ready.  On CUDAQ_RX_SHUTDOWN the loop should end.
//
// tx_acquire: reports the current TX slot to write into.  Does NOT advance the
// TX cursor (tx_commit does), so a dropped request can skip tx_commit.
//
// tx_commit: publishes the current TX slot (ordering fence + doorbell) and
// advances the TX cursor.  Slots use RPCHeader / RPCResponse layout.

typedef cudaq_rx_status_t (*cudaq_host_rx_acquire_fn_t)(
    void *host_ctx, void **out_rx_slot, size_t *out_slot_size);
typedef cudaq_status_t (*cudaq_host_tx_acquire_fn_t)(
    void *host_ctx, void **out_tx_slot, size_t *out_slot_size);
typedef cudaq_status_t (*cudaq_host_tx_commit_fn_t)(void *host_ctx);

/// Host-callable RX/TX ops and host-resident transport state passed to the
/// library generic host loop.
typedef struct {
  void *host_ctx; ///< Host-resident transport state (cursors + ring ptrs).
  cudaq_host_rx_acquire_fn_t rx_acquire;
  cudaq_host_tx_acquire_fn_t tx_acquire;
  cudaq_host_tx_commit_fn_t tx_commit;
} cudaq_host_dataplane_t;

//===------------------------------------------------------------------------===
// Device (GPU) data-plane
//===------------------------------------------------------------------------===
//
// Device analog of the host data-plane above, for the library generic GPU
// dispatch kernel.  Device code cannot call host function pointers, so
// transports expose __device__ function addresses captured with
// cudaMemcpyFromSymbol and bundled in cudaq_device_dataplane_t.  The kernel
// invokes them on-device once per slot.  The transport owns device-resident
// RX/TX cursors in device_ctx; protocol concerns stay in the loop.  Providers
// without a GPU generic path leave get_device_dataplane NULL.
//
// Semantics mirror the host hooks; slots use RPCHeader / RPCResponse layout.
// Device tx_acquire / tx_commit report status via side effects on device_ctx
// rather than cudaq_status_t return codes.

typedef cudaq_rx_status_t (*cudaq_device_rx_acquire_fn_t)(
    void *device_ctx, void **out_rx_slot, size_t *out_slot_size);
typedef void (*cudaq_device_tx_acquire_fn_t)(void *device_ctx,
                                             void **out_tx_slot,
                                             size_t *out_slot_size);
typedef void (*cudaq_device_tx_commit_fn_t)(void *device_ctx);

/// Device-callable RX/TX ops and device-resident transport state passed to the
/// library generic GPU dispatch kernel.
typedef struct {
  void *device_ctx; ///< Device-resident transport state (cursors + ring ptrs).
  cudaq_device_rx_acquire_fn_t rx_acquire; ///< __device__ fn (MemcpyFromSymbol).
  cudaq_device_tx_acquire_fn_t tx_acquire;
  cudaq_device_tx_commit_fn_t tx_commit;
} cudaq_device_dataplane_t;

//===------------------------------------------------------------------------===
// Provider vtable
//===------------------------------------------------------------------------===

/// Interface struct for transport layer providers.  Each provider must
/// implement this interface and provide a getter function
/// (`cudaq_realtime_get_bridge_interface`) that returns a pointer to a
/// statically allocated instance with function pointers set to the provider's
/// implementation.  Trailing optional getters are zero-initialized (NULL) when
/// not supported.
typedef struct {
  int version;

  // Required lifecycle + context export.
  cudaq_status_t (*create)(cudaq_realtime_bridge_handle_t *, int, char **);
  cudaq_status_t (*destroy)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*get_transport_context)(cudaq_realtime_bridge_handle_t,
                                          cudaq_realtime_transport_context_t,
                                          void *);
  cudaq_status_t (*connect)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*launch)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*disconnect)(cudaq_realtime_bridge_handle_t);

  // Optional host (CPU) data-plane — fills *out for the generic host loop.
  cudaq_status_t (*get_host_dataplane)(cudaq_realtime_bridge_handle_t,
                                       cudaq_host_dataplane_t *out);

  // Optional device (GPU) data-plane — fills *out for the generic GPU kernel.
  cudaq_status_t (*get_device_dataplane)(cudaq_realtime_bridge_handle_t,
                                         cudaq_device_dataplane_t *out);

} cudaq_realtime_bridge_interface_t;

//===------------------------------------------------------------------------===
// Consumer bindings
//===------------------------------------------------------------------------===

/// Resolved host data-plane for the transport-agnostic host loop.  Fill via
/// `iface->get_host_dataplane(handle, &binding.dataplane)`, then hand the
/// binding to `cudaq_dispatcher_set_host_dataplane`; the dispatcher then runs
/// `cudaq_host_unified_generic_loop` on its own thread.
typedef struct {
  cudaq_host_dataplane_t dataplane;
} cudaq_host_transport_binding_t;

//===------------------------------------------------------------------------===
// Public bridge API
//===------------------------------------------------------------------------===

cudaq_status_t
cudaq_bridge_create(cudaq_realtime_bridge_handle_t *out_bridge_handle,
                    cudaq_realtime_transport_provider_t provider, int argc,
                    char **argv);

cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge);

cudaq_status_t cudaq_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t bridge,
    cudaq_realtime_transport_context_t context_type, void *out_context);

cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge);

cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge);

cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge);

#ifdef __cplusplus
}
#endif
