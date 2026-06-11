/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file bridge_interface.h
/// @brief Interface Bindings for transport layer providers (e.g. Hololink).
///
/// Different transport providers can be loaded at runtime via `dlopen`,
/// allowing for dynamic selection and initialization of the desired transport
/// layer. Environment variable CUDAQ_REALTIME_BRIDGE_LIB must be set to the
/// path of the shared library implementing the desired transport provider (if
/// not using the built-in Hololink provider).

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#ifdef __cplusplus
extern "C" {
#endif

///@brief Opaque data structure storing the details of the transport layer
/// connection
typedef void *cudaq_realtime_bridge_handle_t;

typedef enum {
  CUDAQ_PROVIDER_HOLOLINK =
      0, /// Hololink GPU-RoCE transceiver (built-in provider)
  CUDAQ_PROVIDER_EXTERNAL = 1, /// Externally managed transport

} cudaq_realtime_transport_provider_t;

typedef enum {
  RING_BUFFER = 0,        /// Ring buffer transport context.
  DEVICE_UNIFIED = 1,     /// Device-side unified dispatch context.
  HOST_UNIFIED_FUSED = 2, /// Provider-owned host unified dispatch loop.
  HOST_UNIFIED_DATA = 3, /// Provider host unified data-plane primitives.
} cudaq_realtime_transport_context_t;

#define CUDAQ_BRIDGE_CAP_HOST_RING (1ull << 0)
#define CUDAQ_BRIDGE_CAP_DEVICE_RING (1ull << 1)
#define CUDAQ_BRIDGE_CAP_HOST_UNIFIED_FUSED (1ull << 2)
#define CUDAQ_BRIDGE_CAP_HOST_UNIFIED_DATA (1ull << 3)
#define CUDAQ_BRIDGE_CAP_DEVICE_UNIFIED_LAUNCH (1ull << 4)

/// @brief Device-side unified bridge dispatch context.
typedef cudaq_unified_dispatch_ctx_t cudaq_device_unified_dispatch_ctx_t;

/// @brief Start a provider-owned host-side unified bridge loop.
typedef cudaq_status_t (*cudaq_bridge_host_unified_start_fn_t)(
    void *transport_ctx, cudaq_function_entry_t *function_table,
    size_t func_count, volatile int *shutdown_flag, uint64_t *stats);

/// @brief Stop a provider-owned host-side unified bridge loop.
typedef cudaq_status_t (*cudaq_bridge_host_unified_stop_fn_t)(
    void *transport_ctx);

/// @brief Provider-owned host-side unified bridge dispatch context.
typedef struct {
  cudaq_bridge_host_unified_start_fn_t start_fn;
  cudaq_bridge_host_unified_stop_fn_t stop_fn;
  void *transport_ctx;
} cudaq_host_unified_fused_ctx_t;

typedef struct {
  void *slot;
  size_t size;
  uint64_t token0;
  uint64_t token1;
} cudaq_bridge_slot_t;

typedef enum {
  CUDAQ_BRIDGE_RX_OK = 0,
  CUDAQ_BRIDGE_RX_EMPTY = 1,
  CUDAQ_BRIDGE_RX_ERROR = 2
} cudaq_bridge_rx_status_t;

/// @brief Acquire one inbound host slot. This function must be non-blocking:
/// return CUDAQ_BRIDGE_RX_EMPTY when no work is available so the bridge-owned
/// session can observe its shutdown flag.
typedef cudaq_bridge_rx_status_t (*cudaq_bridge_rx_acquire_fn_t)(
    void *transport_ctx, cudaq_bridge_slot_t *rx_slot);
/// @brief Release or re-arm a previously acquired inbound host slot.
typedef cudaq_status_t (*cudaq_bridge_rx_release_fn_t)(
    void *transport_ctx, const cudaq_bridge_slot_t *rx_slot);
/// @brief Acquire an outbound host slot for a response to the inbound slot.
typedef cudaq_status_t (*cudaq_bridge_tx_acquire_fn_t)(
    void *transport_ctx, const cudaq_bridge_slot_t *rx_slot,
    cudaq_bridge_slot_t *tx_slot);
/// @brief Commit `bytes` response bytes in the outbound host slot.
typedef cudaq_status_t (*cudaq_bridge_tx_commit_fn_t)(
    void *transport_ctx, const cudaq_bridge_slot_t *tx_slot, size_t bytes);

typedef struct {
  cudaq_bridge_rx_acquire_fn_t rx_acquire;
  cudaq_bridge_rx_release_fn_t rx_release;
  cudaq_bridge_tx_acquire_fn_t tx_acquire;
  cudaq_bridge_tx_commit_fn_t tx_commit;
  void *transport_ctx;
} cudaq_host_unified_dataplane_ctx_t;

typedef struct cudaq_bridge_dispatch_session cudaq_bridge_dispatch_session_t;

typedef struct {
  cudaq_dispatcher_config_t dispatcher_config;
  cudaq_function_table_t function_table;
  volatile int *shutdown_flag;
  uint64_t *stats;
  /// Device-ring dispatch launch function. Required only when the session uses
  /// CUDAQ_DISPATCH_PATH_DEVICE with a non-unified kernel type.
  cudaq_dispatch_launch_fn_t launch_fn;
  /// Optional caller-owned host dispatcher mailbox for graph-launch host paths.
  void **mailbox;
} cudaq_bridge_dispatch_session_config_t;

/// @brief Create and initialize a transport bridge for the specified provider.
/// For the built-in Hololink provider, this loads the Hololink shared library
/// and initializes the transceiver with the provided `args`.  For the EXTERNAL
/// provider, this loads the shared library specified by the
/// CUDAQ_REALTIME_BRIDGE_LIB environment variable and calls its create callback
/// to initialize the bridge.
cudaq_status_t
cudaq_bridge_create(cudaq_realtime_bridge_handle_t *out_bridge_handle,
                    cudaq_realtime_transport_provider_t provider, int argc,
                    char **argv);

/// @brief Destroy the transport bridge and release all associated resources.
cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge);

/// @brief Retrieve the transport context for the given bridge.
/// This could be a ring buffer or unified context.
cudaq_status_t cudaq_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t bridge,
    cudaq_realtime_transport_context_t context_type, void *out_context);

/// @brief Retrieve provider capabilities for the bridge handle.
cudaq_status_t
cudaq_bridge_get_capabilities(cudaq_realtime_bridge_handle_t bridge,
                              uint64_t *out_capabilities);

/// @brief Connect the transport bridge.
cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge);

/// @brief Launch the transport bridge's main processing loop (e.g. start
/// Hololink kernels).
cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge);

/// @brief Disconnect the transport bridge (e.g. stop Hololink kernels and
/// disconnect).
cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge);

/// @brief Create a bridge-owned dispatch session.
///
/// The session selects the provider capability that matches dispatcher_config:
/// host ring, provider-fused host unified, host unified data-plane, device
/// ring, or provider device-unified launch. Applications should prefer sessions
/// over direct cudaq_bridge_get_transport_context calls for dispatch.
cudaq_status_t cudaq_bridge_dispatch_session_create(
    cudaq_realtime_bridge_handle_t bridge,
    const cudaq_bridge_dispatch_session_config_t *config,
    cudaq_bridge_dispatch_session_t **out_session);

/// @brief Start dispatch and launch the provider transport processing path.
cudaq_status_t
cudaq_bridge_dispatch_session_start(cudaq_bridge_dispatch_session_t *session);

/// @brief Stop dispatch processing. The caller still owns bridge disconnect.
cudaq_status_t
cudaq_bridge_dispatch_session_stop(cudaq_bridge_dispatch_session_t *session);

cudaq_status_t
cudaq_bridge_dispatch_session_destroy(cudaq_bridge_dispatch_session_t *session);

cudaq_status_t cudaq_bridge_dispatch_session_get_processed(
    cudaq_bridge_dispatch_session_t *session, uint64_t *out_packets);

#define CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION 2

/// @brief Interface struct for transport layer providers.  Each provider must
/// implement this interface and provide a `getter` function
/// (`cudaq_realtime_get_bridge_interface`) that returns a pointer to a
/// statically allocated instance of this struct with the function pointers set
/// to the provider's implementation.
typedef struct {
  int version;
  cudaq_status_t (*create)(cudaq_realtime_bridge_handle_t *, int, char **);
  cudaq_status_t (*destroy)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*get_transport_context)(cudaq_realtime_bridge_handle_t,
                                          cudaq_realtime_transport_context_t,
                                          void *);
  cudaq_status_t (*connect)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*launch)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*disconnect)(cudaq_realtime_bridge_handle_t);
  uint64_t (*get_capabilities)(cudaq_realtime_bridge_handle_t);

} cudaq_realtime_bridge_interface_t;

#ifdef __cplusplus
}
#endif
