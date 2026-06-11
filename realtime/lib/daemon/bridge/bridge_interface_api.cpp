/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file bridge_interface_api.cpp
/// @brief API implementation for transport layer bridge interface.
///
/// This file provides the implementation of the API functions declared in
/// bridge_interface.h.  It manages the loading of transport provider shared
/// libraries, retrieval of their interface structs, and dispatch of API calls
/// to the appropriate provider implementation based on the bridge handle.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
namespace {
std::unordered_map<cudaq_realtime_transport_provider_t,
                   cudaq_realtime_bridge_interface_t *>
    provider_interface_map;

std::unordered_map<cudaq_realtime_bridge_handle_t,
                   cudaq_realtime_bridge_interface_t *>
    bridge_handle_interface_map;

// Mutex to protect access to global maps (provider_interface_map and
// bridge_handle_interface_map) for thread safety.
std::shared_mutex bridge_interface_mutex;

/// @brief Path to the built-in Hololink bridge library.  This is used when the
/// provider is CUDAQ_PROVIDER_HOLOLINK to load the Hololink implementation of
/// the bridge interface.  The library must be present at the load path (e.g.,
/// LD_LIBRARY_PATH) for the built-in provider to work.
const char *Hololink_Bridge_Lib = "libcudaq-realtime-bridge-hololink.so";

enum class BridgeSessionKind {
  Dispatcher,
  HostRingLoop,
  HostUnifiedFused,
  HostUnifiedData
};

bool has_capability(uint64_t capabilities, uint64_t capability) {
  return (capabilities & capability) != 0;
}

bool has_graph_launch_entries(const cudaq_function_table_t &table) {
  for (uint32_t i = 0; i < table.count; ++i) {
    if (table.entries[i].dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
      return true;
  }
  return false;
}

struct BridgeRPCHeader {
  uint32_t magic;
  uint32_t function_id;
  uint32_t arg_len;
  uint32_t request_id;
  uint64_t ptp_timestamp;
};

struct BridgeRPCResponse {
  uint32_t magic;
  uint32_t status;
  uint32_t result_len;
  uint32_t request_id;
  uint64_t ptp_timestamp;
};

size_t response_bytes(const cudaq_bridge_slot_t &tx_slot) {
  if (!tx_slot.slot || tx_slot.size < sizeof(BridgeRPCResponse))
    return tx_slot.size;

  const auto *response = static_cast<const BridgeRPCResponse *>(tx_slot.slot);
  if (response->magic != CUDAQ_RPC_MAGIC_RESPONSE)
    return tx_slot.size;

  const size_t bytes =
      sizeof(BridgeRPCResponse) + static_cast<size_t>(response->result_len);
  return bytes < tx_slot.size ? bytes : tx_slot.size;
}

bool dispatch_host_dataplane_slot(const cudaq_function_table_t &table,
                                  const cudaq_bridge_slot_t &rx_slot,
                                  const cudaq_bridge_slot_t &tx_slot,
                                  size_t *out_response_bytes) {
  if (!rx_slot.slot || !tx_slot.slot || !out_response_bytes ||
      rx_slot.size < sizeof(BridgeRPCHeader))
    return false;

  const auto *request = static_cast<const BridgeRPCHeader *>(rx_slot.slot);
  if (request->magic != CUDAQ_RPC_MAGIC_REQUEST)
    return false;

  for (uint32_t i = 0; i < table.count; ++i) {
    auto &entry = table.entries[i];
    if (entry.dispatch_mode != CUDAQ_DISPATCH_HOST_CALL ||
        entry.function_id != request->function_id || !entry.handler.host_fn)
      continue;

    const size_t slot_size =
        rx_slot.size < tx_slot.size ? rx_slot.size : tx_slot.size;
    entry.handler.host_fn(rx_slot.slot, tx_slot.slot, slot_size);
    *out_response_bytes = response_bytes(tx_slot);
    return true;
  }

  return false;
}
} // namespace

struct cudaq_bridge_dispatch_session {
  cudaq_realtime_bridge_handle_t bridge = nullptr;
  cudaq_realtime_bridge_interface_t *bridge_interface = nullptr;
  cudaq_bridge_dispatch_session_config_t config{};
  BridgeSessionKind kind = BridgeSessionKind::Dispatcher;
  bool started = false;

  cudaq_dispatch_manager_t *manager = nullptr;
  cudaq_dispatcher_t *dispatcher = nullptr;

  cudaq_ringbuffer_t ringbuffer{};
  cudaq_host_dispatch_loop_ctx_t host_loop_ctx{};
  std::thread host_loop_thread;

  cudaq_host_unified_fused_ctx_t host_fused{};
  cudaq_host_unified_dataplane_ctx_t host_dataplane{};
  std::thread host_dataplane_thread;
};

namespace {
cudaq_status_t
destroy_dispatcher_resources(cudaq_bridge_dispatch_session *session) {
  if (!session)
    return CUDAQ_ERR_INVALID_ARG;

  cudaq_status_t status = CUDAQ_OK;
  if (session->dispatcher) {
    status = cudaq_dispatcher_destroy(session->dispatcher);
    session->dispatcher = nullptr;
  }
  if (session->manager) {
    const auto manager_status =
        cudaq_dispatch_manager_destroy(session->manager);
    if (status == CUDAQ_OK)
      status = manager_status;
    session->manager = nullptr;
  }
  return status;
}

cudaq_status_t create_dispatcher(cudaq_bridge_dispatch_session *session) {
  auto status = cudaq_dispatch_manager_create(&session->manager);
  if (status != CUDAQ_OK)
    return status;

  status = cudaq_dispatcher_create(session->manager,
                                   &session->config.dispatcher_config,
                                   &session->dispatcher);
  if (status != CUDAQ_OK) {
    destroy_dispatcher_resources(session);
    return status;
  }

  status = cudaq_dispatcher_set_function_table(session->dispatcher,
                                               &session->config.function_table);
  if (status != CUDAQ_OK)
    return status;

  status = cudaq_dispatcher_set_control(session->dispatcher,
                                        session->config.shutdown_flag,
                                        session->config.stats);
  if (status != CUDAQ_OK)
    return status;

  if (session->config.mailbox) {
    status = cudaq_dispatcher_set_mailbox(session->dispatcher,
                                          session->config.mailbox);
    if (status != CUDAQ_OK)
      return status;
  }

  session->kind = BridgeSessionKind::Dispatcher;
  return CUDAQ_OK;
}

cudaq_status_t configure_device_session(cudaq_bridge_dispatch_session *session,
                                        uint64_t capabilities) {
  if (session->config.dispatcher_config.kernel_type == CUDAQ_KERNEL_UNIFIED) {
    if (!has_capability(capabilities, CUDAQ_BRIDGE_CAP_DEVICE_UNIFIED_LAUNCH))
      return CUDAQ_ERR_INVALID_ARG;

    cudaq_device_unified_dispatch_ctx_t unified{};
    auto status = session->bridge_interface->get_transport_context(
        session->bridge, DEVICE_UNIFIED, &unified);
    if (status != CUDAQ_OK)
      return status;
    if (!unified.launch_fn || !unified.transport_ctx)
      return CUDAQ_ERR_INVALID_ARG;

    status = create_dispatcher(session);
    if (status != CUDAQ_OK)
      return status;
    return cudaq_dispatcher_set_unified_launch(
        session->dispatcher, unified.launch_fn, unified.transport_ctx);
  }

  if (!has_capability(capabilities, CUDAQ_BRIDGE_CAP_DEVICE_RING))
    return CUDAQ_ERR_INVALID_ARG;

  auto status = session->bridge_interface->get_transport_context(
      session->bridge, RING_BUFFER, &session->ringbuffer);
  if (status != CUDAQ_OK)
    return status;

  status = create_dispatcher(session);
  if (status != CUDAQ_OK)
    return status;

  status = cudaq_dispatcher_set_ringbuffer(session->dispatcher,
                                           &session->ringbuffer);
  if (status != CUDAQ_OK)
    return status;

  cudaq_dispatch_launch_fn_t launch_fn = session->config.launch_fn;
  if (!launch_fn)
    return CUDAQ_ERR_INVALID_ARG;
  return cudaq_dispatcher_set_launch_fn(session->dispatcher, launch_fn);
}

cudaq_status_t
configure_host_ring_loop(cudaq_bridge_dispatch_session *session) {
  if (session->config.dispatcher_config.num_slots == 0 ||
      session->config.dispatcher_config.slot_size == 0)
    return CUDAQ_ERR_INVALID_ARG;

  std::memset(&session->host_loop_ctx, 0, sizeof(session->host_loop_ctx));
  session->host_loop_ctx.ringbuffer = session->ringbuffer;
  session->host_loop_ctx.config = session->config.dispatcher_config;
  session->host_loop_ctx.function_table = session->config.function_table;
  session->host_loop_ctx.shutdown_flag =
      (void *)(uintptr_t)session->config.shutdown_flag;
  session->host_loop_ctx.stats_counter = session->config.stats;
  session->host_loop_ctx.skip_stream_sweep = true;
  session->kind = BridgeSessionKind::HostRingLoop;
  return CUDAQ_OK;
}

cudaq_status_t
configure_host_ring_session(cudaq_bridge_dispatch_session *session,
                            uint64_t capabilities) {
  if (!has_capability(capabilities, CUDAQ_BRIDGE_CAP_HOST_RING))
    return CUDAQ_ERR_INVALID_ARG;

  auto status = session->bridge_interface->get_transport_context(
      session->bridge, RING_BUFFER, &session->ringbuffer);
  if (status != CUDAQ_OK)
    return status;

  if (session->config.dispatcher_config.slot_size == 0) {
    const size_t slot_size =
        session->ringbuffer.rx_stride_sz < session->ringbuffer.tx_stride_sz
            ? session->ringbuffer.rx_stride_sz
            : session->ringbuffer.tx_stride_sz;
    if (slot_size == 0)
      return CUDAQ_ERR_INVALID_ARG;
    session->config.dispatcher_config.slot_size =
        static_cast<uint32_t>(slot_size);
  }

  if (!has_graph_launch_entries(session->config.function_table))
    return configure_host_ring_loop(session);

  status = create_dispatcher(session);
  if (status != CUDAQ_OK)
    return status;
  return cudaq_dispatcher_set_ringbuffer(session->dispatcher,
                                         &session->ringbuffer);
}

cudaq_status_t
configure_host_unified_session(cudaq_bridge_dispatch_session *session,
                               uint64_t capabilities) {
  if (has_capability(capabilities, CUDAQ_BRIDGE_CAP_HOST_UNIFIED_FUSED)) {
    auto status = session->bridge_interface->get_transport_context(
        session->bridge, HOST_UNIFIED_FUSED, &session->host_fused);
    if (status != CUDAQ_OK)
      return status;
    if (!session->host_fused.start_fn || !session->host_fused.stop_fn ||
        !session->host_fused.transport_ctx)
      return CUDAQ_ERR_INVALID_ARG;
    session->kind = BridgeSessionKind::HostUnifiedFused;
    return CUDAQ_OK;
  }

  if (has_capability(capabilities, CUDAQ_BRIDGE_CAP_HOST_UNIFIED_DATA)) {
    auto status = session->bridge_interface->get_transport_context(
        session->bridge, HOST_UNIFIED_DATA, &session->host_dataplane);
    if (status != CUDAQ_OK)
      return status;
    if (!session->host_dataplane.rx_acquire ||
        !session->host_dataplane.rx_release ||
        !session->host_dataplane.tx_acquire ||
        !session->host_dataplane.tx_commit ||
        !session->host_dataplane.transport_ctx)
      return CUDAQ_ERR_INVALID_ARG;
    session->kind = BridgeSessionKind::HostUnifiedData;
    return CUDAQ_OK;
  }

  return CUDAQ_ERR_INVALID_ARG;
}

cudaq_status_t configure_session(cudaq_bridge_dispatch_session *session,
                                 uint64_t capabilities) {
  const auto &dispatcher_config = session->config.dispatcher_config;
  if (dispatcher_config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    if (dispatcher_config.kernel_type == CUDAQ_KERNEL_UNIFIED)
      return configure_host_unified_session(session, capabilities);
    return configure_host_ring_session(session, capabilities);
  }

  if (dispatcher_config.dispatch_path == CUDAQ_DISPATCH_PATH_DEVICE)
    return configure_device_session(session, capabilities);

  return CUDAQ_ERR_INVALID_ARG;
}

void run_host_dataplane_loop(cudaq_bridge_dispatch_session *session) {
  auto &dataplane = session->host_dataplane;
  while (*session->config.shutdown_flag == 0) {
    cudaq_bridge_slot_t rx_slot{};
    const auto rx_status =
        dataplane.rx_acquire(dataplane.transport_ctx, &rx_slot);
    if (rx_status == CUDAQ_BRIDGE_RX_EMPTY) {
      std::this_thread::yield();
      continue;
    }
    if (rx_status != CUDAQ_BRIDGE_RX_OK)
      break;

    cudaq_bridge_slot_t tx_slot{};
    const auto tx_status =
        dataplane.tx_acquire(dataplane.transport_ctx, &rx_slot, &tx_slot);
    if (tx_status == CUDAQ_OK) {
      size_t bytes = 0;
      if (dispatch_host_dataplane_slot(session->config.function_table, rx_slot,
                                       tx_slot, &bytes)) {
        if (dataplane.tx_commit(dataplane.transport_ctx, &tx_slot, bytes) ==
            CUDAQ_OK)
          *session->config.stats += 1;
      }
    }

    if (dataplane.rx_release(dataplane.transport_ctx, &rx_slot) != CUDAQ_OK)
      break;
  }
}
} // namespace

cudaq_status_t
cudaq_bridge_create(cudaq_realtime_bridge_handle_t *out_bridge_handle,
                    cudaq_realtime_transport_provider_t provider, int argc,
                    char **argv) {
  // For create, hold an unique lock.
  std::unique_lock<std::shared_mutex> lock(bridge_interface_mutex);

  if (!out_bridge_handle)
    return CUDAQ_ERR_INVALID_ARG;

  const auto it = provider_interface_map.find(provider);
  if (it != provider_interface_map.end()) {
    auto *bridge_interface = it->second;
    const auto status = bridge_interface->create(out_bridge_handle, argc, argv);
    if (status == CUDAQ_OK) {
      if (!*out_bridge_handle)
        return CUDAQ_ERR_INTERNAL;
      bridge_handle_interface_map[*out_bridge_handle] = bridge_interface;
    }
    return status;
  }

  const std::string lib_name = [&]() {
    if (provider == CUDAQ_PROVIDER_HOLOLINK) {
      return Hololink_Bridge_Lib;
    } else {
      const char *bridgeLibPath = std::getenv("CUDAQ_REALTIME_BRIDGE_LIB");
      if (!bridgeLibPath) {
        std::cerr << "ERROR: CUDAQ_REALTIME_BRIDGE_LIB environment variable "
                     "not set for EXTERNAL provider"
                  << std::endl;
        return "";
      }
      return bridgeLibPath;
    }
  }();

  if (lib_name.empty())
    return CUDAQ_ERR_INVALID_ARG;
  dlerror(); // reset errors

  void *lib_handle = dlopen(lib_name.c_str(), RTLD_NOW);

  if (!lib_handle) {
    std::cerr << "ERROR: Failed to load bridge library '" << lib_name
              << "': " << dlerror() << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  using GetInterfaceFunction = cudaq_realtime_bridge_interface_t *(*)();
  GetInterfaceFunction fcn = (GetInterfaceFunction)(intptr_t)dlsym(
      lib_handle, "cudaq_realtime_get_bridge_interface");
  if (!fcn) {
    std::cerr << "ERROR: Failed to interface getter from '" << lib_name
              << "': " << dlerror() << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  cudaq_realtime_bridge_interface_t *bridge_interface = fcn();

  if (!bridge_interface) {
    std::cerr << "ERROR: Bridge interface getter returned null from '"
              << lib_name << "'" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  provider_interface_map[provider] = bridge_interface;

  // Check interface version compatibility
  if (bridge_interface->version != CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION) {
    std::cerr << "ERROR: Bridge interface version mismatch for '" << lib_name
              << "': expected " << CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION
              << ", got " << bridge_interface->version << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  if (!bridge_interface->create || !bridge_interface->destroy ||
      !bridge_interface->get_transport_context || !bridge_interface->connect ||
      !bridge_interface->launch || !bridge_interface->disconnect ||
      !bridge_interface->get_capabilities) {
    std::cerr << "ERROR: Bridge interface from '" << lib_name
              << "' is missing required callbacks" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  // Run the create callback to allow the bridge to perform any initial setup
  const auto status = bridge_interface->create(out_bridge_handle, argc, argv);
  if (status == CUDAQ_OK) {
    if (!*out_bridge_handle)
      return CUDAQ_ERR_INTERNAL;
    bridge_handle_interface_map[*out_bridge_handle] = bridge_interface;
  }
  return status;
}

cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge) {
  // For destroy, hold an unique lock.
  std::unique_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in destroy" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  const auto status = bridge_interface->destroy(bridge);
  if (status == CUDAQ_OK) {
    bridge_handle_interface_map.erase(it);
  }
  return status;
}

// Retrieve the transport context information for the given bridge.
cudaq_status_t cudaq_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t bridge,
    cudaq_realtime_transport_context_t context_type, void *out_context) {
  // Hold a shared lock since this is a read-only operation on the global maps.
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in get_transport_context"
              << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->get_transport_context(bridge, context_type,
                                                 out_context);
}

cudaq_status_t
cudaq_bridge_get_capabilities(cudaq_realtime_bridge_handle_t bridge,
                              uint64_t *out_capabilities) {
  if (!out_capabilities)
    return CUDAQ_ERR_INVALID_ARG;

  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);
  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in get_capabilities"
              << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }

  auto *bridge_interface = it->second;
  if (!bridge_interface->get_capabilities)
    return CUDAQ_ERR_INTERNAL;
  *out_capabilities = bridge_interface->get_capabilities(bridge);
  return CUDAQ_OK;
}

cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge) {
  // Hold a shared lock since this is a read-only operation on the global maps.
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in connect" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->connect(bridge);
}

cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge) {
  // Hold a shared lock since this is a read-only operation on the global maps.
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in launch" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->launch(bridge);
}

cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge) {
  // Hold a shared lock since this is a read-only operation on the global maps.
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in disconnect" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->disconnect(bridge);
}

cudaq_status_t cudaq_bridge_dispatch_session_create(
    cudaq_realtime_bridge_handle_t bridge,
    const cudaq_bridge_dispatch_session_config_t *config,
    cudaq_bridge_dispatch_session_t **out_session) {
  if (!config || !out_session || !config->function_table.entries ||
      config->function_table.count == 0 || !config->shutdown_flag ||
      !config->stats)
    return CUDAQ_ERR_INVALID_ARG;

  *out_session = nullptr;

  cudaq_realtime_bridge_interface_t *bridge_interface = nullptr;
  {
    std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);
    const auto it = bridge_handle_interface_map.find(bridge);
    if (it == bridge_handle_interface_map.end()) {
      std::cerr << "ERROR: Invalid bridge handle in dispatch_session_create"
                << std::endl;
      return CUDAQ_ERR_INVALID_ARG;
    }
    bridge_interface = it->second;
  }

  if (!bridge_interface || !bridge_interface->get_capabilities)
    return CUDAQ_ERR_INTERNAL;

  auto *session = new (std::nothrow) cudaq_bridge_dispatch_session();
  if (!session)
    return CUDAQ_ERR_INTERNAL;

  session->bridge = bridge;
  session->bridge_interface = bridge_interface;
  session->config = *config;

  const uint64_t capabilities = bridge_interface->get_capabilities(bridge);
  auto status = configure_session(session, capabilities);
  if (status != CUDAQ_OK) {
    destroy_dispatcher_resources(session);
    delete session;
    return status;
  }

  *out_session = session;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_bridge_dispatch_session_start(cudaq_bridge_dispatch_session_t *session) {
  if (!session)
    return CUDAQ_ERR_INVALID_ARG;
  if (session->started)
    return CUDAQ_OK;

  if (session->config.shutdown_flag)
    *session->config.shutdown_flag = 0;

  cudaq_status_t status = CUDAQ_OK;
  switch (session->kind) {
  case BridgeSessionKind::Dispatcher:
    status = cudaq_dispatcher_start(session->dispatcher);
    break;
  case BridgeSessionKind::HostRingLoop:
    session->host_loop_thread = std::thread(
        [session]() { cudaq_host_dispatcher_loop(&session->host_loop_ctx); });
    status = CUDAQ_OK;
    break;
  case BridgeSessionKind::HostUnifiedFused:
    status = session->host_fused.start_fn(
        session->host_fused.transport_ctx,
        session->config.function_table.entries,
        session->config.function_table.count, session->config.shutdown_flag,
        session->config.stats);
    break;
  case BridgeSessionKind::HostUnifiedData:
    session->host_dataplane_thread =
        std::thread([session]() { run_host_dataplane_loop(session); });
    status = CUDAQ_OK;
    break;
  }

  if (status != CUDAQ_OK)
    return status;

  session->started = true;
  status = session->bridge_interface->launch(session->bridge);
  if (status != CUDAQ_OK) {
    cudaq_bridge_dispatch_session_stop(session);
    return status;
  }
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_bridge_dispatch_session_stop(cudaq_bridge_dispatch_session_t *session) {
  if (!session)
    return CUDAQ_ERR_INVALID_ARG;
  if (!session->started)
    return CUDAQ_OK;

  if (session->config.shutdown_flag)
    *session->config.shutdown_flag = 1;

  cudaq_status_t status = CUDAQ_OK;
  switch (session->kind) {
  case BridgeSessionKind::Dispatcher:
    status = cudaq_dispatcher_stop(session->dispatcher);
    break;
  case BridgeSessionKind::HostRingLoop:
    if (session->host_loop_thread.joinable())
      session->host_loop_thread.join();
    break;
  case BridgeSessionKind::HostUnifiedFused:
    status = session->host_fused.stop_fn(session->host_fused.transport_ctx);
    break;
  case BridgeSessionKind::HostUnifiedData:
    if (session->host_dataplane_thread.joinable())
      session->host_dataplane_thread.join();
    break;
  }

  if (status == CUDAQ_OK)
    session->started = false;
  return status;
}

cudaq_status_t cudaq_bridge_dispatch_session_destroy(
    cudaq_bridge_dispatch_session_t *session) {
  if (!session)
    return CUDAQ_ERR_INVALID_ARG;

  auto status = cudaq_bridge_dispatch_session_stop(session);
  const auto destroy_status = destroy_dispatcher_resources(session);
  if (status == CUDAQ_OK)
    status = destroy_status;
  delete session;
  return status;
}

cudaq_status_t cudaq_bridge_dispatch_session_get_processed(
    cudaq_bridge_dispatch_session_t *session, uint64_t *out_packets) {
  if (!session || !out_packets)
    return CUDAQ_ERR_INVALID_ARG;

  if (session->kind == BridgeSessionKind::Dispatcher && session->dispatcher)
    return cudaq_dispatcher_get_processed(session->dispatcher, out_packets);

  if (!session->config.stats)
    return CUDAQ_ERR_INVALID_ARG;
  *out_packets = *session->config.stats;
  return CUDAQ_OK;
}
