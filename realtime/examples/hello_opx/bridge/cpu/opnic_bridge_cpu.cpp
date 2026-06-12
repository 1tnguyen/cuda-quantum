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
/// The bridge exposes a type-erased transport lifecycle. The default
/// `opnic_bridge_cpu_create_context` helper instantiates the incrementer packet
/// layout, while examples with larger packets instantiate
/// `opnic_bridge_cpu_create_context_for_packets<In, Out>` from
/// `opnic_bridge_cpu_typed.hpp` and still use this same vtable.

#include "opnic_bridge_cpu_typed.hpp"
#include "opnic_type.h"

#include <iostream>

namespace {

OpnicBridgeCpuContextBase *
as_context(cudaq_realtime_bridge_handle_t handle) {
  return reinterpret_cast<OpnicBridgeCpuContextBase *>(handle);
}

cudaq_status_t opnic_bridge_cpu_create(cudaq_realtime_bridge_handle_t *,
                                       int /*argc*/, char ** /*argv*/) {
  return CUDAQ_OK;
}

cudaq_status_t opnic_bridge_cpu_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  delete as_context(handle);
  return CUDAQ_OK;
}

cudaq_status_t opnic_bridge_cpu_get_transport_context(
    cudaq_realtime_bridge_handle_t handle,
    cudaq_realtime_transport_context_t context_type, void *out_context) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  return as_context(handle)->get_transport_context(context_type, out_context);
}

cudaq_status_t opnic_bridge_cpu_connect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  return as_context(handle)->connect();
}

cudaq_status_t opnic_bridge_cpu_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  return as_context(handle)->launch();
}

cudaq_status_t opnic_bridge_cpu_disconnect(
    cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  return as_context(handle)->disconnect();
}

cudaq_status_t
opnic_bridge_cpu_get_host_dataplane(cudaq_realtime_bridge_handle_t handle,
                                    cudaq_host_dataplane_t *out) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  return as_context(handle)->get_host_dataplane(out);
}

} // namespace

extern "C" cudaq_realtime_bridge_handle_t
opnic_bridge_cpu_create_context(const OpnicBridgeCpuConfig *cfg) {
  return opnic_bridge_cpu_create_context_for_packets<RPCInputPacket,
                                                     RPCOutputPacket>(cfg);
}

extern "C" cudaq_status_t opnic_bridge_cpu_sync_all() {
  try {
    qm::sync();
  } catch (const std::exception &e) {
    std::cerr << "ERROR: OPNIC global sync failed: " << e.what() << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  return CUDAQ_OK;
}

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
