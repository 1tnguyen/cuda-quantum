/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

/// @file device_dispatch_rpc.cuh
/// @brief Shared __device__ per-slot DEVICE_CALL dispatch.
///
/// Device analog of the host `cudaq_host_dispatch_rpc`: it is the one piece of
/// per-slot logic every GPU dispatch loop needs, factored out so the fused
/// kernels and the library-owned generic kernel cannot drift on framing or
/// lookup semantics.  Header-only because device code that calls it must be
/// device-linked into the same module (RDC); there is no `.so` symbol.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cstddef>
#include <cstdint>

/// @brief Validate RPC_MAGIC_REQUEST on `rx_slot`, look up `function_id` in the
/// table, run the DEVICE_CALL handler, and frame the RPCResponse into
/// `tx_slot`.  Returns the framed length (sizeof(RPCResponse) + result_len), or
/// 0 to drop the slot (bad magic / unknown id / non-DEVICE_CALL / null
/// handler).
///
/// Mirrors the device handler ABI used by the fused kernels: the handler reads
/// args from `rx_slot + sizeof(RPCHeader)` and writes results to
/// `tx_slot + sizeof(RPCResponse)`.  `rx_slot` and `tx_slot` may be distinct
/// (separate RX/TX rings) or alias (in-place); header fields are cached before
/// the response header is written, so aliasing is safe.  When the slots alias,
/// the arg and result regions coincide (RPCHeader and RPCResponse are both
/// 24 B), matching the in-place pattern the OPNIC/DOCA fused kernels already
/// rely on.
__device__ inline std::size_t
cudaq_device_dispatch_rpc(const cudaq_function_entry_t *entries,
                          std::size_t count, const void *rx_slot, void *tx_slot,
                          std::size_t slot_size) {
  using cudaq::realtime::DeviceRPCFunction;
  using cudaq::realtime::RPC_MAGIC_REQUEST;
  using cudaq::realtime::RPC_MAGIC_RESPONSE;
  using cudaq::realtime::RPCHeader;
  using cudaq::realtime::RPCResponse;

  const auto *req = static_cast<const RPCHeader *>(rx_slot);
  if (req->magic != RPC_MAGIC_REQUEST)
    return 0;

  // Cache header fields before any write to tx_slot (rx/tx may alias).
  const std::uint32_t function_id = req->function_id;
  const std::uint32_t arg_len = req->arg_len;
  const std::uint32_t request_id = req->request_id;
  const std::uint64_t ptp_timestamp = req->ptp_timestamp;

  const cudaq_function_entry_t *entry = nullptr;
  for (std::size_t i = 0; i < count; ++i) {
    if (entries[i].function_id == function_id &&
        entries[i].dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL) {
      entry = &entries[i];
      break;
    }
  }

  int status = -1;
  std::uint32_t result_len = 0;
  if (entry != nullptr && entry->handler.device_fn_ptr != nullptr) {
    auto func =
        reinterpret_cast<DeviceRPCFunction>(entry->handler.device_fn_ptr);
    const void *arg_buffer =
        static_cast<const std::uint8_t *>(rx_slot) + sizeof(RPCHeader);
    void *out_buffer =
        static_cast<std::uint8_t *>(tx_slot) + sizeof(RPCResponse);
    const auto max_result_len =
        static_cast<std::uint32_t>(slot_size - sizeof(RPCResponse));
    status = func(arg_buffer, out_buffer, arg_len, max_result_len, &result_len);
  }

  auto *resp = static_cast<RPCResponse *>(tx_slot);
  resp->magic = RPC_MAGIC_RESPONSE;
  resp->status = status;
  resp->result_len = result_len;
  resp->request_id = request_id;
  resp->ptp_timestamp = ptp_timestamp;
  return sizeof(RPCResponse) + result_len;
}
