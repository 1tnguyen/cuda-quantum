/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

/// @file decoder_dispatch.hpp
/// @brief The decoder server's transport-agnostic dispatch core.
///
/// `poll_loop` services two `opnic_cpu_transport_ctx` rings on a single thread,
/// routing each RPC by `function_id` to a HOST_CALL handler in the (dlopen'd)
/// shim function table. It depends only on the realtime dispatch ABI and the
/// plain `opnic_cpu_transport_ctx` struct (pointers + sizes) -- NOT on the
/// OPNIC SDK. See decoder_server_cpu.cpp for the rationale behind the
/// single-thread, two-stream design.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h" // RPCHeader
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h" // CUDAQ_REALTIME_CPU_RELAX

#include "opnic_bridge_cpu.hpp" // opnic_cpu_transport_ctx (plain struct, no SDK)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace opnic_decoder {

/// Per-stream ring cursors (one consumer/producer pair per OPNIC stream).
struct StreamCursor {
  std::uint32_t consumer = 0; // next RX slot to read
  std::uint32_t out = 0;      // next TX slot to write
};

/// Dispatch one HOST_CALL RPC in place: route by function_id to the shim's
/// handler, copy the request into the TX slot, and let the handler rewrite that
/// slot as the RPCResponse (the canonical slot ABI). Returns true if a response
/// was produced (false => unknown function / bad magic => caller drops it).
inline bool dispatch_one(const cudaq_function_table_t &table,
                         const void *rx_slot, void *tx_slot,
                         std::size_t slot_size) {
  const auto *h = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  if (h->magic != cudaq::realtime::RPC_MAGIC_REQUEST)
    return false;
  cudaq_host_rpc_fn_t fn = nullptr;
  for (std::uint32_t i = 0; i < table.count; ++i)
    if (table.entries[i].dispatch_mode == CUDAQ_DISPATCH_HOST_CALL &&
        table.entries[i].function_id == h->function_id) {
      fn = table.entries[i].handler.host_fn;
      break;
    }
  if (!fn)
    return false;

  // Copy only header + the clamped argument bytes. For control packets this
  // avoids copying unused chunk payload capacity.
  std::uint32_t arg_len = h->arg_len;
  const std::size_t max_args = slot_size - sizeof(cudaq::realtime::RPCHeader);
  if (arg_len > max_args)
    arg_len = static_cast<std::uint32_t>(max_args);
  if (tx_slot != rx_slot)
    std::memcpy(tx_slot, rx_slot,
                sizeof(cudaq::realtime::RPCHeader) + arg_len);

  // Time only the HOST_CALL handler. For decoder RPCs, this is shim parsing plus
  // cuda-qx decoder work. It deliberately excludes OPNIC transport, host polling,
  // the RX->TX copy above, and the TX doorbell below. The response's
  // ptp_timestamp field is otherwise unused by these examples, so we reuse it as
  // a low-overhead timing channel back to QUA without changing packet size.
  const auto host_start = std::chrono::steady_clock::now();
  fn(tx_slot, slot_size); // handler reads the request and writes the response
  const auto host_end = std::chrono::steady_clock::now();
  const auto handler_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(host_end -
                                                           host_start)
          .count();
  auto *resp = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  if (resp->magic == cudaq::realtime::RPC_MAGIC_RESPONSE)
    resp->ptp_timestamp = static_cast<std::uint64_t>(handler_ns);
  return true;
}

/// Poll one OPNIC stream once. Returns 1 if a packet was dispatched, 0 if the
/// stream was idle, -1 if OPX sent the phase-complete packet (function_id == 0).
/// That packet is a control convention between the QUA program and this host
/// server; it is not a shim HOST_CALL and does not change the decoder bank.
inline int service_stream(const char *name, const opnic_cpu_transport_ctx &ctx,
                          StreamCursor &cur,
                          const cudaq_function_table_t &table,
                          std::uint64_t *stats) {
  if (static_cast<std::int32_t>(*ctx.pi_ptr - cur.consumer) < 1)
    return 0; // no new packet

  const std::uint32_t rx_idx =
      cur.consumer % static_cast<std::uint32_t>(ctx.buf_count);
  void *rx_slot = reinterpret_cast<std::uint8_t *>(ctx.rx_buffer) +
                  rx_idx * ctx.rx_alloc_size;

  const auto *h = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  if (h->function_id == 0) {
    std::fprintf(stderr,
                 "[HOST] %s phase-complete packet: magic=0x%08x req=%u\n",
                 name, h->magic, h->request_id);
    cur.consumer++;
    return -1; // OPX phase-complete convention
  }
  cur.consumer++;

  const std::uint32_t tx_idx =
      cur.out % static_cast<std::uint32_t>(ctx.buf_count);
  void *tx_slot = reinterpret_cast<std::uint8_t *>(ctx.tx_buffer) +
                  tx_idx * ctx.tx_alloc_size;
  const std::size_t slot_size = std::min(ctx.rx_alloc_size, ctx.tx_alloc_size);

  if (dispatch_one(table, rx_slot, tx_slot, slot_size)) {
    std::atomic_thread_fence(std::memory_order_seq_cst); // PCIe ordering
    *ctx.doorbell_ptr = 1;
    cur.out++;
    if (stats)
      (*stats)++;
  } else {
    const auto *w = static_cast<const std::uint32_t *>(rx_slot);
    std::fprintf(stderr,
                 "[HOST] dropped %s slot: pi=%u consumer=%u magic=0x%08x "
                 "fn=0x%08x arg_len=%u req=%u words=[0x%08x 0x%08x "
                 "0x%08x 0x%08x]\n",
                 name, *ctx.pi_ptr, cur.consumer, h->magic, h->function_id,
                 h->arg_len, h->request_id, w[0], w[1], w[2], w[3]);
  }
  return 1;
}

/// Single-thread loop servicing BOTH streams until `*shutdown_flag`. Data is
/// polled first each turn (hot path priority); configure shares the thread so
/// it can never race a decode against the shared decoder bank. Returns -1 when
/// OPX sends a phase-complete packet, allowing the outer server to tear down
/// this phase's OPNIC bridge contexts and reconnect for the next QUA program
/// while keeping the dlopen'd shim and decoder bank alive.
inline int poll_loop(const opnic_cpu_transport_ctx &data,
                     const opnic_cpu_transport_ctx &control,
                     const cudaq_function_table_t &table,
                     volatile int *shutdown_flag, std::uint64_t *decode_stats,
                     std::uint64_t *config_stats) {
  StreamCursor dcur, ccur;
  while (*shutdown_flag == 0) {
    const int d = service_stream("data", data, dcur, table, decode_stats);
    if (d < 0)
      return -1;
    const int c = service_stream("control", control, ccur, table, config_stats);
    if (c < 0)
      return -1;
    if (d == 0 && c == 0)
      CUDAQ_REALTIME_CPU_RELAX();
  }
  return 0;
}

} // namespace opnic_decoder
