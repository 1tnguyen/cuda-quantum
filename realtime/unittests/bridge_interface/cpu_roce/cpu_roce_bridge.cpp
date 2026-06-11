/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

namespace {

struct CpuRoceBridgeConfig {
  std::string device = "mlx5_0";
  std::string peer_ip = "192.168.0.2";
  std::string local_ip;
  unsigned remote_qp = 2;
  unsigned num_pages = 64;
  std::size_t page_size = 384;
  unsigned payload_size = 24;
  int timeout_sec = 60;
  bool unified = false;
  bool forward = false;
};

bool starts_with(const std::string &s, const char *prefix) {
  const std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool parse_args(int argc, char **argv, CpuRoceBridgeConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h")
      return false;
    if (starts_with(a, "--device="))
      cfg.device = a.substr(9);
    else if (starts_with(a, "--peer-ip="))
      cfg.peer_ip = a.substr(10);
    else if (starts_with(a, "--bridge-ip="))
      cfg.local_ip = a.substr(12);
    else if (starts_with(a, "--local-ip="))
      cfg.local_ip = a.substr(11);
    else if (starts_with(a, "--remote-qp="))
      cfg.remote_qp =
          static_cast<unsigned>(std::stoul(a.substr(12), nullptr, 0));
    else if (starts_with(a, "--num-pages="))
      cfg.num_pages = static_cast<unsigned>(std::stoul(a.substr(12)));
    else if (starts_with(a, "--page-size="))
      cfg.page_size = std::stoull(a.substr(12));
    else if (starts_with(a, "--payload-size="))
      cfg.payload_size = static_cast<unsigned>(std::stoul(a.substr(15)));
    else if (starts_with(a, "--timeout="))
      cfg.timeout_sec = std::stoi(a.substr(10));
    else if (a == "--unified")
      cfg.unified = true;
    else if (a == "--forward")
      cfg.forward = true;
    else {
      std::cerr << "Unknown CPU RoCE bridge argument: " << a << std::endl;
      return false;
    }
  }
  return !(cfg.unified && cfg.forward);
}

struct CpuRoceBridgeContext {
  CpuRoceBridgeConfig cfg;
  cpu_roce_transceiver_t xcvr = nullptr;
  std::thread monitor_thread;
  std::atomic<bool> connected{false};
  std::atomic<bool> launched{false};
  std::atomic<bool> host_unified_ready{false};
  cudaq_function_entry_t *function_table = nullptr;
  std::size_t func_count = 0;
  volatile int *shutdown_flag = nullptr;
  std::uint64_t *stats = nullptr;
  std::uint32_t local_qp = 0;
  std::uint32_t local_rkey = 0;
  std::uint64_t local_buffer = 0;
};

void print_ready(const CpuRoceBridgeContext &ctx) {
  std::cout << "\n=== Bridge Ready ===" << std::endl;
  std::cout << "  QP Number: 0x" << std::hex << ctx.local_qp << std::dec
            << std::endl;
  std::cout << "  RKey: " << ctx.local_rkey << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex << ctx.local_buffer << std::dec
            << std::endl;
  std::cout << "\nWaiting (Ctrl+C to stop, timeout=" << ctx.cfg.timeout_sec
            << "s)..." << std::endl;
  std::cout.flush();
}

std::size_t host_unified_dispatch(void *opaque_ctx, const void *rx_slot,
                                  void *tx_slot, std::size_t slot_size) {
  auto *ctx = static_cast<CpuRoceBridgeContext *>(opaque_ctx);
  if (!ctx || !ctx->function_table)
    return 0;
  if (ctx->shutdown_flag && *ctx->shutdown_flag)
    return 0;

  const auto *request =
      static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  if (request->magic != cudaq::realtime::RPC_MAGIC_REQUEST)
    return 0;

  for (std::size_t i = 0; i < ctx->func_count; ++i) {
    auto &entry = ctx->function_table[i];
    if (entry.dispatch_mode == CUDAQ_DISPATCH_HOST_CALL &&
        entry.function_id == request->function_id && entry.handler.host_fn) {
      entry.handler.host_fn(rx_slot, tx_slot, slot_size);
      if (ctx->stats)
        *ctx->stats += 1;
      return slot_size;
    }
  }
  return 0;
}

cudaq_status_t host_unified_start(void *transport_ctx,
                                  cudaq_function_entry_t *function_table,
                                  std::size_t func_count,
                                  volatile int *shutdown_flag,
                                  std::uint64_t *stats) {
  auto *ctx = static_cast<CpuRoceBridgeContext *>(transport_ctx);
  if (!ctx || !ctx->xcvr || !function_table || func_count == 0 ||
      !shutdown_flag || !stats)
    return CUDAQ_ERR_INVALID_ARG;
  if (ctx->monitor_thread.joinable())
    return CUDAQ_OK;

  ctx->function_table = function_table;
  ctx->func_count = func_count;
  ctx->shutdown_flag = shutdown_flag;
  ctx->stats = stats;
  cpu_roce_set_unified_dispatch(ctx->xcvr, host_unified_dispatch, ctx);
  ctx->host_unified_ready.store(true, std::memory_order_release);
  ctx->monitor_thread =
      std::thread([ctx]() { cpu_roce_blocking_monitor(ctx->xcvr); });
  return CUDAQ_OK;
}

cudaq_status_t host_unified_stop(void *transport_ctx) {
  auto *ctx = static_cast<CpuRoceBridgeContext *>(transport_ctx);
  if (!ctx || !ctx->xcvr)
    return CUDAQ_ERR_INVALID_ARG;
  if (ctx->shutdown_flag)
    *ctx->shutdown_flag = 1;
  cpu_roce_close(ctx->xcvr);
  if (ctx->monitor_thread.joinable())
    ctx->monitor_thread.join();
  ctx->host_unified_ready.store(false, std::memory_order_release);
  return CUDAQ_OK;
}

cudaq_status_t
cpu_roce_bridge_create(cudaq_realtime_bridge_handle_t *out_handle, int argc,
                       char **argv) {
  if (!out_handle)
    return CUDAQ_ERR_INVALID_ARG;

  auto *ctx = new CpuRoceBridgeContext();
  if (!ctx)
    return CUDAQ_ERR_INTERNAL;
  if (!parse_args(argc, argv, ctx->cfg)) {
    delete ctx;
    return CUDAQ_ERR_INVALID_ARG;
  }

  const std::size_t frame_size =
      sizeof(cudaq::realtime::RPCHeader) + ctx->cfg.payload_size;
  ctx->xcvr = cpu_roce_create_transceiver(
      ctx->cfg.device.c_str(), /*ib_port=*/1, ctx->cfg.remote_qp, frame_size,
      ctx->cfg.page_size, ctx->cfg.num_pages, ctx->cfg.peer_ip.c_str(),
      ctx->cfg.forward ? 1 : 0, /*rx_only=*/0, /*tx_only=*/0,
      ctx->cfg.unified ? 1 : 0, CPU_ROCE_TX_MODE_RDMA_SEND,
      /*peer_rx_base_addr=*/0, /*peer_rx_rkey=*/0);
  if (!ctx->xcvr) {
    delete ctx;
    return CUDAQ_ERR_INTERNAL;
  }
  if (!ctx->cfg.local_ip.empty())
    cpu_roce_set_local_ip(ctx->xcvr, ctx->cfg.local_ip.c_str());

  *out_handle = ctx;
  return CUDAQ_OK;
}

cudaq_status_t cpu_roce_bridge_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = static_cast<CpuRoceBridgeContext *>(handle);
  if (ctx->xcvr) {
    cpu_roce_close(ctx->xcvr);
    if (ctx->monitor_thread.joinable())
      ctx->monitor_thread.join();
    cpu_roce_destroy_transceiver(ctx->xcvr);
  }
  delete ctx;
  return CUDAQ_OK;
}

cudaq_status_t cpu_roce_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t handle,
    cudaq_realtime_transport_context_t context_type, void *out_context) {
  if (!handle || !out_context)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = static_cast<CpuRoceBridgeContext *>(handle);

  if (context_type == RING_BUFFER) {
    if (ctx->cfg.unified)
      return CUDAQ_ERR_INVALID_ARG;
    auto *ring = static_cast<cudaq_ringbuffer_t *>(out_context);
    ring->rx_flags_host = reinterpret_cast<volatile std::uint64_t *>(
        cpu_roce_get_rx_ring_flag_addr(ctx->xcvr));
    ring->tx_flags_host = reinterpret_cast<volatile std::uint64_t *>(
        cpu_roce_get_tx_ring_flag_addr(ctx->xcvr));
    ring->rx_data_host = reinterpret_cast<std::uint8_t *>(
        cpu_roce_get_rx_ring_data_addr(ctx->xcvr));
    ring->tx_data_host = reinterpret_cast<std::uint8_t *>(
        cpu_roce_get_tx_ring_data_addr(ctx->xcvr));
    ring->rx_flags = ring->rx_flags_host;
    ring->tx_flags = ring->tx_flags_host;
    ring->rx_data = ring->rx_data_host;
    ring->tx_data = ring->tx_data_host;
    ring->rx_stride_sz = ctx->cfg.page_size;
    ring->tx_stride_sz = ctx->cfg.page_size;
    return CUDAQ_OK;
  }

  if (context_type == HOST_UNIFIED_FUSED) {
    if (!ctx->cfg.unified)
      return CUDAQ_ERR_INVALID_ARG;
    auto *host = static_cast<cudaq_host_unified_fused_ctx_t *>(out_context);
    host->start_fn = host_unified_start;
    host->stop_fn = host_unified_stop;
    host->transport_ctx = ctx;
    return CUDAQ_OK;
  }

  return CUDAQ_ERR_INVALID_ARG;
}

cudaq_status_t cpu_roce_bridge_connect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = static_cast<CpuRoceBridgeContext *>(handle);
  if (!ctx->xcvr)
    return CUDAQ_ERR_INVALID_ARG;
  if (ctx->connected.load(std::memory_order_acquire))
    return CUDAQ_OK;

  if (!cpu_roce_start(ctx->xcvr))
    return CUDAQ_ERR_INTERNAL;
  ctx->local_qp = cpu_roce_get_qp_number(ctx->xcvr);
  ctx->local_rkey = cpu_roce_get_rkey(ctx->xcvr);
  ctx->local_buffer = cpu_roce_get_buffer_addr(ctx->xcvr);
  ctx->connected.store(true, std::memory_order_release);
  return CUDAQ_OK;
}

cudaq_status_t cpu_roce_bridge_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = static_cast<CpuRoceBridgeContext *>(handle);
  if (!ctx->connected.load(std::memory_order_acquire))
    return CUDAQ_ERR_INVALID_ARG;
  if (ctx->launched.exchange(true, std::memory_order_acq_rel))
    return CUDAQ_OK;

  if (ctx->cfg.unified) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!ctx->host_unified_ready.load(std::memory_order_acquire)) {
      if (std::chrono::steady_clock::now() > deadline)
        return CUDAQ_ERR_INTERNAL;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } else {
    ctx->monitor_thread =
        std::thread([ctx]() { cpu_roce_blocking_monitor(ctx->xcvr); });
  }

  print_ready(*ctx);
  return CUDAQ_OK;
}

cudaq_status_t
cpu_roce_bridge_disconnect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = static_cast<CpuRoceBridgeContext *>(handle);
  if (ctx->xcvr)
    cpu_roce_close(ctx->xcvr);
  if (ctx->monitor_thread.joinable())
    ctx->monitor_thread.join();
  ctx->connected.store(false, std::memory_order_release);
  ctx->launched.store(false, std::memory_order_release);
  return CUDAQ_OK;
}

std::uint64_t
cpu_roce_bridge_get_capabilities(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return 0;
  auto *ctx = static_cast<CpuRoceBridgeContext *>(handle);
  if (ctx->cfg.forward)
    return 0;
  if (ctx->cfg.unified)
    return CUDAQ_BRIDGE_CAP_HOST_UNIFIED_FUSED;
  return CUDAQ_BRIDGE_CAP_HOST_RING;
}

} // namespace

extern "C" cudaq_realtime_bridge_interface_t *
cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t iface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      cpu_roce_bridge_create,
      cpu_roce_bridge_destroy,
      cpu_roce_bridge_get_transport_context,
      cpu_roce_bridge_connect,
      cpu_roce_bridge_launch,
      cpu_roce_bridge_disconnect,
      cpu_roce_bridge_get_capabilities,
  };
  return &iface;
}
