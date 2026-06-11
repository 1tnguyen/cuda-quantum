/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#ifndef CUDAQ_REALTIME_CPU_ROCE_BRIDGE_LIB
#error                                                                         \
    "CUDAQ_REALTIME_CPU_ROCE_BRIDGE_LIB must name the CPU RoCE bridge library"
#endif

extern "C" void
setup_rpc_increment_function_table_host(cudaq_function_entry_t *h_entries);

namespace {

struct CpuBridgeAppConfig {
  unsigned num_pages = 64;
  int timeout_sec = 60;
  bool unified = false;
  bool forward = false;
};

bool starts_with(const std::string &s, const char *prefix) {
  const std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool parse_app_args(int argc, char **argv, CpuBridgeAppConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n\n"
          << "CPU RoCE bridge-interface HOST_CALL test binary. Transport "
             "arguments are forwarded to the bridge provider.\n\n"
          << "Options:\n"
          << "  --device=NAME       IB device (default: mlx5_0)\n"
          << "  --peer-ip=ADDR      Peer IPv4 (FPGA / emulator)\n"
          << "  --bridge-ip=ADDR    Local bridge IPv4 for source GID\n"
          << "  --local-ip=ADDR     Alias for --bridge-ip\n"
          << "  --remote-qp=N       Remote QP number\n"
          << "  --num-pages=N       Ring slots, power of two\n"
          << "  --page-size=N       Per-slot stride in bytes\n"
          << "  --payload-size=N    RPC payload bytes\n"
          << "  --timeout=N         Run timeout in seconds\n"
          << "  --unified           Use bridge-owned host unified session\n"
          << "  --forward           Transport echo baseline, no dispatch\n";
      return false;
    }
    if (starts_with(a, "--num-pages="))
      cfg.num_pages = static_cast<unsigned>(std::stoul(a.substr(12)));
    else if (starts_with(a, "--timeout="))
      cfg.timeout_sec = std::stoi(a.substr(10));
    else if (a == "--unified")
      cfg.unified = true;
    else if (a == "--forward")
      cfg.forward = true;
  }
  return !(cfg.unified && cfg.forward);
}

std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

} // namespace

int main(int argc, char **argv) {
  CpuBridgeAppConfig cfg;
  if (!parse_app_args(argc, argv, cfg))
    return 0;

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  if (setenv("CUDAQ_REALTIME_BRIDGE_LIB", CUDAQ_REALTIME_CPU_ROCE_BRIDGE_LIB,
             /*overwrite=*/1) != 0) {
    std::cerr << "ERROR: failed to set CUDAQ_REALTIME_BRIDGE_LIB" << std::endl;
    return 1;
  }

  cudaq_realtime_bridge_handle_t bridge = nullptr;
  if (cudaq_bridge_create(&bridge, CUDAQ_PROVIDER_EXTERNAL, argc, argv) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_bridge_create failed" << std::endl;
    return 1;
  }

  if (cudaq_bridge_connect(bridge) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_bridge_connect failed" << std::endl;
    cudaq_bridge_destroy(bridge);
    return 1;
  }

  cudaq_function_entry_t h_entries[1];
  if (!cfg.forward)
    setup_rpc_increment_function_table_host(h_entries);

  volatile int dispatcher_shutdown = 0;
  std::uint64_t packets_dispatched = 0;
  cudaq_bridge_dispatch_session_t *session = nullptr;

  if (!cfg.forward) {
    cudaq_bridge_dispatch_session_config_t session_config{};
    session_config.dispatcher_config.num_slots = cfg.num_pages;
    session_config.dispatcher_config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    session_config.dispatcher_config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
    session_config.dispatcher_config.kernel_type =
        cfg.unified ? CUDAQ_KERNEL_UNIFIED : CUDAQ_KERNEL_REGULAR;
    session_config.dispatcher_config.skip_tx_markers = 1;
    session_config.function_table.entries = h_entries;
    session_config.function_table.count = 1;
    session_config.shutdown_flag = &dispatcher_shutdown;
    session_config.stats = &packets_dispatched;

    if (cudaq_bridge_dispatch_session_create(bridge, &session_config,
                                             &session) != CUDAQ_OK) {
      std::cerr << "ERROR: cudaq_bridge_dispatch_session_create failed"
                << std::endl;
      cudaq_bridge_disconnect(bridge);
      cudaq_bridge_destroy(bridge);
      return 1;
    }
    if (cudaq_bridge_dispatch_session_start(session) != CUDAQ_OK) {
      std::cerr << "ERROR: cudaq_bridge_dispatch_session_start failed"
                << std::endl;
      cudaq_bridge_dispatch_session_destroy(session);
      cudaq_bridge_disconnect(bridge);
      cudaq_bridge_destroy(bridge);
      return 1;
    }
  } else if (cudaq_bridge_launch(bridge) != CUDAQ_OK) {
    std::cerr << "ERROR: cudaq_bridge_launch failed" << std::endl;
    cudaq_bridge_disconnect(bridge);
    cudaq_bridge_destroy(bridge);
    return 1;
  }

  const auto t0 = std::chrono::steady_clock::now();
  while (g_shutdown.load(std::memory_order_acquire) == 0) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
    if (elapsed > cfg.timeout_sec) {
      std::cout << "\nTimeout reached (" << cfg.timeout_sec << "s)"
                << std::endl;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  std::cout << "\n=== Shutting down ===" << std::endl;
  dispatcher_shutdown = 1;

  if (session)
    cudaq_bridge_dispatch_session_stop(session);
  cudaq_bridge_disconnect(bridge);

  if (!cfg.forward)
    std::cout << "Packets dispatched: " << packets_dispatched << std::endl;

  if (session)
    cudaq_bridge_dispatch_session_destroy(session);
  cudaq_bridge_destroy(bridge);
  std::cout << "Done." << std::endl;
  return 0;
}
