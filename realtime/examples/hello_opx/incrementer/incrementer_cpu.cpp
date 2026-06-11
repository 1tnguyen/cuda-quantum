/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file incrementer_cpu.cpp
/// @brief CPU-only "hello, OPNIC" RPC dispatch tutorial (counterpart of
///        `incrementer_gpu.cu`).
///
/// No GPU in the data plane.  An OPX sends `rpc_increment(int)` requests over
/// the OPNIC transport; this host program services them and returns `int + 1`.
/// It is meant to be read top to bottom: `main` walks through numbered STEPs,
/// each introduced by a banner comment.
///
/// Two host execution shapes are selected at runtime:
///
///   default     3-thread ring   bridge `launch` starts RX/TX transport threads;
///                                `cudaq_dispatcher_*` runs the library host
///                                dispatch loop against the ring from
///                                `get_transport_context(RING_BUFFER)`.
///   --unified   single-thread   `cudaq_dispatcher_*` runs the library generic
///                                unified host loop on a dispatcher-owned thread
///                                against the bridge host data-plane from
///                                `get_host_dataplane`.
///
/// Both paths wire the dispatcher through the same public setter API, then
/// start / block / stop.  The transport bridge constructed in STEP 2 is the
/// only transport-specific piece; swapping it for a different bridge is the
/// intended next exercise.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include "opnic_bridge_cpu.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>

// OPNIC stream IDs
#define RPC_INPUT_STREAM_ID 1
#define RPC_OUTPUT_STREAM_ID 1

namespace {

using cudaq::realtime::fnv1a_hash;
using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

//=============================================================================
// The RPC handler -- your application logic
//=============================================================================
// Each RPC is named by a stable 32-bit id (the FNV-1a hash of its name), shared
// by both ends of the wire. The dispatcher matches an incoming request to this
// id in the function table (STEP 3) and invokes the handler below.

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID = fnv1a_hash("rpc_increment");

/// CPU implementation of `rpc_increment` on the canonical HOST_CALL ABI
/// (`cudaq_host_rpc_fn_t`: void(void *slot, size_t)).  The handler runs in place
/// on the transport slot: it reads the RPCHeader and the int payload, then
/// rewrites the same slot as an RPCResponse echoing request_id / ptp_timestamp.
/// Operating in place is what keeps the dispatch path zero-copy; the layout is
/// wire-compatible with the GPU handler in `incrementer_gpu.cu`.
void rpc_increment_host(void *slot, std::size_t slot_size) {
  auto *h = static_cast<RPCHeader *>(slot);
  if (h->magic != RPC_MAGIC_REQUEST)
    return;
  if (slot_size < sizeof(RPCResponse) + sizeof(std::int32_t))
    return;

  const std::uint32_t request_id = h->request_id;
  const std::uint64_t ptp = h->ptp_timestamp;
  auto *payload = reinterpret_cast<std::int32_t *>(
      static_cast<std::uint8_t *>(slot) + sizeof(RPCHeader));
  const std::int32_t result = *payload + 1;

  auto *resp = static_cast<RPCResponse *>(slot);
  resp->magic = RPC_MAGIC_RESPONSE;
  resp->status = 0;
  resp->result_len = sizeof(std::int32_t);
  resp->request_id = request_id;
  resp->ptp_timestamp = ptp;
  *payload = result;
}

//=============================================================================
// Process-wide shutdown flag
//=============================================================================
// One flag drives every shutdown trigger (signal, timeout, in-band shutdown
// packet) and the dispatcher's run loop.  It is plain `volatile int` because
// that is the type the dispatcher control API expects.
volatile int g_shutdown = 0;
void on_signal(int) { g_shutdown = 1; }

//=============================================================================
// Command-line options
//=============================================================================
struct CliConfig {
  bool unified = false;
  int timeout_sec = 60;
};

CliConfig parse_args(int argc, char **argv) {
  CliConfig cfg;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--unified") {
      cfg.unified = true;
    } else if (a.rfind("--timeout=", 0) == 0) {
      cfg.timeout_sec = std::stoi(a.substr(10));
    } else if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0] << " [--unified] [--timeout=N]\n"
                << "  (default)  3-thread ring  (bridge RX/TX + dispatch loop)\n"
                << "  --unified  single-thread  (library generic unified loop)\n";
      std::exit(0);
    }
  }
  return cfg;
}

} // namespace

int main(int argc, char **argv) {
  //===========================================================================
  // STEP 1 - Parse options and prepare the process
  //===========================================================================
  // Pick the execution shape, require root (OPNIC mmaps PCIe BARs), and install
  // signal handlers so Ctrl+C / SIGTERM trip the shared shutdown flag.
  const CliConfig cli = parse_args(argc, argv);

  const char *mode_label =
      cli.unified ? "single-thread generic unified host loop"
                  : "3-thread ring host loop";
  std::printf("[HOST] Mode: CPU %s (no GPU in data plane)\n", mode_label);

  if (geteuid() != 0) {
    std::fprintf(stderr, "ERROR: Run as root\n");
    return 1;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  //===========================================================================
  // STEP 2 - Construct the transport bridge
  //===========================================================================
  // The bridge owns the transport (here: OPNIC over CPU) and exposes a uniform
  // vtable to the dispatcher.  THIS is the only transport-specific code in the
  // example: to run over a different transport, construct a different bridge
  // here and fetch its interface -- everything below stays the same.
  OpnicBridgeCpuConfig bridge_cfg{};
  bridge_cfg.input_stream_id = RPC_INPUT_STREAM_ID;
  bridge_cfg.output_stream_id = RPC_OUTPUT_STREAM_ID;
  bridge_cfg.unified = cli.unified;
  bridge_cfg.shutdown_flag = &g_shutdown;

  cudaq_realtime_bridge_handle_t bridge_handle =
      opnic_bridge_cpu_create_context(&bridge_cfg);
  if (!bridge_handle) {
    std::fprintf(stderr, "ERROR: Failed to construct CPU OPNIC bridge\n");
    return 1;
  }
  cudaq_realtime_bridge_interface_t *bridge =
      cudaq_realtime_get_opnic_cpu_bridge_interface();

  //===========================================================================
  // STEP 3 - Describe the RPC surface (the function table)
  //===========================================================================
  // The function table maps each RPC id to a handler and its argument/result
  // schema.  The dispatcher consults it for every request.  This example
  // exposes a single RPC; real programs register one entry per verb.
  cudaq_function_entry_t entry{};
  entry.handler.host_fn = &rpc_increment_host;
  entry.function_id = RPC_INCREMENT_FUNCTION_ID;
  entry.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  entry.schema.num_args = 1;
  entry.schema.num_results = 1;
  entry.schema.args[0].type_id = CUDAQ_TYPE_INT32;
  entry.schema.results[0].type_id = CUDAQ_TYPE_INT32;

  cudaq_function_table_t table{};
  table.entries = &entry;
  table.count = 1;

  //===========================================================================
  // STEP 4 - Connect the bridge (handshake with the OPX)
  //===========================================================================
  // connect() performs the OPX <-> OPNIC handshake and prepares the streams.
  // After this returns the transport is live and able to move packets.
  if (bridge->connect(bridge_handle) != CUDAQ_OK) {
    std::fprintf(stderr, "ERROR: Failed to connect bridge\n");
    bridge->destroy(bridge_handle);
    return 1;
  }

  //===========================================================================
  // STEP 5 - Arm the run timeout
  //===========================================================================
  // SIGINT/SIGTERM were armed in STEP 1; this optional watchdog also trips the
  // shutdown flag after --timeout seconds so an unattended run cannot hang.
  std::thread timeout_thread;
  if (cli.timeout_sec > 0) {
    timeout_thread = std::thread([sec = cli.timeout_sec]() {
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(sec);
      while (g_shutdown == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (g_shutdown == 0) {
        std::printf("[HOST] Timeout (%ds) reached\n", sec);
        g_shutdown = 1;
      }
    });
  }

  std::printf("[HOST] Running (Ctrl+C to stop, timeout=%ds)...\n",
              cli.timeout_sec);
  std::cout.flush();

  //===========================================================================
  // STEP 6 - Create, configure, and wire the dispatcher
  //===========================================================================
  // The dispatcher owns the run loop.  Both shapes share one lifecycle
  // (wire -> start -> block-until-shutdown -> stop); they differ only in the
  // transport input wired in and -- for the ring path -- the bridge transport
  // threads launched around the dispatcher.  `fail` runs the shared teardown
  // for the error paths so the happy path below reads top to bottom.
  cudaq_dispatch_manager_t *manager = nullptr;
  cudaq_dispatcher_t *dispatcher = nullptr;

  // Helper function to run the shared teardown for the error paths.
  auto fail = [&](const char *msg) {
    std::fprintf(stderr, "ERROR: %s\n", msg);
    g_shutdown = 1;
    if (dispatcher)
      cudaq_dispatcher_destroy(dispatcher);
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    bridge->disconnect(bridge_handle);
    if (timeout_thread.joinable())
      timeout_thread.join();
    bridge->destroy(bridge_handle);
    return 1;
  };

  cudaq_dispatcher_config_t config{};
  config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

  // Ask the bridge for the transport input for the chosen shape.  The unified
  // loop thread reads `binding` for its whole lifetime, so it must outlive
  // stop()/destroy() below -- hence both live in this scope.
  cudaq_host_transport_binding_t binding{};
  cudaq_ringbuffer_t ringbuffer{};
  if (cli.unified) {
    config.kernel_type = CUDAQ_KERNEL_UNIFIED;
    if (!bridge->get_host_dataplane ||
        bridge->get_host_dataplane(bridge_handle, &binding.dataplane) !=
            CUDAQ_OK)
      return fail("Failed to get host data-plane");
  } else {
    if (bridge->get_transport_context(bridge_handle, RING_BUFFER,
                                      &ringbuffer) != CUDAQ_OK)
      return fail("Failed to get transport context");
    config.num_slots = static_cast<std::uint32_t>(bridge_cfg.buffer_count);
    config.slot_size = static_cast<std::uint32_t>(
        ringbuffer.rx_stride_sz < ringbuffer.tx_stride_sz
            ? ringbuffer.rx_stride_sz
            : ringbuffer.tx_stride_sz);
    config.skip_tx_markers = 1;
  }

  if (cudaq_dispatch_manager_create(&manager) != CUDAQ_OK)
    return fail("Failed to create dispatch manager");
  if (cudaq_dispatcher_create(manager, &config, &dispatcher) != CUDAQ_OK)
    return fail("Failed to create dispatcher");

  // Common wiring: the RPC table and the shutdown flag / stats counter.
  std::uint64_t stats = 0;
  cudaq_dispatcher_set_function_table(dispatcher, &table);
  cudaq_dispatcher_set_control(dispatcher, &g_shutdown, &stats);

  // Transport wiring: this is the one call that differs between the shapes.
  if (cli.unified) {
    // Unified: hand the dispatcher the host data-plane; it then runs its own
    // generic host loop on a dispatcher-owned thread.
    cudaq_dispatcher_set_host_dataplane(dispatcher, &binding);
  } else {
    // Ring: hand the dispatcher the ring buffer, then start the bridge RX/TX
    // transport threads that feed it.
    cudaq_dispatcher_set_ringbuffer(dispatcher, &ringbuffer);
    if (bridge->launch(bridge_handle) != CUDAQ_OK)
      return fail("Failed to launch bridge");
  }

  //===========================================================================
  // STEP 7 - Start, run until shutdown, and stop
  //===========================================================================
  // start() spins up the dispatcher's loop (a dispatcher-owned thread for
  // unified, the library ring loop for ring).  We then block on the shared
  // shutdown flag and stop(), which signals the loop and joins it.
  if (cudaq_dispatcher_start(dispatcher) != CUDAQ_OK)
    return fail("Failed to start dispatcher");

  while (g_shutdown == 0)
    CUDAQ_REALTIME_CPU_RELAX();
  cudaq_dispatcher_stop(dispatcher);

  //===========================================================================
  // STEP 8 - Tear down
  //===========================================================================
  // Mirror image of the setup: destroy the dispatcher/manager, disconnect and
  // destroy the bridge, join the watchdog, and report how many RPCs ran.
  cudaq_dispatcher_destroy(dispatcher);
  cudaq_dispatch_manager_destroy(manager);

  bridge->disconnect(bridge_handle);

  if (timeout_thread.joinable())
    timeout_thread.join();

  std::printf("[HOST] Total RPCs dispatched: %llu\n",
              static_cast<unsigned long long>(stats));

  bridge->destroy(bridge_handle);

  std::printf("[HOST] Done.\n");
  return 0;
}
