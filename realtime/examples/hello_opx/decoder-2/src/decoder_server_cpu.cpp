/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file decoder_server_cpu.cpp
/// @brief CPU OPNIC realtime server that serves CUDA-QX decoder RPCs on a
///        single data stream.
///
/// Configuration is done entirely host-side: the server reads a YAML decoder
/// config with `--config=<path>` and calls the CUDA-QX config API before the
/// first QUA program runs. There is no QUA-driven configure phase and no
/// control stream.
///
/// ===========================================================================
/// SINGLE-STREAM DESIGN
/// ===========================================================================
/// All decode RPCs (reset_decoder / enqueue_syndromes / get_corrections) ride
/// one OPNIC data stream (id --data-stream, default 1). Small 64 B slots
/// (DataInputPacket/DataOutputPacket) keep DMA overhead minimal. Serving is
/// delegated to the library generic host loop through the CPU OPNIC bridge
/// host data-plane, which serializes all RPCs against the CUDA-QX decoder bank
/// on one thread with no locking. The decode handlers self-time and report
/// per-RPC host-processing nanoseconds back through the response
/// `ptp_timestamp` field.
///
/// ===========================================================================
/// QUA PHASES
/// ===========================================================================
/// Each QUA program owns one OPNIC synchronization phase. When the QUA program
/// finishes it sends DECODER_PHASE_DONE_FUNCTION_ID (a dedicated non-zero id --
/// NOT function_id 0, which the shared host loop reserves for OPX shutdown).
/// The registered phase-done handler sets the loop's break flag, so the loop
/// returns; the server then tears down/recreates OPNIC bridge contexts and
/// reconnects for the next QUA program. The decoder bank persists across
/// phases. A single server process handles all data batches; pass
/// `--max-phases=N` to limit phases and exit cleanly after the last batch.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include "opnic_bridge_cpu_typed.hpp"

#include "decoder_packets.h"
#include "decoder_handlers.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

// `g_shutdown` requests a real, permanent stop (signal / timeout). `g_loop_break`
// is the flag the per-phase generic host loop watches: it is set either by the
// phase-done handler (QUA batch finished -> loop returns so main can reconnect
// for the next program) or by shutdown (so an in-flight loop unblocks).
// main distinguishes the two by checking `g_shutdown` after the loop returns.
volatile int g_shutdown = 0;
volatile int g_loop_break = 0;
void on_signal(int) {
  g_shutdown = 1;
  g_loop_break = 1;
}

/// HOST_CALL handler for DECODER_PHASE_DONE_FUNCTION_ID. The QUA program sends
/// this fire-and-forget when its data phase finishes. We frame a trivial OK
/// response (the shared host loop rings the TX doorbell for it; QUA does not
/// read it back) and set the loop break flag so the loop returns to main.
void phase_done_handler(void *slot, std::size_t slot_size) {
  if (slot_size >= sizeof(cudaq::realtime::RPCResponse)) {
    const auto *hdr = static_cast<const cudaq::realtime::RPCHeader *>(slot);
    const std::uint32_t request_id = hdr->request_id;
    auto *resp = static_cast<cudaq::realtime::RPCResponse *>(slot);
    resp->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    resp->status = 0;
    resp->result_len = 0;
    resp->request_id = request_id;
    resp->ptp_timestamp = 0;
  }
  g_loop_break = 1;
}

struct CliConfig {
  std::string config_path;          // --config=<yaml> (required)
  std::uint16_t data_stream = 1;    // --data-stream
  std::size_t data_buffers = 1024;  // --data-buffers
  int timeout_sec = 60;             // --timeout
  int max_phases = 0;               // --max-phases (0 => unlimited)
};

void print_usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " --config=<path> [options]\n"
      << "  --config=<path>     YAML decoder config file (required)\n"
      << "  --data-stream=N     OPNIC stream id for decode RPCs (default 1)\n"
      << "  --data-buffers=N    data-ring slots (default 1024)\n"
      << "  --max-phases=N      exit after N QUA phases (default 0: "
         "unlimited)\n"
      << "  --timeout=N         auto-shutdown after N seconds (default 60)\n";
}

bool parse_args(int argc, char **argv, CliConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--config=", 0) == 0)
      cfg.config_path = a.substr(9);
    else if (a.rfind("--data-stream=", 0) == 0)
      cfg.data_stream =
          static_cast<std::uint16_t>(std::stoi(a.substr(14)));
    else if (a.rfind("--data-buffers=", 0) == 0)
      cfg.data_buffers = static_cast<std::size_t>(std::stoul(a.substr(15)));
    else if (a.rfind("--max-phases=", 0) == 0)
      cfg.max_phases = std::stoi(a.substr(13));
    else if (a.rfind("--timeout=", 0) == 0)
      cfg.timeout_sec = std::stoi(a.substr(10));
    else if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "ERROR: unknown argument '%s'\n", a.c_str());
      return false;
    }
  }
  if (cfg.config_path.empty()) {
    std::fprintf(stderr, "ERROR: --config=<path> is required\n");
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  CliConfig cli;
  if (!parse_args(argc, argv, cli))
    return 1;

  std::printf("[HOST] cuda-qx decoder server (CPU, single stream): "
              "data=%u\n",
              cli.data_stream);

  if (geteuid() != 0) {
    std::fprintf(stderr, "ERROR: Run as root\n");
    return 1;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // ==========================================================================
  // [1] Configure the decoder bank host-side from the YAML file.
  //     This replaces the QUA-driven chunked configure_decoder RPC path.
  // ==========================================================================
  std::printf("[HOST] Configuring decoder bank from '%s'\n",
              cli.config_path.c_str());
  if (decoder_configure_from_file(cli.config_path.c_str()) != DECODER_OK) {
    std::fprintf(stderr, "ERROR: decoder_configure_from_file failed\n");
    return 1;
  }
  std::printf("[HOST] Decoder bank configured\n");

  // ==========================================================================
  // [2] Build the HOST_CALL function table: the 3 decode verbs plus the
  //     phase-done marker the QUA program sends at the end of each batch.
  // ==========================================================================
  constexpr std::uint32_t kTableCapacity = DECODER_FUNCTION_COUNT + 1;
  cudaq_function_entry_t entries[kTableCapacity]{};
  std::uint32_t entry_count = 0;
  if (build_decoder_function_table(entries, DECODER_FUNCTION_COUNT,
                                   &entry_count) != DECODER_OK) {
    std::fprintf(stderr, "ERROR: build_decoder_function_table failed\n");
    decoder_finalize();
    return 1;
  }
  cudaq_function_entry_t &pd = entries[entry_count];
  pd = cudaq_function_entry_t{};
  pd.handler.host_fn = &phase_done_handler;
  pd.function_id = DECODER_PHASE_DONE_FUNCTION_ID;
  pd.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  pd.schema.num_args = 0;
  pd.schema.num_results = 0;
  ++entry_count;

  std::printf("[HOST] Function table built (%u HOST_CALL entries)\n",
              entry_count);

  cudaq_function_table_t table{};
  table.entries = entries;
  table.count = entry_count;

  cudaq_dispatch_manager_t *dispatch_mgr = nullptr;
  if (cudaq_dispatch_manager_create(&dispatch_mgr) != CUDAQ_OK) {
    std::fprintf(stderr, "ERROR: dispatch_manager_create failed\n");
    decoder_finalize();
    return 1;
  }

  // ==========================================================================
  // [3] Optional timeout watcher.
  // ==========================================================================
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
        g_loop_break = 1;
      }
    });
  }

  // ==========================================================================
  // [4] Serve QUA phases. Each QUA program sends DECODER_PHASE_DONE_FUNCTION_ID
  //     when its phase is complete; the phase-done handler sets g_loop_break so
  //     the generic loop returns. The server then recreates OPNIC bridge
  //     contexts and reconnects for the next program while keeping the decoder
  //     bank alive.
  // ==========================================================================
  std::uint64_t decode_stats = 0;
  int phase = 0;
  std::printf("[HOST] Running phased server (Ctrl+C to stop, timeout=%ds, "
              "max_phases=%d)...\n",
              cli.timeout_sec, cli.max_phases);
  std::cout.flush();

  while (g_shutdown == 0 &&
         (cli.max_phases == 0 || phase < cli.max_phases)) {
    ++phase;
    g_loop_break = 0; // re-arm: phase-done handler (or shutdown) sets it

    OpnicBridgeCpuConfig bridge_cfg{};
    bridge_cfg.input_stream_id = cli.data_stream;
    bridge_cfg.output_stream_id = cli.data_stream;
    bridge_cfg.buffer_count = cli.data_buffers;
    bridge_cfg.force_reset = true;
    bridge_cfg.unified = true;
    bridge_cfg.shutdown_flag = &g_loop_break;

    cudaq_realtime_bridge_handle_t bridge_handle =
        opnic_bridge_cpu_create_context_for_packets<DataInputPacket,
                                                    DataOutputPacket>(
            &bridge_cfg);
    if (!bridge_handle) {
      std::fprintf(stderr, "ERROR: OPNIC transport init failed: %s\n",
                   "bridge construction failed");
      decoder_finalize();
      cudaq_dispatch_manager_destroy(dispatch_mgr);
      return 1;
    }
    cudaq_realtime_bridge_interface_t *bridge =
        cudaq_realtime_get_opnic_cpu_bridge_interface();

    opnic_cpu_transport_ctx data_ctx{};
    if (bridge->get_transport_context(bridge_handle, UNIFIED, &data_ctx) !=
        CUDAQ_OK) {
      std::fprintf(stderr, "ERROR: failed to get OPNIC transport context\n");
      bridge->destroy(bridge_handle);
      decoder_finalize();
      cudaq_dispatch_manager_destroy(dispatch_mgr);
      return 1;
    }
    std::printf("[HOST] phase %d: data slot=%zu B x%zu\n", phase,
                data_ctx.rx_alloc_size, data_ctx.buf_count);

    if (bridge->connect(bridge_handle) != CUDAQ_OK) {
      std::fprintf(stderr, "ERROR: bridge connect failed\n");
      bridge->destroy(bridge_handle);
      decoder_finalize();
      cudaq_dispatch_manager_destroy(dispatch_mgr);
      return 1;
    }
    std::printf("[HOST] serving phase %d\n", phase);

    cudaq_dispatcher_config_t cfg{};
    cfg.kernel_type = CUDAQ_KERNEL_UNIFIED;
    cfg.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    cfg.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

    cudaq_dispatcher_t *dispatcher = nullptr;
    if (cudaq_dispatcher_create(dispatch_mgr, &cfg, &dispatcher) != CUDAQ_OK) {
      std::fprintf(stderr, "ERROR: dispatcher_create failed\n");
      decoder_finalize();
      cudaq_dispatch_manager_destroy(dispatch_mgr);
      bridge->destroy(bridge_handle);
      return 1;
    }

    cudaq_host_transport_binding_t binding{};
    if (!bridge->get_host_dataplane ||
        bridge->get_host_dataplane(bridge_handle, &binding.dataplane) !=
            CUDAQ_OK) {
      std::fprintf(stderr, "ERROR: failed to get host data-plane\n");
      cudaq_dispatcher_destroy(dispatcher);
      bridge->destroy(bridge_handle);
      decoder_finalize();
      cudaq_dispatch_manager_destroy(dispatch_mgr);
      return 1;
    }

    if (cudaq_dispatcher_set_function_table(dispatcher, &table) != CUDAQ_OK ||
        cudaq_dispatcher_set_control(dispatcher, &g_loop_break, &decode_stats) !=
            CUDAQ_OK ||
        cudaq_dispatcher_set_host_dataplane(dispatcher, &binding) !=
            CUDAQ_OK) {
      std::fprintf(stderr, "ERROR: dispatcher configuration failed\n");
      cudaq_dispatcher_destroy(dispatcher);
      bridge->destroy(bridge_handle);
      decoder_finalize();
      cudaq_dispatch_manager_destroy(dispatch_mgr);
      return 1;
    }

    if (cudaq_dispatcher_start(dispatcher) != CUDAQ_OK) {
      std::fprintf(stderr, "ERROR: dispatcher_start failed\n");
      cudaq_dispatcher_destroy(dispatcher);
      bridge->destroy(bridge_handle);
      decoder_finalize();
      cudaq_dispatch_manager_destroy(dispatch_mgr);
      return 1;
    }

    while (g_loop_break == 0)
      CUDAQ_REALTIME_CPU_RELAX();
    cudaq_dispatcher_stop(dispatcher);
    cudaq_dispatcher_destroy(dispatcher);
    bridge->disconnect(bridge_handle);
    bridge->destroy(bridge_handle);
    std::printf("[HOST] phase %d complete\n", phase);
  }
  g_shutdown = 1;

  if (timeout_thread.joinable())
    timeout_thread.join();

  // ==========================================================================
  // [5] Report and clean up.
  // ==========================================================================
  std::printf("[HOST] decode RPCs: %llu\n",
              static_cast<unsigned long long>(decode_stats));

  cudaq_dispatch_manager_destroy(dispatch_mgr);
  decoder_finalize();
  std::printf("[HOST] Done.\n");
  return 0;
}
