/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file decoder_server_cpu.cpp
/// @brief CPU OPNIC realtime server that serves cuda-qx decoder RPCs, with the
///        decode data plane and the configure control plane on SEPARATE OPNIC
///        streams.
///
/// The HOST_CALL handlers are not hard-coded: the server **dynamically loads**
/// the out-of-tree decoder shim (`libcudaqx-decoder-hostcall.so`, built from
/// realtime/examples/cudaqx_decoder_hostcall) and injects its 4-entry function
/// table into the dispatcher. So the server has no build-time dependency on
/// cuda-qx; the shim is named at runtime via --shim and resolved with dlopen.
///
/// ===========================================================================
/// TWO-STREAM (data plane / control plane) DESIGN
/// ===========================================================================
/// All configuration comes from the OPX end over OPNIC -- the host has no local
/// config file. But the decode verbs and the configure verb have wildly
/// different size/latency profiles, so they ride two separate OPNIC streams:
///
///   DATA stream  (id --data-stream, default 1)
///       Hot path. Small 64 B slots (DataInputPacket/DataOutputPacket), many of
///       them. Carries the decode verbs: reset_decoder / enqueue_syndromes /
///       get_corrections. Every decode RPC DMAs only 64 B.
///
///   CONTROL stream (id --control-stream, default 2)
///       Off the hot path, rare. Chunk-sized slots
///       (ControlInput/OutputPacket), only a few. Carries configure_decoder,
///       which delivers YAML decoder-bank config chunks that the shim
///       reassembles before it calls cuda-qx. We observed the real QUA
///       external-stream path fail around a 4 KiB packet size; the 1024 byte
///       chunk payload used by the QUA helpers is a conservative chosen value,
///       not a decoder protocol limit.
///
/// Both streams are serviced by ONE poll thread (`poll_loop` below), which polls
/// each stream's OPNIC producer index and dispatches whichever has a packet to
/// the same function table, routing by `function_id`. Using a single thread is
/// deliberate and important: the cuda-qx decoder bank is shared mutable state
/// (decode verbs READ it; configure_decoder REBUILDS it via finalize +
/// reconfigure). A single dispatcher thread serializes decode vs. configure for
/// free -- a reconfigure can never race an in-flight decode -- so NO locking is
/// needed on the bank, and the decode hot path pays no synchronization cost.
/// (Two threads, one per stream, would reintroduce that race and force a
/// reader/writer lock on every decode; we avoid it.)
///
/// This is also why we do NOT reuse the library `cudaq_host_dispatcher_loop`
/// here: that loop services a single ring. The two-ring poll loop below is the
/// minimal generalization (it follows the same direct pi/doorbell polling the
/// GPU/unified paths use), so a long-running server can be reconfigured any
/// number of times over the control stream while it keeps decoding.
///
/// ===========================================================================
/// QUA PHASES
/// ===========================================================================
/// Each QUA program owns one OPNIC synchronization phase. The config program
/// uses the control stream, data playback programs use the data stream, and both
/// declare both streams so their OPX signature matches the standing server.
/// At the end of a QUA program, OPX sends a packet with function_id == 0. The
/// poll loop treats that as "phase complete": it stops polling, the server
/// recreates the OPNIC bridge contexts, and then reconnects for the next QUA
/// program. The shim handle and decoder bank are intentionally kept alive
/// across phases, which is what lets config, data, reconfig, and more data run
/// against one long-running server process.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h" // RPCHeader/RPCResponse
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h" // CUDAQ_REALTIME_CPU_RELAX

#include "opnic_bridge_cpu_typed.hpp"

#include "decoder_packets.h" // Data*/Control* packets

// The server's transport-agnostic dispatch core (poll_loop / service_stream /
// dispatch_one).
#include "decoder_dispatch.hpp"

// Shim C ABI: function-table entry points (resolved via dlopen) + ids/count.
#include "cudaqx_decoder_hostcall.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

// Signatures of the two shim entry points we resolve via dlsym (must match
// cudaqx_decoder_hostcall.h).
using cudaqx_rt_get_function_table_fn = int (*)(cudaq_function_entry_t *,
                                                std::uint32_t, std::uint32_t *);
using cudaqx_rt_finalize_fn = void (*)(void);

// Shutdown wiring shared by the poll loop, the signal handler, and the optional
// timeout watcher.
volatile int g_shutdown = 0;
void on_signal(int) { g_shutdown = 1; }

struct CliConfig {
  std::string shim_path;          // --shim=<libcudaqx-decoder-hostcall.so> (req)
  std::uint16_t data_stream = 1;  // --data-stream    (decode verbs)
  std::uint16_t control_stream = 2; // --control-stream (configure_decoder)
  std::size_t data_buffers = 1024;  // --data-buffers   (small slots, hot path)
  std::size_t control_buffers = 8;  // --control-buffers (few large slots)
  int timeout_sec = 60;             // --timeout
  int max_phases = 0;               // --max-phases (0 => unlimited)
};

void print_usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " --shim=<path> [options]\n"
      << "  --shim=<path>          libcudaqx-decoder-hostcall.so (required)\n"
      << "  --data-stream=N        OPNIC stream id for decode verbs (default 1)\n"
      << "  --control-stream=N     OPNIC stream id for configure   (default 2)\n"
      << "  --data-buffers=N       data-ring slots    (default 1024)\n"
      << "  --control-buffers=N    control-ring slots (default 8)\n"
      << "  --max-phases=N         exit after N QUA phases (default 0: unlimited)\n"
      << "  --timeout=N            auto-shutdown after N seconds (default 60)\n";
}

bool parse_args(int argc, char **argv, CliConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--shim=", 0) == 0)
      cfg.shim_path = a.substr(7);
    else if (a.rfind("--data-stream=", 0) == 0)
      cfg.data_stream = static_cast<std::uint16_t>(std::stoi(a.substr(14)));
    else if (a.rfind("--control-stream=", 0) == 0)
      cfg.control_stream = static_cast<std::uint16_t>(std::stoi(a.substr(17)));
    else if (a.rfind("--data-buffers=", 0) == 0)
      cfg.data_buffers = static_cast<std::size_t>(std::stoul(a.substr(15)));
    else if (a.rfind("--control-buffers=", 0) == 0)
      cfg.control_buffers = static_cast<std::size_t>(std::stoul(a.substr(18)));
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
  if (cfg.shim_path.empty()) {
    std::fprintf(stderr, "ERROR: --shim=<path> is required\n");
    return false;
  }
  if (cfg.data_stream == cfg.control_stream) {
    std::fprintf(stderr, "ERROR: --data-stream and --control-stream must differ\n");
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  CliConfig cli;
  if (!parse_args(argc, argv, cli))
    return 1;

  std::printf("[HOST] cuda-qx decoder server (CPU, 2-stream): data=%u "
              "control=%u\n",
              cli.data_stream, cli.control_stream);

  // OPNIC requires privileged access to mmap PCIe regions.
  if (geteuid() != 0) {
    std::fprintf(stderr, "ERROR: Run as root\n");
    return 1;
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  // ==========================================================================
  // [1] Dynamically load the shim and pull in its HOST_CALL function table.
  // ==========================================================================
  void *shim = dlopen(cli.shim_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!shim) {
    std::fprintf(stderr, "ERROR: dlopen('%s') failed: %s\n",
                 cli.shim_path.c_str(), dlerror());
    return 1;
  }
  auto get_function_table = reinterpret_cast<cudaqx_rt_get_function_table_fn>(
      dlsym(shim, "cudaqx_rt_get_function_table"));
  auto finalize = reinterpret_cast<cudaqx_rt_finalize_fn>(
      dlsym(shim, "cudaqx_rt_finalize"));
  if (!get_function_table || !finalize) {
    std::fprintf(stderr, "ERROR: shim missing cudaqx_rt_* symbols: %s\n",
                 dlerror());
    dlclose(shim);
    return 1;
  }

  cudaq_function_entry_t entries[CUDAQX_RT_FUNCTION_COUNT]{};
  std::uint32_t entry_count = 0;
  if (get_function_table(entries, CUDAQX_RT_FUNCTION_COUNT, &entry_count) !=
      CUDAQX_RT_OK) {
    std::fprintf(stderr, "ERROR: cudaqx_rt_get_function_table failed\n");
    finalize();
    dlclose(shim);
    return 1;
  }
  const cudaq_function_table_t table{entries, entry_count};
  std::printf("[HOST] Loaded shim '%s' (%u HOST_CALL entries)\n",
              cli.shim_path.c_str(), entry_count);
  std::printf("[HOST] Decoder bank starts empty; configure_decoder arrives "
              "as chunks on the control stream\n");

  // ==========================================================================
  // [2] Optional timeout watcher (sets the shared shutdown flag).
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
      }
    });
  }

  // ==========================================================================
  // [3] Serve QUA phases. Each QUA program sends function_id=0 when its phase
  //     is complete. That packet is consumed locally by service_stream; it is
  //     not dispatched into the shim. The server then tears down/recreates
  //     OPNIC contexts and syncs with the next QUA program, while the dlopen'd
  //     shim and decoder bank stay alive across phases.
  // ==========================================================================
  std::uint64_t decode_stats = 0, config_stats = 0;
  int phase = 0;
  std::printf("[HOST] Running phased server (Ctrl+C to stop, timeout=%ds, "
              "max_phases=%d)...\n",
              cli.timeout_sec, cli.max_phases);
  std::cout.flush();

  while (g_shutdown == 0 &&
         (cli.max_phases == 0 || phase < cli.max_phases)) {
    ++phase;

    cudaq_realtime_bridge_interface_t *bridge =
        cudaq_realtime_get_opnic_cpu_bridge_interface();
    cudaq_realtime_bridge_handle_t data_bridge = nullptr;
    cudaq_realtime_bridge_handle_t control_bridge = nullptr;
    auto destroy_phase_bridges = [&]() {
      if (control_bridge) {
        bridge->disconnect(control_bridge);
        bridge->destroy(control_bridge);
        control_bridge = nullptr;
      }
      if (data_bridge) {
        bridge->disconnect(data_bridge);
        bridge->destroy(data_bridge);
        data_bridge = nullptr;
      }
    };

    OpnicBridgeCpuConfig data_cfg{};
    data_cfg.input_stream_id = cli.data_stream;
    data_cfg.output_stream_id = cli.data_stream;
    data_cfg.buffer_count = cli.data_buffers;
    data_cfg.force_reset = true;
    data_cfg.unified = true;
    data_cfg.shutdown_flag = &g_shutdown;

    OpnicBridgeCpuConfig control_cfg{};
    control_cfg.input_stream_id = cli.control_stream;
    control_cfg.output_stream_id = cli.control_stream;
    control_cfg.buffer_count = cli.control_buffers;
    control_cfg.force_reset = false;
    control_cfg.unified = true;
    control_cfg.shutdown_flag = &g_shutdown;

    data_bridge =
        opnic_bridge_cpu_create_context_for_packets<DataInputPacket,
                                                    DataOutputPacket>(
            &data_cfg);
    control_bridge =
        opnic_bridge_cpu_create_context_for_packets<ControlInputPacket,
                                                    ControlOutputPacket>(
            &control_cfg);
    if (!data_bridge || !control_bridge) {
      std::fprintf(stderr, "ERROR: OPNIC transport init failed\n");
      destroy_phase_bridges();
      finalize();
      dlclose(shim);
      return 1;
    }

    opnic_cpu_transport_ctx data_ctx{};
    opnic_cpu_transport_ctx control_ctx{};
    if (bridge->get_transport_context(data_bridge, UNIFIED, &data_ctx) !=
            CUDAQ_OK ||
        bridge->get_transport_context(control_bridge, UNIFIED, &control_ctx) !=
            CUDAQ_OK) {
      std::fprintf(stderr, "ERROR: failed to get OPNIC transport contexts\n");
      destroy_phase_bridges();
      finalize();
      dlclose(shim);
      return 1;
    }
    std::printf("[HOST] phase %d: data slot=%zu B x%zu", phase,
                data_ctx.rx_alloc_size, data_ctx.buf_count);
    std::printf(" | control slot=%zu B x%zu", control_ctx.rx_alloc_size,
                control_ctx.buf_count);
    std::printf("\n");

    constexpr std::size_t kRequiredControlSlotBytes =
        (6 + DECODER_CONTROL_PAYLOAD_WORDS) * sizeof(int);
    if (std::min(control_ctx.rx_alloc_size, control_ctx.tx_alloc_size) <
        kRequiredControlSlotBytes) {
      std::fprintf(stderr,
                   "ERROR: control slot %zu < required control slot %zu\n",
                   std::min(control_ctx.rx_alloc_size,
                            control_ctx.tx_alloc_size),
                   kRequiredControlSlotBytes);
      destroy_phase_bridges();
      finalize();
      dlclose(shim);
      return 1;
    }

    std::printf("[HOST] Waiting for OPX handshake on both streams; start the "
                "qua program\n");
    if (opnic_bridge_cpu_sync_all() != CUDAQ_OK) {
      std::fprintf(stderr, "ERROR: bridge global sync failed\n");
      destroy_phase_bridges();
      finalize();
      dlclose(shim);
      return 1;
    }
    std::printf("[HOST] OPX<->OPNIC sync complete; serving phase %d\n", phase);

    opnic_decoder::poll_loop(data_ctx, control_ctx, table, &g_shutdown,
                             &decode_stats, &config_stats);
    destroy_phase_bridges();
    std::printf("[HOST] phase %d complete\n", phase);
  }
  g_shutdown = 1;

  if (timeout_thread.joinable())
    timeout_thread.join();

  // ==========================================================================
  // [6] Report and clean up.
  // ==========================================================================
  std::printf("[HOST] decode RPCs: %llu | configure RPCs: %llu\n",
              static_cast<unsigned long long>(decode_stats),
              static_cast<unsigned long long>(config_stats));

  finalize(); // tear down the decoder bank
  dlclose(shim);

  std::printf("[HOST] Done.\n");
  return 0;
}
