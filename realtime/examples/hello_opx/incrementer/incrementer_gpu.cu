/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file incrementer_gpu.cu
/// @brief GPU "hello, OPNIC" RPC dispatch tutorial (counterpart of
///        `incrementer_cpu.cpp`).
///
/// The GPU is in the data plane: an OPX sends `rpc_increment(int)` requests over
/// the OPNIC transport and a CUDA dispatch loop services them on-device,
/// returning `int + 1`.  Like the CPU tutorial, `main` walks through numbered
/// STEPs, each introduced by a banner comment.
///
/// Two execution shapes are selected at runtime:
///
///   default     3-kernel ring   bridge `launch` starts RX/TX transport kernels;
///                                the dispatcher runs the library device dispatch
///                                kernel against the ring from
///                                `get_transport_context(RING_BUFFER)`.
///   --unified   single-kernel   the dispatcher runs the library generic unified
///                                device kernel against the bridge device
///                                data-plane from `get_device_dataplane`.
///
/// Both shapes share one dispatcher lifecycle (wire -> connect -> start ->
/// sync -> stop); they differ only in the dispatcher config and the single
/// transport-wiring call.  The transport bridge constructed in STEP 2 is the
/// only transport-specific piece; swapping it for a different bridge is the
/// intended next exercise.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/device_unified_generic.cuh"
#include "opnic_kernels.cuh"
#include "opnic_bridge.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unistd.h>

#undef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

//=============================================================================
// The RPC handler -- your application logic (device side)
//=============================================================================
// On the GPU the handler is a __device__ function the dispatch loop invokes
// on-device, once per request.  It reads the request payload and writes the
// result in place (zero-copy).  Each RPC is named by a stable 32-bit id (the
// FNV-1a hash of its name), shared by both ends of the wire; it is
// wire-compatible with the CPU handler in `incrementer_cpu.cpp`.

__device__ int rpc_increment_handler(const void *input, void *output,
                                     std::uint32_t arg_len,
                                     std::uint32_t max_result_len,
                                     std::uint32_t *result_len) {
  const int input_val = *static_cast<const int *>(input);
  int *output_val = static_cast<int *>(output);
  *output_val = input_val + 1;
  *result_len = sizeof(int);
  return 0;
}

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("rpc_increment");

//=============================================================================
// Publishing the handler into the function table
//=============================================================================
// A __device__ function address is only meaningful on the device, so it cannot
// be taken from host code.  This one-shot kernel runs on the GPU to record the
// handler pointer (and its argument/result schema) into the device-resident
// function table the dispatcher consults for every request.

__global__ void init_function_table_kernel(cudaq_function_entry_t *entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.device_fn_ptr =
        reinterpret_cast<void *>(&rpc_increment_handler);
    entries[0].function_id = RPC_INCREMENT_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[0].schema.num_args = 1;
    entries[0].schema.num_results = 1;
    entries[0].schema.args[0].type_id = CUDAQ_TYPE_INT32;
    entries[0].schema.results[0].type_id = CUDAQ_TYPE_INT32;
  }
}

} // namespace

int main(int argc, char **argv) {
  //===========================================================================
  // STEP 1 - Parse options and prepare the process
  //===========================================================================
  // Pick the execution shape and require root (OPNIC mmaps PCIe BARs).
  bool use_unified = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--unified") == 0) {
      use_unified = true;
    }
  }

  if (use_unified)
    printf("[HOST] Mode: unified generic (library device loop) dispatch\n");
  else
    printf("[HOST] Mode: three-kernel dispatch (default)\n");

  if (geteuid() != 0) {
    printf("ERROR: Run as root\n");
    return 1;
  }

  //===========================================================================
  // STEP 2 - Construct the transport bridge
  //===========================================================================
  // The bridge owns the transport (here: OPNIC on the GPU) and exposes a
  // uniform vtable to the dispatcher.  THIS is the only transport-specific code
  // in the example: to run over a different transport, swap in a different
  // bridge's interface getter -- everything below stays the same.  create()
  // reads argv (here just --unified) and hands back the opaque handle; we cast
  // it back only to read the device-resident shutdown flag the dispatch kernel
  // polls (also passed to set_control below).
  const std::size_t page_size = RING_BUFFER_PAGE_SIZE;
  const std::size_t num_pages = RING_BUFFER_NUM_PAGES;

  cudaq_realtime_bridge_interface_t *bridge_iface =
      cudaq_realtime_get_bridge_interface();

  cudaq_realtime_bridge_handle_t bridge_handle = nullptr;
  if (bridge_iface->create(&bridge_handle, argc, argv) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to create OPNIC bridge" << std::endl;
    return 1;
  }
  auto *bridge = reinterpret_cast<OpnicBridgeContext *>(bridge_handle);

  //===========================================================================
  // STEP 3 - Describe the RPC surface (the device function table)
  //===========================================================================
  // Allocate the device-resident function table and populate it on-device via
  // the init kernel (see above).  Also allocate the device-side stats counter
  // the dispatch loop increments per serviced request.
  cudaq_function_entry_t *d_function_entries = nullptr;
  CUDA_CHECK(cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t)));
  init_function_table_kernel<<<1, 1>>>(d_function_entries);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::uint64_t *d_stats = nullptr;
  CUDA_CHECK(cudaMalloc(&d_stats, sizeof(std::uint64_t)));
  CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(std::uint64_t)));

  cudaq_function_table_t table{};
  table.entries = d_function_entries;
  table.count = 1;

  //===========================================================================
  // STEP 4 - Create, configure, and wire the dispatcher
  //===========================================================================
  // The dispatcher owns the run loop.  Both shapes share one config skeleton
  // (device path, device-call mode) and differ only in `kernel_type` and the
  // single transport-wiring call below.
  cudaq_dispatch_manager_t *manager = nullptr;
  if (cudaq_dispatch_manager_create(&manager) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to create dispatch manager" << std::endl;
    return 1;
  }

  cudaq_dispatcher_config_t config{};
  config.device_id = 0;
  config.vp_id = 0;
  config.num_blocks = 1;
  config.threads_per_block = 1;
  config.num_slots = static_cast<uint32_t>(num_pages);
  config.slot_size = static_cast<uint32_t>(page_size);
  config.dispatch_path = CUDAQ_DISPATCH_PATH_DEVICE;
  config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  config.kernel_type = use_unified ? CUDAQ_KERNEL_UNIFIED : CUDAQ_KERNEL_REGULAR;

  cudaq_dispatcher_t *dispatcher = nullptr;
  if (cudaq_dispatcher_create(manager, &config, &dispatcher) != CUDAQ_OK) {
    printf("ERROR: Failed to create dispatcher\n");
    return 1;
  }

  // Common wiring: the RPC table and the shutdown flag / stats counter.  The
  // shutdown flag is device-resident here (the kernel polls it on-device).
  cudaq_dispatcher_set_function_table(dispatcher, &table);
  cudaq_dispatcher_set_control(
      dispatcher, static_cast<volatile int *>(bridge->shutdown_device), d_stats);

  // Transport wiring: this is the one call that differs between the shapes.
  // Both `dp` and `ringbuffer` must outlive cudaq_dispatcher_start below, so
  // they live in this scope.
  cudaq_device_dataplane_t dp{};
  cudaq_ringbuffer_t ringbuffer{};
  if (use_unified) {
    // Unified: hand the dispatcher the device data-plane plus a launch wrapper
    // for the library's generic dispatch kernel; start() invokes the wrapper.
    //
    // Why a wrapper here, when the CPU example just calls set_host_dataplane
    // and lets the dispatcher own the loop?  The host loop is plain host code
    // the library .so owns and calls directly.  The GPU kernel instead makes
    // indirect __device__ calls through the transport's device ops, captured by
    // get_device_dataplane via cudaMemcpyFromSymbol.  CUDA requires the kernel
    // and those device functions to be linked into the SAME device module
    // (RDC), so the launch must be instantiated in this nvcc TU -- which is
    // device-linked with the bridge -- and cannot live in the prebuilt
    // libcudaq-realtime.so.  cudaq_launch_unified_generic_device_loop is the
    // library-provided launch wrapper; the example supplies only the launch
    // site and the resolved data-plane `dp`.
    if (!bridge_iface->get_device_dataplane ||
        bridge_iface->get_device_dataplane(bridge_handle, &dp) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to get OPNIC device data-plane" << std::endl;
      return 1;
    }
    cudaq_dispatcher_set_unified_launch(
        dispatcher, cudaq_launch_unified_generic_device_loop, &dp);
  } else {
    // Ring: hand the dispatcher the ring buffer and the regular dispatch
    // kernel launcher.  The occupancy query forces eager CUDA module loading
    // to avoid lazy-loading stalls once the persistent kernels are running.
    if (bridge_iface->get_transport_context(bridge_handle, RING_BUFFER,
                                            &ringbuffer) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to get ring buffer context" << std::endl;
      return 1;
    }

    int dispatch_blocks = 0;
    cudaError_t occ_err =
        cudaq_dispatch_kernel_query_occupancy(&dispatch_blocks, 1);
    if (occ_err != cudaSuccess) {
      printf("ERROR: Dispatch kernel occupancy query failed: %s\n",
             cudaGetErrorString(occ_err));
      return 1;
    }

    cudaq_dispatcher_set_ringbuffer(dispatcher, &ringbuffer);
    cudaq_dispatcher_set_launch_fn(dispatcher,
                                   &cudaq_launch_dispatch_kernel_regular);
  }

  //===========================================================================
  // STEP 5 - Connect the bridge (and launch RX/TX kernels for the ring)
  //===========================================================================
  // connect() performs the OPX <-> OPNIC handshake.  In ring mode the bridge
  // also launches its persistent RX/TX transport kernels that feed the
  // dispatcher; unified mode needs none -- the single kernel owns transport.
  if (bridge_iface->connect(bridge_handle) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to connect bridge" << std::endl;
    return 1;
  }

  if (!use_unified) {
    if (bridge_iface->launch(bridge_handle) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to launch bridge" << std::endl;
      return 1;
    }
  }

  //===========================================================================
  // STEP 6 - Start, run until shutdown, and stop
  //===========================================================================
  // start() launches the dispatch kernel(s) on the dispatcher's stream.  We
  // then wait for the device-driven session and stop the dispatcher, which
  // signals shutdown and synchronizes.
  if (cudaq_dispatcher_start(dispatcher) != CUDAQ_OK) {
    printf("ERROR: Failed to start dispatcher\n");
    return 1;
  }

  CUDA_CHECK(cudaStreamSynchronize(0));
  cudaq_dispatcher_stop(dispatcher);

  //===========================================================================
  // STEP 7 - Report and tear down
  //===========================================================================
  // Read the serviced-request count, then mirror the setup: disconnect the
  // bridge transport (ring only -- unified launched no transport kernels),
  // destroy the dispatcher/manager, free device memory, and destroy the bridge.
  std::uint64_t processed = 0;
  cudaq_dispatcher_get_processed(dispatcher, &processed);
  std::cout << "[HOST] Total RPCs dispatched: " << processed << "\n";

  if (!use_unified) {
    if (bridge_iface->disconnect(bridge_handle) != CUDAQ_OK) {
      std::cerr << "ERROR: Failed to disconnect bridge" << std::endl;
      return 1;
    }
  }

  cudaq_dispatcher_destroy(dispatcher);
  cudaq_dispatch_manager_destroy(manager);

  cudaFree(d_function_entries);
  cudaFree(d_stats);

  bridge_iface->destroy(bridge_handle);

  std::cout << "[HOST] Done.\n";

  return 0;
}
