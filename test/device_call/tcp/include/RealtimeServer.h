/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "TcpArgParsing.h"
#include "cudaq_internal/device_call/DeviceCallService.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

extern "C" int
cudaq_realtime_get_service(cudaq_realtime_device_call_service *out);

namespace cudaq_internal::device_call {

class DeviceCallServer {
public:
  struct Options {
    std::string readyFile;
    std::uint32_t numSlots = 2;
    std::uint32_t numBlocks = 1;
    std::uint32_t threadsPerBlock = 64;
    std::chrono::milliseconds pollInterval{50};
    std::string logPrefix = "device_call realtime server";
  };

  struct Callback {
    using Function = int (*)(cudaq_realtime_bridge_handle_t, void *);

    Function function = nullptr;
    void *userData = nullptr;

    int operator()(cudaq_realtime_bridge_handle_t bridge) const {
      return function ? function(bridge, userData) : 0;
    }
  };

  struct Hooks {
    Callback afterConnect;
    Callback afterLaunch;
  };

  DeviceCallServer() = default;
  DeviceCallServer(const DeviceCallServer &) = delete;
  DeviceCallServer &operator=(const DeviceCallServer &) = delete;
  ~DeviceCallServer() { cleanup(); }

  static const char *consumeValue(int &index, int argc, char **argv,
                                  const char *current, const char *option) {
    return cudaq_internal::device_call::consumeValue(index, argc, argv, current,
                                                     option);
  }

  static bool parseCommonOptions(int argc, char **argv, Options &options) {
    static constexpr cudaq_internal::device_call::CliOption<Options>
        commonOptions[] = {
            {"--ready-file", cudaq_internal::device_call::parseStringOption<
                                 Options, &Options::readyFile>},
            {"--num-slots", cudaq_internal::device_call::parseUIntOption<
                                Options, std::uint32_t, &Options::numSlots, 1>},
            {"--dispatch-blocks",
             cudaq_internal::device_call::parseUIntOption<
                 Options, std::uint32_t, &Options::numBlocks, 1>},
            {"--dispatch-threads",
             cudaq_internal::device_call::parseUIntOption<
                 Options, std::uint32_t, &Options::threadsPerBlock, 1>},
        };
    return cudaq_internal::device_call::parseCliOptions(argc, argv,
                                                        commonOptions, options);
  }

  static bool writeTextFile(const std::string &path, const std::string &value) {
    std::ofstream file(path);
    if (!file)
      return false;
    file << value;
    return static_cast<bool>(file);
  }

  int run(
      int argc, char **argv, cudaq_realtime_bridge_interface_t *bridgeInterface,
      const Options &options, const Hooks &hooks,
      cudaq_realtime_get_service_fn_t getService = cudaq_realtime_get_service) {
    cleanup();
    shutdownRequested().store(false);
    logPrefix = options.logPrefix;
    bridgeInterfaceRef = bridgeInterface;
    getServiceFactory = getService;

    if (!getServiceFactory)
      return fail("invalid device-call service factory");

    if (!bridgeInterfaceRef ||
        bridgeInterfaceRef->version != CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION)
      return fail("invalid bridge interface");

    installSignalHandlers();

    if (bridgeInterfaceRef->create(&bridge, argc, argv) != CUDAQ_OK)
      return fail("failed to create bridge");

    if (bridgeInterfaceRef->connect(bridge) != CUDAQ_OK)
      return cleanupAndFail("failed to connect bridge");

    if (int status = hooks.afterConnect(bridge)) {
      cleanup();
      return status;
    }

    cudaq_ringbuffer_t ringbuffer{};
    if (bridgeInterfaceRef->get_transport_context(bridge, RING_BUFFER,
                                                  &ringbuffer) != CUDAQ_OK)
      return cleanupAndFail("failed to get bridge ring buffer");

    if (!initializeDispatchState())
      return cleanupAndFail("failed to initialize dispatch control state");

    if (!initializeService())
      return cleanupAndFail("failed to initialize device-call service");

    launchFn(ringbuffer.rx_flags, ringbuffer.tx_flags, ringbuffer.rx_data,
             ringbuffer.tx_data, ringbuffer.rx_stride_sz,
             ringbuffer.tx_stride_sz, functionEntries, functionCount,
             shutdownFlag, stats, options.numSlots, options.numBlocks,
             options.threadsPerBlock, stream);
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess)
      return cleanupAndFailCuda("failed to launch dispatch kernel", err);

    if (bridgeInterfaceRef->launch(bridge) != CUDAQ_OK)
      return cleanupAndFail("failed to launch bridge");

    if (!options.readyFile.empty() &&
        !writeTextFile(options.readyFile, "ready\n"))
      return cleanupAndFail("failed to write ready file");

    if (int status = hooks.afterLaunch(bridge)) {
      cleanup();
      return status;
    }

    while (!shutdownRequested().load())
      std::this_thread::sleep_for(options.pollInterval);

    cleanup();
    return 0;
  }

private:
  static std::atomic_bool &shutdownRequested() {
    static std::atomic_bool requested{false};
    return requested;
  }

  static void signalHandler(int) { shutdownRequested().store(true); }

  static void installSignalHandlers() {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
  }

  int fail(const char *message) {
    std::cerr << logPrefix << ": " << message << '\n';
    return 1;
  }

  int failCuda(const char *message, cudaError_t err) {
    std::cerr << logPrefix << ": " << message << ": " << cudaGetErrorString(err)
              << '\n';
    return 1;
  }

  int cleanupAndFail(const char *message) {
    cleanup();
    return fail(message);
  }

  int cleanupAndFailCuda(const char *message, cudaError_t err) {
    cleanup();
    return failCuda(message, err);
  }

  bool initializeDispatchState() {
    void *shutdownStorage = nullptr;
    if (cudaMalloc(&shutdownStorage, sizeof(int)) != cudaSuccess ||
        cudaMemset(shutdownStorage, 0, sizeof(int)) != cudaSuccess) {
      if (shutdownStorage)
        (void)cudaFree(shutdownStorage);
      return false;
    }
    shutdownFlag = static_cast<volatile int *>(shutdownStorage);

    if (cudaMalloc(&stats, sizeof(std::uint64_t)) != cudaSuccess ||
        cudaMemset(stats, 0, sizeof(std::uint64_t)) != cudaSuccess ||
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
            cudaSuccess)
      return false;

    return true;
  }

  bool initializeService() {
    if (getServiceFactory(&service) != 0 || !service.get_function_count ||
        !service.populate_table || !service.get_device_dispatch_launch) {
      std::cerr << logPrefix << ": device-call service is incomplete\n";
      return false;
    }

    if (service.create && service.create(nullptr, 0, &serviceHandle) != 0)
      return false;

    functionCount = service.get_function_count(serviceHandle);
    if (functionCount == 0) {
      std::cerr << logPrefix
                << ": device-call library has no exported functions\n";
      return false;
    }

    if (cudaMalloc(&functionEntries,
                   functionCount * sizeof(cudaq_function_entry_t)) !=
        cudaSuccess)
      return false;

    if (service.populate_table(serviceHandle, functionEntries, functionCount,
                               stream) != 0 ||
        cudaStreamSynchronize(stream) != cudaSuccess)
      return false;

    (void)cudaGetLastError();
    launchFn = service.get_device_dispatch_launch(serviceHandle);
    if (!launchFn)
      return false;

    if (service.start && service.start(serviceHandle) != 0)
      return false;
    serviceStarted = true;
    return true;
  }

  void cleanup() {
    if (shutdownFlag) {
      int shutdown = 1;
      (void)cudaMemcpy(const_cast<int *>(shutdownFlag), &shutdown,
                       sizeof(shutdown), cudaMemcpyHostToDevice);
    }
    if (serviceStarted && service.stop)
      (void)service.stop(serviceHandle);
    serviceStarted = false;

    if (service.get_device_dispatch_synchronize && !service.stop) {
      if (auto sync = service.get_device_dispatch_synchronize(serviceHandle))
        (void)sync();
    } else if (stream) {
      (void)cudaStreamSynchronize(stream);
    } else if (functionEntries || shutdownFlag || stats) {
      (void)cudaDeviceSynchronize();
    }

    if (bridgeInterfaceRef && bridge) {
      (void)bridgeInterfaceRef->disconnect(bridge);
      (void)bridgeInterfaceRef->destroy(bridge);
      bridge = nullptr;
    }
    if (stream) {
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
    if (functionEntries) {
      (void)cudaFree(functionEntries);
      functionEntries = nullptr;
    }
    if (shutdownFlag) {
      (void)cudaFree(const_cast<int *>(shutdownFlag));
      shutdownFlag = nullptr;
    }
    if (stats) {
      (void)cudaFree(stats);
      stats = nullptr;
    }
    if (serviceHandle && service.destroy)
      (void)service.destroy(serviceHandle);
    serviceHandle = nullptr;
    service = {};
    getServiceFactory = nullptr;
    launchFn = nullptr;
    functionCount = 0;
  }

  std::string logPrefix = "device_call realtime server";
  cudaq_realtime_bridge_interface_t *bridgeInterfaceRef = nullptr;
  cudaq_realtime_bridge_handle_t bridge = nullptr;
  cudaq_function_entry_t *functionEntries = nullptr;
  volatile int *shutdownFlag = nullptr;
  std::uint64_t *stats = nullptr;
  cudaStream_t stream = nullptr;
  cudaq_realtime_device_call_service service{};
  void *serviceHandle = nullptr;
  cudaq_realtime_get_service_fn_t getServiceFactory = nullptr;
  cudaq_dispatch_launch_fn_t launchFn = nullptr;
  std::uint32_t functionCount = 0;
  bool serviceStarted = false;
};

} // namespace cudaq_internal::device_call
