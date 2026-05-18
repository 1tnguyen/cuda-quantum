/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "RealtimeServer.h"

#include <cstdint>
#include <iostream>
#include <string>

extern "C" cudaq_status_t cudaq_realtime_tcp_transport_get_bound_port(
    cudaq_realtime_bridge_handle_t handle, std::uint16_t *outPort);

namespace cudaq_internal::device_call {

class TcpDeviceCallServer {
public:
  int run(int argc, char **argv,
          cudaq_realtime_bridge_interface_t *bridgeInterface,
          cudaq_realtime_get_service_fn_t getService =
              cudaq_realtime_get_service) {
    DeviceCallServer::Options options;
    options.logPrefix = "device_call TCP server";

    std::string portFile;
    if (!parseOptions(argc, argv, options, portFile))
      return fail("expected --port-file, --ready-file, and valid options");

    PortFileContext context{portFile, options.logPrefix};
    DeviceCallServer::Hooks hooks;
    hooks.afterConnect = {writeBoundPort, &context};

    DeviceCallServer server;
    return server.run(argc, argv, bridgeInterface, options, hooks, getService);
  }

private:
  struct PortFileContext {
    std::string portFile;
    std::string logPrefix;
  };

  static bool parseOptions(int argc, char **argv,
                           DeviceCallServer::Options &options,
                           std::string &portFile) {
    if (!DeviceCallServer::parseCommonOptions(argc, argv, options))
      return false;

    for (int i = 1; i < argc; ++i) {
      const char *arg = argv ? argv[i] : nullptr;
      if (!arg)
        continue;
      if (const char *value = DeviceCallServer::consumeValue(i, argc, argv, arg,
                                                             "--port-file")) {
        portFile = value;
        continue;
      }
    }

    return !portFile.empty() && !options.readyFile.empty();
  }

  static int writeBoundPort(cudaq_realtime_bridge_handle_t bridge,
                            void *userData) {
    auto *context = static_cast<PortFileContext *>(userData);
    std::uint16_t boundPort = 0;
    if (cudaq_realtime_tcp_transport_get_bound_port(bridge, &boundPort) !=
        CUDAQ_OK) {
      std::cerr << context->logPrefix << ": failed to query bound TCP port\n";
      return 1;
    }
    if (!DeviceCallServer::writeTextFile(context->portFile,
                                         std::to_string(boundPort))) {
      std::cerr << context->logPrefix << ": failed to write port file\n";
      return 1;
    }
    return 0;
  }

  static int fail(const char *message) {
    std::cerr << "device_call TCP server: " << message << '\n';
    return 1;
  }
};

} // namespace cudaq_internal::device_call
