/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RealtimeTcpServer.h"

extern "C" cudaq_realtime_bridge_interface_t *
cudaq_realtime_get_bridge_interface();

extern "C" int cudaq_realtime_get_service_test_device_call(
    cudaq_realtime_device_call_service *out);

int main(int argc, char **argv) {
  cudaq_internal::device_call::TcpDeviceCallServer server;
  return server.run(
      argc, argv, cudaq_realtime_get_bridge_interface(),
      cudaq_realtime_get_service_test_device_call);
}
