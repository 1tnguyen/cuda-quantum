/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallRuntimePlugin.h"

namespace cudaq_internal::device_call {
void initializeDeviceCallRuntime(int argc, char **argv);
void finalizeDeviceCallRuntime();
} // namespace cudaq_internal::device_call

namespace {

class DeviceCallRuntimePluginImpl final
    : public cudaq_internal::device_call::DeviceCallRuntimePlugin {
public:
  void initialize(int argc, char **argv) override {
    cudaq_internal::device_call::initializeDeviceCallRuntime(argc, argv);
  }

  void finalize() override {
    cudaq_internal::device_call::finalizeDeviceCallRuntime();
  }
};

cudaq_internal::device_call::DeviceCallRuntimePlugin *
getDeviceCallRuntimePlugin() {
  static DeviceCallRuntimePluginImpl plugin;
  return &plugin;
}

} // namespace

extern "C" cudaq_internal::device_call::DeviceCallRuntimePlugin *
getCudaqDeviceCallRuntime_device_call() {
  return getDeviceCallRuntimePlugin();
}
