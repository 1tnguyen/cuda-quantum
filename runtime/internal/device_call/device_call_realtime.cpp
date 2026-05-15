/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallRuntime.h"
#include "cudaq_internal/device_call/DeviceCallRuntimePlugin.h"

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

void initializeNoOp() {}

void initializeSharedMemoryRuntime() {
  cudaq_internal::device_call::initializeDeviceCallService();
}

void teardownNoOp() noexcept {}

void teardownConfiguredRuntime() noexcept {
  cudaq_internal::device_call::shutdownDeviceCallRuntime();
}

struct RuntimeModeActions {
  void (*initialize)();
  void (*teardown)() noexcept;
};

RuntimeModeActions
getRuntimeModeActions(cudaq_internal::device_call::DeviceCallRuntimeMode mode) {
  using cudaq_internal::device_call::DeviceCallRuntimeMode;
  switch (mode) {
  case DeviceCallRuntimeMode::Off:
    return {initializeNoOp, teardownNoOp};
  case DeviceCallRuntimeMode::SharedMemory:
    return {initializeSharedMemoryRuntime, teardownConfiguredRuntime};
  case DeviceCallRuntimeMode::ExternalChannel:
    return {initializeNoOp, teardownConfiguredRuntime};
  }
  return {initializeNoOp, teardownNoOp};
}

} // namespace

namespace cudaq_internal::device_call {

void initializeDeviceCallRuntime(int argc, char **argv) {
  configureDeviceCallRuntime(argc, argv);
  getRuntimeModeActions(getConfiguredDeviceCallRuntimeMode()).initialize();
}

void initializeDeviceCallRuntime() { initializeDeviceCallRuntime(0, nullptr); }

void finalizeDeviceCallRuntime() {
  getRuntimeModeActions(getConfiguredDeviceCallRuntimeMode()).teardown();
}

} // namespace cudaq_internal::device_call

extern "C" cudaq_internal::device_call::DeviceCallRuntimePlugin *
getCudaqDeviceCallRuntime_device_call() {
  return getDeviceCallRuntimePlugin();
}
