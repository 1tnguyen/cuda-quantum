/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <string_view>

namespace cudaq_internal::device_call {

// Internal runtime controls for realtime device_call setup. These are not part
// of the lowered per-call ABI; callers should use the C++ interface and let
// exceptions propagate with their original diagnostics.
enum class DeviceCallRuntimeMode : std::int32_t {
  Off = 0,
  SharedMemory = 1,
  ExternalChannel = 2,
};

void configureDeviceCallRuntime(int argc, char **argv);
DeviceCallRuntimeMode getConfiguredDeviceCallRuntimeMode();
void shutdownDeviceCallRuntime() noexcept;

void initializeDeviceCallRuntime(int argc, char **argv);
void initializeDeviceCallRuntime();
void finalizeDeviceCallRuntime();

void initializeDeviceCallService(void *symbolScope = nullptr,
                                 std::string_view servicePostfix = {});
void finalizeDeviceCallService();

} // namespace cudaq_internal::device_call
