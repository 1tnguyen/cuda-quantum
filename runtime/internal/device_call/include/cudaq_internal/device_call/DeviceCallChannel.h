/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Registry.h"
#include "cudaq_internal/device_call/DeviceCallService.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudaq_internal::device_call {

constexpr std::uint32_t DefaultNumSlots = 2;
constexpr std::uint64_t DefaultSlotSize = 4096;
constexpr std::uint64_t DefaultTimeoutMs = 10000;

struct DeviceCallChannelConfig {
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint64_t slotSize = DefaultSlotSize;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
};

struct DeviceCallChannelCreateArgs {
  cudaq_function_entry_t *functionTable = nullptr;
  std::uint32_t functionCount = 0;
  int deviceId = 0;
  cudaq_dispatch_launch_fn_t launchFn = nullptr;
  cudaq_device_call_dispatch_synchronize_fn_t synchronizeFn = nullptr;
  std::string channelName;
  std::vector<std::string> arguments;
  DeviceCallChannelConfig channelConfig;
};

inline constexpr const char DeviceDispatchSharedMemoryChannelName[] =
    "device_dispatch";

class DeviceCallChannel
    : public cudaq::registry::RegisteredType<DeviceCallChannel> {
public:
  virtual ~DeviceCallChannel() = default;

  struct TransportBuffer {
    std::byte *data = nullptr;
    std::uint64_t capacity = 0;
  };

  struct DeviceCallFrame {
    std::uint32_t functionId = 0;
    TransportBuffer request;
    TransportBuffer response;
    void *channelPrivate = nullptr;
  };

  virtual void initialize(DeviceCallChannelCreateArgs &&args) = 0;
  virtual void acquireFrame(std::uint32_t functionId,
                            std::uint64_t requestBytes,
                            std::uint64_t responseCapacity,
                            DeviceCallFrame &frame) = 0;
  virtual std::uint64_t dispatchFrame(DeviceCallFrame &frame) = 0;
  virtual void releaseFrame(DeviceCallFrame &frame) noexcept = 0;

  virtual void stop() noexcept = 0;
};

std::unique_ptr<DeviceCallChannel>
createDeviceCallChannel(const std::string &name,
                        DeviceCallChannelCreateArgs args);

} // namespace cudaq_internal::device_call
