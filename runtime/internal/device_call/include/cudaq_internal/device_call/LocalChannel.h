/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/MappedRingBuffer.h"

#include <chrono>
#include <cstdint>
#include <thread>

namespace cudaq_internal::device_call {

class SharedMemoryChannelBase : public DeviceCallChannel {
protected:
  void initializeFunctionTable(const DeviceCallChannelCreateArgs &args,
                               bool requireDeviceLaunch);

  const cudaq_function_entry_t *lookup(std::uint32_t functionId) const;

  void configureRingStorage(std::uint64_t requiredSlotSize);

  void resetRingStorage();

  void waitForReusableSlot(std::uint32_t slot);

  void waitForFireAndForgetCompletion(std::uint32_t slot);

  template <typename ResponseState>
  void waitForResponse(std::uint32_t slot, ResponseState responseState) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (true) {
      if (responseState(loadFlag(&txRing.hostFlags()[slot])))
        return;
      if (std::chrono::steady_clock::now() > deadline)
        throw DeviceCallError(DeviceCallStatus::Timeout,
                              "timed out waiting for device_call response");
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  void clearSlot(std::uint32_t slot);

  std::uint8_t *rxHostSlot(std::uint32_t slot) const;

  std::uint8_t *txHostSlot(std::uint32_t slot) const;

  std::uint64_t rxHostSlotAddress(std::uint32_t slot) const;

  std::uint64_t rxDeviceSlotAddress(std::uint32_t slot) const;

  cudaq_function_entry_t *functionTable = nullptr;
  std::uint32_t functionCount = 0;
  int deviceId = 0;
  DeviceCallChannelConfig channelConfig;
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint64_t slotSize = 0;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
  std::uint32_t nextSlot = 0;
  MappedRingBuffer rxRing;
  MappedRingBuffer txRing;
};

} // namespace cudaq_internal::device_call
