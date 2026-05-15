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

// Base for shared-memory device_call channels.  These channels bridge
// DeviceCallChannel frames to the CUDA-Q realtime slot protocol represented by
// cudaq_ringbuffer_t, with function lookup based on cudaq_function_entry_t.
class SharedMemoryChannelBase : public DeviceCallChannel {
protected:
  // Initialize the function table, target device, and channel configuration
  // shared by local channels.  When requireDeviceLaunch is true, args must also
  // provide the dispatch launch hook used by channels that start realtime work.
  void initializeFunctionTable(const DeviceCallChannelCreateArgs &args,
                               bool requireDeviceLaunch);

  // Find the realtime table entry selected by the RPC function_id.
  const cudaq_function_entry_t *lookup(std::uint32_t functionId) const;

  // Allocate mapped RX/TX ring storage with slots large enough for the
  // requested frame size.  Ring flags carry slot state; ring data stores RPC
  // payloads.
  void configureRingStorage(std::uint64_t requiredSlotSize);

  // Release mapped ring storage and clear cached sizing state.
  void resetRingStorage();

  // Wait until both RX and TX flags are clear so a slot can be reused.
  void waitForReusableSlot(std::uint32_t slot);

  // Wait for a no-response frame to leave the realtime in-flight state before
  // its slot is reused.
  void waitForFireAndForgetCompletion(std::uint32_t slot);

  // Poll the TX flag until responseState accepts the observed slot state, or
  // the channel timeout is reached.  Derived channels provide the predicate
  // that matches their realtime dispatch path.
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

  // Clear both flags after the caller has consumed the frame, returning the
  // slot to the realtime ring-buffer idle state.
  void clearSlot(std::uint32_t slot);

  // Host-visible slot addresses used to serialize/deserialize RPC payloads.
  std::uint8_t *rxHostSlot(std::uint32_t slot) const;

  std::uint8_t *txHostSlot(std::uint32_t slot) const;

  // RX slot address values for channels whose signaling protocol publishes slot
  // addresses in realtime ring flags.
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
