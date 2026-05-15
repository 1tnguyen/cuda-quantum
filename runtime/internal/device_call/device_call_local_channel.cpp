/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/LocalChannel.h"

#include <cuda_runtime.h>

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace cudaq_internal::device_call {

void SharedMemoryChannelBase::initializeFunctionTable(
    const DeviceCallChannelCreateArgs &args, bool requireDeviceLaunch) {
  if (!args.functionTable || args.functionCount == 0)
    throw std::invalid_argument("device_call function table is empty");
  if (requireDeviceLaunch && !args.launchFn)
    throw std::invalid_argument("device_call dispatch launch hook is missing");
  functionTable = args.functionTable;
  functionCount = args.functionCount;
  deviceId = args.deviceId;
  channelConfig = args.channelConfig;
}

const cudaq_function_entry_t *
SharedMemoryChannelBase::lookup(std::uint32_t functionId) const {
  for (std::uint32_t i = 0; i < functionCount; ++i)
    if (functionTable[i].function_id == functionId)
      return &functionTable[i];
  return nullptr;
}

void SharedMemoryChannelBase::configureRingStorage(
    std::uint64_t requiredSlotSize) {
  numSlots = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(DefaultNumSlots, channelConfig.numSlots));
  slotSize = llvm::alignTo(
      std::max({DefaultSlotSize, requiredSlotSize, channelConfig.slotSize}),
      256);
  timeoutMs = channelConfig.timeoutMs;

  if (cudaSetDevice(deviceId) != cudaSuccess)
    throw DeviceCallError(DeviceCallStatus::CudaError,
                          "failed to set CUDA device for device_call");

  if (!rxRing.allocate(numSlots, slotSize) ||
      !txRing.allocate(numSlots, slotSize))
    throw DeviceCallError(DeviceCallStatus::CudaError,
                          "failed to allocate device_call ring storage");
}

void SharedMemoryChannelBase::resetRingStorage() {
  rxRing.reset();
  txRing.reset();
  slotSize = 0;
}

void SharedMemoryChannelBase::waitForReusableSlot(std::uint32_t slot) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (loadFlag(&rxRing.hostFlags()[slot]) != 0 ||
         loadFlag(&txRing.hostFlags()[slot]) != 0) {
    if (std::chrono::steady_clock::now() > deadline)
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            "timed out waiting for reusable device_call "
                            "ring slot");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void SharedMemoryChannelBase::waitForFireAndForgetCompletion(
    std::uint32_t slot) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (true) {
    const std::uint64_t rxFlag = loadFlag(&rxRing.hostFlags()[slot]);
    const std::uint64_t txFlag = loadFlag(&txRing.hostFlags()[slot]);
    if (rxFlag == 0 && txFlag != CUDAQ_TX_FLAG_IN_FLIGHT)
      return;
    if (std::chrono::steady_clock::now() > deadline)
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            "timed out waiting for fire-and-forget "
                            "device_call completion");
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void SharedMemoryChannelBase::clearSlot(std::uint32_t slot) {
  storeFlag(&txRing.hostFlags()[slot], 0);
  storeFlag(&rxRing.hostFlags()[slot], 0);
}

std::uint8_t *SharedMemoryChannelBase::rxHostSlot(std::uint32_t slot) const {
  return rxRing.hostData() + slot * slotSize;
}

std::uint8_t *SharedMemoryChannelBase::txHostSlot(std::uint32_t slot) const {
  return txRing.hostData() + slot * slotSize;
}

std::uint64_t
SharedMemoryChannelBase::rxHostSlotAddress(std::uint32_t slot) const {
  return reinterpret_cast<std::uint64_t>(rxHostSlot(slot));
}

std::uint64_t
SharedMemoryChannelBase::rxDeviceSlotAddress(std::uint32_t slot) const {
  return reinterpret_cast<std::uint64_t>(rxRing.deviceData() + slot * slotSize);
}

} // namespace cudaq_internal::device_call
