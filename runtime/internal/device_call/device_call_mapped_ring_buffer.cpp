/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/MappedRingBuffer.h"

#include <cuda_runtime.h>

#include <cstring>

namespace cudaq_internal::device_call {

std::uint64_t loadFlag(volatile std::uint64_t *flag) {
  return __atomic_load_n(const_cast<std::uint64_t *>(flag), __ATOMIC_ACQUIRE);
}

void storeFlag(volatile std::uint64_t *flag, std::uint64_t value) {
  __atomic_store_n(const_cast<std::uint64_t *>(flag), value, __ATOMIC_RELEASE);
}

MappedAllocation::~MappedAllocation() { reset(); }

bool MappedAllocation::allocate(std::size_t size) {
  reset();

  void *hostPtr = nullptr;
  cudaError_t err = cudaHostAlloc(&hostPtr, size, cudaHostAllocMapped);
  if (err != cudaSuccess)
    return false;

  void *devicePtr = nullptr;
  err = cudaHostGetDevicePointer(&devicePtr, hostPtr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(hostPtr);
    return false;
  }

  std::memset(hostPtr, 0, size);
  host = hostPtr;
  device = devicePtr;
  bytes = size;
  return true;
}

void MappedAllocation::reset() {
  if (host)
    cudaFreeHost(host);
  host = nullptr;
  device = nullptr;
  bytes = 0;
}

bool MappedRingBuffer::allocate(std::uint32_t slots,
                                std::uint64_t bytesPerSlot) {
  reset();
  const auto flagsBytes =
      static_cast<std::size_t>(slots) * sizeof(std::uint64_t);
  const auto dataBytes =
      static_cast<std::size_t>(slots) * static_cast<std::size_t>(bytesPerSlot);
  if (!flags.allocate(flagsBytes) || !data.allocate(dataBytes)) {
    reset();
    return false;
  }
  numSlots = slots;
  slotSize = bytesPerSlot;
  return true;
}

void MappedRingBuffer::reset() {
  flags.reset();
  data.reset();
  numSlots = 0;
  slotSize = 0;
}

volatile std::uint64_t *MappedRingBuffer::hostFlags() const {
  return static_cast<volatile std::uint64_t *>(flags.host);
}

volatile std::uint64_t *MappedRingBuffer::deviceFlags() const {
  return static_cast<volatile std::uint64_t *>(flags.device);
}

std::uint8_t *MappedRingBuffer::hostData() const {
  return static_cast<std::uint8_t *>(data.host);
}

std::uint8_t *MappedRingBuffer::deviceData() const {
  return static_cast<std::uint8_t *>(data.device);
}

} // namespace cudaq_internal::device_call
