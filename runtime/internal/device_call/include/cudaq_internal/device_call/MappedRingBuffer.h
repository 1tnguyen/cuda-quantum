/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

namespace cudaq_internal::device_call {

// Acquire/release helpers for ring flags shared across host and device-visible
// dispatch state.
std::uint64_t loadFlag(volatile std::uint64_t *flag);

void storeFlag(volatile std::uint64_t *flag, std::uint64_t value);

// Owning mapped allocation with host and device views of the same storage.
// The device pointer is valid for CUDA code that needs to observe or update the
// storage directly; bytes records the size of the active allocation.
struct MappedAllocation {
  void *host = nullptr;
  void *device = nullptr;
  std::size_t bytes = 0;

  MappedAllocation() = default;
  MappedAllocation(const MappedAllocation &) = delete;
  MappedAllocation &operator=(const MappedAllocation &) = delete;
  ~MappedAllocation();

  // Replace any existing allocation with zero-initialized mapped storage.
  bool allocate(std::size_t size);

  // Release the allocation and clear both address-space views.
  void reset();
};

// Realtime ring storage with separate flag and data arrays.  flags contains one
// std::uint64_t slot state per ring entry; data contains numSlots contiguous
// payload slots, each slotSize bytes wide.
struct MappedRingBuffer {
  MappedAllocation flags;
  MappedAllocation data;
  std::uint32_t numSlots = 0;
  std::uint64_t slotSize = 0;

  // Allocate flags and payload storage for slots entries.
  bool allocate(std::uint32_t slots, std::uint64_t bytesPerSlot);

  // Release all ring storage and clear cached shape metadata.
  void reset();

  // Host and device views remain aliases of the same mapped storage and are
  // valid until reset or the next successful allocate call.
  volatile std::uint64_t *hostFlags() const;

  volatile std::uint64_t *deviceFlags() const;

  std::uint8_t *hostData() const;

  std::uint8_t *deviceData() const;
};

} // namespace cudaq_internal::device_call
