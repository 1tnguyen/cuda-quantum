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

std::uint64_t loadFlag(volatile std::uint64_t *flag);

void storeFlag(volatile std::uint64_t *flag, std::uint64_t value);

struct MappedAllocation {
  void *host = nullptr;
  void *device = nullptr;
  std::size_t bytes = 0;

  MappedAllocation() = default;
  MappedAllocation(const MappedAllocation &) = delete;
  MappedAllocation &operator=(const MappedAllocation &) = delete;
  ~MappedAllocation();

  bool allocate(std::size_t size);

  void reset();
};

struct MappedRingBuffer {
  MappedAllocation flags;
  MappedAllocation data;
  std::uint32_t numSlots = 0;
  std::uint64_t slotSize = 0;

  bool allocate(std::uint32_t slots, std::uint64_t bytesPerSlot);

  void reset();

  volatile std::uint64_t *hostFlags() const;

  volatile std::uint64_t *deviceFlags() const;

  std::uint8_t *hostData() const;

  std::uint8_t *deviceData() const;
};

} // namespace cudaq_internal::device_call
