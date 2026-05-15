/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cstddef>
#include <cstdint>

namespace cudaq_internal::device_call {

static_assert(sizeof(cudaq::realtime::RPCHeader) == CUDAQ_RPC_HEADER_SIZE);

std::uint64_t requiredSlotSize(std::uint64_t requestBytes,
                               std::uint64_t responseCapacity);

bool frameLengthsFitU32(std::uint64_t requestBytes,
                        std::uint64_t responseCapacity);

void initializeRequestHeader(void *frame, std::uint32_t functionId,
                             std::uint64_t requestBytes,
                             std::uint32_t requestId,
                             std::uint64_t timestamp = 0);

std::byte *requestPayload(void *frame);

std::byte *responsePayload(void *frame);

std::uint64_t validateResponseFrame(const void *frame, std::uint32_t requestId,
                                    std::uint64_t responseCapacity,
                                    std::uint64_t availableFrameBytes);

std::uint64_t validateResponseFrame(const void *frame, std::uint32_t requestId,
                                    std::uint64_t responseCapacity);

} // namespace cudaq_internal::device_call
