/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/RpcFrame.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cudaq_internal::device_call {

std::uint64_t requiredSlotSize(std::uint64_t requestBytes,
                               std::uint64_t responseCapacity) {
  return std::max<std::uint64_t>(CUDAQ_RPC_HEADER_SIZE + requestBytes,
                                 sizeof(cudaq::realtime::RPCResponse) +
                                     responseCapacity);
}

bool frameLengthsFitU32(std::uint64_t requestBytes,
                        std::uint64_t responseCapacity) {
  const std::uint64_t requestFrameLen = CUDAQ_RPC_HEADER_SIZE + requestBytes;
  const std::uint64_t responseFrameLen =
      sizeof(cudaq::realtime::RPCResponse) + responseCapacity;
  return requestFrameLen <= std::numeric_limits<std::uint32_t>::max() &&
         responseFrameLen <= std::numeric_limits<std::uint32_t>::max();
}

void initializeRequestHeader(void *frame, std::uint32_t functionId,
                             std::uint64_t requestBytes,
                             std::uint32_t requestId, std::uint64_t timestamp) {
  auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(frame);
  header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header->function_id = functionId;
  header->arg_len = static_cast<std::uint32_t>(requestBytes);
  header->request_id = requestId;
  header->ptp_timestamp = timestamp;
}

std::byte *requestPayload(void *frame) {
  return reinterpret_cast<std::byte *>(static_cast<std::uint8_t *>(frame) +
                                       CUDAQ_RPC_HEADER_SIZE);
}

std::byte *responsePayload(void *frame) {
  return reinterpret_cast<std::byte *>(static_cast<std::uint8_t *>(frame) +
                                       sizeof(cudaq::realtime::RPCResponse));
}

std::uint64_t validateResponseFrame(const void *frame, std::uint32_t requestId,
                                    std::uint64_t responseCapacity,
                                    std::uint64_t availableFrameBytes) {
  if (!frame || availableFrameBytes < sizeof(cudaq::realtime::RPCResponse))
    throw std::invalid_argument("invalid device_call response frame");

  const auto *response =
      reinterpret_cast<const cudaq::realtime::RPCResponse *>(frame);
  if (response->magic != cudaq::realtime::RPC_MAGIC_RESPONSE ||
      response->request_id != requestId)
    throw std::runtime_error("mismatched device_call response frame");

  if (!isSuccessStatus(response->status))
    throwDeviceCallStatus(response->status, "device_call remote endpoint");

  const std::uint64_t responseFrameBytes =
      sizeof(cudaq::realtime::RPCResponse) + response->result_len;
  if (responseFrameBytes > availableFrameBytes)
    throw std::runtime_error("truncated device_call response frame");
  if (response->result_len > responseCapacity)
    throw std::length_error("device_call response exceeds caller capacity");

  return response->result_len;
}

std::uint64_t validateResponseFrame(const void *frame, std::uint32_t requestId,
                                    std::uint64_t responseCapacity) {
  return validateResponseFrame(frame, requestId, responseCapacity,
                               std::numeric_limits<std::uint64_t>::max());
}

} // namespace cudaq_internal::device_call
