/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/RpcFrame.h"

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace cudaq_internal::device_call {

static_assert(sizeof(cudaq::realtime::RPCHeader) == CUDAQ_RPC_HEADER_SIZE);

void initializeRequestHeader(void *frame, std::uint32_t functionId,
                             std::uint64_t requestBytes,
                             std::uint32_t requestId, std::uint64_t timestamp) {
  if (requestBytes > std::numeric_limits<std::uint32_t>::max())
    throw std::invalid_argument("device_call request length exceeds 32 bits");

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

  if (response->status != toAbiStatus(DeviceCallStatus::Success)) {
    std::string message = "device_call remote endpoint: ";
    switch (response->status) {
    case toAbiStatus(DeviceCallStatus::InvalidArgument):
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            message + "invalid request");
    case toAbiStatus(DeviceCallStatus::NotInitialized):
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            message + "endpoint is not initialized");
    case toAbiStatus(DeviceCallStatus::CudaError):
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            message + "CUDA error");
    case toAbiStatus(DeviceCallStatus::Timeout):
      throw DeviceCallError(DeviceCallStatus::Timeout, message + "timed out");
    case toAbiStatus(DeviceCallStatus::ResponseTooLarge):
      throw DeviceCallError(DeviceCallStatus::ResponseTooLarge,
                            message + "response exceeds caller capacity");
    case toAbiStatus(DeviceCallStatus::RemoteError):
      throw DeviceCallError(DeviceCallStatus::RemoteError,
                            message + "remote endpoint error");
    default:
      throw DeviceCallError(DeviceCallStatus::RemoteError,
                            message + "returned device_call status " +
                                std::to_string(response->status));
    }
  }

  const std::uint64_t responseFrameBytes =
      sizeof(cudaq::realtime::RPCResponse) + response->result_len;
  if (responseFrameBytes > availableFrameBytes)
    throw std::runtime_error("truncated device_call response frame");
  if (response->result_len > responseCapacity)
    throw std::length_error("device_call response exceeds caller capacity");

  return response->result_len;
}

} // namespace cudaq_internal::device_call
