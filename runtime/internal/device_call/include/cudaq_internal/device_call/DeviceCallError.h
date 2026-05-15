/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace cudaq_internal::device_call {

enum class DeviceCallStatus : std::int32_t {
  Success = 0,
  InvalidArgument = 1,
  NotInitialized = 2,
  CudaError = 3,
  Timeout = 4,
  ResponseTooLarge = 5,
  RemoteError = 6,
};

constexpr std::int32_t toAbiStatus(DeviceCallStatus status) noexcept {
  return static_cast<std::int32_t>(status);
}

constexpr bool isSuccessStatus(std::int32_t status) noexcept {
  return status == toAbiStatus(DeviceCallStatus::Success);
}

class DeviceCallError : public std::runtime_error {
public:
  DeviceCallError(DeviceCallStatus status, std::string message)
      : std::runtime_error(std::move(message)), statusCode(status) {}

  DeviceCallStatus status() const noexcept { return statusCode; }

private:
  DeviceCallStatus statusCode;
};

DeviceCallStatus
deviceCallStatusFromException(const std::exception &exception) noexcept;

std::int32_t abiStatusFromException(const std::exception &exception) noexcept;

std::string makeDeviceCallStatusMessage(DeviceCallStatus status,
                                        std::string_view source);

[[noreturn]] void throwDeviceCallStatus(std::int32_t status,
                                        std::string_view source);

} // namespace cudaq_internal::device_call
