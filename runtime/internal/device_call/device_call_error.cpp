/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallError.h"

namespace cudaq_internal::device_call {

DeviceCallStatus
deviceCallStatusFromException(const std::exception &exception) noexcept {
  if (const auto *error = dynamic_cast<const DeviceCallError *>(&exception))
    return error->status();
  if (dynamic_cast<const std::invalid_argument *>(&exception))
    return DeviceCallStatus::InvalidArgument;
  if (dynamic_cast<const std::length_error *>(&exception))
    return DeviceCallStatus::ResponseTooLarge;
  return DeviceCallStatus::RemoteError;
}

std::int32_t abiStatusFromException(const std::exception &exception) noexcept {
  return toAbiStatus(deviceCallStatusFromException(exception));
}

std::string makeDeviceCallStatusMessage(DeviceCallStatus status,
                                        std::string_view source) {
  std::string message(source);
  if (!message.empty())
    message += ": ";
  switch (status) {
  case DeviceCallStatus::Success:
    return message + "unexpected success status";
  case DeviceCallStatus::InvalidArgument:
    return message + "invalid request";
  case DeviceCallStatus::NotInitialized:
    return message + "endpoint is not initialized";
  case DeviceCallStatus::CudaError:
    return message + "CUDA error";
  case DeviceCallStatus::Timeout:
    return message + "timed out";
  case DeviceCallStatus::ResponseTooLarge:
    return message + "response exceeds caller capacity";
  case DeviceCallStatus::RemoteError:
    return message + "remote endpoint error";
  }
  return message + "unknown status";
}

[[noreturn]] void throwDeviceCallStatus(std::int32_t status,
                                        std::string_view source) {
  switch (status) {
  case toAbiStatus(DeviceCallStatus::InvalidArgument):
    throw DeviceCallError(
        DeviceCallStatus::InvalidArgument,
        makeDeviceCallStatusMessage(DeviceCallStatus::InvalidArgument, source));
  case toAbiStatus(DeviceCallStatus::NotInitialized):
    throw DeviceCallError(
        DeviceCallStatus::NotInitialized,
        makeDeviceCallStatusMessage(DeviceCallStatus::NotInitialized, source));
  case toAbiStatus(DeviceCallStatus::CudaError):
    throw DeviceCallError(
        DeviceCallStatus::CudaError,
        makeDeviceCallStatusMessage(DeviceCallStatus::CudaError, source));
  case toAbiStatus(DeviceCallStatus::Timeout):
    throw DeviceCallError(
        DeviceCallStatus::Timeout,
        makeDeviceCallStatusMessage(DeviceCallStatus::Timeout, source));
  case toAbiStatus(DeviceCallStatus::ResponseTooLarge):
    throw DeviceCallError(DeviceCallStatus::ResponseTooLarge,
                          makeDeviceCallStatusMessage(
                              DeviceCallStatus::ResponseTooLarge, source));
  case toAbiStatus(DeviceCallStatus::RemoteError):
    throw DeviceCallError(
        DeviceCallStatus::RemoteError,
        makeDeviceCallStatusMessage(DeviceCallStatus::RemoteError, source));
  default:
    throw DeviceCallError(DeviceCallStatus::RemoteError,
                          std::string(source) +
                              ": returned device_call status " +
                              std::to_string(status));
  }
}

} // namespace cudaq_internal::device_call
