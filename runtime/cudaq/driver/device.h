/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <type_traits>
#include <utility>

namespace cudaq {

template <typename DeviceCode, typename... Args>
auto device_call(DeviceCode &&code, Args &&...args)
    -> std::invoke_result_t<DeviceCode, Args...> {
  return std::invoke(std::forward<DeviceCode>(code),
                     std::forward<Args>(args)...);
}

template <typename DeviceCode, typename... Args>
auto device_call(std::size_t device_id, DeviceCode &&code, Args &&...args)
    -> std::invoke_result_t<DeviceCode, Args...> {
  return std::invoke(std::forward<DeviceCode>(code),
                     std::forward<Args>(args)...);
}

#if defined(CUDAQ_QUANTUM_DEVICE)
// For quantum devices, `device_call` can refer to a callback function that is
// registered on the device. We provide an overload that takes a callback name
// as a string, which can be used to look up the corresponding function on the
// device.
template <typename ReturnType, typename... Args>
ReturnType device_call(std::size_t device_id, const char *callbackName,
                       Args &&...args) {
  // This should never be executed, as the JIT compiler should recognize this
  // pattern and replace it with the appropriate device call.
  throw std::runtime_error(
      "device_call with callback name is not implemented in this environment.");
  return ReturnType{};
}
#else
// If not compiling for a quantum device, we output a compile-time error if the callback name overload is used, as it is not intended to be used in this context.
template <typename ReturnType, typename... Args>
ReturnType device_call(std::size_t device_id, const char *callbackName,
                       Args &&...args) {
  static_assert(sizeof...(Args) == 0, "device_call with callback name is only supported on quantum devices.");
  return ReturnType{};
}
#endif

// --- GPU Overloads ---
template <std::size_t BlockSize, std::size_t GridSize, typename DeviceCode,
          typename... Args>
auto device_call(DeviceCode &&code, Args &&...args)
    -> std::invoke_result_t<DeviceCode, Args...> {
  return std::invoke(std::forward<DeviceCode>(code),
                     std::forward<Args>(args)...);
}
template <std::size_t BlockSize, std::size_t GridSize, typename DeviceCode,
          typename... Args>
auto device_call(std::size_t device_id, DeviceCode &&code, Args &&...args)
    -> std::invoke_result_t<DeviceCode, Args...> {
  return std::invoke(std::forward<DeviceCode>(code),
                     std::forward<Args>(args)...);
}

} // namespace cudaq
