/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallServiceUtils.h"

#include <cctype>
#include <dlfcn.h>
#include <string>

namespace cudaq_internal::device_call {

namespace {
constexpr std::string_view DeviceCallServiceFactorySymbol =
    "cudaq_realtime_get_service";

std::string sanitizeDeviceCallServiceName(std::string_view name) {
  std::string result;
  result.reserve(name.size());
  for (unsigned char c : name)
    result.push_back(std::isalnum(c) ? static_cast<char>(c) : '_');
  return result;
}

std::string deviceCallServiceFactorySymbol(std::string_view servicePostfix) {
  if (servicePostfix.empty())
    return std::string(DeviceCallServiceFactorySymbol);
  return std::string(DeviceCallServiceFactorySymbol) + "_" +
         sanitizeDeviceCallServiceName(servicePostfix);
}
} // namespace

DeviceCallServiceFactoryFn
resolveDeviceCallServiceFactory(void *symbolScope,
                                std::string_view servicePostfix) {
  void *scope = symbolScope ? symbolScope : RTLD_DEFAULT;
  auto resolve =
      [&](std::string_view symbolName) -> DeviceCallServiceFactoryFn {
    (void)::dlerror();
    void *symbol = ::dlsym(scope, std::string(symbolName).c_str());
    if (!symbol)
      return nullptr;
    return reinterpret_cast<DeviceCallServiceFactoryFn>(symbol);
  };

  if (!servicePostfix.empty())
    if (auto factory = resolve(deviceCallServiceFactorySymbol(servicePostfix)))
      return factory;

  return resolve(DeviceCallServiceFactorySymbol);
}

} // namespace cudaq_internal::device_call
