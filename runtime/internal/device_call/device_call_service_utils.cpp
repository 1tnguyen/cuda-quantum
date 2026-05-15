/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallServiceUtils.h"

#include <cctype>
#include <cstddef>
#include <dlfcn.h>

namespace cudaq_internal::device_call {

std::string sanitizeDeviceCallServiceName(std::string_view name) {
  std::string result;
  result.reserve(name.size());
  for (unsigned char c : name)
    result.push_back(std::isalnum(c) ? static_cast<char>(c) : '_');
  return result;
}

std::string deviceCallServicePostfixFromLibraryPath(std::string_view path) {
  std::string_view filename = path;
  const std::size_t slash = filename.find_last_of("/\\");
  if (slash != std::string_view::npos)
    filename.remove_prefix(slash + 1);
  if (filename.starts_with("lib"))
    filename.remove_prefix(3);

  const std::size_t so = filename.find(".so");
  if (so != std::string_view::npos)
    filename = filename.substr(0, so);
  else if (filename.ends_with(".dylib"))
    filename.remove_suffix(6);
  else if (filename.ends_with(".dll"))
    filename.remove_suffix(4);

  return sanitizeDeviceCallServiceName(filename);
}

std::string deviceCallServiceFactorySymbol(std::string_view servicePostfix) {
  if (servicePostfix.empty())
    return std::string(DeviceCallServiceFactorySymbol);
  return std::string(DeviceCallServiceFactorySymbol) + "_" +
         sanitizeDeviceCallServiceName(servicePostfix);
}

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
