/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallChannel.h"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using cudaq_internal::device_call::createDeviceCallChannel;
using cudaq_internal::device_call::DeviceCallChannel;

std::string canonicalPluginName(const std::string &name) {
  return "libcudaq-device-call-channel-" + name + ".so";
}

std::string directoryName(const char *path) {
  if (!path)
    return {};
  std::string text(path);
  const auto slash = text.find_last_of('/');
  if (slash == std::string::npos)
    return {};
  return text.substr(0, slash);
}

void appendPathList(std::vector<std::string> &paths, const char *value) {
  if (!value || !*value)
    return;

  std::string text(value);
  std::size_t begin = 0;
  while (begin <= text.size()) {
    const auto end = text.find(':', begin);
    const auto count =
        end == std::string::npos ? std::string::npos : end - begin;
    std::string entry = text.substr(begin, count);
    if (!entry.empty())
      paths.push_back(std::move(entry));
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
}

std::vector<std::string> defaultPluginSearchPaths() {
  std::vector<std::string> paths;
  appendPathList(paths, std::getenv("CUDAQ_DEVICE_CALL_PLUGIN_PATH"));

  Dl_info info{};
  if (::dladdr(reinterpret_cast<void *>(&createDeviceCallChannel), &info) &&
      info.dli_fname) {
    std::string libDir = directoryName(info.dli_fname);
    if (!libDir.empty()) {
      paths.push_back(libDir + "/plugins");
      paths.push_back(std::move(libDir));
    }
  }

  return paths;
}

bool tryLoadChannelPlugin(const std::string &name) {
  static std::mutex mutex;
  static std::vector<void *> loadedPlugins;

  std::lock_guard<std::mutex> lock(mutex);
  if (cudaq::registry::isRegistered<DeviceCallChannel>(name))
    return true;

  const std::string libraryName = canonicalPluginName(name);
  std::vector<std::string> candidates;
  for (const std::string &dir : defaultPluginSearchPaths()) {
    if (!dir.empty() && dir.back() == '/')
      candidates.push_back(dir + libraryName);
    else
      candidates.push_back(dir + "/" + libraryName);
  }
  candidates.push_back(libraryName);

  for (const std::string &candidate : candidates) {
    (void)::dlerror();
    void *handle = ::dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle)
      continue;
    loadedPlugins.push_back(handle);
    if (cudaq::registry::isRegistered<DeviceCallChannel>(name))
      return true;
  }

  return false;
}

} // namespace

namespace cudaq_internal::device_call {

std::unique_ptr<DeviceCallChannel>
createDeviceCallChannel(const std::string &name,
                        DeviceCallChannelCreateArgs args) {
  if (!cudaq::registry::isRegistered<DeviceCallChannel>(name) &&
      !tryLoadChannelPlugin(name))
    throw std::invalid_argument("unknown CUDA-Q device_call channel '" + name +
                                "'");

  auto nextChannel = cudaq::registry::get<DeviceCallChannel>(name);
  if (!nextChannel)
    throw std::invalid_argument(
        "failed to create CUDA-Q device_call channel '" + name + "'");

  nextChannel->initialize(std::move(args));
  return nextChannel;
}

} // namespace cudaq_internal::device_call

CUDAQ_INSTANTIATE_REGISTRY(
    cudaq_internal::device_call::DeviceCallChannel::RegistryType)
