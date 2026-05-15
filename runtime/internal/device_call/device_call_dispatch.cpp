/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq_internal/device_call/ArgParsing.h"
#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/DeviceCallRuntime.h"
#include "cudaq_internal/device_call/DeviceCallService.h"
#include "cudaq_internal/device_call/DeviceCallServiceUtils.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace cudaq_internal::device_call;

template <typename Fn>
std::int32_t runDeviceCallAbi(Fn &&fn) noexcept {
  try {
    fn();
    return toAbiStatus(DeviceCallStatus::Success);
  } catch (const std::exception &exception) {
    return abiStatusFromException(exception);
  } catch (...) {
    return toAbiStatus(DeviceCallStatus::RemoteError);
  }
}

struct DeviceCallRuntimeConfig {
  bool enabled = true;
  std::string channelName = DeviceDispatchSharedMemoryChannelName;
  std::vector<std::string> arguments;
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint64_t slotSize = DefaultSlotSize;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
};

DeviceCallChannelConfig
makeChannelConfig(const DeviceCallRuntimeConfig &config) {
  DeviceCallChannelConfig channelConfig;
  channelConfig.numSlots = config.numSlots;
  channelConfig.slotSize = config.slotSize;
  channelConfig.timeoutMs = config.timeoutMs;
  return channelConfig;
}

bool setChannelName(DeviceCallRuntimeConfig &config, const char *value) {
  if (!value || !*value)
    return false;
  if (std::strcmp(value, "off") == 0 || std::strcmp(value, "none") == 0 ||
      std::strcmp(value, "disabled") == 0) {
    config.enabled = false;
    config.channelName.clear();
    return true;
  }

  config.enabled = true;
  if (std::strcmp(value, "shared-memory") == 0 ||
      std::strcmp(value, "shared_memory") == 0)
    config.channelName = DeviceDispatchSharedMemoryChannelName;
  else
    config.channelName = value;
  return true;
}

bool parseChannelOption(DeviceCallRuntimeConfig &config, const char *value) {
  return setChannelName(config, value);
}

template <typename T>
bool parseEnvUInt(const char *name, T &out, std::uint64_t minValue = 0) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    return true;

  std::uint64_t parsed = 0;
  if (!parseUInt(value,
                 static_cast<std::uint64_t>(std::numeric_limits<T>::max()),
                 parsed) ||
      parsed < minValue)
    return false;
  out = static_cast<T>(parsed);
  return true;
}

bool applyDeviceCallEnvironment(DeviceCallRuntimeConfig &config) {
  if (const char *channel = std::getenv("CUDAQ_DEVICE_CALL_CHANNEL"))
    if (!setChannelName(config, channel))
      return false;

  return parseEnvUInt("CUDAQ_DEVICE_CALL_SLOTS", config.numSlots, 1) &&
         parseEnvUInt("CUDAQ_DEVICE_CALL_SLOT_SIZE", config.slotSize, 1) &&
         parseEnvUInt("CUDAQ_DEVICE_CALL_TIMEOUT_MS", config.timeoutMs, 1);
}

std::vector<std::string> collectArguments(int argc, char **argv) {
  std::vector<std::string> arguments;
  if (argc <= 0 || !argv)
    return arguments;
  arguments.reserve(argc);
  for (int i = 0; i < argc; ++i)
    if (argv[i])
      arguments.emplace_back(argv[i]);
  return arguments;
}

DeviceCallRuntimeConfig parseDeviceCallArgs(int argc, char **argv) {
  DeviceCallRuntimeConfig config;
  config.arguments = collectArguments(argc, argv);
  if (!applyDeviceCallEnvironment(config))
    throw std::invalid_argument("invalid CUDA-Q device_call environment");

  static constexpr CliOption<DeviceCallRuntimeConfig> options[] = {
      {"--cudaq-device-call", parseChannelOption},
      {"--cudaq-device-call-channel", parseChannelOption},
      {"--cudaq-device-call-slots",
       parseUIntOption<DeviceCallRuntimeConfig, std::uint32_t,
                       &DeviceCallRuntimeConfig::numSlots, 1>},
      {"--cudaq-device-call-slot-size",
       parseUIntOption<DeviceCallRuntimeConfig, std::uint64_t,
                       &DeviceCallRuntimeConfig::slotSize, 1>},
      {"--cudaq-device-call-timeout-ms",
       parseUIntOption<DeviceCallRuntimeConfig, std::uint64_t,
                       &DeviceCallRuntimeConfig::timeoutMs, 1>},
  };

  if (!parseCliOptions(argc, argv, options, config))
    throw std::invalid_argument("invalid CUDA-Q device_call command line");

  if (config.enabled && config.channelName.empty())
    throw std::invalid_argument("CUDA-Q device_call channel name is empty");
  return config;
}

int getPointerDevice(cudaq_function_entry_t *entries) {
  int device = 0;
  cudaGetDevice(&device);

  cudaPointerAttributes attributes{};
  auto err = cudaPointerGetAttributes(&attributes, entries);
  if (err == cudaSuccess) {
    if (attributes.type == cudaMemoryTypeDevice ||
        attributes.type == cudaMemoryTypeManaged)
      return attributes.device;
  } else {
    cudaGetLastError();
  }

  return device;
}

struct DeviceId {
  std::uint32_t value = 0;
};

constexpr DeviceId DefaultDeviceId{};

void registerShutdownHandler();

class DeviceCallServiceRecord {
public:
  DeviceCallServiceRecord() = default;
  DeviceCallServiceRecord(const DeviceCallServiceRecord &) = delete;
  DeviceCallServiceRecord &operator=(const DeviceCallServiceRecord &) = delete;
  DeviceCallServiceRecord(DeviceCallServiceRecord &&other) noexcept {
    moveFrom(other);
  }
  DeviceCallServiceRecord &operator=(DeviceCallServiceRecord &&other) noexcept {
    if (this != &other) {
      reset();
      moveFrom(other);
    }
    return *this;
  }
  ~DeviceCallServiceRecord() { reset(); }

  void initialize(cudaq_realtime_get_service_fn_t factory) {
    if (!factory)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call service factory is not initialized");

    if (factory(&service) != 0)
      throw std::invalid_argument("device_call service factory failed");
    if (!service.get_function_count || !service.populate_table)
      throw std::invalid_argument("device_call service is incomplete");

    if (service.create) {
      int hookStatus = service.create(nullptr, 0, &handle);
      if (hookStatus != 0) {
        reset();
        throw std::runtime_error("device_call service create hook failed");
      }
    }

    functionCount = service.get_function_count(handle);
    if (functionCount == 0) {
      reset();
      throw std::invalid_argument("device_call service exported no functions");
    }

    if (cudaMalloc(&functionEntries,
                   functionCount * sizeof(cudaq_function_entry_t)) !=
        cudaSuccess) {
      reset();
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to allocate device_call function table");
    }
    if (cudaMemset(functionEntries, 0,
                   functionCount * sizeof(cudaq_function_entry_t)) !=
        cudaSuccess) {
      reset();
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to initialize device_call function table");
    }
    if (service.populate_table(handle, functionEntries, functionCount,
                               nullptr) != 0 ||
        cudaDeviceSynchronize() != cudaSuccess) {
      reset();
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to populate device_call function table");
    }

    if (service.get_device_dispatch_launch)
      launchFn = service.get_device_dispatch_launch(handle);
    if (!launchFn) {
      reset();
      throw std::invalid_argument(
          "device_call service is missing dispatch launch hook");
    }

    if (service.get_device_dispatch_synchronize)
      synchronizeFn = service.get_device_dispatch_synchronize(handle);
  }

  void start() {
    if (started)
      return;
    if (service.start) {
      int hookStatus = service.start(handle);
      if (hookStatus != 0)
        throw std::runtime_error("device_call service start hook failed");
    }
    started = true;
  }

  void reset() noexcept {
    if (started && service.stop)
      (void)service.stop(handle);
    started = false;

    if (functionEntries) {
      (void)cudaFree(functionEntries);
      functionEntries = nullptr;
    }

    if (handle && service.destroy)
      (void)service.destroy(handle);

    service = {};
    handle = nullptr;
    functionCount = 0;
    launchFn = nullptr;
    synchronizeFn = nullptr;
  }

  cudaq_function_entry_t *entries() const noexcept { return functionEntries; }
  std::uint32_t count() const noexcept { return functionCount; }
  cudaq_dispatch_launch_fn_t launch() const noexcept { return launchFn; }
  cudaq_device_call_dispatch_synchronize_fn_t synchronize() const noexcept {
    return synchronizeFn;
  }
  bool isInitialized() const noexcept { return functionEntries != nullptr; }

private:
  void moveFrom(DeviceCallServiceRecord &other) noexcept {
    service = other.service;
    handle = std::exchange(other.handle, nullptr);
    functionEntries = std::exchange(other.functionEntries, nullptr);
    functionCount = std::exchange(other.functionCount, 0);
    launchFn = std::exchange(other.launchFn, nullptr);
    synchronizeFn = std::exchange(other.synchronizeFn, nullptr);
    started = std::exchange(other.started, false);
    other.service = {};
  }

  cudaq_realtime_device_call_service service{};
  void *handle = nullptr;
  cudaq_function_entry_t *functionEntries = nullptr;
  std::uint32_t functionCount = 0;
  cudaq_dispatch_launch_fn_t launchFn = nullptr;
  cudaq_device_call_dispatch_synchronize_fn_t synchronizeFn = nullptr;
  bool started = false;
};

class DeviceCallEndpoint {
public:
  explicit DeviceCallEndpoint(DeviceId id) : endpointId(id) {}
  DeviceCallEndpoint(const DeviceCallEndpoint &) = delete;
  DeviceCallEndpoint &operator=(const DeviceCallEndpoint &) = delete;
  ~DeviceCallEndpoint() { reset(); }

  DeviceId id() const noexcept { return endpointId; }

  void setChannel(std::unique_ptr<DeviceCallChannel> nextChannel) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureSessionNoLock();
    session->channel = std::move(nextChannel);
  }

  void addService(DeviceCallServiceRecord service) {
    std::lock_guard<std::mutex> lock(mutex);
    ensureSessionNoLock();
    session->services.push_back(std::move(service));
  }

  void startServices() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!session)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call endpoint has no active session");
    for (auto &service : session->services) {
      try {
        service.start();
      } catch (...) {
        resetUnlocked();
        throw;
      }
    }
  }

  void reset() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    resetUnlocked();
  }

  bool hasChannel() const noexcept {
    // The result is a snapshot; callers must not rely on it remaining true.
    std::lock_guard<std::mutex> lock(mutex);
    return session && static_cast<bool>(session->channel);
  }

  bool hasService() const noexcept {
    // The result is a snapshot; callers must not rely on it remaining true.
    std::lock_guard<std::mutex> lock(mutex);
    return session && !session->services.empty();
  }

private:
  friend class DeviceCallManager;

  struct Session {
    ~Session() { reset(); }

    void reset() noexcept {
      if (channel) {
        channel->stop();
        channel.reset();
      }
      for (auto &service : services)
        service.reset();
      services.clear();
    }

    std::unique_ptr<DeviceCallChannel> channel;
    std::vector<DeviceCallServiceRecord> services;
  };

  void ensureSessionNoLock() {
    if (!session)
      session = std::make_shared<Session>();
  }

  std::shared_ptr<Session> currentSession() const noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    return session;
  }

  void resetUnlocked() noexcept { session.reset(); }

  DeviceId endpointId;
  mutable std::mutex mutex;
  std::shared_ptr<Session> session;
};

class DeviceCallManager {
public:
  static DeviceCallManager &instance() {
    static DeviceCallManager manager;
    return manager;
  }

  DeviceCallManager(const DeviceCallManager &) = delete;
  DeviceCallManager &operator=(const DeviceCallManager &) = delete;
  DeviceCallManager(DeviceCallManager &&) = delete;
  DeviceCallManager &operator=(DeviceCallManager &&) = delete;

  void configure(int argc, char **argv) {
    DeviceCallRuntimeConfig parsed = parseDeviceCallArgs(argc, argv);
    std::lock_guard<std::mutex> lock(mutex);
    resetAllEndpointsNoLock();
    config = parsed;

    if (config.enabled &&
        config.channelName != DeviceDispatchSharedMemoryChannelName) {
      auto endpoint = std::make_shared<DeviceCallEndpoint>(DefaultDeviceId);
      DeviceCallChannelCreateArgs args;
      args.channelName = config.channelName;
      args.arguments = config.arguments;
      args.channelConfig = makeChannelConfig(config);
      auto channel =
          createDeviceCallChannel(config.channelName, std::move(args));
      endpoint->setChannel(std::move(channel));
      endpoints[endpoint->id().value] = std::move(endpoint);
      registerShutdownHandler();
    }
  }

  DeviceCallRuntimeMode configuredMode() const {
    std::lock_guard<std::mutex> lock(mutex);
    if (!config.enabled)
      return DeviceCallRuntimeMode::Off;
    if (config.channelName == DeviceDispatchSharedMemoryChannelName)
      return DeviceCallRuntimeMode::SharedMemory;
    return DeviceCallRuntimeMode::ExternalChannel;
  }

  void registerServiceFactory(cudaq_realtime_get_service_fn_t factory) {
    registerServiceFactoryForDevice(DefaultDeviceId.value, factory);
  }

  void
  registerServiceFactoryForDevice(std::uint32_t deviceId,
                                  cudaq_realtime_get_service_fn_t factory) {
    std::lock_guard<std::mutex> lock(mutex);
    auto endpoint = findEndpointNoLock(deviceId);
    if (factory && endpoint && endpoint->hasChannel())
      throw std::invalid_argument(
          "cannot register a device_call service factory after channel setup");
    if (factory)
      registeredFactories[deviceId] = factory;
    else
      registeredFactories.erase(deviceId);
  }

  void initializeService(void *symbolScope = nullptr,
                         std::string_view servicePostfix = {}) {
    initializeServiceForDevice(DefaultDeviceId.value, symbolScope,
                               servicePostfix);
  }

  void initializeServiceForDevice(std::uint32_t deviceId,
                                  void *symbolScope = nullptr,
                                  std::string_view servicePostfix = {}) {
    std::lock_guard<std::mutex> lock(mutex);
    auto existingEndpoint = findEndpointNoLock(deviceId);
    if (existingEndpoint && existingEndpoint->hasService())
      return;
    if (config.enabled &&
        config.channelName != DeviceDispatchSharedMemoryChannelName) {
      if (existingEndpoint && existingEndpoint->hasChannel()) {
        return;
      } else {
        throw DeviceCallError(
            DeviceCallStatus::NotInitialized,
            "external device_call channel is not initialized");
      }
    }

    auto factory =
        resolveServiceFactoryNoLock(deviceId, symbolScope, servicePostfix);
    if (!factory)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call service factory was not found");

    DeviceCallServiceRecord service;
    service.initialize(factory);

    auto endpoint = std::make_shared<DeviceCallEndpoint>(DeviceId{deviceId});
    DeviceCallChannelCreateArgs args;
    args.functionTable = service.entries();
    args.functionCount = service.count();
    args.deviceId = getPointerDevice(service.entries());
    args.launchFn = service.launch();
    args.synchronizeFn = service.synchronize();
    args.channelName = DeviceDispatchSharedMemoryChannelName;
    args.arguments = config.arguments;
    args.channelConfig = makeChannelConfig(config);
    auto channel = createDeviceCallChannel(
        DeviceDispatchSharedMemoryChannelName, std::move(args));
    endpoint->setChannel(std::move(channel));
    endpoint->addService(std::move(service));
    endpoint->startServices();

    resetEndpointNoLock(deviceId);
    endpoints[deviceId] = std::move(endpoint);
    registerShutdownHandler();
  }

  void finalizeService() { finalizeServiceForDevice(DefaultDeviceId.value); }

  void finalizeServiceForDevice(std::uint32_t deviceId) {
    std::lock_guard<std::mutex> lock(mutex);
    auto endpoint = findEndpointNoLock(deviceId);
    if (!endpoint || !endpoint->hasService())
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call service is not initialized");
    resetEndpointNoLock(deviceId);
  }

  void setFunctionTableWithLauncher(
      cudaq_function_entry_t *entries, std::uint32_t count,
      cudaq_dispatch_launch_fn_t launchFn,
      cudaq_device_call_dispatch_synchronize_fn_t synchronizeFn) {
    setFunctionTableWithLauncherForDevice(DefaultDeviceId.value, entries, count,
                                          launchFn, synchronizeFn);
  }

  void setFunctionTableWithLauncherForDevice(
      std::uint32_t deviceId, cudaq_function_entry_t *entries,
      std::uint32_t count, cudaq_dispatch_launch_fn_t launchFn,
      cudaq_device_call_dispatch_synchronize_fn_t synchronizeFn) {
    if (!entries || count == 0)
      throw std::invalid_argument("device_call function table is empty");
    if (!launchFn)
      throw std::invalid_argument(
          "device_call dispatch launch hook is missing");

    std::lock_guard<std::mutex> lock(mutex);
    if (config.enabled &&
        config.channelName != DeviceDispatchSharedMemoryChannelName) {
      auto endpoint = findEndpointNoLock(deviceId);
      if (endpoint && endpoint->hasChannel())
        return;
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "external device_call channel is not initialized");
    }

    auto endpoint = std::make_shared<DeviceCallEndpoint>(DeviceId{deviceId});
    DeviceCallChannelCreateArgs args;
    args.functionTable = entries;
    args.functionCount = count;
    args.deviceId = getPointerDevice(entries);
    args.launchFn = launchFn;
    args.synchronizeFn = synchronizeFn;
    args.channelName = DeviceDispatchSharedMemoryChannelName;
    args.arguments = config.arguments;
    args.channelConfig = makeChannelConfig(config);
    auto channel = createDeviceCallChannel(
        DeviceDispatchSharedMemoryChannelName, std::move(args));
    endpoint->setChannel(std::move(channel));
    resetEndpointNoLock(deviceId);
    endpoints[deviceId] = std::move(endpoint);
    registerShutdownHandler();
  }

  void shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    resetAllEndpointsNoLock();
  }

  struct DeviceCallFrameHandle {
    std::shared_ptr<DeviceCallEndpoint::Session> session;
    DeviceCallChannel::DeviceCallFrame frame;
  };

  void acquireFrameForDevice(std::uint32_t deviceId, std::uint32_t functionId,
                             std::uint64_t requestBytes,
                             std::uint64_t responseCapacity, void **frameHandle,
                             void **requestPayload, void **responsePayload) {
    if (!frameHandle || !requestPayload || !responsePayload)
      throw std::invalid_argument(
          "device_call frame output pointers must be non-null");
    *frameHandle = nullptr;
    *requestPayload = nullptr;
    *responsePayload = nullptr;

    std::shared_ptr<DeviceCallEndpoint> endpoint;
    {
      std::lock_guard<std::mutex> lock(mutex);
      endpoint = findEndpointNoLock(deviceId);
    }
    if (!endpoint)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call endpoint is not initialized");

    auto handle = std::make_unique<DeviceCallFrameHandle>();
    handle->session = endpoint->currentSession();
    if (!handle->session || !handle->session->channel)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call channel is not initialized");

    handle->session->channel->acquireFrame(functionId, requestBytes,
                                           responseCapacity, handle->frame);

    *requestPayload = handle->frame.request.data;
    *responsePayload =
        responseCapacity == 0 ? nullptr : handle->frame.response.data;
    *frameHandle = handle.release();
  }

  static std::uint64_t dispatchFrame(void *opaqueFrame) {
    if (!opaqueFrame)
      throw std::invalid_argument("invalid device_call frame handle");
    auto *handle = static_cast<DeviceCallFrameHandle *>(opaqueFrame);
    if (!handle->session || !handle->session->channel)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "device_call channel is not initialized");
    return handle->session->channel->dispatchFrame(handle->frame);
  }

  static void releaseFrame(void *opaqueFrame) noexcept {
    if (!opaqueFrame)
      return;
    auto *handle = static_cast<DeviceCallFrameHandle *>(opaqueFrame);
    if (handle->session && handle->session->channel)
      handle->session->channel->releaseFrame(handle->frame);
    delete handle;
  }

private:
  DeviceCallManager() = default;

  cudaq_realtime_get_service_fn_t
  resolveServiceFactoryNoLock(std::uint32_t deviceId, void *symbolScope,
                              std::string_view servicePostfix) const {
    if (!symbolScope) {
      auto foundFactory = registeredFactories.find(deviceId);
      if (foundFactory != registeredFactories.end())
        return foundFactory->second;
    }

    return resolveDeviceCallServiceFactory(symbolScope, servicePostfix);
  }

  std::shared_ptr<DeviceCallEndpoint>
  findEndpointNoLock(std::uint32_t deviceId) const {
    auto found = endpoints.find(deviceId);
    if (found == endpoints.end())
      return nullptr;
    return found->second;
  }

  void resetEndpointNoLock(std::uint32_t deviceId) noexcept {
    auto found = endpoints.find(deviceId);
    if (found == endpoints.end())
      return;
    found->second->reset();
    endpoints.erase(found);
  }

  void resetAllEndpointsNoLock() noexcept {
    for (auto &[deviceId, endpoint] : endpoints) {
      (void)deviceId;
      endpoint->reset();
    }
    endpoints.clear();
  }

  mutable std::mutex mutex;
  DeviceCallRuntimeConfig config;
  std::unordered_map<std::uint32_t, std::shared_ptr<DeviceCallEndpoint>>
      endpoints;
  std::unordered_map<std::uint32_t, cudaq_realtime_get_service_fn_t>
      registeredFactories;
};

void shutdownManagerNoThrow() { DeviceCallManager::instance().shutdown(); }

void registerShutdownHandler() {
  static std::once_flag once;
  // The runtime library is process-resident, so the atexit handler cannot
  // outlive unloaded library text.
  std::call_once(once, [] { std::atexit(shutdownManagerNoThrow); });
}

} // namespace

namespace cudaq_internal::device_call {

void configureDeviceCallRuntime(int argc, char **argv) {
  DeviceCallManager::instance().configure(argc, argv);
}

DeviceCallRuntimeMode getConfiguredDeviceCallRuntimeMode() {
  return DeviceCallManager::instance().configuredMode();
}

void shutdownDeviceCallRuntime() noexcept {
  DeviceCallManager::instance().shutdown();
}

void registerDeviceCallServiceFactory(cudaq_realtime_get_service_fn_t factory) {
  DeviceCallManager::instance().registerServiceFactory(factory);
}

void initializeDeviceCallService(void *symbolScope,
                                 std::string_view servicePostfix) {
  DeviceCallManager::instance().initializeService(symbolScope, servicePostfix);
}

void finalizeDeviceCallService() {
  DeviceCallManager::instance().finalizeService();
}

void setDeviceCallFunctionTableWithLauncher(
    cudaq_function_entry_t *entries, std::uint32_t count,
    cudaq_dispatch_launch_fn_t launchFn) {
  DeviceCallManager::instance().setFunctionTableWithLauncher(entries, count,
                                                             launchFn, nullptr);
}

void setDeviceCallFunctionTableWithLauncherForDevice(
    std::uint32_t deviceId, cudaq_function_entry_t *entries,
    std::uint32_t count, cudaq_dispatch_launch_fn_t launchFn) {
  DeviceCallManager::instance().setFunctionTableWithLauncherForDevice(
      deviceId, entries, count, launchFn, nullptr);
}

} // namespace cudaq_internal::device_call

extern "C" std::int32_t __cudaq_device_call_acquire_realtime_frame(
    std::uint32_t deviceId, std::uint32_t functionId,
    std::uint64_t requestBytes, std::uint64_t responseCapacity,
    void **frameHandle, void **requestPayload, void **responsePayload) {
  return runDeviceCallAbi([&] {
    DeviceCallManager::instance().acquireFrameForDevice(
        deviceId, functionId, requestBytes, responseCapacity, frameHandle,
        requestPayload, responsePayload);
  });
}

extern "C" std::int32_t
__cudaq_device_call_dispatch_realtime_frame(void *frameHandle,
                                            std::uint64_t *responseBytes) {
  return runDeviceCallAbi([&] {
    if (!responseBytes)
      throw std::invalid_argument(
          "device_call response byte pointer must be non-null");
    *responseBytes = DeviceCallManager::dispatchFrame(frameHandle);
  });
}

extern "C" void
__cudaq_device_call_safely_release_realtime_frame(void *frameHandle) {
  DeviceCallManager::releaseFrame(frameHandle);
}
