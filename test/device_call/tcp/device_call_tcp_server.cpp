/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/ArgParsing.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/DeviceCallRuntime.h"
#include "cudaq_internal/device_call/DeviceCallServiceUtils.h"
#include "cudaq_internal/device_call/RpcFrame.h"
#include "SocketUtils.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>

#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <signal.h>
#include <string>
#include <utility>
#include <vector>

extern "C" std::int32_t __cudaq_device_call_acquire_realtime_frame(
    std::uint32_t deviceId, std::uint32_t functionId,
    std::uint64_t requestBytes, std::uint64_t responseCapacity,
    void **frameHandle, void **requestPayload, void **responsePayload);
extern "C" std::int32_t
__cudaq_device_call_dispatch_realtime_frame(void *frameHandle,
                                            std::uint64_t *responseBytes);
extern "C" void
__cudaq_device_call_safely_release_realtime_frame(void *frameHandle);

namespace {

using namespace cudaq_internal::device_call;

constexpr std::uint32_t DefaultNumSlots = 2;
constexpr std::uint32_t DefaultSlotSize = 4096;
constexpr std::uint64_t DefaultTimeoutMs = 10000;

using ServiceConfigureFn = int (*)(int, char **);

struct Options {
  std::string host = "127.0.0.1";
  std::uint16_t port = 0;
  std::string portFile;
  std::string readyFile;
  std::string serviceLibrary;
  int gpu = -1;
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint32_t slotSize = DefaultSlotSize;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
};

struct LoadedService {
  void *handle = nullptr;
  bool serviceInitialized = false;

  ~LoadedService() { reset(); }

  void reset() {
    if (serviceInitialized) {
      try {
        finalizeDeviceCallService();
      } catch (...) {
      }
      serviceInitialized = false;
    }
    shutdownDeviceCallRuntime();
    if (handle) {
      (void)::dlclose(handle);
      handle = nullptr;
    }
  }
};

struct RealtimeFrameLease {
  ~RealtimeFrameLease() {
    if (handle)
      __cudaq_device_call_safely_release_realtime_frame(handle);
  }

  void *handle = nullptr;
};

struct ListenSocket {
  TcpSocket socket;
  std::uint16_t boundPort = 0;
};

std::atomic_bool &shutdownRequested() {
  static std::atomic_bool requested{false};
  return requested;
}

void signalHandler(int) { shutdownRequested().store(true); }

void installSignalHandlers() {
  struct sigaction action{};
  action.sa_handler = signalHandler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  (void)::sigaction(SIGINT, &action, nullptr);
  (void)::sigaction(SIGTERM, &action, nullptr);
}

bool parseSlotSizeOption(Options &options, const char *value) {
  std::uint64_t parsed = 0;
  if (!parseUInt(value, std::numeric_limits<std::uint32_t>::max(), parsed) ||
      parsed < CUDAQ_RPC_HEADER_SIZE)
    return false;
  options.slotSize = static_cast<std::uint32_t>(parsed);
  return true;
}

bool parseOptions(int argc, char **argv, Options &options) {
  static constexpr CliOption<Options> optionSpecs[] = {
      {"--host", parseStringOption<Options, &Options::host>},
      {"--port", parseUIntOption<Options, std::uint16_t, &Options::port>},
      {"--port-file", parseStringOption<Options, &Options::portFile>},
      {"--ready-file", parseStringOption<Options, &Options::readyFile>},
      {"--service", parseStringOption<Options, &Options::serviceLibrary>},
      {"--service-lib", parseStringOption<Options, &Options::serviceLibrary>},
      {"--gpu", parseNonNegativeIntOption<Options, &Options::gpu>},
      {"--num-slots",
       parseUIntOption<Options, std::uint32_t, &Options::numSlots, 1>},
      {"--slot-size", parseSlotSizeOption},
      {"--timeout-ms",
       parseUIntOption<Options, std::uint64_t, &Options::timeoutMs, 1>},
  };

  if (!parseCliOptions(argc, argv, optionSpecs, options))
    return false;

  if (options.serviceLibrary.empty()) {
    if (std::optional<std::string> value =
            llvm::sys::Process::GetEnv("CUDAQ_DEVICE_CALL_SERVICE_LIB"))
      options.serviceLibrary = std::move(*value);
  }
  return !options.serviceLibrary.empty();
}

bool writeTextFile(const std::string &path, const std::string &value) {
  if (path.empty())
    return true;
  return !llvm::sys::writeFileWithEncoding(path, value);
}

int fail(const std::string &message) {
  std::cerr << "cudaq-device-call-tcp-server: " << message << '\n';
  return 1;
}

void configureLocalRuntime(const Options &options) {
  std::vector<std::string> args;
  args.emplace_back("cudaq-device-call-tcp-server");
  args.emplace_back("--cudaq-device-call=shared-memory");
  args.emplace_back("--cudaq-device-call-slots=" +
                    std::to_string(options.numSlots));
  args.emplace_back("--cudaq-device-call-slot-size=" +
                    std::to_string(options.slotSize));
  args.emplace_back("--cudaq-device-call-timeout-ms=" +
                    std::to_string(options.timeoutMs));

  std::vector<char *> runtimeArgv;
  runtimeArgv.reserve(args.size() + 1);
  for (auto &arg : args)
    runtimeArgv.push_back(arg.data());
  runtimeArgv.push_back(nullptr);
  configureDeviceCallRuntime(static_cast<int>(args.size()), runtimeArgv.data());
}

std::unique_ptr<LoadedService>
loadAndInitializeService(int argc, char **argv, const Options &options) {
  auto service = std::make_unique<LoadedService>();
  service->handle =
      ::dlopen(options.serviceLibrary.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!service->handle) {
    const char *error = ::dlerror();
    fail("failed to load service library '" + options.serviceLibrary +
         "': " + (error ? error : "unknown dlopen error"));
    return nullptr;
  }

  if (options.gpu >= 0 && cudaSetDevice(options.gpu) != cudaSuccess) {
    fail("failed to set CUDA device " + std::to_string(options.gpu));
    return nullptr;
  }

  if (auto *symbol =
          ::dlsym(service->handle, "__cudaq_realtime_device_call_configure")) {
    auto *configure = reinterpret_cast<ServiceConfigureFn>(symbol);
    if (int status = configure(argc, argv)) {
      fail("service configure hook failed with status " +
           std::to_string(status));
      return nullptr;
    }
  }

  try {
    configureLocalRuntime(options);
  } catch (const std::exception &exception) {
    fail("local device-call runtime configuration failed: " +
         std::string(exception.what()));
    return nullptr;
  }

  const std::string servicePostfix =
      deviceCallServicePostfixFromLibraryPath(options.serviceLibrary);
  if (!resolveDeviceCallServiceFactory(service->handle, servicePostfix)) {
    fail("service library does not export " +
         deviceCallServiceFactorySymbol(servicePostfix) + " or " +
         std::string(DeviceCallServiceFactorySymbol));
    return nullptr;
  }

  try {
    initializeDeviceCallService(service->handle, servicePostfix);
  } catch (const std::exception &exception) {
    fail("service factory initialization failed: " +
         std::string(exception.what()));
    return nullptr;
  }

  service->serviceInitialized = true;
  return service;
}

std::optional<ListenSocket> makeListenSocket(const Options &options) {
  TcpSocket socket;
  auto boundPort = socket.listen(options.host, options.port, 128);
  if (!boundPort)
    return std::nullopt;
  return ListenSocket{std::move(socket), *boundPort};
}

std::vector<std::uint8_t>
makeTcpErrorResponseFromAbiStatus(const cudaq::realtime::RPCHeader &request,
                                  std::int32_t status) {
  std::vector<std::uint8_t> response(sizeof(cudaq::realtime::RPCResponse), 0);
  auto *rpcResponse =
      reinterpret_cast<cudaq::realtime::RPCResponse *>(response.data());
  rpcResponse->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  rpcResponse->status = status;
  rpcResponse->result_len = 0;
  rpcResponse->request_id = request.request_id;
  rpcResponse->ptp_timestamp = request.ptp_timestamp;
  return response;
}

std::vector<std::uint8_t>
makeTcpErrorResponse(const cudaq::realtime::RPCHeader &request,
                     DeviceCallStatus status) {
  return makeTcpErrorResponseFromAbiStatus(request, toAbiStatus(status));
}

std::vector<std::uint8_t>
makeTcpSuccessResponse(const cudaq::realtime::RPCHeader &request,
                       const void *payload, std::uint64_t payloadBytes) {
  std::vector<std::uint8_t> response(sizeof(cudaq::realtime::RPCResponse) +
                                     payloadBytes);
  auto *rpcResponse =
      reinterpret_cast<cudaq::realtime::RPCResponse *>(response.data());
  rpcResponse->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  rpcResponse->status = toAbiStatus(DeviceCallStatus::Success);
  rpcResponse->result_len = static_cast<std::uint32_t>(payloadBytes);
  rpcResponse->request_id = request.request_id;
  rpcResponse->ptp_timestamp = request.ptp_timestamp;
  if (payloadBytes > 0)
    std::memcpy(response.data() + sizeof(cudaq::realtime::RPCResponse), payload,
                payloadBytes);
  return response;
}

std::vector<std::uint8_t>
dispatchRequest(const Options &options, const std::vector<std::uint8_t> &req) {
  const auto *header =
      reinterpret_cast<const cudaq::realtime::RPCHeader *>(req.data());
  if (header->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      header->arg_len != req.size() - CUDAQ_RPC_HEADER_SIZE)
    return makeTcpErrorResponse(*header, DeviceCallStatus::InvalidArgument);

  const std::uint8_t *payload = req.data() + CUDAQ_RPC_HEADER_SIZE;
  const std::uint32_t responseCapacity =
      options.slotSize - sizeof(cudaq::realtime::RPCResponse);
  RealtimeFrameLease frame;
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  std::uint64_t responseLen = 0;
  std::int32_t status = __cudaq_device_call_acquire_realtime_frame(
      0, header->function_id, header->arg_len, responseCapacity, &frame.handle,
      &requestPayload, &responsePayload);

  if (!isSuccessStatus(status))
    return makeTcpErrorResponseFromAbiStatus(*header, status);
  if ((header->arg_len > 0 && !requestPayload) ||
      (responseCapacity > 0 && !responsePayload))
    return makeTcpErrorResponse(*header, DeviceCallStatus::InvalidArgument);

  if (header->arg_len > 0)
    std::memcpy(requestPayload, payload, header->arg_len);

  status =
      __cudaq_device_call_dispatch_realtime_frame(frame.handle, &responseLen);

  if (!isSuccessStatus(status))
    return makeTcpErrorResponseFromAbiStatus(*header, status);
  if (responseLen > responseCapacity)
    return makeTcpErrorResponse(*header, DeviceCallStatus::ResponseTooLarge);

  return makeTcpSuccessResponse(*header, responsePayload, responseLen);
}

void serveClient(TcpSocket &client, const Options &options) {
  client.setTimeout(options.timeoutMs);
  auto shouldStop = [] { return shutdownRequested().load(); };
  while (!shutdownRequested().load()) {
    std::vector<std::uint8_t> request;
    if (!readLengthPrefixedFrame(client, request, options.slotSize,
                                 shouldStop) ||
        request.size() < CUDAQ_RPC_HEADER_SIZE)
      break;
    std::vector<std::uint8_t> response = dispatchRequest(options, request);
    if (!writeLengthPrefixedFrame(client, response.data(), response.size(),
                                  shouldStop))
      break;
  }
}

int serveTcp(const Options &options) {
  auto listenSocket = makeListenSocket(options);
  if (!listenSocket)
    return fail("failed to listen on " + options.host + ":" +
                std::to_string(options.port) + ": " + std::strerror(errno));

  if (!writeTextFile(options.portFile,
                     std::to_string(listenSocket->boundPort))) {
    listenSocket->socket.close();
    return fail("failed to write port file '" + options.portFile + "'");
  }
  if (!writeTextFile(options.readyFile, "ready\n")) {
    listenSocket->socket.close();
    return fail("failed to write ready file '" + options.readyFile + "'");
  }

  while (!shutdownRequested().load()) {
    const SocketWaitStatus status =
        listenSocket->socket.waitForReadable(std::chrono::milliseconds(100));
    if (status == SocketWaitStatus::Timeout)
      continue;
    if (status == SocketWaitStatus::Error)
      break;

    TcpSocket client = listenSocket->socket.accept();
    if (!client.isValid())
      break;
    serveClient(client, options);
  }

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parseOptions(argc, argv, options))
    return fail("expected --service, --service-lib, or "
                "CUDAQ_DEVICE_CALL_SERVICE_LIB plus valid TCP options");

  installSignalHandlers();

  auto service = loadAndInitializeService(argc, argv, options);
  if (!service)
    return 1;

  return serveTcp(options);
}
