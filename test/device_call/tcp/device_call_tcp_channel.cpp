/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "TcpArgParsing.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/RpcFrame.h"
#include "SocketUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using namespace cudaq_internal::device_call;

std::uint64_t checkedFrameLength(std::uint64_t headerBytes,
                                 std::uint64_t payloadBytes,
                                 const char *message) {
  constexpr std::uint64_t maxFrameBytes =
      std::numeric_limits<std::uint32_t>::max();
  if (payloadBytes > maxFrameBytes - headerBytes)
    throw std::invalid_argument(message);
  return headerBytes + payloadBytes;
}

enum class TcpLaunchMode { External, Auto };

struct TcpEndpoint {
  std::string host = "127.0.0.1";
  std::uint16_t port = 0;
};

struct TcpChannelOptions {
  TcpLaunchMode launch = TcpLaunchMode::External;
  TcpEndpoint endpoint;
  std::string server;
  std::string serviceLibrary;
  int gpu = -1;
};

struct AutoLaunchedServer {
  ~AutoLaunchedServer() { stop(); }
  void stop();

  pid_t pid = -1;
  std::string workDir;
  std::string portFile;
  std::string readyFile;
};

struct TcpServerLaunchConfig {
  TcpEndpoint endpoint;
  std::string server;
  std::string serviceLibrary;
  int gpu = -1;
  DeviceCallChannelConfig channelConfig;
};

struct TcpServerLaunchResult {
  TcpEndpoint endpoint;
  std::unique_ptr<AutoLaunchedServer> server;
};

std::optional<TcpEndpoint> readBoundPort(const std::string &path,
                                         TcpEndpoint endpoint) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file)
    return std::nullopt;
  llvm::StringRef text = (*file)->getBuffer().trim();
  std::uint16_t parsed = 0;
  if (text.getAsInteger(10, parsed) || parsed == 0)
    return std::nullopt;
  endpoint.port = parsed;
  return endpoint;
}

bool parseEndpoint(const char *value, TcpEndpoint &endpoint,
                   bool allowZeroPort) {
  if (!value || !*value)
    return false;
  llvm::StringRef text(value);
  auto [host, portText] = text.rsplit(':');
  if (host.empty() || portText.empty() || host == text)
    return false;

  std::uint16_t parsed = 0;
  if (portText.getAsInteger(10, parsed))
    return false;
  if (!allowZeroPort && parsed == 0)
    return false;

  endpoint.host = host.str();
  endpoint.port = parsed;
  return true;
}

bool parseLaunchMode(TcpChannelOptions &options, const char *value) {
  if (!value)
    return false;
  if (std::strcmp(value, "external") == 0) {
    options.launch = TcpLaunchMode::External;
    return true;
  }
  if (std::strcmp(value, "auto") == 0) {
    options.launch = TcpLaunchMode::Auto;
    return true;
  }
  return false;
}

bool parseEndpointOption(TcpChannelOptions &options, const char *value) {
  return parseEndpoint(value, options.endpoint, true);
}

bool parseHostOption(TcpChannelOptions &options, const char *value) {
  if (!value)
    return false;
  options.endpoint.host = value;
  return true;
}

bool parsePortOption(TcpChannelOptions &options, const char *value) {
  std::uint64_t parsed = 0;
  if (!parseUInt(value, std::numeric_limits<std::uint16_t>::max(), parsed))
    return false;
  options.endpoint.port = static_cast<std::uint16_t>(parsed);
  return true;
}

bool parseGpuOption(TcpChannelOptions &options, const char *value) {
  std::uint64_t parsed = 0;
  if (!parseUInt(value,
                 static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
                 parsed))
    return false;
  options.gpu = static_cast<int>(parsed);
  return true;
}

template <typename ParseFn>
bool applyEnvOption(TcpChannelOptions &options, const char *name,
                    ParseFn parse) {
  if (const char *value = std::getenv(name))
    return parse(options, value);
  return true;
}

bool parseTcpChannelOptions(const std::vector<std::string> &arguments,
                            TcpChannelOptions &options) {
  if (!applyEnvOption(options, "CUDAQ_DEVICE_CALL_TCP_ENDPOINT",
                      parseEndpointOption) ||
      !applyEnvOption(options, "CUDAQ_DEVICE_CALL_TCP_HOST", parseHostOption) ||
      !applyEnvOption(options, "CUDAQ_DEVICE_CALL_TCP_PORT", parsePortOption) ||
      !applyEnvOption(options, "CUDAQ_DEVICE_CALL_TCP_LAUNCH",
                      parseLaunchMode) ||
      !applyEnvOption(
          options, "CUDAQ_DEVICE_CALL_TCP_SERVER",
          parseStringOption<TcpChannelOptions, &TcpChannelOptions::server>) ||
      !applyEnvOption(options, "CUDAQ_DEVICE_CALL_TCP_SERVICE_LIB",
                      parseStringOption<TcpChannelOptions,
                                        &TcpChannelOptions::serviceLibrary>) ||
      !applyEnvOption(options, "CUDAQ_DEVICE_CALL_TCP_GPU", parseGpuOption))
    return false;

  std::vector<char *> argv;
  argv.reserve(arguments.size() + 1);
  for (const std::string &arg : arguments)
    argv.push_back(const_cast<char *>(arg.c_str()));
  argv.push_back(nullptr);

  static constexpr CliOption<TcpChannelOptions> optionsSpec[] = {
      {"--cudaq-device-call-tcp-endpoint", parseEndpointOption},
      {"--cudaq-device-call-tcp-host", parseHostOption},
      {"--cudaq-device-call-tcp-port", parsePortOption},
      {"--cudaq-device-call-tcp-launch", parseLaunchMode},
      {"--cudaq-device-call-tcp-server",
       parseStringOption<TcpChannelOptions, &TcpChannelOptions::server>},
      {"--cudaq-device-call-tcp-service",
       parseStringOption<TcpChannelOptions,
                         &TcpChannelOptions::serviceLibrary>},
      {"--cudaq-device-call-tcp-service-lib",
       parseStringOption<TcpChannelOptions,
                         &TcpChannelOptions::serviceLibrary>},
      {"--cudaq-device-call-tcp-gpu", parseGpuOption},
  };

  return parseCliOptions(static_cast<int>(arguments.size()), argv.data(),
                         optionsSpec, options);
}

TcpServerLaunchResult launchTcpServer(const TcpServerLaunchConfig &config);

class TcpDeviceCallChannel : public DeviceCallChannel {
public:
  ~TcpDeviceCallChannel() override { stop(); }

  void initialize(DeviceCallChannelCreateArgs &&args) override {
    TcpChannelOptions options;
    if (!parseTcpChannelOptions(args.arguments, options))
      throw std::invalid_argument("invalid TCP device_call channel options");

    if (options.launch == TcpLaunchMode::Auto) {
      if (options.server.empty())
        throw std::invalid_argument(
            "TCP device_call auto-launch requires a server executable");
      TcpServerLaunchConfig launchConfig;
      launchConfig.endpoint = std::move(options.endpoint);
      launchConfig.server = std::move(options.server);
      launchConfig.serviceLibrary = std::move(options.serviceLibrary);
      launchConfig.gpu = options.gpu;
      launchConfig.channelConfig = args.channelConfig;
      auto launchResult = launchTcpServer(launchConfig);
      endpoint = std::move(launchResult.endpoint);
      server = std::move(launchResult.server);
    } else {
      if (options.endpoint.port == 0)
        throw std::invalid_argument(
            "TCP device_call external launch requires a nonzero port");
      endpoint = std::move(options.endpoint);
    }

    timeoutMs = args.channelConfig.timeoutMs;
  }

  void acquireFrame(std::uint32_t functionId, std::uint64_t requestBytes,
                    std::uint64_t responseCapacity,
                    DeviceCallFrame &frame) override {
    std::lock_guard<std::mutex> lock(mutex);
    frame = {};
    const std::uint64_t requestFrameLen = checkedFrameLength(
        CUDAQ_RPC_HEADER_SIZE, requestBytes,
        "TCP device_call request frame length exceeds 32 bits");
    const std::uint64_t responseFrameLen = checkedFrameLength(
        sizeof(cudaq::realtime::RPCResponse), responseCapacity,
        "TCP device_call response frame length exceeds 32 bits");

    auto state = std::make_unique<TcpFrameState>();
    state->requestId = nextRequestId++;
    state->requestFrame.assign(static_cast<std::size_t>(requestFrameLen), 0);
    state->responseFrame.assign(static_cast<std::size_t>(responseFrameLen), 0);

    initializeRequestHeader(state->requestFrame.data(), functionId,
                            requestBytes, state->requestId);

    frame.functionId = functionId;
    frame.request.data = requestPayload(state->requestFrame.data());
    frame.request.capacity = requestBytes;
    frame.response.data = responsePayload(state->responseFrame.data());
    frame.response.capacity = responseCapacity;
    frame.channelPrivate = state.release();
  }

  std::uint64_t dispatchFrame(DeviceCallFrame &frame) override {
    std::lock_guard<std::mutex> lock(mutex);
    if (!frame.channelPrivate)
      throw std::invalid_argument("invalid TCP device_call frame");
    auto *state = static_cast<TcpFrameState *>(frame.channelPrivate);

    ensureConnected();

    if (!writeLengthPrefixedFrame(socket, state->requestFrame.data(),
                                  state->requestFrame.size())) {
      socket.close();
      throw std::runtime_error("failed to write TCP device_call request");
    }

    if (frame.response.capacity == 0) {
      socket.close();
      return 0;
    }

    std::uint32_t responseFrameLen = 0;
    if (!readLengthPrefixedFrame(socket, state->responseFrame.data(),
                                 state->responseFrame.size(),
                                 responseFrameLen)) {
      socket.close();
      throw std::system_error(std::make_error_code(std::errc::timed_out),
                              "timed out reading TCP device_call response");
    }

    if (responseFrameLen < sizeof(cudaq::realtime::RPCResponse)) {
      socket.close();
      throw std::runtime_error("truncated TCP device_call response");
    }

    return validateResponseFrame(state->responseFrame.data(), state->requestId,
                                 frame.response.capacity, responseFrameLen);
  }

  void releaseFrame(DeviceCallFrame &frame) noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    delete static_cast<TcpFrameState *>(frame.channelPrivate);
    frame = {};
  }

  void stop() noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    socket.close();
  }

private:
  struct TcpFrameState {
    std::uint32_t requestId = 0;
    std::vector<std::uint8_t> requestFrame;
    std::vector<std::uint8_t> responseFrame;
  };

  void ensureConnected() {
    if (socket.isValid())
      return;

    if (!socket.connect(endpoint.host, endpoint.port, timeoutMs)) {
      if (errno == EINVAL)
        throw std::invalid_argument("invalid TCP device_call endpoint");
      throw std::runtime_error("failed to connect TCP device_call endpoint");
    }
  }

  TcpEndpoint endpoint;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
  std::uint32_t nextRequestId = 1;
  TcpSocket socket;
  std::unique_ptr<AutoLaunchedServer> server;
  std::mutex mutex;
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, TcpDeviceCallChannel, tcp)

void AutoLaunchedServer::stop() {
  if (pid > 0) {
    ::kill(pid, SIGTERM);
    for (int i = 0; i < 100; ++i) {
      int status = 0;
      pid_t result = ::waitpid(pid, &status, WNOHANG);
      if (result == pid) {
        pid = -1;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (pid > 0) {
      ::kill(pid, SIGKILL);
      (void)::waitpid(pid, nullptr, 0);
      pid = -1;
    }
  }

  if (!portFile.empty())
    (void)llvm::sys::fs::remove(portFile);
  if (!readyFile.empty())
    (void)llvm::sys::fs::remove(readyFile);
  if (!workDir.empty())
    (void)llvm::sys::fs::remove_directories(workDir);
}

TcpServerLaunchResult launchTcpServer(const TcpServerLaunchConfig &config) {
  llvm::SmallString<128> workDir;
  if (llvm::sys::fs::createUniqueDirectory("cudaq-device-call", workDir))
    throw std::runtime_error("failed to create TCP device_call work directory");

  auto launched = std::make_unique<AutoLaunchedServer>();
  launched->workDir = workDir.str().str();
  llvm::SmallString<128> portFile(workDir);
  llvm::sys::path::append(portFile, "port");
  launched->portFile = portFile.str().str();
  llvm::SmallString<128> readyFile(workDir);
  llvm::sys::path::append(readyFile, "ready");
  launched->readyFile = readyFile.str().str();

  const std::string port = std::to_string(config.endpoint.port);
  const std::string slots = std::to_string(config.channelConfig.numSlots);
  const std::string slotSize = std::to_string(config.channelConfig.slotSize);
  const std::string timeout = std::to_string(config.channelConfig.timeoutMs);
  const std::string gpu = std::to_string(config.gpu);

  std::vector<std::string> args = {
      config.server,      "--host",       config.endpoint.host,
      "--port",           port,           "--port-file",
      launched->portFile, "--ready-file", launched->readyFile,
      "--num-slots",      slots,          "--slot-size",
      slotSize,           "--timeout-ms", timeout};
  if (!config.serviceLibrary.empty()) {
    args.push_back("--service");
    args.push_back(config.serviceLibrary);
  }
  if (config.gpu >= 0) {
    args.push_back("--gpu");
    args.push_back(gpu);
  }

  pid_t pid = ::fork();
  if (pid < 0)
    throw std::runtime_error("failed to fork TCP device_call server");

  if (pid == 0) {
    std::vector<char *> execArgs;
    execArgs.reserve(args.size() + 1);
    for (auto &arg : args)
      execArgs.push_back(const_cast<char *>(arg.c_str()));
    execArgs.push_back(nullptr);
    if (config.server.find('/') == std::string::npos)
      ::execvp(config.server.c_str(), execArgs.data());
    else
      ::execv(config.server.c_str(), execArgs.data());
    _exit(127);
  }

  launched->pid = pid;
  const auto deadline =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(config.channelConfig.timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    int status = 0;
    pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid) {
      launched->pid = -1;
      throw std::runtime_error("TCP device_call server exited during startup");
    }
    if (llvm::sys::fs::exists(launched->readyFile)) {
      auto endpoint = readBoundPort(launched->portFile, config.endpoint);
      if (!endpoint)
        throw std::runtime_error(
            "TCP device_call server did not report a bound port");
      return {std::move(*endpoint), std::move(launched)};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  throw std::system_error(std::make_error_code(std::errc::timed_out),
                          "timed out launching TCP device_call server");
}

} // namespace
