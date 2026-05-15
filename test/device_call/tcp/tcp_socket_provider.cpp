/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/ArgParsing.h"
#include "cudaq_internal/device_call/MappedRingBuffer.h"
#include "SocketUtils.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/rpc_wire_format.h"

#include <cuda_runtime.h>

#include "llvm/ADT/StringSwitch.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace cudaq_internal::device_call;

constexpr std::uint32_t DefaultNumSlots = 2;
constexpr std::uint32_t DefaultSlotSize = 256;
constexpr std::uint64_t DefaultTimeoutMs = 5000;

enum class SignalAddress { Device, Host };

struct TcpBridgeContext {
  std::string host = "127.0.0.1";
  std::uint16_t requestedPort = 0;
  std::uint16_t boundPort = 0;
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint32_t slotSize = DefaultSlotSize;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
  int gpu = -1;
  SignalAddress signalAddress = SignalAddress::Device;

  TcpSocket listenSocket;
  std::thread worker;
  std::atomic_bool stopRequested{false};

  MappedAllocation rxFlags;
  MappedAllocation txFlags;
  MappedAllocation rxData;
  MappedAllocation txData;
  cudaq_ringbuffer_t ringbuffer{};
  std::uint32_t nextSlot = 0;
};

bool parseSlotSizeOption(TcpBridgeContext &ctx, const char *value) {
  std::uint64_t parsed = 0;
  if (!parseUInt(value, std::numeric_limits<std::uint32_t>::max(), parsed) ||
      parsed < CUDAQ_RPC_HEADER_SIZE)
    return false;
  ctx.slotSize = static_cast<std::uint32_t>(parsed);
  return true;
}

bool parseSignalAddressOption(TcpBridgeContext &ctx, const char *value) {
  const auto signalAddress =
      llvm::StringSwitch<std::optional<SignalAddress>>(value)
          .Case("device", SignalAddress::Device)
          .Case("host", SignalAddress::Host)
          .Default(std::nullopt);
  if (!signalAddress)
    return false;
  ctx.signalAddress = *signalAddress;
  return true;
}

cudaq_status_t parseArgs(TcpBridgeContext &ctx, int argc, char **argv) {
  static constexpr CliOption<TcpBridgeContext> options[] = {
      {"--host", parseStringOption<TcpBridgeContext, &TcpBridgeContext::host>},
      {"--port", parseUIntOption<TcpBridgeContext, std::uint16_t,
                                 &TcpBridgeContext::requestedPort>},
      {"--num-slots", parseUIntOption<TcpBridgeContext, std::uint32_t,
                                      &TcpBridgeContext::numSlots, 1>},
      {"--slot-size", parseSlotSizeOption},
      {"--timeout-ms", parseUIntOption<TcpBridgeContext, std::uint64_t,
                                       &TcpBridgeContext::timeoutMs, 1>},
      {"--gpu",
       parseNonNegativeIntOption<TcpBridgeContext, &TcpBridgeContext::gpu>},
      {"--signal-address", parseSignalAddressOption},
  };

  return parseCliOptions(argc, argv, options, ctx) ? CUDAQ_OK
                                                   : CUDAQ_ERR_INVALID_ARG;
}

bool allocateRingBuffer(TcpBridgeContext &ctx) {
  const std::size_t flagsBytes = ctx.numSlots * sizeof(std::uint64_t);
  const std::size_t dataBytes =
      static_cast<std::size_t>(ctx.numSlots) * ctx.slotSize;

  if (!ctx.rxFlags.allocate(flagsBytes) || !ctx.txFlags.allocate(flagsBytes) ||
      !ctx.rxData.allocate(dataBytes) || !ctx.txData.allocate(dataBytes))
    return false;

  ctx.ringbuffer.rx_flags =
      static_cast<volatile std::uint64_t *>(ctx.rxFlags.device);
  ctx.ringbuffer.tx_flags =
      static_cast<volatile std::uint64_t *>(ctx.txFlags.device);
  ctx.ringbuffer.rx_data = static_cast<std::uint8_t *>(ctx.rxData.device);
  ctx.ringbuffer.tx_data = static_cast<std::uint8_t *>(ctx.txData.device);
  ctx.ringbuffer.rx_stride_sz = ctx.slotSize;
  ctx.ringbuffer.tx_stride_sz = ctx.slotSize;
  ctx.ringbuffer.rx_flags_host =
      static_cast<volatile std::uint64_t *>(ctx.rxFlags.host);
  ctx.ringbuffer.tx_flags_host =
      static_cast<volatile std::uint64_t *>(ctx.txFlags.host);
  ctx.ringbuffer.rx_data_host = static_cast<std::uint8_t *>(ctx.rxData.host);
  ctx.ringbuffer.tx_data_host = static_cast<std::uint8_t *>(ctx.txData.host);
  return true;
}

bool bindListenSocket(TcpBridgeContext &ctx) {
  auto boundPort = ctx.listenSocket.listen(ctx.host, ctx.requestedPort, 8);
  if (!boundPort)
    return false;
  ctx.boundPort = *boundPort;
  return true;
}

bool shouldStop(const TcpBridgeContext &ctx) {
  return ctx.stopRequested.load(std::memory_order_acquire);
}

bool waitForReusableSlot(TcpBridgeContext &ctx, std::uint32_t slot) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(ctx.timeoutMs);
  while (!shouldStop(ctx)) {
    if (loadFlag(&ctx.ringbuffer.rx_flags_host[slot]) == 0 &&
        loadFlag(&ctx.ringbuffer.tx_flags_host[slot]) == 0)
      return true;
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return false;
}

bool waitForResponse(TcpBridgeContext &ctx, std::uint32_t slot) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(ctx.timeoutMs);
  while (!shouldStop(ctx)) {
    const std::uint64_t txFlag = loadFlag(&ctx.ringbuffer.tx_flags_host[slot]);
    if (txFlag != 0 && txFlag != CUDAQ_TX_FLAG_IN_FLIGHT)
      return (txFlag >> 48) != CUDAQ_TX_FLAG_ERROR_TAG;
    if (std::chrono::steady_clock::now() > deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  return false;
}

void signalSlot(TcpBridgeContext &ctx, std::uint32_t slot) {
  const std::uint64_t slotAddress =
      ctx.signalAddress == SignalAddress::Device
          ? reinterpret_cast<std::uint64_t>(ctx.ringbuffer.rx_data +
                                            slot * ctx.ringbuffer.rx_stride_sz)
          : reinterpret_cast<std::uint64_t>(ctx.ringbuffer.rx_data_host +
                                            slot * ctx.ringbuffer.rx_stride_sz);
  __sync_synchronize();
  storeFlag(&ctx.ringbuffer.rx_flags_host[slot], slotAddress);
}

bool dispatchFrame(TcpBridgeContext &ctx, TcpSocket &client,
                   const std::vector<std::uint8_t> &frame) {
  auto *header =
      reinterpret_cast<const cudaq::realtime::RPCHeader *>(frame.data());
  if (header->magic != cudaq::realtime::RPC_MAGIC_REQUEST)
    return false;
  if (header->arg_len != frame.size() - CUDAQ_RPC_HEADER_SIZE)
    return false;

  const std::uint32_t slot = ctx.nextSlot;
  if (!waitForReusableSlot(ctx, slot))
    return false;

  std::uint8_t *rxSlot =
      ctx.ringbuffer.rx_data_host + slot * ctx.ringbuffer.rx_stride_sz;
  std::memset(rxSlot, 0, ctx.ringbuffer.rx_stride_sz);
  std::memcpy(rxSlot, frame.data(), frame.size());
  signalSlot(ctx, slot);

  if (!waitForResponse(ctx, slot))
    return false;

  __sync_synchronize();
  std::uint8_t *txSlot =
      ctx.ringbuffer.tx_data_host + slot * ctx.ringbuffer.tx_stride_sz;
  auto *response = reinterpret_cast<cudaq::realtime::RPCResponse *>(txSlot);
  const std::uint64_t responseBytes =
      sizeof(cudaq::realtime::RPCResponse) + response->result_len;
  if (responseBytes > ctx.ringbuffer.tx_stride_sz)
    return false;

  auto stop = [&ctx] { return shouldStop(ctx); };
  const bool sent =
      writeLengthPrefixedFrame(client, txSlot, responseBytes, stop);
  storeFlag(&ctx.ringbuffer.tx_flags_host[slot], 0);
  ctx.nextSlot = (slot + 1) % ctx.numSlots;
  return sent;
}

void handleClient(TcpBridgeContext &ctx, TcpSocket &client) {
  std::vector<std::uint8_t> frame;
  auto stop = [&ctx] { return shouldStop(ctx); };
  while (!shouldStop(ctx)) {
    if (!readLengthPrefixedFramePolling(client, frame, ctx.slotSize, stop) ||
        frame.size() < CUDAQ_RPC_HEADER_SIZE)
      return;
    if (!dispatchFrame(ctx, client, frame))
      return;
  }
}

void runSocketPump(TcpBridgeContext *ctx) {
  while (!shouldStop(*ctx)) {
    const SocketWaitStatus status =
        ctx->listenSocket.waitForReadable(std::chrono::milliseconds(100));
    if (status == SocketWaitStatus::Timeout)
      continue;
    if (status == SocketWaitStatus::Error)
      return;

    TcpSocket client = ctx->listenSocket.accept();
    if (!client.isValid())
      return;

    handleClient(*ctx, client);
  }
}

void releaseResources(TcpBridgeContext &ctx) {
  ctx.rxFlags.reset();
  ctx.txFlags.reset();
  ctx.rxData.reset();
  ctx.txData.reset();
  ctx.ringbuffer = {};
}

cudaq_status_t tcpBridgeCreate(cudaq_realtime_bridge_handle_t *outHandle,
                               int argc, char **argv) {
  if (!outHandle)
    return CUDAQ_ERR_INVALID_ARG;

  auto *ctx = new (std::nothrow) TcpBridgeContext();
  if (!ctx)
    return CUDAQ_ERR_INTERNAL;

  const cudaq_status_t status = parseArgs(*ctx, argc, argv);
  if (status != CUDAQ_OK) {
    delete ctx;
    return status;
  }

  *outHandle = reinterpret_cast<cudaq_realtime_bridge_handle_t>(ctx);
  return CUDAQ_OK;
}

cudaq_status_t tcpBridgeDestroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<TcpBridgeContext *>(handle);
  ctx->stopRequested.store(true, std::memory_order_release);
  ctx->listenSocket.close();
  if (ctx->worker.joinable())
    ctx->worker.join();
  releaseResources(*ctx);
  delete ctx;
  return CUDAQ_OK;
}

cudaq_status_t
tcpBridgeGetTransportContext(cudaq_realtime_bridge_handle_t handle,
                             cudaq_realtime_transport_context_t contextType,
                             void *outContext) {
  if (!handle || !outContext)
    return CUDAQ_ERR_INVALID_ARG;
  if (contextType != RING_BUFFER)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<TcpBridgeContext *>(handle);
  *reinterpret_cast<cudaq_ringbuffer_t *>(outContext) = ctx->ringbuffer;
  return CUDAQ_OK;
}

cudaq_status_t tcpBridgeConnect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<TcpBridgeContext *>(handle);
  if (ctx->gpu >= 0 && cudaSetDevice(ctx->gpu) != cudaSuccess)
    return CUDAQ_ERR_CUDA;
  if (!allocateRingBuffer(*ctx))
    return CUDAQ_ERR_CUDA;
  if (!bindListenSocket(*ctx)) {
    releaseResources(*ctx);
    return CUDAQ_ERR_INTERNAL;
  }
  return CUDAQ_OK;
}

cudaq_status_t tcpBridgeLaunch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<TcpBridgeContext *>(handle);
  if (!ctx->listenSocket.isValid())
    return CUDAQ_ERR_INVALID_ARG;
  if (ctx->worker.joinable())
    return CUDAQ_OK;
  ctx->stopRequested.store(false, std::memory_order_release);
  ctx->worker = std::thread(runSocketPump, ctx);
  return CUDAQ_OK;
}

cudaq_status_t tcpBridgeDisconnect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<TcpBridgeContext *>(handle);
  ctx->stopRequested.store(true, std::memory_order_release);
  ctx->listenSocket.close();
  if (ctx->worker.joinable())
    ctx->worker.join();
  releaseResources(*ctx);
  return CUDAQ_OK;
}

} // namespace

extern "C" cudaq_realtime_bridge_interface_t *
cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t tcpBridgeInterface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      tcpBridgeCreate,
      tcpBridgeDestroy,
      tcpBridgeGetTransportContext,
      tcpBridgeConnect,
      tcpBridgeLaunch,
      tcpBridgeDisconnect};
  return &tcpBridgeInterface;
}

extern "C" cudaq_status_t cudaq_realtime_tcp_transport_get_bound_port(
    cudaq_realtime_bridge_handle_t handle, std::uint16_t *outPort) {
  if (!handle || !outPort)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<TcpBridgeContext *>(handle);
  if (ctx->boundPort == 0)
    return CUDAQ_ERR_INVALID_ARG;
  *outPort = ctx->boundPort;
  return CUDAQ_OK;
}
