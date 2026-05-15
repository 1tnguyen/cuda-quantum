/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/LocalChannel.h"
#include "cudaq_internal/device_call/RpcFrame.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

using namespace cudaq_internal::device_call;

std::uint64_t requiredSlotSize(std::uint64_t requestBytes,
                               std::uint64_t responseCapacity) {
  constexpr std::uint64_t maxFrameBytes =
      std::numeric_limits<std::uint32_t>::max();
  if (requestBytes > maxFrameBytes - CUDAQ_RPC_HEADER_SIZE ||
      responseCapacity > maxFrameBytes - sizeof(cudaq::realtime::RPCResponse))
    throw std::invalid_argument("device_call frame length exceeds 32 bits");

  return std::max<std::uint64_t>(CUDAQ_RPC_HEADER_SIZE + requestBytes,
                                 sizeof(cudaq::realtime::RPCResponse) +
                                     responseCapacity);
}

class GpuDispatchChannel : public SharedMemoryChannelBase {
public:
  ~GpuDispatchChannel() override { stop(); }

  void initialize(DeviceCallChannelCreateArgs &&args) override {
    initializeFunctionTable(args, true);
    launchFn = args.launchFn;
    synchronizeFn = args.synchronizeFn;
  }

  void acquireFrame(std::uint32_t functionId, std::uint64_t requestBytes,
                    std::uint64_t responseCapacity,
                    DeviceCallFrame &frame) override {
    std::lock_guard<std::mutex> lock(mutex);
    frame = {};
    const std::uint64_t requiredBytes =
        requiredSlotSize(requestBytes, responseCapacity);
    ensureStarted(requiredBytes);

    std::uint32_t slot = numSlots;
    for (std::uint32_t attempt = 0; attempt < numSlots; ++attempt) {
      const std::uint32_t candidate = nextSlot;
      nextSlot = (nextSlot + 1) % numSlots;
      if (frameStates[candidate].inUse)
        continue;
      prepareReusableSlot(candidate);
      slot = candidate;
      break;
    }
    if (slot == numSlots)
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            "no reusable device_call ring slot available");

    const std::uint32_t requestId = nextRequestId++;
    std::uint8_t *rxSlot = rxHostSlot(slot);
    std::uint8_t *txSlot = txHostSlot(slot);
    std::memset(txSlot, 0, sizeof(cudaq::realtime::RPCResponse));
    initializeRequestHeader(rxSlot, functionId, requestBytes, requestId);

    auto &state = frameStates[slot];
    state.slot = slot;
    state.requestId = requestId;
    state.inUse = true;
    state.submittedNoResponse = false;
    state.needsFireAndForgetCleanup = false;

    frame.functionId = functionId;
    frame.request.data = requestPayload(rxSlot);
    frame.request.capacity = requestBytes;
    frame.response.data = responsePayload(txSlot);
    frame.response.capacity = responseCapacity;
    frame.channelPrivate = &state;
  }

  std::uint64_t dispatchFrame(DeviceCallFrame &frame) override {
    std::lock_guard<std::mutex> lock(mutex);
    if (!frame.channelPrivate)
      throw std::invalid_argument("invalid device_call frame");
    auto *state = static_cast<FrameState *>(frame.channelPrivate);
    if (!state->inUse)
      throw std::invalid_argument("device_call frame is not active");

    const std::uint32_t slot = state->slot;
    storeFlag(&rxRing.hostFlags()[slot], rxDeviceSlotAddress(slot));

    if (frame.response.capacity == 0) {
      state->submittedNoResponse = true;
      return 0;
    }

    try {
      waitForResponse(slot,
                      [](std::uint64_t flag) -> bool { return flag != 0; });
    } catch (const DeviceCallError &err) {
      if (err.status() == DeviceCallStatus::Timeout) {
        frame.channelPrivate = nullptr;
        stopNoLock();
      }
      throw;
    }

    std::uint8_t *txSlot = txHostSlot(slot);
    return validateResponseFrame(txSlot, state->requestId,
                                 frame.response.capacity, slotSize);
  }

  void releaseFrame(DeviceCallFrame &frame) noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    if (!frame.channelPrivate)
      return;
    auto *state = static_cast<FrameState *>(frame.channelPrivate);
    if (state->inUse) {
      if (state->submittedNoResponse) {
        state->submittedNoResponse = false;
        state->needsFireAndForgetCleanup = true;
      } else {
        clearSlot(state->slot);
      }
      state->inUse = false;
    }
    frame = {};
  }

  void stop() noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    stopNoLock();
  }

private:
  struct FrameState {
    std::uint32_t slot = 0;
    std::uint32_t requestId = 0;
    bool inUse = false;
    bool submittedNoResponse = false;
    bool needsFireAndForgetCleanup = false;
  };

  bool hasActiveFrame() const {
    return std::any_of(frameStates.begin(), frameStates.end(),
                       [](const auto &state) {
                         return state.inUse || state.needsFireAndForgetCleanup;
                       });
  }

  void prepareReusableSlot(std::uint32_t slot) {
    auto &state = frameStates[slot];
    if (state.needsFireAndForgetCleanup) {
      waitForFireAndForgetCompletion(slot);
      clearSlot(slot);
      state.needsFireAndForgetCleanup = false;
    }
    waitForReusableSlot(slot);
  }

  void ensureStarted(std::uint64_t requiredSlotSize) {
    if (started && requiredSlotSize <= slotSize)
      return;
    if (started && hasActiveFrame())
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            "cannot resize device_call ring with active "
                            "frames");

    stopNoLock();

    configureRingStorage(requiredSlotSize);

    void *shutdown = nullptr;
    if (cudaMalloc(&shutdown, sizeof(int)) != cudaSuccess)
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to allocate device_call shutdown flag");
    shutdownFlag = static_cast<volatile int *>(shutdown);
    if (cudaMemset(const_cast<int *>(shutdownFlag), 0, sizeof(int)) !=
        cudaSuccess)
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to initialize device_call shutdown flag");

    if (cudaMalloc(&stats, sizeof(std::uint64_t)) != cudaSuccess)
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to allocate device_call stats");
    if (cudaMemset(stats, 0, sizeof(std::uint64_t)) != cudaSuccess)
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to initialize device_call stats");

    if (!synchronizeFn)
      if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
          cudaSuccess)
        throw DeviceCallError(DeviceCallStatus::CudaError,
                              "failed to create device_call dispatch stream");

    if (!launchFn)
      throw std::invalid_argument(
          "device_call dispatch launch hook is missing");

    launchFn(rxRing.deviceFlags(), txRing.deviceFlags(), rxRing.deviceData(),
             txRing.deviceData(), slotSize, slotSize, functionTable,
             functionCount, shutdownFlag, stats, numSlots, 1, 64, stream);

    if (!synchronizeFn && cudaGetLastError() != cudaSuccess)
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to launch device_call dispatcher");

    started = true;
    nextSlot = 0;
    frameStates.assign(numSlots, FrameState{});
  }

  void stopNoLock() noexcept {
    if (started && shutdownFlag) {
      int shutdown = 1;
      cudaMemcpy(const_cast<int *>(shutdownFlag), &shutdown, sizeof(int),
                 cudaMemcpyHostToDevice);
      if (synchronizeFn)
        (void)synchronizeFn();
      else if (stream)
        cudaStreamSynchronize(stream);
    }

    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    if (shutdownFlag) {
      cudaFree(const_cast<int *>(shutdownFlag));
      shutdownFlag = nullptr;
    }
    if (stats) {
      cudaFree(stats);
      stats = nullptr;
    }

    resetRingStorage();
    frameStates.clear();
    started = false;
  }

  cudaq_dispatch_launch_fn_t launchFn = nullptr;
  cudaq_device_call_dispatch_synchronize_fn_t synchronizeFn = nullptr;
  std::uint32_t nextRequestId = 1;
  bool started = false;
  std::vector<FrameState> frameStates;
  volatile int *shutdownFlag = nullptr;
  std::uint64_t *stats = nullptr;
  cudaStream_t stream = nullptr;
  mutable std::mutex mutex;
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, GpuDispatchChannel, device_dispatch)

} // namespace
