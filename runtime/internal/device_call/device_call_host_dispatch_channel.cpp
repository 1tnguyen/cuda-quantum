/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
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

cudaq_ringbuffer_t makeCudaqRingBuffer(const MappedRingBuffer &rx,
                                       const MappedRingBuffer &tx,
                                       std::uint64_t slotSize) {
  cudaq_ringbuffer_t ringbuffer{};
  ringbuffer.rx_flags = rx.deviceFlags();
  ringbuffer.tx_flags = tx.deviceFlags();
  ringbuffer.rx_data = rx.deviceData();
  ringbuffer.tx_data = tx.deviceData();
  ringbuffer.rx_stride_sz = slotSize;
  ringbuffer.tx_stride_sz = slotSize;
  ringbuffer.rx_flags_host = rx.hostFlags();
  ringbuffer.tx_flags_host = tx.hostFlags();
  ringbuffer.rx_data_host = rx.hostData();
  ringbuffer.tx_data_host = tx.hostData();
  return ringbuffer;
}

DeviceCallStatus statusFromCudaqStatus(cudaq_status_t status) {
  switch (status) {
  case CUDAQ_OK:
    return DeviceCallStatus::Success;
  case CUDAQ_ERR_INVALID_ARG:
    return DeviceCallStatus::InvalidArgument;
  case CUDAQ_ERR_CUDA:
    return DeviceCallStatus::CudaError;
  case CUDAQ_ERR_INTERNAL:
  default:
    return DeviceCallStatus::RemoteError;
  }
}

void checkDispatcherStatus(cudaq_status_t status, const char *message) {
  if (status == CUDAQ_OK)
    return;
  throw DeviceCallError(statusFromCudaqStatus(status), message);
}

class HostDispatchChannel : public SharedMemoryChannelBase {
public:
  ~HostDispatchChannel() override { stop(); }

  void initialize(DeviceCallChannelCreateArgs &&args) override {
    initializeFunctionTable(args, false);
    mailbox = args.mailbox;
  }

  void acquireFrame(std::uint32_t functionId, std::uint64_t requestBytes,
                    std::uint64_t responseCapacity,
                    DeviceCallFrame &frame) override {
    std::lock_guard<std::mutex> lock(mutex);
    frame = {};
    const cudaq_function_entry_t *entry = lookup(functionId);
    if (!entry || entry->dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH ||
        !entry->handler.graph_exec)
      throw std::invalid_argument(
          "host device_call dispatch requires a graph-launch table entry");

    const std::uint64_t requiredBytes =
        requiredSlotSize(requestBytes, responseCapacity);
    ensureGraphDispatcherStarted(requiredBytes);

    std::uint32_t slot = numSlots;
    for (std::uint32_t attempt = 0; attempt < numSlots; ++attempt) {
      const std::uint32_t candidate = nextSlot;
      nextSlot = (nextSlot + 1) % numSlots;
      if (frameStates[candidate].inUse)
        continue;
      waitForReusableSlot(candidate);
      slot = candidate;
      break;
    }
    if (slot == numSlots)
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            "no reusable host device_call ring slot available");

    const std::uint32_t requestId = nextRequestId++;
    std::uint8_t *rxSlot = rxHostSlot(slot);
    std::uint8_t *txSlot = txHostSlot(slot);
    std::memset(rxSlot, 0, slotSize);
    std::memset(txSlot, 0, slotSize);
    initializeRequestHeader(rxSlot, functionId, requestBytes, requestId);

    auto &state = frameStates[slot];
    state.slot = slot;
    state.requestId = requestId;
    state.inUse = true;

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
    cudaq_host_ringbuffer_signal_slot(&ringbuffer, slot);

    try {
      waitForResponse(slot, [this, slot](std::uint64_t) -> bool {
        int cudaError = 0;
        switch (
            cudaq_host_ringbuffer_poll_tx_flag(&ringbuffer, slot, &cudaError)) {
        case CUDAQ_TX_EMPTY:
        case CUDAQ_TX_IN_FLIGHT:
          return false;
        case CUDAQ_TX_READY:
          return true;
        case CUDAQ_TX_ERROR:
          throw DeviceCallError(DeviceCallStatus::CudaError,
                                "host device_call graph launch failed");
        }
        throw DeviceCallError(DeviceCallStatus::RemoteError,
                              "host device_call dispatcher returned an "
                              "unknown TX status");
      });
    } catch (const DeviceCallError &err) {
      frame.channelPrivate = nullptr;
      state->inUse = false;
      if (err.status() == DeviceCallStatus::Timeout ||
          err.status() == DeviceCallStatus::CudaError)
        stopGraphDispatcher();
      throw;
    }

    return validateResponseFrame(txHostSlot(slot), state->requestId,
                                 frame.response.capacity, slotSize);
  }

  void releaseFrame(DeviceCallFrame &frame) noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    if (!frame.channelPrivate)
      return;
    auto *state = static_cast<FrameState *>(frame.channelPrivate);
    if (state->inUse) {
      clearSlot(state->slot);
      state->inUse = false;
    }
    frame = {};
  }

  void stop() noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    stopGraphDispatcher();
  }

private:
  struct FrameState {
    std::uint32_t slot = 0;
    std::uint32_t requestId = 0;
    bool inUse = false;
  };

  bool hasActiveFrame() const {
    return std::any_of(frameStates.begin(), frameStates.end(),
                       [](const auto &state) { return state.inUse; });
  }

  void ensureGraphDispatcherStarted(std::uint64_t requiredSlotSize) {
    if (graphDispatcherStarted && requiredSlotSize <= slotSize)
      return;
    if (graphDispatcherStarted && hasActiveFrame())
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            "cannot resize host device_call ring with active "
                            "frames");

    stopGraphDispatcher();
    configureRingStorage(requiredSlotSize);
    ringbuffer = makeCudaqRingBuffer(rxRing, txRing, slotSize);

    cudaq_dispatcher_config_t dispatchConfig{};
    dispatchConfig.device_id = deviceId;
    dispatchConfig.num_slots = numSlots;
    dispatchConfig.slot_size = static_cast<std::uint32_t>(slotSize);
    dispatchConfig.kernel_type = CUDAQ_KERNEL_REGULAR;
    dispatchConfig.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    dispatchConfig.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;

    checkDispatcherStatus(cudaq_dispatch_manager_create(&dispatchManager),
                          "failed to create host device_call dispatch manager");
    checkDispatcherStatus(cudaq_dispatcher_create(dispatchManager,
                                                  &dispatchConfig,
                                                  &graphDispatcher),
                          "failed to create host device_call dispatcher");
    checkDispatcherStatus(
        cudaq_dispatcher_set_ringbuffer(graphDispatcher, &ringbuffer),
        "failed to configure host device_call ring buffer");

    cudaq_function_table_t table{};
    table.entries = functionTable;
    table.count = functionCount;
    checkDispatcherStatus(
        cudaq_dispatcher_set_function_table(graphDispatcher, &table),
        "failed to configure host device_call function table");

    shutdownFlag = 0;
    stats = 0;
    checkDispatcherStatus(
        cudaq_dispatcher_set_control(graphDispatcher, &shutdownFlag, &stats),
        "failed to configure host device_call dispatcher control");
    if (mailbox)
      checkDispatcherStatus(
          cudaq_dispatcher_set_mailbox(graphDispatcher, mailbox),
          "failed to configure host device_call mailbox");
    checkDispatcherStatus(cudaq_dispatcher_start(graphDispatcher),
                          "failed to start host device_call dispatcher");

    graphDispatcherStarted = true;
    nextSlot = 0;
    frameStates.assign(numSlots, FrameState{});
  }

  void stopGraphDispatcher() noexcept {
    if (graphDispatcherStarted && graphDispatcher) {
      __atomic_store_n(const_cast<int *>(&shutdownFlag), 1, __ATOMIC_RELEASE);
      (void)cudaq_dispatcher_stop(graphDispatcher);
    }
    if (graphDispatcher) {
      (void)cudaq_dispatcher_destroy(graphDispatcher);
      graphDispatcher = nullptr;
    }
    if (dispatchManager) {
      (void)cudaq_dispatch_manager_destroy(dispatchManager);
      dispatchManager = nullptr;
    }

    resetRingStorage();
    ringbuffer = {};
    frameStates.clear();
    graphDispatcherStarted = false;
  }

  void **mailbox = nullptr;
  std::uint32_t nextRequestId = 1;
  std::vector<FrameState> frameStates;
  bool graphDispatcherStarted = false;
  cudaq_dispatch_manager_t *dispatchManager = nullptr;
  cudaq_dispatcher_t *graphDispatcher = nullptr;
  cudaq_ringbuffer_t ringbuffer{};
  volatile int shutdownFlag = 0;
  std::uint64_t stats = 0;
  mutable std::mutex mutex;
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, HostDispatchChannel, host_dispatch)

} // namespace
