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
#include "cudaq_internal/device_call/DeviceCallService.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace cudaq_internal::device_call {
void shutdownDeviceCallRuntime() noexcept;
void registerDeviceCallServiceFactory(cudaq_realtime_get_service_fn_t factory);
void initializeDeviceCallService(void *symbolScope,
                                 std::string_view servicePostfix);
void finalizeDeviceCallService();
void setDeviceCallFunctionTableWithLauncherForDevice(
    std::uint32_t deviceId, cudaq_function_entry_t *entries,
    std::uint32_t count, cudaq_dispatch_launch_fn_t launchFn);
void setHostDeviceCallFunctionTableForDevice(
    std::uint32_t deviceId, cudaq_function_entry_t *entries,
    std::uint32_t count, void **mailbox);
} // namespace cudaq_internal::device_call

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

constexpr std::uint32_t AddThemFunctionId =
    cudaq::realtime::fnv1a_hash("addThem");
constexpr std::uint32_t GraphAddThemFunctionId =
    cudaq::realtime::fnv1a_hash("graphAddThem");
constexpr std::int32_t DeviceCallSuccessStatus =
    toAbiStatus(DeviceCallStatus::Success);
constexpr std::int32_t DeviceCallInvalidArgumentStatus =
    toAbiStatus(DeviceCallStatus::InvalidArgument);
constexpr std::int32_t DeviceCallNotInitializedStatus =
    toAbiStatus(DeviceCallStatus::NotInitialized);
constexpr std::int32_t DeviceCallResponseTooLargeStatus =
    toAbiStatus(DeviceCallStatus::ResponseTooLarge);

#define ASSERT_CUDA_SUCCESS(expr)                                              \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    ASSERT_EQ(cudaSuccess, err)                                                \
        << #expr << " failed at " << __FILE__ << ":" << __LINE__ << ": "       \
        << cudaGetErrorString(err);                                            \
  } while (false)

__device__ int addThemHandler(const void *input, void *output,
                              std::uint32_t argLen, std::uint32_t maxResultLen,
                              std::uint32_t *resultLen) {
  if (argLen != 2 * sizeof(std::int32_t))
    return 101;
  if (maxResultLen < sizeof(std::int32_t))
    return 102;

  auto *args = static_cast<const std::int32_t *>(input);
  *static_cast<std::int32_t *>(output) = args[0] + args[1];
  *resultLen = sizeof(std::int32_t);
  return 0;
}

__device__ int addThemOffsetHandler(const void *input, void *output,
                                    std::uint32_t argLen,
                                    std::uint32_t maxResultLen,
                                    std::uint32_t *resultLen) {
  if (argLen != 2 * sizeof(std::int32_t))
    return 101;
  if (maxResultLen < sizeof(std::int32_t))
    return 102;

  auto *args = static_cast<const std::int32_t *>(input);
  *static_cast<std::int32_t *>(output) = args[0] + args[1] + 100;
  *resultLen = sizeof(std::int32_t);
  return 0;
}

__global__ void graphAddThemHandler(void **mailboxSlotPtr) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto *ioContext =
      reinterpret_cast<cudaq::realtime::GraphIOContext *>(*mailboxSlotPtr);
  if (!ioContext || !ioContext->rx_slot || !ioContext->tx_slot)
    return;

  auto *request = static_cast<cudaq::realtime::RPCHeader *>(ioContext->rx_slot);
  auto *response =
      reinterpret_cast<cudaq::realtime::RPCResponse *>(ioContext->tx_slot);
  auto *result = reinterpret_cast<std::int32_t *>(
      ioContext->tx_slot + sizeof(cudaq::realtime::RPCResponse));

  std::int32_t status = 0;
  std::uint32_t resultLen = sizeof(std::int32_t);
  if (request->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      request->arg_len != 2 * sizeof(std::int32_t) ||
      ioContext->tx_stride_sz <
          sizeof(cudaq::realtime::RPCResponse) + sizeof(std::int32_t)) {
    status = 103;
    resultLen = 0;
  } else {
    auto *args = reinterpret_cast<const std::int32_t *>(request + 1);
    *result = args[0] + args[1];
  }

  response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  response->request_id = request->request_id;
  response->ptp_timestamp = request->ptp_timestamp;
  response->status = status;
  response->result_len = resultLen;

  __threadfence_system();
  if (ioContext->tx_flag)
    *ioContext->tx_flag = ioContext->tx_flag_value;
}

bool createGraphAddThem(void **dMailbox, cudaGraph_t *graphOut,
                        cudaGraphExec_t *execOut) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  cudaKernelNodeParams params{};
  void *kernelArgs[] = {&dMailbox};
  params.func = reinterpret_cast<void *>(graphAddThemHandler);
  params.gridDim = dim3(1, 1, 1);
  params.blockDim = dim3(32, 1, 1);
  params.kernelParams = kernelArgs;

  cudaGraphNode_t node = nullptr;
  if (cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params) !=
      cudaSuccess) {
    cudaGraphDestroy(graph);
    return false;
  }

  cudaGraphExec_t exec = nullptr;
  if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
    cudaGraphDestroy(graph);
    return false;
  }

  *graphOut = graph;
  *execOut = exec;
  return true;
}

std::int32_t dispatchUsingFrameLease(std::uint32_t deviceId,
                                     std::uint32_t functionId,
                                     const void *request,
                                     std::uint64_t requestLen, void *response,
                                     std::uint64_t responseCapacity,
                                     std::uint64_t *responseLen) {
  if ((requestLen > 0 && !request) || !responseLen)
    return DeviceCallInvalidArgumentStatus;
  if (responseCapacity > 0 && !response)
    return DeviceCallInvalidArgumentStatus;

  void *frame = nullptr;
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  std::int32_t status = __cudaq_device_call_acquire_realtime_frame(
      deviceId, functionId, requestLen, responseCapacity, &frame,
      &requestPayload, &responsePayload);
  if (status != DeviceCallSuccessStatus)
    return status;
  if ((requestLen > 0 && !requestPayload) ||
      (responseCapacity > 0 && !responsePayload)) {
    __cudaq_device_call_safely_release_realtime_frame(frame);
    return DeviceCallInvalidArgumentStatus;
  }

  if (requestLen > 0)
    std::memcpy(requestPayload, request, requestLen);

  status = __cudaq_device_call_dispatch_realtime_frame(frame, responseLen);
  if (status == DeviceCallSuccessStatus && *responseLen > responseCapacity)
    status = DeviceCallResponseTooLargeStatus;
  if (status == DeviceCallSuccessStatus && *responseLen > 0)
    std::memcpy(response, responsePayload, *responseLen);

  __cudaq_device_call_safely_release_realtime_frame(frame);
  return status;
}

__global__ void initDeviceCallTable(cudaq_function_entry_t *entries) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  entries[0].handler.device_fn_ptr = reinterpret_cast<void *>(&addThemHandler);
  entries[0].function_id = AddThemFunctionId;
  entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  entries[0].reserved[0] = 0;
  entries[0].reserved[1] = 0;
  entries[0].reserved[2] = 0;
  entries[0].schema.num_args = 2;
  entries[0].schema.num_results = 1;
  entries[0].schema.reserved = 0;
  entries[0].schema.args[0].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.args[0].size_bytes = sizeof(std::int32_t);
  entries[0].schema.args[0].num_elements = 1;
  entries[0].schema.args[1].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.args[1].size_bytes = sizeof(std::int32_t);
  entries[0].schema.args[1].num_elements = 1;
  entries[0].schema.results[0].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.results[0].size_bytes = sizeof(std::int32_t);
  entries[0].schema.results[0].num_elements = 1;
}

__global__ void initDeviceCallTableWithOffset(cudaq_function_entry_t *entries) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  entries[0].handler.device_fn_ptr =
      reinterpret_cast<void *>(&addThemOffsetHandler);
  entries[0].function_id = AddThemFunctionId;
  entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  entries[0].reserved[0] = 0;
  entries[0].reserved[1] = 0;
  entries[0].reserved[2] = 0;
  entries[0].schema.num_args = 2;
  entries[0].schema.num_results = 1;
  entries[0].schema.reserved = 0;
  entries[0].schema.args[0].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.args[0].size_bytes = sizeof(std::int32_t);
  entries[0].schema.args[0].num_elements = 1;
  entries[0].schema.args[1].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.args[1].size_bytes = sizeof(std::int32_t);
  entries[0].schema.args[1].num_elements = 1;
  entries[0].schema.results[0].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.results[0].size_bytes = sizeof(std::int32_t);
  entries[0].schema.results[0].num_elements = 1;
}

void fillHostGraphAddEntry(cudaq_function_entry_t &entry,
                           cudaGraphExec_t graphExec) {
  entry = {};
  entry.handler.graph_exec = graphExec;
  entry.function_id = GraphAddThemFunctionId;
  entry.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
  entry.schema.num_args = 2;
  entry.schema.num_results = 1;
  entry.schema.args[0].type_id = CUDAQ_TYPE_INT32;
  entry.schema.args[0].size_bytes = sizeof(std::int32_t);
  entry.schema.args[0].num_elements = 1;
  entry.schema.args[1].type_id = CUDAQ_TYPE_INT32;
  entry.schema.args[1].size_bytes = sizeof(std::int32_t);
  entry.schema.args[1].num_elements = 1;
  entry.schema.results[0].type_id = CUDAQ_TYPE_INT32;
  entry.schema.results[0].size_bytes = sizeof(std::int32_t);
  entry.schema.results[0].num_elements = 1;
}

int testServiceCreate(const void *, std::size_t, void **handle) {
  if (handle)
    *handle = nullptr;
  return 0;
}

int testServiceDestroy(void *) { return 0; }

std::uint32_t testServiceGetCount(void *) { return 1; }

int testServicePopulateTable(void *, cudaq_function_entry_t *entries,
                             std::uint32_t capacity, cudaStream_t stream) {
  if (!entries || capacity < 1)
    return 1;
  initDeviceCallTable<<<1, 1, 0, stream>>>(entries);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

cudaq_dispatch_launch_fn_t testServiceGetLaunch(void *) {
  return cudaq_launch_dispatch_kernel_regular;
}

int getTestRealtimeService(cudaq_realtime_device_call_service *out) {
  if (!out)
    return 1;
  *out = {};
  out->create = testServiceCreate;
  out->destroy = testServiceDestroy;
  out->get_function_count = testServiceGetCount;
  out->populate_table = testServicePopulateTable;
  out->get_device_dispatch_launch = testServiceGetLaunch;
  return 0;
}

class DeviceCallDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_CUDA_SUCCESS(
        cudaMalloc(&functionEntries, sizeof(cudaq_function_entry_t)));
    initDeviceCallTable<<<1, 1>>>(functionEntries);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
    ASSERT_NO_THROW(setDeviceCallFunctionTableWithLauncherForDevice(
        0, functionEntries, 1, cudaq_launch_dispatch_kernel_regular));
  }

  void TearDown() override {
    shutdownDeviceCallRuntime();
    if (functionEntries)
      cudaFree(functionEntries);
  }

  cudaq_function_entry_t *functionEntries = nullptr;
};

TEST_F(DeviceCallDispatchTest, DispatchesI32AddHandler) {
  std::array<std::int32_t, 2> request{};
  auto *args = request.data();
  args[0] = 19;
  args[1] = 23;

  std::int32_t response = 0;
  std::uint64_t responseLen = 0;
  ASSERT_EQ(0,
            dispatchUsingFrameLease(0, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
  EXPECT_EQ(sizeof(response), responseLen);
  EXPECT_EQ(42, response);
}

TEST_F(DeviceCallDispatchTest, DispatchesI32AddHandlerThroughFrameLease) {
  void *frame = nullptr;
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                   0, AddThemFunctionId, 2 * sizeof(std::int32_t),
                   sizeof(std::int32_t), &frame, &requestPayload,
                   &responsePayload));
  ASSERT_NE(nullptr, frame);
  ASSERT_NE(nullptr, requestPayload);
  ASSERT_NE(nullptr, responsePayload);

  auto *args = static_cast<std::int32_t *>(requestPayload);
  args[0] = 19;
  args[1] = 23;

  std::uint64_t responseLen = 0;
  ASSERT_EQ(0,
            __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
  EXPECT_EQ(sizeof(std::int32_t), responseLen);
  EXPECT_EQ(42, *static_cast<std::int32_t *>(responsePayload));

  __cudaq_device_call_safely_release_realtime_frame(frame);
}

TEST_F(DeviceCallDispatchTest, DispatchesVoidFireAndForgetThroughFrameLease) {
  void *frame = nullptr;
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                   0, AddThemFunctionId, 2 * sizeof(std::int32_t), 0, &frame,
                   &requestPayload, &responsePayload));
  ASSERT_NE(nullptr, frame);
  ASSERT_NE(nullptr, requestPayload);
  EXPECT_EQ(nullptr, responsePayload);

  auto *args = static_cast<std::int32_t *>(requestPayload);
  args[0] = 19;
  args[1] = 23;

  std::uint64_t responseLen = 123;
  ASSERT_EQ(0,
            __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
  EXPECT_EQ(0u, responseLen);

  __cudaq_device_call_safely_release_realtime_frame(frame);

  for (int i = 0; i < 2; ++i) {
    frame = nullptr;
    requestPayload = nullptr;
    responsePayload = nullptr;
    ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                     0, AddThemFunctionId, 2 * sizeof(std::int32_t),
                     sizeof(std::int32_t), &frame, &requestPayload,
                     &responsePayload));
    args = static_cast<std::int32_t *>(requestPayload);
    args[0] = 19;
    args[1] = 23;

    responseLen = 0;
    ASSERT_EQ(0,
              __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
    EXPECT_EQ(sizeof(std::int32_t), responseLen);
    EXPECT_EQ(42, *static_cast<std::int32_t *>(responsePayload));

    __cudaq_device_call_safely_release_realtime_frame(frame);
  }
}

class DeviceCallEndpointRoutingTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_CUDA_SUCCESS(
        cudaMalloc(&defaultEntries, sizeof(cudaq_function_entry_t)));
    ASSERT_CUDA_SUCCESS(
        cudaMalloc(&remoteEntries, sizeof(cudaq_function_entry_t)));
    initDeviceCallTable<<<1, 1>>>(defaultEntries);
    initDeviceCallTableWithOffset<<<1, 1>>>(remoteEntries);
    ASSERT_CUDA_SUCCESS(cudaGetLastError());
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
  }

  void TearDown() override {
    shutdownDeviceCallRuntime();
    if (defaultEntries)
      cudaFree(defaultEntries);
    if (remoteEntries)
      cudaFree(remoteEntries);
  }

  cudaq_function_entry_t *defaultEntries = nullptr;
  cudaq_function_entry_t *remoteEntries = nullptr;
};

TEST_F(DeviceCallEndpointRoutingTest, DispatchesByDeviceId) {
  std::array<std::int32_t, 2> request{19, 23};
  std::int32_t response = 0;
  std::uint64_t responseLen = 0;

  ASSERT_NO_THROW(setDeviceCallFunctionTableWithLauncherForDevice(
      0, defaultEntries, 1, cudaq_launch_dispatch_kernel_regular));
  ASSERT_EQ(0,
            dispatchUsingFrameLease(0, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
  EXPECT_EQ(sizeof(response), responseLen);
  EXPECT_EQ(42, response);
  shutdownDeviceCallRuntime();

  response = 0;
  responseLen = 0;
  ASSERT_NO_THROW(setDeviceCallFunctionTableWithLauncherForDevice(
      7, remoteEntries, 1, cudaq_launch_dispatch_kernel_regular));
  ASSERT_EQ(0,
            dispatchUsingFrameLease(7, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
  EXPECT_EQ(sizeof(response), responseLen);
  EXPECT_EQ(142, response);
}

class HostGraphDispatchFrameTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_CUDA_SUCCESS(
        cudaHostAlloc(&hMailbox, sizeof(void *), cudaHostAllocMapped));
    std::memset(hMailbox, 0, sizeof(void *));
    ASSERT_CUDA_SUCCESS(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&dMailbox), hMailbox, 0));
    ASSERT_TRUE(createGraphAddThem(dMailbox, &graph, &graphExec));

    fillHostGraphAddEntry(entries[0], graphExec);
    ASSERT_NO_THROW(setHostDeviceCallFunctionTableForDevice(
        0, entries.data(), entries.size(), hMailbox));
  }

  void TearDown() override {
    if (frame)
      __cudaq_device_call_safely_release_realtime_frame(frame);
    shutdownDeviceCallRuntime();
    if (graphExec)
      cudaGraphExecDestroy(graphExec);
    if (graph)
      cudaGraphDestroy(graph);
    if (hMailbox)
      cudaFreeHost(hMailbox);
  }

  void **hMailbox = nullptr;
  void **dMailbox = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  std::array<cudaq_function_entry_t, 1> entries{};
  void *frame = nullptr;
};

TEST_F(HostGraphDispatchFrameTest, DispatchesGraphLaunchThroughFrameLease) {
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  ASSERT_EQ(DeviceCallSuccessStatus,
            __cudaq_device_call_acquire_realtime_frame(
                0, GraphAddThemFunctionId, 2 * sizeof(std::int32_t),
                sizeof(std::int32_t), &frame, &requestPayload,
                &responsePayload));
  ASSERT_NE(nullptr, frame);
  ASSERT_NE(nullptr, requestPayload);
  ASSERT_NE(nullptr, responsePayload);

  auto *args = static_cast<std::int32_t *>(requestPayload);
  args[0] = 19;
  args[1] = 23;

  std::uint64_t responseLen = 0;
  ASSERT_EQ(DeviceCallSuccessStatus,
            __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
  EXPECT_EQ(sizeof(std::int32_t), responseLen);
  EXPECT_EQ(42, *static_cast<std::int32_t *>(responsePayload));

  __cudaq_device_call_safely_release_realtime_frame(frame);
  frame = nullptr;
}

class DeviceCallServiceFactoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_NO_THROW(registerDeviceCallServiceFactory(getTestRealtimeService));
    ASSERT_NO_THROW(initializeDeviceCallService(nullptr, {}));
  }

  void TearDown() override {
    try {
      finalizeDeviceCallService();
    } catch (...) {
    }
    ASSERT_NO_THROW(registerDeviceCallServiceFactory(nullptr));
  }
};

TEST_F(DeviceCallServiceFactoryTest, DispatchesThroughRegisteredFactory) {
  std::array<std::int32_t, 2> request{19, 23};
  std::int32_t response = 0;
  std::uint64_t responseLen = 0;

  ASSERT_EQ(0,
            dispatchUsingFrameLease(0, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
  EXPECT_EQ(sizeof(response), responseLen);
  EXPECT_EQ(42, response);
}

TEST_F(DeviceCallServiceFactoryTest, FinalizeClearsFactorySession) {
  ASSERT_NO_THROW(finalizeDeviceCallService());

  std::array<std::int32_t, 2> request{19, 23};
  std::int32_t response = 0;
  std::uint64_t responseLen = 0;
  EXPECT_EQ(DeviceCallNotInitializedStatus,
            dispatchUsingFrameLease(0, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
}

} // namespace
