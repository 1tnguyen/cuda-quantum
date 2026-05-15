/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/RpcFrame.h"
#include "SocketUtils.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

extern "C" cudaq_realtime_bridge_interface_t *
cudaq_realtime_get_bridge_interface();

extern "C" cudaq_status_t cudaq_realtime_tcp_transport_get_bound_port(
    cudaq_realtime_bridge_handle_t handle, std::uint16_t *outPort);

namespace {

using namespace cudaq_internal::device_call;

constexpr std::uint32_t RpcIncrementFunctionId =
    cudaq::realtime::fnv1a_hash("rpc_increment");

__device__ int rpcIncrementHandler(const void *input, void *output,
                                   std::uint32_t argLen,
                                   std::uint32_t maxResultLen,
                                   std::uint32_t *resultLen) {
  const auto *in = static_cast<const std::uint8_t *>(input);
  auto *out = static_cast<std::uint8_t *>(output);
  const std::uint32_t len = argLen < maxResultLen ? argLen : maxResultLen;
  for (std::uint32_t i = 0; i < len; ++i)
    out[i] = static_cast<std::uint8_t>(in[i] + 1);
  *resultLen = len;
  return 0;
}

__global__ void initIncrementTable(cudaq_function_entry_t *entries) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  entries[0].handler.device_fn_ptr =
      reinterpret_cast<void *>(&rpcIncrementHandler);
  entries[0].function_id = RpcIncrementFunctionId;
  entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  entries[0].reserved[0] = 0;
  entries[0].reserved[1] = 0;
  entries[0].reserved[2] = 0;
  entries[0].schema.num_args = 1;
  entries[0].schema.num_results = 1;
  entries[0].schema.reserved = 0;
  entries[0].schema.args[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  entries[0].schema.args[0].reserved[0] = 0;
  entries[0].schema.args[0].reserved[1] = 0;
  entries[0].schema.args[0].reserved[2] = 0;
  entries[0].schema.args[0].size_bytes = 0;
  entries[0].schema.args[0].num_elements = 0;
  entries[0].schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  entries[0].schema.results[0].reserved[0] = 0;
  entries[0].schema.results[0].reserved[1] = 0;
  entries[0].schema.results[0].reserved[2] = 0;
  entries[0].schema.results[0].size_bytes = 0;
  entries[0].schema.results[0].num_elements = 0;
}

TcpSocket connectToLocalhost(std::uint16_t port) {
  TcpSocket socket;
  (void)socket.connect("127.0.0.1", port, 5000);
  return socket;
}

std::vector<std::uint8_t>
makeRpcFrame(const std::vector<std::uint8_t> &payload) {
  std::vector<std::uint8_t> frame(sizeof(cudaq::realtime::RPCHeader) +
                                  payload.size());
  initializeRequestHeader(frame.data(), RpcIncrementFunctionId, payload.size(),
                          7);
  std::memcpy(requestPayload(frame.data()), payload.data(), payload.size());
  return frame;
}

class TcpRealtimeTransportTest : public ::testing::Test {
protected:
  void TearDown() override {
    if (dShutdownFlag) {
      int shutdown = 1;
      (void)cudaMemcpy(dShutdownFlag, &shutdown, sizeof(shutdown),
                       cudaMemcpyHostToDevice);
    }
    if (stream)
      (void)cudaStreamSynchronize(stream);
    if (stream) {
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
    if (bridgeInterface && bridgeHandle) {
      (void)bridgeInterface->disconnect(bridgeHandle);
      (void)bridgeInterface->destroy(bridgeHandle);
      bridgeHandle = nullptr;
    }
    if (dFunctionEntries) {
      (void)cudaFree(dFunctionEntries);
      dFunctionEntries = nullptr;
    }
    if (dShutdownFlag) {
      (void)cudaFree(dShutdownFlag);
      dShutdownFlag = nullptr;
    }
    if (dStats) {
      (void)cudaFree(dStats);
      dStats = nullptr;
    }
  }

  void initializeFunctionTable() {
    ASSERT_EQ(cudaMalloc(&dFunctionEntries, sizeof(cudaq_function_entry_t)),
              cudaSuccess);
    initIncrementTable<<<1, 1>>>(dFunctionEntries);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  void startDispatchKernel(const cudaq_ringbuffer_t &ringbuffer) {
    ASSERT_EQ(cudaMalloc(&dShutdownFlag, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dShutdownFlag, 0, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dStats, sizeof(std::uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(dStats, 0, sizeof(std::uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              cudaSuccess);

    cudaq_launch_dispatch_kernel_regular(
        ringbuffer.rx_flags, ringbuffer.tx_flags, ringbuffer.rx_data,
        ringbuffer.tx_data, ringbuffer.rx_stride_sz, ringbuffer.tx_stride_sz,
        dFunctionEntries, 1, dShutdownFlag, dStats, 2, 1, 64, stream);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  }

  cudaq_realtime_bridge_interface_t *bridgeInterface = nullptr;
  cudaq_realtime_bridge_handle_t bridgeHandle = nullptr;
  cudaq_function_entry_t *dFunctionEntries = nullptr;
  int *dShutdownFlag = nullptr;
  std::uint64_t *dStats = nullptr;
  cudaStream_t stream = nullptr;
};

TEST_F(TcpRealtimeTransportTest, DispatchesIncrementRpcOverTcp) {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0)
    GTEST_SKIP() << "CUDA device is not available";

  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (!prop.canMapHostMemory)
    GTEST_SKIP() << "GPU does not support mapped host memory";

  bridgeInterface = cudaq_realtime_get_bridge_interface();
  ASSERT_NE(bridgeInterface, nullptr);
  ASSERT_EQ(bridgeInterface->version, CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION);

  const char *argv[] = {"tcp-transport-test",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "0",
                        "--num-slots",
                        "2",
                        "--slot-size",
                        "256",
                        "--timeout-ms",
                        "5000",
                        "--gpu",
                        "0"};
  const int argc = static_cast<int>(sizeof(argv) / sizeof(argv[0]));

  ASSERT_EQ(
      bridgeInterface->create(&bridgeHandle, argc, const_cast<char **>(argv)),
      CUDAQ_OK);
  ASSERT_EQ(bridgeInterface->connect(bridgeHandle), CUDAQ_OK);

  std::uint16_t port = 0;
  ASSERT_EQ(cudaq_realtime_tcp_transport_get_bound_port(bridgeHandle, &port),
            CUDAQ_OK);
  ASSERT_GT(port, 0);

  cudaq_ringbuffer_t ringbuffer{};
  ASSERT_EQ(bridgeInterface->get_transport_context(bridgeHandle, RING_BUFFER,
                                                   &ringbuffer),
            CUDAQ_OK);
  ASSERT_NE(ringbuffer.rx_flags, nullptr);
  ASSERT_NE(ringbuffer.tx_flags, nullptr);
  ASSERT_NE(ringbuffer.rx_data, nullptr);
  ASSERT_NE(ringbuffer.tx_data, nullptr);

  initializeFunctionTable();
  startDispatchKernel(ringbuffer);
  ASSERT_EQ(bridgeInterface->launch(bridgeHandle), CUDAQ_OK);

  TcpSocket client = connectToLocalhost(port);
  ASSERT_TRUE(client.isValid());

  const std::vector<std::uint8_t> payload = {0, 1, 2, 3};
  auto requestFrame = makeRpcFrame(payload);
  ASSERT_TRUE(writeLengthPrefixedFrame(client, requestFrame.data(),
                                       requestFrame.size()));

  std::vector<std::uint8_t> responseFrame;
  ASSERT_TRUE(readLengthPrefixedFrame(
      client, responseFrame, std::numeric_limits<std::uint32_t>::max()));
  ASSERT_GE(responseFrame.size(), sizeof(cudaq::realtime::RPCResponse));

  const auto *response = reinterpret_cast<const cudaq::realtime::RPCResponse *>(
      responseFrame.data());
  EXPECT_EQ(response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE);
  EXPECT_EQ(response->status, toAbiStatus(DeviceCallStatus::Success));
  EXPECT_EQ(response->result_len, payload.size());
  EXPECT_EQ(response->request_id, 7u);

  const std::vector<std::uint8_t> result(
      responseFrame.begin() + sizeof(cudaq::realtime::RPCResponse),
      responseFrame.end());
  const std::vector<std::uint8_t> expected = {1, 2, 3, 4};
  EXPECT_EQ(result, expected);
}

} // namespace
