/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"
#include <map>

namespace cudaq {
CuDensityMatTimeStepper::CuDensityMatTimeStepper(
    cudensitymatHandle_t handle, cudensitymatOperator_t liouvillian)
    : m_handle(handle), m_liouvillian(liouvillian) {};

CuDensityMatTimeStepper::~CuDensityMatTimeStepper() {
  for (auto &buffer : m_parameterBuffers) {
    if (buffer.device)
      cudaq::dynamics::DeviceAllocator::free(buffer.device);
    if (buffer.host)
      cudaFreeHost(buffer.host);
    if (buffer.ready)
      cudaEventDestroy(buffer.ready);
  }
  if (m_workspace)
    cudensitymatDestroyWorkspace(m_workspace);
}

CuDensityMatTimeStepper::ParameterBuffer &
CuDensityMatTimeStepper::acquireParameterBuffer(std::size_t elementCount) {
  ParameterBuffer *availableBuffer = nullptr;
  for (std::size_t i = 0; i < m_parameterBuffers.size(); ++i) {
    const std::size_t index =
        (m_nextParameterBuffer + i) % m_parameterBuffers.size();
    auto &buffer = m_parameterBuffers[index];
    if (!buffer.inUse) {
      availableBuffer = &buffer;
      m_nextParameterBuffer = index + 1;
      break;
    }

    const cudaError_t eventStatus = cudaEventQuery(buffer.ready);
    if (eventStatus == cudaSuccess) {
      availableBuffer = &buffer;
      m_nextParameterBuffer = index + 1;
      break;
    }
    if (eventStatus != cudaErrorNotReady)
      HANDLE_CUDA_ERROR(eventStatus);
  }

  if (!availableBuffer) {
    m_parameterBuffers.emplace_back();
    availableBuffer = &m_parameterBuffers.back();
    HANDLE_CUDA_ERROR(cudaEventCreateWithFlags(&availableBuffer->ready,
                                               cudaEventDisableTiming));
    m_nextParameterBuffer = m_parameterBuffers.size();
  }

  if (availableBuffer->capacity < elementCount) {
    if (availableBuffer->device)
      cudaq::dynamics::DeviceAllocator::free(availableBuffer->device);
    if (availableBuffer->host)
      HANDLE_CUDA_ERROR(cudaFreeHost(availableBuffer->host));

    const std::size_t bufferSize = elementCount * sizeof(std::complex<double>);
    availableBuffer->device =
        cudaq::dynamics::DeviceAllocator::allocate(bufferSize);
    HANDLE_CUDA_ERROR(cudaMallocHost(
        reinterpret_cast<void **>(&availableBuffer->host), bufferSize));
    availableBuffer->capacity = elementCount;
  }

  return *availableBuffer;
}

state CuDensityMatTimeStepper::compute(
    const state &inputState, double t,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  auto *simState =
      cudaq::state_helper::getSimulationState(const_cast<state *>(&inputState));
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");
  CuDensityMatState &state = *castSimState;

  // Create a new state for the next step
  auto next_state = CuDensityMatState::zero_like(state);
  assert(next_state.getBatchSize() == state.getBatchSize());
  computeImpl(state.get_impl(), next_state.get_impl(), t, parameters,
              state.getBatchSize());
  return cudaq::state(
      std::make_unique<CuDensityMatState>(std::move(next_state)).release());
}

void CuDensityMatTimeStepper::computeImpl(
    cudensitymatState_t inState, cudensitymatState_t outState, double t,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    int64_t batchSize) {
  // The prepared action is valid for any input and output states with the same
  // shape, kind, and factorization. Reuse it across integration stages.
  if (!m_workspace || m_preparedBatchSize != batchSize) {
    if (m_workspace)
      HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(m_workspace));

    m_workspace = nullptr;
    m_workspaceBuffer = nullptr;
    m_requiredBufferSize = 0;
    m_preparedBatchSize = 0;
    HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(m_handle, &m_workspace));

    {
      cudaq::dynamics::PerfMetricScopeTimer metricTimer(
          "cudensitymatOperatorPrepareAction");
      HANDLE_CUDM_ERROR(cudensitymatOperatorPrepareAction(
          m_handle, m_liouvillian, inState, outState, CUDENSITYMAT_COMPUTE_64F,
          dynamics::Context::getRecommendedWorkSpaceLimit(), m_workspace, 0x0));
    }

    HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(
        m_handle, m_workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
        CUDENSITYMAT_WORKSPACE_SCRATCH, &m_requiredBufferSize));
    m_preparedBatchSize = batchSize;
  }

  if (m_requiredBufferSize > 0) {
    void *const workspaceBuffer =
        dynamics::Context::getCurrentContext()->getScratchSpace(
            m_requiredBufferSize);
    if (workspaceBuffer != m_workspaceBuffer) {
      HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(
          m_handle, m_workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
          CUDENSITYMAT_WORKSPACE_SCRATCH, workspaceBuffer,
          m_requiredBufferSize));
      m_workspaceBuffer = workspaceBuffer;
    }
  }

  // Apply the operator action
  std::map<std::string, std::complex<double>> sortedParameters(
      parameters.begin(), parameters.end());
  const auto numComplexParams = sortedParameters.size();
  ParameterBuffer *parameterBuffer = nullptr;
  if (numComplexParams > 0)
    parameterBuffer = &acquireParameterBuffer(numComplexParams * batchSize);
  // Note: for batch, params is F-order 2d-array of user-defined real parameter
  // values: params[numParams, batchSize].
  std::size_t parameterIndex = 0;
  for (int i = 0; i < batchSize; ++i) {
    for (const auto &[k, v] : sortedParameters)
      parameterBuffer->host[parameterIndex++] = v;
  }
  double *param_d = nullptr;
  if (parameterBuffer) {
    const std::size_t parameterSize =
        parameterIndex * sizeof(std::complex<double>);
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(parameterBuffer->device,
                                      parameterBuffer->host, parameterSize,
                                      cudaMemcpyHostToDevice, 0));
    param_d = static_cast<double *>(parameterBuffer->device);
  }
  {
    cudaq::dynamics::PerfMetricScopeTimer metricTimer(
        "cudensitymatOperatorComputeAction");
    HANDLE_CUDM_ERROR(cudensitymatOperatorComputeAction(
        m_handle, m_liouvillian, t, batchSize, numComplexParams * 2, param_d,
        inState, outState, m_workspace, 0x0));
  }

  if (parameterBuffer) {
    HANDLE_CUDA_ERROR(cudaEventRecord(parameterBuffer->ready, 0));
    parameterBuffer->inUse = true;
  }
}

} // namespace cudaq
