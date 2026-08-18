/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuDensityMatState.h"
#include "cudaq/algorithms/base_time_stepper.h"
#include <cudensitymat.h>
#include <memory>
#include <vector>

namespace cudaq::dynamics {
class CuSparseStateVectorRhs;
}

namespace cudaq {
class CuDensityMatTimeStepper : public base_time_stepper {
public:
  explicit CuDensityMatTimeStepper(cudensitymatHandle_t handle,
                                   cudensitymatOperator_t liouvillian);
  ~CuDensityMatTimeStepper() override;

  CuDensityMatTimeStepper(const CuDensityMatTimeStepper &) = delete;
  CuDensityMatTimeStepper &operator=(const CuDensityMatTimeStepper &) = delete;

  state compute(const state &inputState, double t,
                const std::unordered_map<std::string, std::complex<double>>
                    &parameters) override;
  void computeImpl(
      cudensitymatState_t inState, cudensitymatState_t outState, double t,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      int64_t batchSize, void *inputData, void *outputData);

  bool overwritesOutput(int64_t batchSize) const;

private:
  struct ParameterBuffer {
    void *device{nullptr};
    std::complex<double> *host{nullptr};
    std::size_t capacity{0};
    cudaEvent_t ready{nullptr};
    bool inUse{false};
  };

  ParameterBuffer &acquireParameterBuffer(std::size_t elementCount);

  cudensitymatHandle_t m_handle;
  cudensitymatOperator_t m_liouvillian;
  cudensitymatWorkspaceDescriptor_t m_workspace{nullptr};
  void *m_workspaceBuffer{nullptr};
  std::size_t m_requiredBufferSize{0};
  int64_t m_preparedBatchSize{0};
  std::vector<ParameterBuffer> m_parameterBuffers;
  std::size_t m_nextParameterBuffer{0};
  std::shared_ptr<dynamics::CuSparseStateVectorRhs> m_fullSystemSparseRhs;
};
} // namespace cudaq
