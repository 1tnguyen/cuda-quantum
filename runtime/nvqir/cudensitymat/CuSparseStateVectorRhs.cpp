/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuSparseStateVectorRhs.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"

#include <cuComplex.h>
#include <cusparse.h>
#include <stdexcept>

namespace cudaq::dynamics {
namespace {

void checkCusparse(cusparseStatus_t status, const char *operation) {
  if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error(
        cudaq_fmt::format("[cusparse] {} failed with status {}", operation,
                          cudaq_fmt::underlying(status)));
}

void *copyToDevice(const void *data, std::size_t size) {
  void *const result = DeviceAllocator::allocate(size);
  try {
    HANDLE_CUDA_ERROR(cudaMemcpy(result, data, size, cudaMemcpyHostToDevice));
  } catch (...) {
    DeviceAllocator::free(result);
    throw;
  }
  return result;
}

} // namespace

struct CuSparseStateVectorRhs::Impl {
  std::size_t dimension{0};
  std::size_t nonzeroCount{0};
  cusparseHandle_t handle{nullptr};
  cusparseSpMatDescr_t matrix{nullptr};
  cusparseDnVecDescr_t inputVector{nullptr};
  cusparseDnVecDescr_t outputVector{nullptr};
  void *rowOffsets{nullptr};
  void *columnIndices{nullptr};
  void *values{nullptr};
  void *workspace{nullptr};
  std::size_t workspaceSize{0};
  bool prepared{false};

  ~Impl() {
    if (inputVector)
      cusparseDestroyDnVec(inputVector);
    if (outputVector)
      cusparseDestroyDnVec(outputVector);
    if (matrix)
      cusparseDestroySpMat(matrix);
    if (workspace)
      DeviceAllocator::free(workspace);
    if (values)
      DeviceAllocator::free(values);
    if (columnIndices)
      DeviceAllocator::free(columnIndices);
    if (rowOffsets)
      DeviceAllocator::free(rowOffsets);
    if (handle)
      cusparseDestroy(handle);
  }
};

CuSparseStateVectorRhs::CuSparseStateVectorRhs(
    std::size_t dimension, std::vector<std::int32_t> rowOffsets,
    std::vector<std::int32_t> columnIndices,
    std::vector<std::complex<double>> values)
    : m_impl(std::make_unique<Impl>()) {
  if (dimension == 0 || rowOffsets.size() != dimension + 1 ||
      columnIndices.size() != values.size() || values.empty() ||
      rowOffsets.back() != static_cast<std::int32_t>(values.size()))
    throw std::invalid_argument("Invalid full-system CSR operator.");

  m_impl->dimension = dimension;
  m_impl->nonzeroCount = values.size();
  checkCusparse(cusparseCreate(&m_impl->handle), "cusparseCreate");

  m_impl->rowOffsets = copyToDevice(rowOffsets.data(),
                                    rowOffsets.size() * sizeof(rowOffsets[0]));
  m_impl->columnIndices = copyToDevice(
      columnIndices.data(), columnIndices.size() * sizeof(columnIndices[0]));
  m_impl->values =
      copyToDevice(values.data(), values.size() * sizeof(values[0]));

  checkCusparse(
      cusparseCreateCsr(&m_impl->matrix, static_cast<std::int64_t>(dimension),
                        static_cast<std::int64_t>(dimension),
                        static_cast<std::int64_t>(m_impl->nonzeroCount),
                        m_impl->rowOffsets, m_impl->columnIndices,
                        m_impl->values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F),
      "cusparseCreateCsr");
}

CuSparseStateVectorRhs::~CuSparseStateVectorRhs() = default;

void CuSparseStateVectorRhs::compute(void *input, void *output) {
  if (!input || !output)
    throw std::invalid_argument(
        "The full-system sparse RHS requires device state storage.");

  constexpr cuDoubleComplex alpha{1.0, 0.0};
  constexpr cuDoubleComplex beta{0.0, 0.0};
  if (!m_impl->prepared) {
    checkCusparse(
        cusparseCreateDnVec(&m_impl->inputVector,
                            static_cast<std::int64_t>(m_impl->dimension), input,
                            CUDA_C_64F),
        "cusparseCreateDnVec(input)");
    checkCusparse(
        cusparseCreateDnVec(&m_impl->outputVector,
                            static_cast<std::int64_t>(m_impl->dimension),
                            output, CUDA_C_64F),
        "cusparseCreateDnVec(output)");
    checkCusparse(cusparseSpMV_bufferSize(
                      m_impl->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                      m_impl->matrix, m_impl->inputVector, &beta,
                      m_impl->outputVector, CUDA_C_64F,
                      CUSPARSE_SPMV_ALG_DEFAULT, &m_impl->workspaceSize),
                  "cusparseSpMV_bufferSize");
    if (m_impl->workspaceSize > 0)
      m_impl->workspace = DeviceAllocator::allocate(m_impl->workspaceSize);
    checkCusparse(cusparseSpMV_preprocess(
                      m_impl->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                      m_impl->matrix, m_impl->inputVector, &beta,
                      m_impl->outputVector, CUDA_C_64F,
                      CUSPARSE_SPMV_ALG_DEFAULT, m_impl->workspace),
                  "cusparseSpMV_preprocess");
    m_impl->prepared = true;
  } else {
    checkCusparse(cusparseDnVecSetValues(m_impl->inputVector, input),
                  "cusparseDnVecSetValues(input)");
    checkCusparse(cusparseDnVecSetValues(m_impl->outputVector, output),
                  "cusparseDnVecSetValues(output)");
  }

  checkCusparse(cusparseSpMV(m_impl->handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, m_impl->matrix, m_impl->inputVector, &beta,
                             m_impl->outputVector, CUDA_C_64F,
                             CUSPARSE_SPMV_ALG_DEFAULT, m_impl->workspace),
                "cusparseSpMV");
}

} // namespace cudaq::dynamics
