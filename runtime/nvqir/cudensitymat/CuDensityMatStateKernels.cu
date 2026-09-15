/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatStateKernels.h"
#include <cuComplex.h>

namespace {
__global__ void completeHermitianMatrixKernel(cuDoubleComplex *matrix,
                                              std::size_t dimension) {
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;
  const auto col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= dimension || col >= dimension || row > col)
    return;

  const auto upperIndex = row + col * dimension;
  const auto lowerIndex = col + row * dimension;
  const auto upper = matrix[upperIndex];
  const auto lower = matrix[lowerIndex];
  const cuDoubleComplex sum = {upper.x + lower.x, upper.y - lower.y};
  matrix[upperIndex] = sum;
  if (row != col)
    matrix[lowerIndex] = {sum.x, -sum.y};
}
} // namespace

void cudaq::dynamics::completeHermitianMatrix(void *deviceData,
                                              std::size_t dimension) {
  constexpr dim3 block(32, 8);
  const dim3 grid((dimension + block.x - 1) / block.x,
                  (dimension + block.y - 1) / block.y);
  completeHermitianMatrixKernel<<<grid, block>>>(
      static_cast<cuDoubleComplex *>(deviceData), dimension);
  HANDLE_CUDA_ERROR(cudaGetLastError());
}
