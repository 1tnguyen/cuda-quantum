/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatErrorHandling.h"
#include "adaptive_integrator_kernels.h"

#include <algorithm>
#include <cuComplex.h>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <stdexcept>

namespace cudaq::integrators::detail {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaximumBlocks = 4096;

__device__ __forceinline__ void addScaledComplex(double &real, double &imag,
                                                 double scale,
                                                 const cuDoubleComplex *values,
                                                 std::size_t index) {
  real = fma(scale, values[index].x, real);
  imag = fma(scale, values[index].y, imag);
}

__device__ __forceinline__ void addStage(double &real, double &imag,
                                         double coefficient, double dt,
                                         const cuDoubleComplex *values,
                                         std::size_t index) {
  if (coefficient != 0.0)
    addScaledComplex(real, imag, dt * coefficient, values, index);
}

__global__ void combineStageKernel(cuDoubleComplex *__restrict__ output,
                                   const cuDoubleComplex *__restrict__ y,
                                   const cuDoubleComplex *__restrict__ k0,
                                   const cuDoubleComplex *__restrict__ k1,
                                   const cuDoubleComplex *__restrict__ k2,
                                   const cuDoubleComplex *__restrict__ k3,
                                   const cuDoubleComplex *__restrict__ k4,
                                   const cuDoubleComplex *__restrict__ k5,
                                   double a0, double a1, double a2, double a3,
                                   double a4, double a5, double dt,
                                   std::size_t elementCount) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elementCount;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double real = y[i].x;
    double imag = y[i].y;
    addStage(real, imag, a0, dt, k0, i);
    addStage(real, imag, a1, dt, k1, i);
    addStage(real, imag, a2, dt, k2, i);
    addStage(real, imag, a3, dt, k3, i);
    addStage(real, imag, a4, dt, k4, i);
    addStage(real, imag, a5, dt, k5, i);
    output[i] = make_cuDoubleComplex(real, imag);
  }
}

__global__ void errorRatioKernel(const cuDoubleComplex *__restrict__ y0,
                                 const cuDoubleComplex *__restrict__ y1,
                                 const cuDoubleComplex *__restrict__ k0,
                                 const cuDoubleComplex *__restrict__ k2,
                                 const cuDoubleComplex *__restrict__ k3,
                                 const cuDoubleComplex *__restrict__ k4,
                                 const cuDoubleComplex *__restrict__ k5,
                                 const cuDoubleComplex *__restrict__ k6,
                                 double dt, double rtol, double atol,
                                 std::size_t elementCount,
                                 double *__restrict__ sum) {
  using BlockReduce = cub::BlockReduce<double, kThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage reductionStorage;

  // These are torchdiffeq's Dormand-Prince-Shampine error coefficients.
  constexpr double e0 = 35.0 / 384.0 - 1951.0 / 21600.0;
  constexpr double e2 = 500.0 / 1113.0 - 22642.0 / 50085.0;
  constexpr double e3 = 125.0 / 192.0 - 451.0 / 720.0;
  constexpr double e4 = -2187.0 / 6784.0 + 12231.0 / 42400.0;
  constexpr double e5 = 11.0 / 84.0 - 649.0 / 6300.0;
  constexpr double e6 = -1.0 / 60.0;

  double threadSum = 0.0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elementCount;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double errorReal = e0 * k0[i].x;
    double errorImag = e0 * k0[i].y;
    addScaledComplex(errorReal, errorImag, e2, k2, i);
    addScaledComplex(errorReal, errorImag, e3, k3, i);
    addScaledComplex(errorReal, errorImag, e4, k4, i);
    addScaledComplex(errorReal, errorImag, e5, k5, i);
    errorReal = dt * fma(e6, k6[i].x, errorReal);
    errorImag = dt * fma(e6, k6[i].y, errorImag);

    const double y0Magnitude = hypot(y0[i].x, y0[i].y);
    const double y1Magnitude = hypot(y1[i].x, y1[i].y);
    const double tolerance = atol + rtol * fmax(y0Magnitude, y1Magnitude);
    threadSum += (errorReal * errorReal + errorImag * errorImag) /
                 (tolerance * tolerance);
  }

  const double blockSum = BlockReduce(reductionStorage).Sum(threadSum);
  if (threadIdx.x == 0)
    atomicAdd(sum, blockSum);
}

__global__ void finishErrorRatioKernel(double *sum, std::size_t elementCount) {
  *sum = sqrt(*sum / static_cast<double>(elementCount));
}

int blockCount(std::size_t elementCount) {
  const std::size_t requiredBlocks =
      (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
  return static_cast<int>(
      std::min(requiredBlocks, static_cast<std::size_t>(kMaximumBlocks)));
}

const cuDoubleComplex *asComplex(const void *pointer) {
  return static_cast<const cuDoubleComplex *>(pointer);
}

} // namespace

Dopri5DeviceOps::Dopri5DeviceOps(std::size_t elementCount)
    : m_elementCount(elementCount), m_deviceErrorRatio(nullptr) {
  if (elementCount == 0)
    throw std::invalid_argument("Dopri5 requires a non-empty state.");
  HANDLE_CUDA_ERROR(cudaMalloc(&m_deviceErrorRatio, sizeof(double)));
}

Dopri5DeviceOps::~Dopri5DeviceOps() {
  if (m_deviceErrorRatio)
    cudaFree(m_deviceErrorRatio);
}

void Dopri5DeviceOps::combineStage(void *output, const void *y,
                                   const std::array<const void *, 6> &stages,
                                   const std::array<double, 6> &coefficients,
                                   double dt) const {
  combineStageKernel<<<blockCount(m_elementCount), kThreadsPerBlock>>>(
      static_cast<cuDoubleComplex *>(output), asComplex(y),
      asComplex(stages[0]), asComplex(stages[1]), asComplex(stages[2]),
      asComplex(stages[3]), asComplex(stages[4]), asComplex(stages[5]),
      coefficients[0], coefficients[1], coefficients[2], coefficients[3],
      coefficients[4], coefficients[5], dt, m_elementCount);
  HANDLE_CUDA_ERROR(cudaGetLastError());
}

void Dopri5DeviceOps::clear(void *buffer) const {
  HANDLE_CUDA_ERROR(
      cudaMemsetAsync(buffer, 0, m_elementCount * sizeof(cuDoubleComplex)));
}

double Dopri5DeviceOps::errorRatio(const void *y0, const void *y1,
                                   const std::array<const void *, 7> &stages,
                                   double dt, double rtol, double atol) const {
  HANDLE_CUDA_ERROR(cudaMemsetAsync(m_deviceErrorRatio, 0, sizeof(double)));
  errorRatioKernel<<<blockCount(m_elementCount), kThreadsPerBlock>>>(
      asComplex(y0), asComplex(y1), asComplex(stages[0]), asComplex(stages[2]),
      asComplex(stages[3]), asComplex(stages[4]), asComplex(stages[5]),
      asComplex(stages[6]), dt, rtol, atol, m_elementCount, m_deviceErrorRatio);
  HANDLE_CUDA_ERROR(cudaGetLastError());
  finishErrorRatioKernel<<<1, 1>>>(m_deviceErrorRatio, m_elementCount);
  HANDLE_CUDA_ERROR(cudaGetLastError());

  double errorRatio = 0.0;
  HANDLE_CUDA_ERROR(cudaMemcpy(&errorRatio, m_deviceErrorRatio, sizeof(double),
                               cudaMemcpyDeviceToHost));
  return errorRatio;
}

} // namespace cudaq::integrators::detail
