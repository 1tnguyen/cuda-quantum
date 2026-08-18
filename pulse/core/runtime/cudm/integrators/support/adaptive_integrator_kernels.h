/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <cstddef>

namespace cudaq::integrators::detail {

/// Device-resident vector operations used by the native Dormand-Prince
/// integrator. The only device-to-host transfer is the final scalar error ratio
/// required by the host-side adaptive-step controller.
class Dopri5DeviceOps {
public:
  explicit Dopri5DeviceOps(std::size_t elementCount);
  ~Dopri5DeviceOps();

  Dopri5DeviceOps(const Dopri5DeviceOps &) = delete;
  Dopri5DeviceOps &operator=(const Dopri5DeviceOps &) = delete;

  /// Form y + dt * sum_i coefficients[i] * stages[i] in one device pass.
  void combineStage(void *output, const void *y,
                    const std::array<const void *, 6> &stages,
                    const std::array<double, 6> &coefficients, double dt) const;

  /// Zero a derivative buffer before a cuDensityMat operator action.
  void clear(void *buffer) const;

  /// Compute the torchdiffeq Dopri5 scaled RMS error ratio directly from the
  /// stage derivatives without materializing an embedded fourth-order state.
  double errorRatio(const void *y0, const void *y1,
                    const std::array<const void *, 7> &stages, double dt,
                    double rtol, double atol) const;

private:
  std::size_t m_elementCount;
  double *m_deviceErrorRatio;
};

} // namespace cudaq::integrators::detail
