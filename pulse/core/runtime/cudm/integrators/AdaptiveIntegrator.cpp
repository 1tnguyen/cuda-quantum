/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatIntegratorBase.h"
#include "CuDensityMatUtils.h"
#include "support/adaptive_integrator_kernels.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/cudaq_mpi.h"
#include "cudaq/runtime/logger/logger.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace cudaq::integrators {

// Dormand-Prince RK5(4) adaptive integrator.
// Reference: Dormand, J. R.; Prince, P. J. (1980), "A family of embedded
// Runge-Kutta formulae", Journal of Computational and Applied Mathematics.
//
// Uses the mainlined cuDensityMat time stepper to evaluate the Liouvillian
// action f(t, y) = L(t) y, forms the embedded 5th- and 4th-order solutions,
// and adapts the step size from the scaled error between them.

using cudmIntHelp = CuDensityMatIntegratorHelper;

namespace {
// Butcher tableau (nodes).
constexpr std::array<double, 7> kNodes = {0.0,       0.2, 0.3, 0.8,
                                          8.0 / 9.0, 1.0, 1.0};

// Lower-triangular a_{ij} coefficients.
constexpr std::array<std::array<double, 6>, 7> kA = {
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {0.2, 0.0, 0.0, 0.0, 0.0, 0.0},
     {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
     {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0},
     {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
      0.0, 0.0},
     {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
      -5103.0 / 18656.0, 0.0},
     {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
      11.0 / 84.0}}};

// Step-size controller parameters. These match torchdiffeq's Dopri5 defaults.
constexpr double kSafetyFactor = 0.9;
constexpr double kMaxScale = 10.0;
constexpr double kMinScale = 0.2;
constexpr double kOrder = 5.0;

/// @brief Select the next step size from the scaled error estimate.
double adaptStepSize(double errorNorm, double dtCurrent, double dtMin,
                     double dtMax) {
  double factor;
  if (errorNorm == 0.0) {
    factor = kMaxScale;
  } else {
    factor = kSafetyFactor * std::pow(1.0 / errorNorm, 1.0 / kOrder);
    const double minimumFactor = errorNorm < 1.0 ? 1.0 : kMinScale;
    factor = std::max(minimumFactor, std::min(kMaxScale, factor));
  }
  return std::max(dtMin, std::min(dtMax, dtCurrent * factor));
}
} // namespace

dopri5::dopri5(double rtol, double atol, double dt_initial, double dt_min,
               double dt_max)
    : m_rtol(rtol), m_atol(atol), m_dt_initial(dt_initial), m_dt(dt_initial),
      m_dt_min(dt_min), m_dt_max(dt_max), m_t(0.0) {
  if (rtol <= 0.0 || atol <= 0.0)
    throw std::invalid_argument(
        "dopri5 integrator requires positive rtol and atol.");
  if (dt_min <= 0.0 || dt_max <= 0.0 || dt_min > dt_max)
    throw std::invalid_argument(
        "dopri5 integrator requires 0 < dt_min <= dt_max.");
}

std::shared_ptr<base_integrator> dopri5::clone() {
  auto cloned = std::make_shared<cudaq::integrators::dopri5>(
      m_rtol, m_atol, m_dt_initial, m_dt_min, m_dt_max);
  cloned->m_t = this->m_t;
  cloned->m_dt = this->m_dt;
  cloned->m_state = this->m_state;
  cloned->m_system = this->m_system;
  cloned->m_schedule = this->m_schedule;
  cloned->m_stats = this->m_stats;
  return cloned;
}

void dopri5::setState(const cudaq::state &initialState, double t0) {
  cudmIntHelp::setState(m_state, m_t, initialState, t0);
  m_dt = m_dt_initial;
  resetStats();
}

std::pair<double, cudaq::state> dopri5::getState() {
  return cudmIntHelp::getState(m_state, m_t);
}

void dopri5::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer("dopri5::integrate");
  cudmIntHelp::ensureStepper(m_stepper, m_state, m_system, m_schedule);

  if (m_t >= targetTime)
    return;

  auto *cudmStepper = dynamic_cast<CuDensityMatTimeStepper *>(m_stepper.get());
  if (!cudmStepper)
    throw std::runtime_error("dopri5 requires a cuDensityMat time stepper.");

  auto current = CuDensityMatState::clone(*cudmIntHelp::asCudmState(*m_state));
  auto candidate = std::make_unique<CuDensityMatState>(
      CuDensityMatState::zero_like(*current));
  std::array<std::unique_ptr<CuDensityMatState>, 7> stages;
  for (auto &stage : stages)
    stage = std::make_unique<CuDensityMatState>(
        CuDensityMatState::zero_like(*current));

  const std::size_t elementCount = current->getTensor().get_num_elements();
  detail::Dopri5DeviceOps deviceOps(elementCount);

  const bool isDistributed =
      cudaq::dynamics::Context::getCurrentContext()->isDistributed();
  std::size_t globalElementCount = elementCount;
  if (isDistributed) {
    std::size_t stateDimension = 1;
    for (const auto extent : current->get_hilbert_space_dims())
      stateDimension *= static_cast<std::size_t>(extent);
    globalElementCount = stateDimension * current->getBatchSize();
    if (current->is_density_matrix())
      globalElementCount *= stateDimension;
  }

  auto evaluate = [&](const CuDensityMatState &input, CuDensityMatState &output,
                      double time) {
    auto params = cudmIntHelp::scheduleParamsAt(m_schedule, time);
    if (!cudmStepper->overwritesOutput(input.getBatchSize()))
      deviceOps.clear(output.get_device_pointer());
    cudmStepper->computeImpl(input.get_impl(), output.get_impl(), time, params,
                             input.getBatchSize(), input.get_device_pointer(),
                             output.get_device_pointer());
  };

  // TorchDiffeq evaluates f(t0, y0) once, then reuses the final stage of every
  // accepted Dormand-Prince step as the first stage of the next step (FSAL).
  evaluate(*current, *stages[0], m_t);

  // Guard against runaway step rejection (e.g. dt driven to dt_min).
  constexpr std::size_t MAX_ITERATIONS = 100000;
  std::size_t iterations = 0;

  while (m_t < targetTime) {
    if (++iterations > MAX_ITERATIONS)
      throw std::runtime_error(
          "dopri5 integrator exceeded maximum iterations; possible "
          "convergence issue or step size driven below dt_min.");

    const double dt = std::min(m_dt, targetTime - m_t);

    // Stage zero is the cached derivative. Each remaining stage is assembled
    // with one fused device kernel and evaluated into a preallocated buffer.
    for (int j = 1; j < 7; ++j) {
      std::array<const void *, 6> stagePointers{};
      for (int i = 0; i < j; ++i)
        stagePointers[i] = stages[i]->get_device_pointer();
      deviceOps.combineStage(candidate->get_device_pointer(),
                             current->get_device_pointer(), stagePointers,
                             kA[j], dt);
      evaluate(*candidate, *stages[j], m_t + kNodes[j] * dt);
    }

    // The seventh stage input is also the fifth-order solution because the
    // final tableau row equals the solution weights. Compute TorchDiffeq's
    // Shampine error estimate directly from the stage derivatives on device.
    std::array<const void *, 7> stagePointers;
    for (int j = 0; j < 7; ++j)
      stagePointers[j] = stages[j]->get_device_pointer();

    double errorNorm;
    {
      cudaq::dynamics::PerfMetricScopeTimer errorTimer(
          "dopri5::error_norm.device");
      errorNorm = deviceOps.errorRatio(current->get_device_pointer(),
                                       candidate->get_device_pointer(),
                                       stagePointers, dt, m_rtol, m_atol);
      if (isDistributed) {
        const double localErrorSum =
            errorNorm * errorNorm * static_cast<double>(elementCount);
        const double globalErrorSum =
            cudaq::mpi::all_reduce(localErrorSum, std::plus<double>());
        errorNorm =
            std::sqrt(globalErrorSum / static_cast<double>(globalElementCount));
      }
    }
    const double dtNext = adaptStepSize(errorNorm, dt, m_dt_min, m_dt_max);
    const bool accept = errorNorm <= 1.0 || dt <= m_dt_min;

    if (accept) {
      std::swap(current, candidate);
      // k7 = f(t + dt, y5) becomes the next step's k1. The old k1 buffer is
      // recycled for the next k7 output.
      std::swap(stages[0], stages[6]);
      m_t += dt;
      m_dt = dtNext;
      ++m_stats.accepted_steps;
      m_stats.min_dt_used = std::min(m_stats.min_dt_used, dt);
      m_stats.max_dt_used = std::max(m_stats.max_dt_used, dt);
      m_stats.avg_dt = (m_stats.avg_dt * (m_stats.accepted_steps - 1) + dt) /
                       m_stats.accepted_steps;
    } else {
      m_dt = dtNext;
      ++m_stats.rejected_steps;
    }
  }

  m_state = std::make_shared<cudaq::state>(current.release());
}

} // namespace cudaq::integrators
