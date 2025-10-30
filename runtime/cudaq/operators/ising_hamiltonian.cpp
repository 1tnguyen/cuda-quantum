/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <sstream>
#include <stdexcept>

namespace cudaq {
annealing_hamiltonian::annealing_hamiltonian(
    const std::vector<cudaq::sum_op<cudaq::spin_handler>> &hamiltonians,
    const std::vector<scalar_operator> &scalingFactors)
    : spin_hamiltonians(hamiltonians), scaling_factors(scalingFactors) {
  if (spin_hamiltonians.size() != scaling_factors.size())
    throw std::runtime_error(
        "The number of spin Hamiltonians must match the number of scaling "
        "factors.");

  if (hamiltonians.empty())
    throw std::runtime_error("At least one spin Hamiltonian must be provided.");

  for (const auto &spin_op : spin_hamiltonians)
    validate_spin_hamiltonian(spin_op);
}

const std::vector<cudaq::sum_op<cudaq::spin_handler>> &
annealing_hamiltonian::get_spin_hamiltonians() const {
  return spin_hamiltonians;
}
const std::vector<scalar_operator> &
annealing_hamiltonian::get_scaling_factors() const {
  return scaling_factors;
}

void annealing_hamiltonian::validate_spin_hamiltonian(
    const cudaq::sum_op<cudaq::spin_handler> &spin_op) const {
  for (const auto &product_op : spin_op) {

    auto coeff = product_op.get_coefficient();
    // Coefficient must be constant
    if (!coeff.is_constant())
      throw std::runtime_error(
          "Annealing Hamiltonian terms must have constant coefficients.");
    // The coefficient must be real
    if (std::imag(coeff.evaluate({})) != 0.0)
      throw std::runtime_error(
          "Annealing Hamiltonian terms must have real coefficients.");

    // Terms must be 2-body or less
    if (product_op.num_ops() > 2)
      throw std::runtime_error(
          "Annealing Hamiltonian terms must be 2-body or less.");
  }
}

std::vector<std::pair<std::string, std::unordered_map<std::string, double>>>
annealing_hamiltonian::get_spin_hamiltonians_representation() const {
  std::vector<std::pair<std::string, std::unordered_map<std::string, double>>>
      representation;

  for (std::size_t i = 0; i < spin_hamiltonians.size(); ++i) {
    const auto &spin_op = spin_hamiltonians[i];
    std::unordered_map<std::string, double> terms;
    for (const auto &product_op : spin_op) {
      auto coeff = product_op.get_coefficient().evaluate({});
      terms[product_op.get_term_id()] = std::real(coeff);
    }

    representation.emplace_back("H_" + std::to_string(i), std::move(terms));
  }

  return representation;
}

std::vector<std::unordered_map<std::string, std::vector<double>>>
annealing_hamiltonian::get_annealing_schedule(cudaq::schedule &schedule) const {
  std::unordered_map<std::string, std::vector<double>> schedule_data;
  for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
    auto &factor = scaling_factors[i];
    std::vector<double> values;
    for (const auto &step : schedule) {
      std::unordered_map<std::string, std::complex<double>> params;
      for (const auto &param : schedule.get_parameters()) {
        params[param] = schedule.get_value_function()(param, step);
      }
      values.emplace_back(std::real(factor.evaluate(params)));
    }
    schedule.reset();
    schedule_data["H_" + std::to_string(i)] = std::move(values);
  }
  return {schedule_data};
}

scalar_operator annealing_hamiltonian::create_linear_field(
    cudaq::schedule &time_schedule, double initial_value, double final_value,
    const std::string &time_parameter_name) {
  const double tmax = (time_schedule.end() - 1)->real();
  if (std::find(time_schedule.get_parameters().begin(),
                time_schedule.get_parameters().end(),
                time_parameter_name) == time_schedule.get_parameters().end())
    throw std::runtime_error("The time schedule does not contain the time "
                             "parameter '" +
                             time_parameter_name + "'.");

  auto callback =
      [initial_value, final_value, time_parameter_name,
       tmax](const std::unordered_map<std::string, std::complex<double>>
                 &parameters) {
        // Evaluate the linear field based on the time parameter
        const double t = std::real(parameters.at(time_parameter_name));
        return initial_value + (final_value - initial_value) * t / tmax;
      };

  return scalar_operator(std::move(callback));
}

scalar_operator annealing_hamiltonian::create_piecewise_linear_field(
    cudaq::schedule &time_schedule,
    const std::vector<std::pair<double, double>> &fixed_points,
    const std::string &time_parameter_name) {
  const double tmax = (time_schedule.end() - 1)->real();
  if (std::find(time_schedule.get_parameters().begin(),
                time_schedule.get_parameters().end(),
                time_parameter_name) == time_schedule.get_parameters().end())
    throw std::runtime_error("The time schedule does not contain the time "
                             "parameter '" +
                             time_parameter_name + "'.");
  // Check that fixed points are sorted by time
  for (std::size_t i = 1; i < fixed_points.size(); ++i) {
    if (fixed_points[i].first <= fixed_points[i - 1].first)
      throw std::runtime_error(
          "Fixed points must be sorted in increasing order of time.");
  }
  auto callback =
      [fixed_points, time_parameter_name,
       tmax](const std::unordered_map<std::string, std::complex<double>>
                 &parameters) {
        // Evaluate the piecewise linear field based on the time parameter
        const double t = std::real(parameters.at(time_parameter_name));
        // Handle boundary cases
        if (t <= fixed_points.front().first)
          return fixed_points.front().second;
        if (t >= fixed_points.back().first)
          return fixed_points.back().second;
        // Find the interval [t_i, t_{i+1}] that contains t
        for (std::size_t i = 0; i < fixed_points.size() - 1; ++i) {
          if (t >= fixed_points[i].first && t <= fixed_points[i + 1].first) {
            // Linear interpolation
            const double t0 = fixed_points[i].first;
            const double v0 = fixed_points[i].second;
            const double t1 = fixed_points[i + 1].first;
            const double v1 = fixed_points[i + 1].second;
            const double slope = (v1 - v0) / (t1 - t0);
            return v0 + slope * (t - t0);
          }
        }
        throw std::runtime_error("Internal error: unreachable condition when "
                                 "evaluating piecewise linear field.");
      };

  return scalar_operator(std::move(callback));
}
} // namespace cudaq
