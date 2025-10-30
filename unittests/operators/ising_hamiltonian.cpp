/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved. *
 * *
 * This source code and the accompanying materials are made available under *
 * the terms of the Apache License 2.0 which accompanies this distribution. *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <gtest/gtest.h>

TEST(IsingHamiltonianTest, ConstructorValidInputs) {
  {
    auto H1 = 0.1 * cudaq::spin_op::x(0) + 0.2 * cudaq::spin_op::x(1) +
              0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
    auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
              0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);

    cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2}, {cudaq::scalar_operator(1.0), cudaq::scalar_operator(2.0)});

    EXPECT_EQ(hamiltonian.get_spin_hamiltonians().size(), 2);
    EXPECT_EQ(hamiltonian.get_scaling_factors().size(), 2);
  }
  {
    // Spin simplification should be active
    // Note: product of pauli operators will produce an `i` factor, so must
    // cancel it out with an explicit `i`.
    auto H1 = 0.1 * std::complex<double>(0.0, 1.0) * cudaq::spin_op::x(0) *
                  cudaq::spin_op::z(0) +
              0.2 * std::complex<double>(0.0, 1.0) * cudaq::spin_op::x(1) *
                  cudaq::spin_op::y(1) +
              0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
    auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
              0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1) *
                  cudaq::spin_op::y(0) * cudaq::spin_op::y(1);

    cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2}, {cudaq::scalar_operator(1.0), cudaq::scalar_operator(2.0)});

    EXPECT_EQ(hamiltonian.get_spin_hamiltonians().size(), 2);
    EXPECT_EQ(hamiltonian.get_scaling_factors().size(), 2);
  }
}

TEST(IsingHamiltonianTest, ConstructorInvalidInputs) {
  {
    auto H1 = 0.1 * cudaq::spin_op::x(0) + 0.2 * cudaq::spin_op::x(1) +
              0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
    auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
              0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);

    // Wrong number of scaling factors (3 vs 2)
    EXPECT_ANY_THROW(cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2}, {cudaq::scalar_operator(1.0), cudaq::scalar_operator(2.0),
                   cudaq::scalar_operator(3.0)}));
  }

  {
    // Complex coefficient
    auto H1 = 0.1 * std::complex<double>(0.0, 1.0) * cudaq::spin_op::x(0) +
              0.2 * cudaq::spin_op::x(1) +
              0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
    auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
              0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);

    EXPECT_ANY_THROW(cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2}, {cudaq::scalar_operator(1.0), cudaq::scalar_operator(2.0)}));
  }

  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("value");
    if (entry == parameters.end())
      throw std::runtime_error("'value' not defined in parameters");
    return entry->second;
  };

  {
    // Non-constant coefficient
    auto H1 = cudaq::scalar_operator(function) * cudaq::spin_op::x(0) +
              0.2 * cudaq::spin_op::x(1) +
              0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
    auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
              0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);

    EXPECT_ANY_THROW(cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2}, {cudaq::scalar_operator(1.0), cudaq::scalar_operator(2.0)}));
  }

  {
    auto H1 = 0.1 * cudaq::spin_op::x(0) + 0.2 * cudaq::spin_op::x(1) +
              0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) *
                  cudaq::spin_op::x(3); // 3-body term
    auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
              0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);

    EXPECT_ANY_THROW(cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2}, {cudaq::scalar_operator(1.0), cudaq::scalar_operator(2.0)}));
  }
}

TEST(IsingHamiltonianTest, HamiltonianRepresentation) {
  auto H1 = 0.1 * cudaq::spin_op::x(0) + 0.2 * cudaq::spin_op::x(1) +
            0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
  auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
            0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);

  cudaq::annealing_hamiltonian hamiltonian(
      {H1, H2}, {cudaq::scalar_operator(1.0), cudaq::scalar_operator(2.0)});
  auto repr = hamiltonian.get_spin_hamiltonians_representation();
  EXPECT_EQ(repr.size(), 2);
  auto &[termId1, terms1] = repr[0];
  EXPECT_EQ(terms1.size(), 3);
  EXPECT_TRUE(terms1.find("X0") != terms1.end());
  EXPECT_NEAR(terms1["X0"], 0.1, 1e-6);
  EXPECT_TRUE(terms1.find("X1") != terms1.end());
  EXPECT_NEAR(terms1["X1"], 0.2, 1e-6);
  EXPECT_TRUE(terms1.find("X0X1") != terms1.end());
  EXPECT_NEAR(terms1["X0X1"], 0.3, 1e-6);

  auto &[termId2, terms2] = repr[1];
  EXPECT_EQ(terms2.size(), 3);
  EXPECT_TRUE(terms2.find("Z0") != terms2.end());
  EXPECT_NEAR(terms2["Z0"], 0.1, 1e-6);
  EXPECT_TRUE(terms2.find("Z1") != terms2.end());
  EXPECT_NEAR(terms2["Z1"], 0.2, 1e-6);
  EXPECT_TRUE(terms2.find("Z0Z1") != terms2.end());
  EXPECT_NEAR(terms2["Z0Z1"], 0.3, 1e-6);
}

TEST(IsingHamiltonianTest, ScheduleRepresentation) {
  auto H1 = 0.1 * cudaq::spin_op::x(0) + 0.2 * cudaq::spin_op::x(1) +
            0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
  auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
            0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);
  constexpr double tmax = 100.0;
  const std::vector<double> steps = cudaq::linspace(0.0, tmax, 101);
  cudaq::schedule schedule(steps, {"t"});
  {
    // Constant schedules
    cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2}, {cudaq::scalar_operator(1.0), cudaq::scalar_operator(2.0)});
    const auto annealing_schedule =
        hamiltonian.get_annealing_schedule(schedule);
    EXPECT_EQ(annealing_schedule.size(), 1);
    const auto &schedule = annealing_schedule[0];
    EXPECT_EQ(schedule.size(), 2);
    EXPECT_TRUE(schedule.find("H_0") != schedule.end());
    const auto &values1 = schedule.at("H_0");
    EXPECT_EQ(values1.size(), steps.size());
    for (const auto &val : values1)
      EXPECT_NEAR(val, 1.0, 1e-6);
    EXPECT_TRUE(schedule.find("H_1") != schedule.end());
    const auto &values2 = schedule.at("H_1");
    EXPECT_EQ(values2.size(), steps.size());
    for (const auto &val : values2)
      EXPECT_NEAR(val, 2.0, 1e-6);
  }
  {
    // Time-dependent schedules
    cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2},
        {cudaq::scalar_operator(
             [tmax](const std::unordered_map<std::string, std::complex<double>>
                        &parameters) -> std::complex<double> {
               const double t = std::real(parameters.at("t"));
               // ramp down from 1.0 to 0.0
               return 1.0 - t / tmax;
             }),
         cudaq::scalar_operator(
             [tmax](const std::unordered_map<std::string, std::complex<double>>
                        &parameters) -> std::complex<double> {
               const double t = std::real(parameters.at("t"));
               // ramp up from 0.0 to 1.0
               return t / tmax;
             })});
    const auto annealing_schedule =
        hamiltonian.get_annealing_schedule(schedule);
    EXPECT_EQ(annealing_schedule.size(), 1);
    const auto &schedule = annealing_schedule[0];
    EXPECT_EQ(schedule.size(), 2);
    EXPECT_TRUE(schedule.find("H_0") != schedule.end());
    const auto &values1 = schedule.at("H_0");
    EXPECT_EQ(values1.size(), steps.size());
    for (std::size_t i = 0; i < steps.size(); ++i) {
      const double expected_value = 1.0 - steps[i] / tmax;
      EXPECT_NEAR(values1[i], expected_value, 1e-6);
    }
    EXPECT_TRUE(schedule.find("H_1") != schedule.end());
    const auto &values2 = schedule.at("H_1");
    EXPECT_EQ(values2.size(), steps.size());
    for (std::size_t i = 0; i < steps.size(); ++i) {
      const double expected_value = steps[i] / tmax;
      EXPECT_NEAR(values2[i], expected_value, 1e-6);
    }
  }
}

TEST(IsingHamiltonianTest, LinearFieldSchedule) {
  auto H1 = 0.1 * cudaq::spin_op::x(0) + 0.2 * cudaq::spin_op::x(1) +
            0.3 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1);
  auto H2 = 0.1 * cudaq::spin_op::z(0) + 0.2 * cudaq::spin_op::z(1) +
            0.3 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1);
  constexpr double tmax = 100.0;
  const std::vector<double> steps = cudaq::linspace(0.0, tmax, 101);
  cudaq::schedule schedule(steps, {"t"});
  {
    cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2},
        {cudaq::annealing_hamiltonian::create_linear_field(schedule, 0.0, 5.0),
         cudaq::annealing_hamiltonian::create_linear_field(schedule, 10.0,
                                                           0.0)});
    const auto annealing_schedule =
        hamiltonian.get_annealing_schedule(schedule);
    EXPECT_EQ(annealing_schedule.size(), 1);
    const auto &schedule = annealing_schedule[0];
    EXPECT_EQ(schedule.size(), 2);
    EXPECT_TRUE(schedule.find("H_0") != schedule.end());
    const auto &values1 = schedule.at("H_0");
    EXPECT_EQ(values1.size(), steps.size());
    for (std::size_t i = 0; i < steps.size(); ++i) {
      const double expected_value = (5.0 / tmax) * steps[i];
      EXPECT_NEAR(values1[i], expected_value, 1e-6);
    }
    EXPECT_TRUE(schedule.find("H_1") != schedule.end());
    const auto &values2 = schedule.at("H_1");
    EXPECT_EQ(values2.size(), steps.size());
    for (std::size_t i = 0; i < steps.size(); ++i) {
      const double expected_value = 10.0 - (10.0 / tmax) * steps[i];
      EXPECT_NEAR(values2[i], expected_value, 1e-6);
    }
  }

  {
    // piecewise linear with only one points
    cudaq::annealing_hamiltonian hamiltonian(
        {H1, H2}, {cudaq::annealing_hamiltonian::create_piecewise_linear_field(
                       schedule, {{0.0, 0.0}, {50.0, 1.0}, {100.0, 2.0}}),
                   cudaq::annealing_hamiltonian::create_piecewise_linear_field(
                       schedule, {{0.0, 10.0}})});
    const auto annealing_schedule =
        hamiltonian.get_annealing_schedule(schedule);
    EXPECT_EQ(annealing_schedule.size(), 1);
    const auto &schedule = annealing_schedule[0];
    EXPECT_EQ(schedule.size(), 2);
    EXPECT_TRUE(schedule.find("H_0") != schedule.end());
    const auto &values1 = schedule.at("H_0");
    EXPECT_EQ(values1.size(), steps.size());
    // t = 0.0 -> 0.0, t = 50.0 -> 1.0, t = 100.0 -> 2.0
    for (std::size_t i = 0; i < steps.size(); ++i) {
      double expected_value;
      if (steps[i] <= 50.0)
        expected_value = (steps[i] / 50.0) * 1.0;
      else
        expected_value = 1.0 + ((steps[i] - 50.0) / 50.0) * 1.0;
      EXPECT_NEAR(values1[i], expected_value, 1e-6);
    }
    EXPECT_TRUE(schedule.find("H_1") != schedule.end());
    const auto &values2 = schedule.at("H_1");
    EXPECT_EQ(values2.size(), steps.size());
    for (std::size_t i = 0; i < steps.size(); ++i) {
      const double expected_value = 10.0;
      EXPECT_NEAR(values2[i], expected_value, 1e-6);
    }
  }
}
