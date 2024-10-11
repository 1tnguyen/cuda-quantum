/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "ahs/program/program.h"
#include <gtest/gtest.h>

namespace {
// Sample payload
const std::string referenceJson = R"(
  {
  "setup": {
    "ahs_register": {
      "sites": [
        [
          "0.0",
          "0.0"
        ],
        [
          "0.0",
          "0.000003"
        ],
        [
          "0.0",
          "0.000006"
        ],
        [
          "0.000003",
          "0.0"
        ],
        [
          "0.000003",
          "0.000003"
        ],
        [
          "0.000003",
          "0.000003"
        ],
        [
          "0.000003",
          "0.000006"
        ]
      ],
      "filling": [
        1,
        1,
        1,
        1,
        1,
        0,
        0
      ]
    }
  },
  "hamiltonian": {
    "drivingFields": [
      {
        "amplitude": {
          "time_series": {
            "values": [
              "0.0",
              "25132700.0",
              "25132700.0",
              "0.0"
            ],
            "times": [
              "0.0",
              "3E-7",
              "0.0000027",
              "0.000003"
            ]
          },
          "pattern": "uniform"
        },
        "phase": {
          "time_series": {
            "values": [
              "0",
              "0"
            ],
            "times": [
              "0.0",
              "0.000003"
            ]
          },
          "pattern": "uniform"
        },
        "detuning": {
          "time_series": {
            "values": [
              "-125664000.0",
              "-125664000.0",
              "125664000.0",
              "125664000.0"
            ],
            "times": [
              "0.0",
              "3E-7",
              "0.0000027",
              "0.000003"
            ]
          },
          "pattern": "uniform"
        }
      }
    ],
    "localDetuning": [
      {
        "magnitude": {
          "time_series": {
            "values": [
              "-125664000.0",
              "125664000.0"
            ],
            "times": [
              "0.0",
              "0.000003"
            ]
          },
          "pattern": [
            "0.5",
            "1.0",
            "0.5",
            "0.5",
            "0.5",
            "0.5"
          ]
        }
      }
    ]
  }
}
  )";
} // namespace

CUDAQ_TEST(QueraTester, checkProgramJsonify) {
  cudaq::ahs::AtomArrangement atoms;
  atoms.sites = {{0.0, 0.0},      {0.0, 3.0e-6},    {0.0, 6.0e-6},
                 {3.0e-6, 0.0},   {3.0e-6, 3.0e-6}, {3.0e-6, 3.0e-6},
                 {3.0e-6, 6.0e-6}};
  atoms.filling = {1, 1, 1, 1, 1, 0, 0};
  cudaq::ahs::Program program;
  program.setup.ahs_register = atoms;

  cudaq::ahs::PhysicalField Omega;
  Omega.time_series = std::vector<std::pair<double, double>>{
      {0.0, 0.0}, {3.0e-7, 2.51327e7}, {2.7e-6, 2.51327e7}, {3.0e-6, 0.0}};

  cudaq::ahs::PhysicalField Phi;
  Phi.time_series =
      std::vector<std::pair<double, double>>{{0.0, 0.0}, {3.0e-6, 0.0}};

  cudaq::ahs::PhysicalField Delta;
  Delta.time_series =
      std::vector<std::pair<double, double>>{{0.0, -1.25664e8},
                                             {3.0e-7, -1.25664e8},
                                             {2.7e-6, 1.25664e8},
                                             {3.0e-6, 1.25664e8}};
  cudaq::ahs::DrivingField drive;
  drive.amplitude = Omega;
  drive.phase = Phi;
  drive.detuning = Delta;
  program.hamiltonian.drivingFields = {drive};

  cudaq::ahs::PhysicalField localDetuning;
  localDetuning.time_series = std::vector<std::pair<double, double>>{
      {0.0, -1.25664e8}, {3.0e-6, 1.25664e8}};
  localDetuning.pattern = std::vector<double>{0.5, 1.0, 0.5, 0.5, 0.5, 0.5};
  cudaq::ahs::LocalDetuning detuning;
  detuning.magnitude = localDetuning;

  program.hamiltonian.localDetuning = {detuning};
  nlohmann::json j = program;
  std::cout << j.dump(4) << std::endl;

  cudaq::ahs::Program refProgram = nlohmann::json::parse(referenceJson);
  EXPECT_EQ(refProgram.setup.ahs_register.sites,
            program.setup.ahs_register.sites);
  EXPECT_EQ(refProgram.setup.ahs_register.filling,
            program.setup.ahs_register.filling);
  EXPECT_EQ(refProgram.hamiltonian.drivingFields.size(),
            program.hamiltonian.drivingFields.size());
  EXPECT_EQ(refProgram.hamiltonian.localDetuning.size(),
            program.hamiltonian.localDetuning.size());

  const auto checkField = [](const cudaq::ahs::PhysicalField &field1,
                             const cudaq::ahs::PhysicalField &field2) {
    EXPECT_EQ(field1.pattern, field2.pattern);
    EXPECT_TRUE(field1.time_series.almostEqual(field2.time_series));
  };
  for (size_t i = 0; i < program.hamiltonian.drivingFields.size(); ++i) {
    auto refField = refProgram.hamiltonian.drivingFields[i];
    auto field = program.hamiltonian.drivingFields[i];
    checkField(refField.amplitude, field.amplitude);
    checkField(refField.phase, field.phase);
    checkField(refField.detuning, field.detuning);
  }
  for (size_t i = 0; i < program.hamiltonian.localDetuning.size(); ++i) {
    auto refField = refProgram.hamiltonian.localDetuning[i];
    auto field = program.hamiltonian.localDetuning[i];
    checkField(refField.magnitude, field.magnitude);
  }
}
