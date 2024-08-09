/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// REQUIRES: c++20

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>
#include <iostream>

struct rwpe {
  double operator()() __qpu__ {
    int n_iter = 24;
    double mu = 0.7951, sigma = 0.6065;
    int iteration = 0;

    // Allocate the qubits
    cudaq::qvector q(2);

    // Alias them
    auto &aux = q.front();
    auto &target = q.back();

    x(q[1]);

    while (iteration < n_iter) {
      h(aux);
      rz(1.0 - (mu / sigma), aux);
      rz(.25 / sigma, target);
      x<cudaq::ctrl>(aux, target);
      rz(-.25 / sigma, target);
      x<cudaq::ctrl>(aux, target);
      h(aux);
      if (mz(aux)) {
        x(aux);
        mu += sigma * .6065;
      } else {
        mu -= sigma * .6065;
      }

      sigma *= .7951;
      iteration += 1;
    }

    return 2. * mu;
  }
};

int main() {
  cudaq::set_random_seed(123);
  auto phases = cudaq::run(10, rwpe{});
  for (const auto &phase : phases) {
    std::cout << "Phase: " << phase << "\n";
  }
  return 0;
}
