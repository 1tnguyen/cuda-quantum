/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <gtest/gtest.h>
#include <random>

TEST(MQPUTester, checkSimple) {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  double result = cudaq::observe<cudaq::parallel::thread>(ansatz, h, 0.59);
  EXPECT_NEAR(result, -1.7487, 1e-3);
  printf("Get energy directly as double %.16lf\n", result);
}

TEST(MQPUTester, checkLarge) {

  // This will warm up the GPUs, we don't time this
  auto &platform = cudaq::get_platform();
  printf("Num QPUs %lu\n", platform.num_qpus());
  int nQubits = 12;
  int nTerms = 1000; /// Scale this on multiple gpus to see speed up
  auto H = cudaq::spin_op::random(nQubits, nTerms, std::mt19937::default_seed);

  printf("Total Terms = %lu\n", H.num_terms());
  auto kernel = [](const int n_qubits, const int layers,
                   std::vector<int> cnot_pairs,
                   std::vector<double> params) __qpu__ {
    // Allocate the qubits
    cudaq::qvector q(n_qubits);

    // We can only handle 1d vectors so
    // count the params manually
    int param_counter = 0;
    for (int i = 0; i < n_qubits; i++) {
      rx(params[param_counter], q[i]);
      rz(params[param_counter + 1], q[i]);
      param_counter = param_counter + 2;
    }

    for (std::size_t i = 0; i < cnot_pairs.size(); i += 2) {
      x<cudaq::ctrl>(q[cnot_pairs[i]], q[cnot_pairs[i + 1]]);
    }

    // Apply layers of rotation+entangling
    for (int i = 1; i < layers; i++) {
      // Apply rotation layer
      for (int j = 0; j < n_qubits; j++) {
        rz(params[param_counter], q[j]);
        rx(params[param_counter + 1], q[j]);
        rz(params[param_counter + 2], q[j]);
        param_counter = param_counter + 3;
      }

      // Apply entangling layer
      for (std::size_t p = 0; p < cnot_pairs.size(); p += 2) {
        x<cudaq::ctrl>(q[cnot_pairs[i]], q[cnot_pairs[i + 1]]);
      }
    }
  };

  int nLayers = 2;
  auto execParams = cudaq::random_vector(
      -M_PI, M_PI, nQubits * (3 * nLayers + 2), std::mt19937::default_seed);

  std::vector<int> cnot_pairs(nQubits);
  std::iota(cnot_pairs.begin(), cnot_pairs.end(), 0);
  std::mt19937 g{std::mt19937::default_seed + 1};
  std::shuffle(cnot_pairs.begin(), cnot_pairs.end(), g);

  auto t1 = std::chrono::high_resolution_clock::now();
  cudaq::observe<cudaq::parallel::thread>(kernel, H, nQubits, nLayers,
                                          cnot_pairs, execParams);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> ms_double = t2 - t1;
  printf("Time %lf s\n", ms_double.count() * 1e-3);
}

TEST(MQPUTester, checkAsyncWithKernelBuilder) {
  auto [kernel, numIters] = cudaq::make_kernel<int>();
  constexpr std::size_t numQubits = 1;
  auto qubits = kernel.qalloc(numQubits);
  auto theta = 0.2;
  const auto rotateStep = [&](auto index) {
    for (std::size_t i = 0; i < numQubits; ++i) {
      kernel.rx(theta, qubits[i]);
    };
  };
  kernel.for_loop(0, numIters, rotateStep);
  auto &platform = cudaq::get_platform();
  int numSteps = 1;
  // Query the number of QPUs in the system
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);
  std::vector<cudaq::async_state_result> stateFutures;
  // QPU 0: 1 step, QPU 1: 2 steps, etc.
  for (std::size_t i = 0; i < num_qpus; i++) {
    stateFutures.emplace_back(cudaq::get_state_async(i, kernel, numSteps));
    numSteps++;
  }

  auto angle = 0.0;
  for (auto &stateFutures : stateFutures) {
    // Each run add 0.2 rad to the rotation
    angle += 0.2;
    const std::complex<double> expectedState[2] = {{std::cos(angle / 2), 0.0},
                                                   {0.0, -std::sin(angle / 2)}};
    auto gotState = stateFutures.get();
    EXPECT_NEAR(std::abs(gotState[0] - expectedState[0]), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(gotState[1] - expectedState[1]), 0.0, 1e-6);
  }
}

struct iqft {
  void operator()(cudaq::qview<> &q) __qpu__ {
    int N = q.size();
    // Swap qubits
    for (int i = 0; i < N / 2; ++i) {
      swap(q[i], q[N - i - 1]);
    }

    for (int i = 0; i < N - 1; ++i) {
      h(q[i]);
      int j = i + 1;
      for (int y = i; y >= 0; --y) {
        const double theta = -M_PI / std::pow(2.0, j - y);
        r1<cudaq::ctrl>(theta, q[j], q[y]);
      }
    }

    h(q[N - 1]);
  }
};

struct qpe {
  double operator()(const int n_c, const int n_q) __qpu__ {
    // Allocate a register of qubits
    cudaq::qvector q(n_c + n_q);

    // Extract sub-registers, one for the counting qubits
    // another for the eigenstate register
    auto counting_qubits = q.front(n_c);

    auto &state_register = q.back();

    // Prepare the eigenstate
    x(state_register);

    // Put the counting register into uniform superposition
    h(counting_qubits);

    // Perform ctrl-U^j
    for (int i = 0; i < n_c; ++i) {
      for (int j = 0; j < std::pow(2, i); ++j) {
        t<cudaq::ctrl>(counting_qubits[i], state_register);
      }
    }

    // Apply inverse quantum fourier transform
    iqft{}(counting_qubits);
    // Measure and compute the phase...
    return cudaq::to_integer(mz(counting_qubits)) / std::pow(2, n_c);
  }
};

TEST(MQPUTester, checkRunAsync) {
  // Query the number of QPUs in the system
  auto &platform = cudaq::get_platform();
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);

  for (int qpu_id = 0; qpu_id < num_qpus; ++qpu_id) {
    const int numCountingQubits = 3 + qpu_id;
    auto phases_future =
        cudaq::run_async(qpu_id, 1, qpe{}, numCountingQubits, 1);
    for (const auto &phase : phases_future.get()) {
      printf("Phase = %lf\n", phase.get());
      EXPECT_TRUE(phase.isOk());
      EXPECT_NEAR(phase.get(), .125, 1e-4);
    }
  }
}
