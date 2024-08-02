/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/


#include <cudaq.h>
#include <iostream>
struct h_qec {
  void operator()(cudaq::qvector<>& q) __qpu__ {
    if (q.size() != 3) 
      throw std::invalid_argument("Must be 3 qubits");
    // QEC H
    h(q[0]);
    cx(q[0], q[1]);
    cx(q[0], q[2]);
  }
};

struct cx_qec {
  void operator()(cudaq::qvector<>& q1, cudaq::qvector<>& q2) __qpu__ {
    if (q1.size() != 3 || q2.size() != 3) 
      throw std::invalid_argument("Must be 3 qubits");
    x(q2[2]);
    for (int i =0; i < 3; ++i)
      cx(q1[i], q2[i]);
  }
};

struct measure_syndrome {
  int operator()(cudaq::qvector<>& q, cudaq::qvector<>& aux) __qpu__ {
    if (q.size() != 3) 
      throw std::invalid_argument("Must be 3 qubits");
    if (aux.size() != 2) 
      throw std::invalid_argument("Aux register must have 2 qubits");
    cx(q[0], aux[0]);
    cx(q[1], aux[0]);
    
    cx(q[0], aux[1]);
    cx(q[2], aux[1]);

    auto syndromes = mz(aux);
    reset(aux[0]);
    reset(aux[1]);
    if (!syndromes[0] && !syndromes[1])
      return 0;
    else if (syndromes[0] && syndromes[1]) 
     return 3;
    else if (syndromes[0])
      return 1;
    else 
    return 2;
  }
};

struct bell_qec {
  int operator()() __qpu__ {
    cudaq::qvector q1(3);
    cudaq::qvector q2(3);
    cudaq::qvector aux(2);
    // QEC H
    h_qec{}(q1);

    // QEC round
    if (measure_syndrome{}(q1, aux) != 0 || measure_syndrome{}(q2, aux) != 0)
      throw std::runtime_error("Failed QEC check");

    // QEC CX
    cx_qec{}(q1, q2);

    // QEC round
    if (measure_syndrome{}(q1, aux) != 0 || measure_syndrome{}(q2, aux) != 0)
      throw std::runtime_error("Failed QEC check");

    // Bell measurement
    auto bit1 = mz(q1[0]);
    auto bit2 = mz(q2[0]);

    if (!bit1 && !bit2)
      return 0;

    if (bit1 && bit2)
      return 1;

    throw std::runtime_error("Unreachable: corrupted entanglement due to noise!");
  }
};

int main() {
  cudaq::noise_model noise;
  constexpr double error_rate = 0.75;
  cudaq::depolarization_channel depolarization(error_rate);
  for (std::size_t i = 0; i < 8; ++i) 
    noise.add_channel<cudaq::types::h>({i}, depolarization);
  
  cudaq::set_noise(noise);
  
  auto results = cudaq::run(10, bell_qec{});
  for (const auto& result: results) {
    if (result.isOk())
      std::cout << "Bit = " << result.get() << "\n";
    else
      std::cout << "Error: " << result.getError() << "\n";
  }
}
