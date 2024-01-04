/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target remote-sim --remote-sim-auto-launch 1 %s -o %t && %t 
// RUN: nvq++ --enable-mlir --target remote-sim --remote-sim-auto-launch 1 %s -o %t && %t 
// clang-format on

#include <cudaq.h>

template <std::size_t N>
struct ghz {
  auto operator()() __qpu__ {
    cudaq::qarray<N> q;
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  auto kernel = ghz<10>{};
  auto counts = cudaq::sample(kernel);
  counts.dump();
#ifndef SYNTAX_CHECK
  assert(counts.size() == 2);  
#endif
  return 0;
}