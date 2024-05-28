/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --target ionq                     --emulate %s -o %t && %t 2>&1 | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate %s -o %t && %t 2>&1 | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t 2>&1 | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t 2>&1 | FileCheck %s
// clang-format on

#include <cudaq.h>

struct kernel {
  void operator()() __qpu__ {
    const std::vector<cudaq::complex> stateVector{1.0, 0.0, 0.0, 0.0};
    cudaq::qvector v(stateVector);
    h(v);
    mz(v);
  }
};

int main() {
  auto counts = cudaq::sample(kernel{});
  counts.dump();
  return 0;
}

// CHECK: state initialization is not supported
