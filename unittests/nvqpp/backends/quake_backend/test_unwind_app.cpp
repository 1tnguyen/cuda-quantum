/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// Expected nop payload: one structured loop and one non-local return. The
// server validates these annotations before performing backend lowering.
__qpu__ int
unwind_return_expected_1_unwind_expected_1_if_expected_1_loop_expected_0_cfg(
    int stop) {
  cudaq::qubit q;

  for (int i = 0; i < 3; ++i) {
    x(q);
    if (i == stop)
      return 1;
  }
  return 2;
}

int main() {
  auto results = cudaq::run(
      1,
      unwind_return_expected_1_unwind_expected_1_if_expected_1_loop_expected_0_cfg,
      1);
  return results.size() == 1 && results[0] == 1 ? 0 : 1;
}
