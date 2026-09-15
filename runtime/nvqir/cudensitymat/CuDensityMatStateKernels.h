/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>

namespace cudaq::dynamics {
/// Complete an in-place, column-major matrix as matrix + matrix^dagger.
void completeHermitianMatrix(void *deviceData, std::size_t dimension);
} // namespace cudaq::dynamics
