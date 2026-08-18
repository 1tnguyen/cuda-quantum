/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudaq::dynamics {

class CuSparseStateVectorRhs {
public:
  CuSparseStateVectorRhs(std::size_t dimension,
                         std::vector<std::int32_t> rowOffsets,
                         std::vector<std::int32_t> columnIndices,
                         std::vector<std::complex<double>> values);
  ~CuSparseStateVectorRhs();

  CuSparseStateVectorRhs(const CuSparseStateVectorRhs &) = delete;
  CuSparseStateVectorRhs &operator=(const CuSparseStateVectorRhs &) = delete;

  void compute(void *input, void *output);

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace cudaq::dynamics
