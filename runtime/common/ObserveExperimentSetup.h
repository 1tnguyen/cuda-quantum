/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/spin_op.h"
#include <unordered_map>

namespace cudaq {
class sample_result;
class observe_experiment_setup {
public:
  // sample_result indexing key + bit locations (for marginal bit strings) +
  // sign multiplication (in case a specialized diagonalizing circuit is
  // required)
  using term_result_lookup_t =
      std::tuple<std::string, std::vector<std::size_t>, double>;
  using term_result_map_t =
      std::unordered_map<spin_op::spin_op_term, term_result_lookup_t>;
  observe_experiment_setup(pauli_partition_strategy partition)
      : partition_scheme(partition){};
  void add_result_mapping_for_term(const spin_op::spin_op_term &term,
                                   const std::string &result_key,
                                   const std::vector<std::size_t> &bit_map,
                                   double sign = 1.0);
  double retrieve_term_expectation(const spin_op::spin_op_term &term,
                                   sample_result &result) const;
  pauli_partition_strategy get_partition_scheme() const {
    return partition_scheme;
  }

  /// Whether result mapping data has been populated.
  bool has_result_mapping() const { return !result_map.empty(); }

private:
  pauli_partition_strategy partition_scheme;
  term_result_map_t result_map;
};
} // namespace cudaq
