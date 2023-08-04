/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ObserveExperimentSetup.h"
#include "MeasureCounts.h"
#include <assert.h>

namespace cudaq {

void observe_experiment_setup::add_result_mapping_for_term(
    const spin_op::spin_op_term &term, const std::string &result_key,
    const std::vector<std::size_t> &bit_map, double sign) {
  assert(result_map.find(term) == result_map.end() && "Duplicate observe term");
  result_map[term] = std::make_tuple(result_key, bit_map, sign);
}

double observe_experiment_setup::retrieve_term_expectation(
    const spin_op::spin_op_term &term, sample_result &result) const {
  const auto term_iter = result_map.find(term);
  if (term_iter == result_map.end())
    throw std::logic_error(spin_op(term, 1.0).to_string(false) +
                           " is not part of this observe_experiment_setup.");

  const auto &[key, bit_map, sign] = term_iter->second;
  const auto all_keys = result.register_names();
  if (std::find(all_keys.begin(), all_keys.end(), key) == all_keys.end())
    throw std::logic_error(std::string("Result register named '") + key +
                           "' for the term " +
                           spin_op(term, 1.0).to_string(false) +
                           " cannot be retrieved from the sample_result.");
  auto counts = result.get_marginal(bit_map, key);
  const auto exp_val = counts.exp_val_z();
  return sign * exp_val;
}
double observe_experiment_setup::compute_exp_val(const spin_op &ham,
                                                 sample_result &result) const {
  double sum = 0.0;
  ham.for_each_term([&](cudaq::spin_op &term) {
    if (term.is_identity())
      sum += term.get_coefficient().real();
    else {
      assert(term.num_terms() == 1 && "Expected a single term spin_op");
      const double term_exp_val_from_group =
          retrieve_term_expectation(term.get_raw_data().first.front(), result);
      sum += (term.get_coefficient().real() * term_exp_val_from_group);
    }
  });
  return sum;
}
} // namespace cudaq
