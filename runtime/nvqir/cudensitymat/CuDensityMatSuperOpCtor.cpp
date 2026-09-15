/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatOpConverter.h"
#include "CuDensityMatUtils.h"
#include "cudaq/runtime/logger/logger.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <ranges>
#include <set>

namespace {
constexpr auto localFusionMaxModesEnv =
    "CUDAQ_CUDM_LOCAL_LEFT_FUSION_MAX_MODES";

std::size_t getLocalFusionMaxModes() {
  const auto *value = std::getenv(localFusionMaxModesEnv);
  if (!value)
    return 0;

  char *end = nullptr;
  errno = 0;
  const auto parsed = std::strtol(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0' || parsed < 0 ||
      parsed > INT_MAX)
    throw std::invalid_argument(
        cudaq_fmt::format("{} must be a non-negative integer, got '{}'.",
                          localFusionMaxModesEnv, value));
  return static_cast<std::size_t>(parsed);
}

cudaq::product_op<cudaq::matrix_handler>
computeDagger(const cudaq::matrix_handler &op) {
  const std::string daggerOpName = op.to_string(false) + "_dagger";
  try {
    auto func = [op](const std::vector<int64_t> &dimensions,
                     const std::unordered_map<std::string, std::complex<double>>
                         &params) {
      cudaq::dimension_map dims;
      if (dimensions.size() != op.degrees().size())
        throw std::runtime_error("Dimension mismatched");

      for (std::size_t i = 0; i < dimensions.size(); ++i) {
        dims[op.degrees()[i]] = dimensions[i];
      }
      auto originalMat = op.to_matrix(dims, params);
      return originalMat.adjoint();
    };

    auto dia_func =
        [op](const std::vector<int64_t> &dimensions,
             const std::unordered_map<std::string, std::complex<double>>
                 &params) {
          cudaq::dimension_map dims;
          if (dimensions.size() != op.degrees().size())
            throw std::runtime_error("Dimension mismatched");

          for (std::size_t i = 0; i < dimensions.size(); ++i) {
            dims[op.degrees()[i]] = dimensions[i];
          }
          auto diaMat = op.to_diagonal_matrix(dims, params);
          for (auto &offset : diaMat.second)
            offset *= -1;
          for (auto &element : diaMat.first)
            element = std::conj(element);
          return diaMat;
        };
    cudaq::matrix_handler::define(daggerOpName, {-1}, std::move(func),
                                  std::move(dia_func));
  } catch (...) {
    // Nothing, this has been define
  }
  return cudaq::matrix_handler::instantiate(daggerOpName, op.degrees());
}

cudaq::scalar_operator computeDagger(const cudaq::scalar_operator &scalar) {
  if (scalar.is_constant()) {
    return cudaq::scalar_operator(std::conj(scalar.evaluate()));
  } else {
    return cudaq::scalar_operator(
        [scalar](
            const std::unordered_map<std::string, std::complex<double>> &params)
            -> std::complex<double> {
          return std::conj(scalar.evaluate(params));
        });
  }
}

cudaq::product_op<cudaq::matrix_handler>
computeDagger(const cudaq::product_op<cudaq::matrix_handler> &productOp) {
  std::vector<cudaq::product_op<cudaq::matrix_handler>> daggerOps;
  for (const auto &component : productOp) {
    if (const auto *elemOp =
            dynamic_cast<const cudaq::matrix_handler *>(&component)) {
      daggerOps.emplace_back(computeDagger(*elemOp));
    } else {
      throw std::runtime_error("Unhandled type!");
    }
  }
  std::reverse(daggerOps.begin(), daggerOps.end());

  if (daggerOps.empty())
    throw std::runtime_error("Empty product operator");
  cudaq::product_op<cudaq::matrix_handler> daggerProduct = daggerOps[0];
  for (std::size_t i = 1; i < daggerOps.size(); ++i) {
    daggerProduct *= daggerOps[i];
  }
  daggerProduct *= computeDagger(productOp.get_coefficient());
  return daggerProduct;
}

cudaq::sum_op<cudaq::matrix_handler>
computeDagger(const cudaq::sum_op<cudaq::matrix_handler> &sumOp) {
  cudaq::sum_op<cudaq::matrix_handler> daggerSum =
      cudaq::sum_op<cudaq::matrix_handler>::empty();
  for (const cudaq::product_op<cudaq::matrix_handler> &prodOp : sumOp)
    daggerSum += computeDagger(prodOp);

  return daggerSum;
}

} // namespace

std::vector<
    std::pair<std::vector<cudaq::scalar_operator>, cudensitymatOperatorTerm_t>>
cudaq::dynamics::CuDensityMatOpConverter::computeLindbladTerms(
    const std::vector<sum_op<cudaq::matrix_handler>> &batchedCollapseOps,
    const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  if (batchedCollapseOps.empty())
    return {};
  // Split the collapse operators into batched product terms.
  auto batchedCollapsedProdTerms = splitToBatch(batchedCollapseOps);
  std::vector<std::pair<std::vector<cudaq::scalar_operator>,
                        cudensitymatOperatorTerm_t>>
      lindbladTerms;

  for (const auto &collapseOp : batchedCollapsedProdTerms) {
    const auto allSameDegrees =
        std::all_of(collapseOp.begin(), collapseOp.end(),
                    [&](const product_op<matrix_handler> &prodOp) {
                      return prodOp.degrees() == collapseOp[0].degrees();
                    });
    if (!allSameDegrees) {
      throw std::invalid_argument("All product terms in a collapse operator "
                                  "must have the same degrees.");
    }
  }

  const auto batchedSize = batchedCollapsedProdTerms.size();
  const auto numberProductTerms = batchedCollapsedProdTerms[0].size();
  for (std::size_t leftProdTermIdx = 0; leftProdTermIdx < numberProductTerms;
       ++leftProdTermIdx) {
    for (std::size_t rightProdTermIdx = 0;
         rightProdTermIdx < numberProductTerms; ++rightProdTermIdx) {
      std::vector<product_op<matrix_handler>> l_ops;
      std::vector<product_op<matrix_handler>> r_ops;
      l_ops.reserve(batchedSize);
      r_ops.reserve(batchedSize);
      for (std::size_t i = 0; i < batchedSize; ++i) {
        l_ops.push_back(batchedCollapsedProdTerms[i][leftProdTermIdx]);
        r_ops.push_back(batchedCollapsedProdTerms[i][rightProdTermIdx]);
      }
      // L * rho * L_dagger
      {
        std::vector<scalar_operator> coeffs;
        coeffs.reserve(batchedSize);
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities;
        for (std::size_t i = 0; i < batchedSize; ++i) {
          coeffs.push_back(l_ops[i].get_coefficient() *
                           computeDagger(r_ops[i].get_coefficient()));
        }
        const auto leftNumOps = l_ops[0].num_ops();
        for (std::size_t i = 0; i < leftNumOps; ++i) {
          std::vector<cudaq::matrix_handler> leftOpComponents;
          for (const auto &leftOp : l_ops) {
            const auto &component = leftOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              leftOpComponents.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp = createElementaryOperator(leftOpComponents,
                                                     parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(l_ops[0][i].degrees());
          all_action_dual_modalities.emplace_back(
              std::vector<int>(l_ops[0][i].degrees().size(), 0));
        }
        auto ldags = r_ops;
        for (auto &ldag : ldags) {
          ldag = computeDagger(ldag);
        }
        const auto rightNumOps = ldags[0].num_ops();
        for (std::size_t i = 0; i < rightNumOps; ++i) {
          std::vector<cudaq::matrix_handler> rightOpComponents;
          for (const auto &rightOp : ldags) {
            const auto &component = rightOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              rightOpComponents.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp = createElementaryOperator(rightOpComponents,
                                                     parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(ldags[0][i].degrees());
          all_action_dual_modalities.emplace_back(
              std::vector<int>(ldags[0][i].degrees().size(), 1));
        }
        cudensitymatOperatorTerm_t D1_term = createProductOperatorTerm(
            elemOps, modeExtents, allDegrees, all_action_dual_modalities);
        lindbladTerms.emplace_back(std::make_pair(coeffs, D1_term));
      }

      std::vector<product_op<matrix_handler>> L_daggerTimesL;
      std::vector<scalar_operator> L_daggerTimesL_coeffs;
      L_daggerTimesL.reserve(batchedSize);
      L_daggerTimesL_coeffs.reserve(batchedSize);
      for (std::size_t i = 0; i < batchedSize; ++i) {
        // -0.5 * L_dagger * L
        auto ldag = computeDagger(r_ops[i]);
        auto l_op = l_ops[i];
        L_daggerTimesL.emplace_back(-0.5 * ldag * l_op);
        L_daggerTimesL_coeffs.push_back(-0.5 * l_ops[i].get_coefficient() *
                                        ldag.get_coefficient());
      }

      {
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities_left;
        std::vector<std::vector<int>> all_action_dual_modalities_right;

        const auto numOps = L_daggerTimesL[0].num_ops();
        for (std::size_t i = 0; i < numOps; ++i) {
          std::vector<cudaq::matrix_handler> components;
          for (const auto &prodOp : L_daggerTimesL) {
            const auto &component = prodOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              components.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp =
              createElementaryOperator(components, parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(L_daggerTimesL[0][i].degrees());
          all_action_dual_modalities_left.emplace_back(
              std::vector<int>(L_daggerTimesL[0][i].degrees().size(), 0));
          all_action_dual_modalities_right.emplace_back(
              std::vector<int>(L_daggerTimesL[0][i].degrees().size(), 1));
        }

        {
          // For left side, we need to reverse the order
          std::vector<cudensitymatElementaryOperator_t> d2Ops(elemOps);
          std::reverse(d2Ops.begin(), d2Ops.end());
          std::vector<std::vector<std::size_t>> d2Degrees(allDegrees);
          std::reverse(d2Degrees.begin(), d2Degrees.end());
          cudensitymatOperatorTerm_t D2_term = createProductOperatorTerm(
              d2Ops, modeExtents, d2Degrees, all_action_dual_modalities_left);
          lindbladTerms.emplace_back(
              std::make_pair(L_daggerTimesL_coeffs, D2_term));
        }
        {
          cudensitymatOperatorTerm_t D3_term =
              createProductOperatorTerm(elemOps, modeExtents, allDegrees,
                                        all_action_dual_modalities_right);
          lindbladTerms.emplace_back(
              std::make_pair(L_daggerTimesL_coeffs, D3_term));
        }
      }
    }
  }
  return lindbladTerms;
}

cudensitymatOperator_t
cudaq::dynamics::CuDensityMatOpConverter::constructLiouvillian(
    const std::vector<sum_op<cudaq::matrix_handler>> &hamOperators,
    const std::vector<std::vector<sum_op<cudaq::matrix_handler>>>
        &collapseOperators,
    const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool isMasterEquation, bool *requiresHermitianCompletion) {
  LOG_API_TIME();
  if (requiresHermitianCompletion)
    *requiresHermitianCompletion = false;
  if (hamOperators.empty()) {
    throw std::invalid_argument(
        "Cannot construct Liouvillian operator from an empty list of "
        "Hamiltonians.");
  }
  const auto batchSize = hamOperators.size();
  const auto numberProductTerms = hamOperators[0].num_terms();
  // Check if all Hamiltonians have the same number of product terms
  for (const auto &hamiltonian : hamOperators) {
    if (hamiltonian.num_terms() != numberProductTerms) {
      throw std::invalid_argument(
          "All Hamiltonians must have the same number of product terms.");
    }
  }

  const bool noCollapseOperators =
      collapseOperators.empty() ||
      std::all_of(collapseOperators.begin(), collapseOperators.end(),
                  [](const std::vector<sum_op<cudaq::matrix_handler>> &ops) {
                    return ops.empty();
                  });

  const auto fusionMaxModes = getLocalFusionMaxModes();
  if (isMasterEquation && batchSize == 1 && fusionMaxModes > 0 &&
      requiresHermitianCompletion) {
    if (!collapseOperators.empty() && collapseOperators.size() != batchSize)
      throw std::invalid_argument(
          "Collapse-operator batch size must match Hamiltonian batch size.");

    // For a Hermitian density matrix, the Lindblad equation can be evaluated
    // from one half-action and its adjoint:
    //   T = (-i H - 1/2 sum_j L_j^dagger L_j) rho
    //       + 1/2 sum_j L_j rho L_j^dagger,
    //   d rho / dt = T + T^dagger.
    // This cuts the effective-Hamiltonian work in half. The super-operator
    // converter below then fuses compatible static left-action terms up to the
    // locality cap. Parameter-dependent or wider terms use its exact ordinary
    // lowering.
    auto effectiveHamiltonian =
        hamOperators[0] * std::complex<double>(0.0, -1.0);
    const std::vector<sum_op<matrix_handler>> emptyCollapseOps;
    const auto &collapseOps =
        collapseOperators.empty() ? emptyCollapseOps : collapseOperators[0];
    for (const auto &collapseOp : collapseOps)
      effectiveHamiltonian += -0.5 * computeDagger(collapseOp) * collapseOp;

    auto halfLiouvillian = super_op::left_multiply(effectiveHamiltonian);
    for (const auto &collapseOp : collapseOps)
      halfLiouvillian += super_op::left_right_multiply(
          0.5 * collapseOp, computeDagger(collapseOp));

    *requiresHermitianCompletion = true;
    CUDAQ_INFO("Enabled local left-action fusion with a maximum of {} modes.",
               fusionMaxModes);
    return constructLiouvillian({halfLiouvillian}, modeExtents, parameters);
  }

  if (!isMasterEquation && noCollapseOperators) {
    CUDAQ_INFO("Construct state vector Liouvillian");
    std::vector<sum_op<cudaq::matrix_handler>> liouvillians;
    liouvillians.reserve(batchSize);
    for (const auto &ham : hamOperators) {
      liouvillians.emplace_back(ham * std::complex<double>(0.0, -1.0));
    }
    return convertToCudensitymatOperator(parameters, liouvillians, modeExtents);
  } else {
    CUDAQ_INFO("Construct density matrix Liouvillian");
    cudensitymatOperator_t liouvillian;
    HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
        m_handle, static_cast<int32_t>(modeExtents.size()), modeExtents.data(),
        &liouvillian));
    // Append an operator term to the operator (super-operator)
    // Handle the Hamiltonian
    const std::map<std::string, std::complex<double>> sortedParameters(
        parameters.begin(), parameters.end());
    auto ks = std::views::keys(sortedParameters);
    const std::vector<std::string> keys{ks.begin(), ks.end()};
    std::vector<sum_op<cudaq::matrix_handler>> leftHam;
    std::vector<sum_op<cudaq::matrix_handler>> rightHam;
    leftHam.reserve(batchSize);
    rightHam.reserve(batchSize);
    for (const auto &ham : hamOperators) {
      leftHam.emplace_back(ham * std::complex<double>(0.0, -1.0));
      rightHam.emplace_back(computeDagger(ham) *
                            std::complex<double>(0.0, 1.0));
    }
    // -i constant (left multiplication)
    appendToCudensitymatOperator(liouvillian, parameters, leftHam, modeExtents,
                                 /*duality=*/0);
    // +i constant (right multiplication, i.e., dual)
    appendToCudensitymatOperator(liouvillian, parameters, rightHam, modeExtents,
                                 /*duality=*/1);

    // Check that all collapsed operator vectors have the same size
    if (!collapseOperators.empty()) {
      const auto collapseSize = collapseOperators[0].size();
      for (const auto &collapseOperator : collapseOperators) {
        if (collapseOperator.size() != collapseSize) {
          throw std::invalid_argument(
              "All collapse operator vectors must have the same size.");
        }
      }
      // Handle collapsed operators
      for (std::size_t i = 0; i < collapseSize; ++i) {
        std::vector<sum_op<cudaq::matrix_handler>> batchedCollapseTerms;
        for (const auto &collapseOperator : collapseOperators) {
          batchedCollapseTerms.push_back(collapseOperator[i]);
        }
        for (auto &[coeffs, term] : computeLindbladTerms(
                 batchedCollapseTerms, modeExtents, parameters)) {
          assert(coeffs.size() == batchSize);
          appendBatchedTermToOperator(liouvillian, term, coeffs, keys);
        }
      }
    }

    return liouvillian;
  }
}

cudensitymatOperator_t
cudaq::dynamics::CuDensityMatOpConverter::constructLiouvillian(
    const std::vector<super_op> &superOps,
    const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  LOG_API_TIME();
  if (superOps.empty())
    throw std::invalid_argument(
        "Super-operator cannot be empty. At least one super-operator is "
        "required.");

  cudensitymatOperator_t liouvillian;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      m_handle, static_cast<int32_t>(modeExtents.size()), modeExtents.data(),
      &liouvillian));
  // Append an operator term to the operator (super-operator)
  // Handle the Hamiltonian
  const std::map<std::string, std::complex<double>> sortedParameters(
      parameters.begin(), parameters.end());
  auto ks = std::views::keys(sortedParameters);
  const std::vector<std::string> keys{ks.begin(), ks.end()};

  const auto batchSize = superOps.size();
  const auto hasLeftMultiplication = [](const super_op::term &term) {
    return term.first.has_value();
  };
  const auto hasRightMultiplication = [](const super_op::term &term) {
    return term.second.has_value();
  };
  for (std::size_t i = 1; i < batchSize; ++i) {
    if (superOps[i].num_terms() != superOps[0].num_terms()) {
      throw std::invalid_argument(
          "All super-operators in the batch must have the same number of "
          "terms.");
    }

    for (std::size_t j = 0; j < superOps[i].num_terms(); ++j) {
      if (hasLeftMultiplication(superOps[i][j]) !=
              hasLeftMultiplication(superOps[0][j]) ||
          hasRightMultiplication(superOps[i][j]) !=
              hasRightMultiplication(superOps[0][j])) {
        throw std::invalid_argument(
            "All super-operators in the batch must have the same structure");
      }
    }
  }

  // Greedily group compatible static left-action terms whose union support
  // fits the requested locality cap. Missing modes are padded with identities,
  // so each group becomes one dense local elementary operator. The feature is
  // disabled by default. Wider and parameter-dependent terms retain the exact
  // ordinary lowering below.
  const auto fusionMaxModes = getLocalFusionMaxModes();
  std::unordered_set<std::size_t> preLoweredTermIds;
  if (batchSize == 1 && fusionMaxModes > 0) {
    struct FusionInput {
      std::size_t termId;
      std::vector<std::size_t> support;
    };
    struct FusionGroup {
      std::vector<std::size_t> support;
      std::vector<std::size_t> termIds;
    };

    std::vector<FusionInput> inputs;
    inputs.reserve(superOps[0].num_terms());
    for (std::size_t termId = 0; termId < superOps[0].num_terms(); ++termId) {
      const auto &term = superOps[0][termId];
      if (!term.first.has_value() || term.second.has_value())
        continue;
      std::set<std::size_t> supportSet;
      if (!term.first->get_parameter_descriptions().empty())
        continue;
      const auto degrees = term.first->degrees();
      supportSet.insert(degrees.begin(), degrees.end());
      if (supportSet.empty() || supportSet.size() > fusionMaxModes)
        continue;
      inputs.push_back({termId, {supportSet.begin(), supportSet.end()}});
    }

    if (!inputs.empty()) {
      // Seed groups with the widest terms.  This lets one-body terms naturally
      // fold into nearby interaction blocks instead of consuming the locality
      // budget before the interactions are seen.
      std::stable_sort(inputs.begin(), inputs.end(),
                       [](const auto &a, const auto &b) {
                         return a.support.size() > b.support.size();
                       });
      std::vector<FusionGroup> groups;
      for (const auto &input : inputs) {
        std::size_t bestGroup = groups.size();
        std::size_t bestUnionSize = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> bestUnion;
        for (std::size_t groupId = 0; groupId < groups.size(); ++groupId) {
          std::vector<std::size_t> supportUnion;
          std::set_union(groups[groupId].support.begin(),
                         groups[groupId].support.end(), input.support.begin(),
                         input.support.end(), std::back_inserter(supportUnion));
          if (supportUnion.size() <= fusionMaxModes &&
              supportUnion.size() < bestUnionSize) {
            bestGroup = groupId;
            bestUnionSize = supportUnion.size();
            bestUnion = std::move(supportUnion);
          }
        }
        if (bestGroup == groups.size()) {
          groups.push_back({input.support, {input.termId}});
        } else {
          groups[bestGroup].support = std::move(bestUnion);
          groups[bestGroup].termIds.push_back(input.termId);
        }
      }

      cudaq::dimension_map dimensions;
      for (std::size_t degree = 0; degree < modeExtents.size(); ++degree)
        dimensions[degree] = modeExtents[degree];

      const auto embedMatrix =
          [&](const cudaq::product_op<cudaq::matrix_handler> &op,
              const std::vector<std::size_t> &support) {
            const auto opDegrees = op.degrees();
            const auto opMatrix = op.to_matrix(dimensions, parameters);
            std::size_t blockDim = 1;
            for (const auto degree : support)
              blockDim *= modeExtents[degree];
            std::vector<std::complex<double>> embedded(blockDim * blockDim,
                                                       {0.0, 0.0});
            std::vector<std::size_t> blockStrides(support.size(), 1);
            for (std::size_t i = 1; i < support.size(); ++i)
              blockStrides[i] =
                  blockStrides[i - 1] * modeExtents[support[i - 1]];
            std::vector<std::size_t> opStrides(opDegrees.size(), 1);
            for (std::size_t i = 1; i < opDegrees.size(); ++i)
              opStrides[i] = opStrides[i - 1] * modeExtents[opDegrees[i - 1]];

            for (std::size_t col = 0; col < blockDim; ++col) {
              for (std::size_t row = 0; row < blockDim; ++row) {
                bool identityMatches = true;
                std::size_t opRow = 0;
                std::size_t opCol = 0;
                std::size_t opPos = 0;
                for (std::size_t blockPos = 0; blockPos < support.size();
                     ++blockPos) {
                  const auto extent = modeExtents[support[blockPos]];
                  const auto rowDigit = (row / blockStrides[blockPos]) % extent;
                  const auto colDigit = (col / blockStrides[blockPos]) % extent;
                  if (opPos < opDegrees.size() &&
                      opDegrees[opPos] == support[blockPos]) {
                    opRow += rowDigit * opStrides[opPos];
                    opCol += colDigit * opStrides[opPos];
                    ++opPos;
                  } else if (rowDigit != colDigit) {
                    identityMatches = false;
                    break;
                  }
                }
                if (identityMatches)
                  embedded[row * blockDim + col] = opMatrix[{opRow, opCol}];
              }
            }
            return embedded;
          };

      for (const auto &group : groups) {
        std::size_t localDim = 1;
        std::vector<int64_t> fusedExtents;
        fusedExtents.reserve(group.support.size());
        for (const auto degree : group.support) {
          localDim *= modeExtents[degree];
          fusedExtents.push_back(modeExtents[degree]);
        }
        std::vector<std::complex<double>> localMatrix(localDim * localDim,
                                                      {0.0, 0.0});
        for (const auto termId : group.termIds) {
          const auto embedded =
              embedMatrix(*superOps[0][termId].first, group.support);
          for (std::size_t row = 0; row < localDim; ++row)
            for (std::size_t col = 0; col < localDim; ++col)
              localMatrix[col * localDim + row] +=
                  embedded[row * localDim + col];
          preLoweredTermIds.insert(termId);
        }

        auto *localMatrix_d = cudaq::dynamics::createArrayGpu(localMatrix);
        cudensitymatElementaryOperator_t localElem = nullptr;
        HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
            m_handle, static_cast<int32_t>(fusedExtents.size()),
            fusedExtents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0,
            nullptr, CUDA_C_64F, localMatrix_d, cudensitymatTensorCallbackNone,
            cudensitymatTensorGradientCallbackNone, &localElem));
        m_elementaryOperators.emplace(localElem);
        m_deviceBuffers.emplace(localMatrix_d);
        auto term = createProductOperatorTerm(
            {localElem}, modeExtents, {group.support},
            {std::vector<int>(group.support.size(), 0)});
        appendBatchedTermToOperator(liouvillian, term,
                                    {cudaq::scalar_operator(1.0)}, keys);
      }
      CUDAQ_INFO("Fused {} static left-action terms into {} local actions "
                 "(maximum {} modes).",
                 preLoweredTermIds.size(), groups.size(), fusionMaxModes);
    }
  }

  for (std::size_t termId = 0; termId < superOps[0].num_terms(); ++termId) {
    if (preLoweredTermIds.contains(termId))
      continue;
    std::vector<cudaq::product_op<cudaq::matrix_handler>> leftOps;
    std::vector<cudaq::product_op<cudaq::matrix_handler>> rightOps;
    leftOps.reserve(batchSize);
    rightOps.reserve(batchSize);
    for (std::size_t i = 0; i < batchSize; ++i) {
      if (superOps[i][termId].first.has_value())
        leftOps.push_back(superOps[i][termId].first.value());
      if (superOps[i][termId].second.has_value())
        rightOps.push_back(superOps[i][termId].second.value());
    }

    const auto allSameDegrees =
        [](const std::vector<cudaq::product_op<cudaq::matrix_handler>> &ops) {
          return std::all_of(
              ops.begin(), ops.end(),
              [&](const cudaq::product_op<cudaq::matrix_handler> &op) {
                return op.degrees() == ops[0].degrees();
              });
        };

    if (!leftOps.empty()) {
      if (!rightOps.empty()) {
        if (leftOps.size() != rightOps.size()) {
          throw std::invalid_argument(
              "Left and right product terms in a super-operator must have the "
              "same number of terms.");
        }
        if (!allSameDegrees(leftOps) || !allSameDegrees(rightOps)) {
          throw std::invalid_argument(
              "All product terms in a super-operator must have the same "
              "degrees.");
        }

        std::vector<cudaq::scalar_operator> coeffs;
        coeffs.reserve(batchSize);

        // cuDensityMat can apply a local L * rho * R product as one
        // elementary tensor by listing the same Hilbert-space mode twice,
        // once with ket modality and once with bra modality.  Preserve that
        // representation instead of lowering it to two full-state actions.
        // Start with the common non-batched, static, one-mode case. This is the
        // form used by local unitary jump operators such as X_i rho X_i. Like
        // left-action fusion, it is disabled when the locality cap is zero.
        const bool canFuseMixedDuality =
            fusionMaxModes > 0 && batchSize == 1 && leftOps[0].num_ops() == 1 &&
            rightOps[0].num_ops() == 1 && leftOps[0][0].degrees().size() == 1 &&
            leftOps[0][0].degrees() == rightOps[0][0].degrees() &&
            leftOps[0][0].get_parameter_descriptions().empty() &&
            rightOps[0][0].get_parameter_descriptions().empty();
        if (canFuseMixedDuality) {
          const auto &leftComponent = leftOps[0][0];
          const auto &rightComponent = rightOps[0][0];
          const auto *leftElem =
              dynamic_cast<const cudaq::matrix_handler *>(&leftComponent);
          const auto *rightElem =
              dynamic_cast<const cudaq::matrix_handler *>(&rightComponent);
          if (leftElem && rightElem) {
            cudaq::dimension_map dimensions;
            for (std::size_t degree = 0; degree < modeExtents.size(); ++degree)
              dimensions[degree] = modeExtents[degree];
            const auto leftMatrix = leftElem->to_matrix(dimensions, parameters);
            const auto rightMatrix =
                rightElem->to_matrix(dimensions, parameters);
            const auto localDim = leftMatrix.rows();
            if (leftMatrix.cols() == localDim &&
                rightMatrix.rows() == localDim &&
                rightMatrix.cols() == localDim) {
              const auto fusedDim = localDim * localDim;
              std::vector<std::complex<double>> fusedMatrix(fusedDim *
                                                            fusedDim);
              // Column-major storage of kron(leftMatrix, rightMatrix),
              // matching cuDensityMat's fused-noise examples.
              for (std::size_t leftCol = 0; leftCol < localDim; ++leftCol)
                for (std::size_t rightCol = 0; rightCol < localDim; ++rightCol)
                  for (std::size_t leftRow = 0; leftRow < localDim; ++leftRow)
                    for (std::size_t rightRow = 0; rightRow < localDim;
                         ++rightRow) {
                      const auto row = leftRow + localDim * rightRow;
                      const auto col = leftCol + localDim * rightCol;
                      fusedMatrix[col * fusedDim + row] =
                          leftMatrix[{leftRow, leftCol}] *
                          rightMatrix[{rightRow, rightCol}];
                    }

              auto *fusedMatrix_d =
                  cudaq::dynamics::createArrayGpu(fusedMatrix);
              const std::vector<int64_t> fusedExtents = {
                  static_cast<int64_t>(localDim),
                  static_cast<int64_t>(localDim)};
              cudensitymatElementaryOperator_t fusedElem = nullptr;
              HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
                  m_handle, 2, fusedExtents.data(),
                  CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr, CUDA_C_64F,
                  fusedMatrix_d, cudensitymatTensorCallbackNone,
                  cudensitymatTensorGradientCallbackNone, &fusedElem));
              m_elementaryOperators.emplace(fusedElem);
              m_deviceBuffers.emplace(fusedMatrix_d);

              const auto degree = leftElem->degrees()[0];
              auto term = createProductOperatorTerm(
                  {fusedElem}, modeExtents, {{degree, degree}}, {{0, 1}});
              coeffs.push_back(leftOps[0].get_coefficient() *
                               rightOps[0].get_coefficient());
              appendBatchedTermToOperator(liouvillian, term, coeffs, keys);
              continue;
            }
          }
        }

        // L * rho * R
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities;

        const auto leftNumOps = leftOps[0].num_ops();
        for (std::size_t i = 0; i < leftNumOps; ++i) {
          std::vector<cudaq::matrix_handler> leftOpComponents;
          for (const auto &leftOp : leftOps) {
            const auto &component = leftOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              leftOpComponents.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp = createElementaryOperator(leftOpComponents,
                                                     parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(leftOps[0][i].degrees());
          all_action_dual_modalities.emplace_back(
              std::vector<int>(leftOps[0][i].degrees().size(), 0));
        }

        const auto rightNumOps = rightOps[0].num_ops();
        for (std::size_t i = 0; i < rightNumOps; ++i) {
          std::vector<cudaq::matrix_handler> rightOpComponents;
          for (const auto &rightOp : rightOps) {
            const auto &component = rightOp[i];
            if (const auto *elemOp =
                    dynamic_cast<const cudaq::matrix_handler *>(&component)) {
              rightOpComponents.emplace_back(*elemOp);
            } else {
              // Catch anything that we don't know
              throw std::runtime_error("Unhandled type!");
            }
          }

          auto cudmElemOp = createElementaryOperator(rightOpComponents,
                                                     parameters, modeExtents);
          elemOps.emplace_back(cudmElemOp);
          allDegrees.emplace_back(rightOps[0][i].degrees());
          all_action_dual_modalities.emplace_back(
              std::vector<int>(rightOps[0][i].degrees().size(), 1));
        }

        for (std::size_t i = 0; i < batchSize; ++i) {
          coeffs.push_back(leftOps[i].get_coefficient() *
                           rightOps[i].get_coefficient());
        }

        cudensitymatOperatorTerm_t term = createProductOperatorTerm(
            elemOps, modeExtents, allDegrees, all_action_dual_modalities);
        appendBatchedTermToOperator(liouvillian, term, coeffs, keys);
      } else {
        std::vector<sum_op<cudaq::matrix_handler>> ops;
        ops.reserve(batchSize);
        for (const auto &leftOp : leftOps) {
          ops.emplace_back(sum_op<cudaq::matrix_handler>(leftOp));
        }
        const auto duality = 0; // No duality for left multiplication
        appendToCudensitymatOperator(liouvillian, parameters, ops, modeExtents,
                                     duality);
      }
    } else {
      if (!rightOps.empty()) {
        std::vector<sum_op<cudaq::matrix_handler>> ops;
        ops.reserve(batchSize);
        for (const auto &rightOp : rightOps) {
          ops.emplace_back(sum_op<cudaq::matrix_handler>(rightOp));
        }
        const auto duality = 1; // Duality for right multiplication
        appendToCudensitymatOperator(liouvillian, parameters, ops, modeExtents,
                                     duality);
      } else {
        throw std::runtime_error("Invalid super-operator term encountered: no "
                                 "operation action is specified.");
      }
    }
  }

  return liouvillian;
}
