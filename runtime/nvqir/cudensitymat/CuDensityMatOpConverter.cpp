/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatOpConverter.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"
#include "common/Logger.h"
#include <iostream>
#include <map>
#include <ranges>

namespace {
std::vector<int64_t>
getSubspaceExtents(const std::vector<int64_t> &modeExtents,
                   const std::vector<std::size_t> &degrees) {
  std::vector<int64_t> subspaceExtents;

  for (std::size_t degree : degrees) {
    if (degree >= modeExtents.size())
      throw std::out_of_range("Degree exceeds modeExtents size.");

    subspaceExtents.push_back(modeExtents[degree]);
  }

  return subspaceExtents;
}

cudaq::dimension_map
convertDimensions(const std::vector<int64_t> &modeExtents) {

  cudaq::dimension_map dimensions;
  for (size_t i = 0; i < modeExtents.size(); ++i)
    dimensions[i] = static_cast<std::size_t>(modeExtents[i]);

  return dimensions;
}

// Function to flatten a matrix into a 1D array (column major)
std::vector<std::complex<double>>
flattenMatrixColumnMajor(const cudaq::complex_matrix &matrix) {
  std::vector<std::complex<double>> flatMatrix;
  flatMatrix.reserve(matrix.size());
  for (size_t col = 0; col < matrix.cols(); col++) {
    for (size_t row = 0; row < matrix.rows(); row++) {
      flatMatrix.push_back(matrix[{row, col}]);
    }
  }

  return flatMatrix;
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

cudaq::dynamics::CuDensityMatOpConverter::CuDensityMatOpConverter(
    cudensitymatHandle_t handle)
    : m_handle(handle) {
  const auto getIntEnvVarIfPresent =
      [](const char *envName) -> std::optional<int> {
    if (auto *envVal = std::getenv(envName)) {
      const std::string envValStr(envVal);
      const char *nptr = envValStr.data();
      char *endptr = nullptr;
      errno = 0; // reset errno to 0 before call
      auto envIntVal = strtol(nptr, &endptr, 10);

      if (nptr == endptr || errno != 0 || envIntVal < 0)
        throw std::runtime_error(fmt::format(
            "Invalid {} setting. Expected a non-negative number. Got: '{}'",
            envName, envValStr));

      return envIntVal;
    }
    // The environment variable is not set.
    return std::nullopt;
  };

  {
    const auto minDim =
        getIntEnvVarIfPresent("CUDAQ_DYNAMICS_MIN_MULTIDIAGONAL_DIMENSION");
    if (minDim.has_value()) {
      cudaq::info("Setting multi-diagonal min dimension to {}.",
                  minDim.value());
      m_minDimensionDiag = minDim.value();
    }
  }

  {
    const auto maxDiags = getIntEnvVarIfPresent(
        "CUDAQ_DYNAMICS_MAX_DIAGONAL_COUNT_FOR_MULTIDIAGONAL");
    if (maxDiags.has_value()) {
      cudaq::info("Setting multi-diagonal max number of diagonals to {}.",
                  maxDiags.value());
      m_maxDiagonalsDiag = maxDiags.value();
    }
  }
}

cudensitymatOperator_t
cudaq::dynamics::CuDensityMatOpConverter::constructLiouvillian(
    const sum_op<cudaq::matrix_handler> &ham,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapseOperators,
    const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    bool isMasterEquation) {
  LOG_API_TIME();
  if (!isMasterEquation && collapseOperators.empty()) {
    cudaq::info("Construct state vector Liouvillian");
    auto liouvillian = ham * std::complex<double>(0.0, -1.0);
    return convertToCudensitymatOperator(parameters, liouvillian, modeExtents);
  } else {
    cudaq::info("Construct density matrix Liouvillian");
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
    for (auto &[coeff, term] :
         convertToCudensitymat(ham, parameters, modeExtents)) {
      cudensitymatWrappedScalarCallback_t wrappedCallback =
          cudensitymatScalarCallbackNone;
      if (coeff.is_constant()) {
        const auto coeffVal = coeff.evaluate();
        const auto leftCoeff = std::complex<double>(0.0, -1.0) * coeffVal;
        // -i constant (left multiplication)
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            m_handle, liouvillian, term, 0,
            make_cuDoubleComplex(leftCoeff.real(), leftCoeff.imag()),
            wrappedCallback, cudensitymatScalarGradientCallbackNone));

        // +i constant (right multiplication, i.e., dual)
        const auto rightCoeff = std::complex<double>(0.0, 1.0) * coeffVal;
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            m_handle, liouvillian, term, 1,
            make_cuDoubleComplex(rightCoeff.real(), rightCoeff.imag()),
            wrappedCallback, cudensitymatScalarGradientCallbackNone));
      } else {
        wrappedCallback = wrapScalarCallback(coeff, keys);
        // -i constant (left multiplication)
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            m_handle, liouvillian, term, 0, make_cuDoubleComplex(0.0, -1.0),
            wrappedCallback, cudensitymatScalarGradientCallbackNone));

        // +i constant (right multiplication, i.e., dual)
        HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
            m_handle, liouvillian, term, 1, make_cuDoubleComplex(0.0, 1.0),
            wrappedCallback, cudensitymatScalarGradientCallbackNone));
      }
    }

    // Handle collapsed operators
    for (auto &collapseOperator : collapseOperators) {
      for (auto &[coeff, term] :
           computeLindbladTerms(collapseOperator, modeExtents, parameters)) {
        cudensitymatWrappedScalarCallback_t wrappedCallback =
            cudensitymatScalarCallbackNone;
        if (coeff.is_constant()) {
          const auto coeffVal = coeff.evaluate();
          HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
              m_handle, liouvillian, term, 0,
              make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
              wrappedCallback, cudensitymatScalarGradientCallbackNone));
        } else {
          wrappedCallback = wrapScalarCallback(coeff, keys);
          HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
              m_handle, liouvillian, term, 0, make_cuDoubleComplex(1.0, 0.0),
              wrappedCallback, cudensitymatScalarGradientCallbackNone));
        }
      }
    }

    return liouvillian;
  }
}

void cudaq::dynamics::CuDensityMatOpConverter::clearCallbackContext() {
  m_scalarCallbacks.clear();
  m_tensorCallbacks.clear();
}

cudaq::dynamics::CuDensityMatOpConverter::~CuDensityMatOpConverter() {
  for (auto term : m_operatorTerms)
    cudensitymatDestroyOperatorTerm(term);

  for (auto op : m_elementaryOperators)
    cudensitymatDestroyElementaryOperator(op);

  for (auto *buffer : m_deviceBuffers) {
    cudaq::dynamics::DeviceAllocator::free(buffer);
  }
}

cudensitymatElementaryOperator_t
cudaq::dynamics::CuDensityMatOpConverter::createElementaryOperator(
    const cudaq::matrix_handler &elemOp,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<int64_t> &modeExtents) {
  auto subspaceExtents = getSubspaceExtents(modeExtents, elemOp.degrees());
  cudaq::dimension_map dimensions = convertDimensions(modeExtents);
  cudensitymatWrappedTensorCallback_t wrappedTensorCallback =
      cudensitymatTensorCallbackNone;

  static const std::vector<std::string> g_knownNonParametricOps = []() {
    std::vector<std::string> opNames;
    opNames.emplace_back(
        cudaq::boson_op::identity(0).begin()->to_string(false));
    // These are ops that we created during lindblad generation
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::boson_op::create(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(
        cudaq::boson_op::annihilate(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::boson_op::number(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::spin_op::i(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::spin_op::x(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::spin_op::y(0).begin()->to_string(false));
    opNames.emplace_back(opNames.back() + "_dagger");
    opNames.emplace_back(cudaq::spin_op::z(0).begin()->to_string(false));
    return opNames;
  }();

  // This is a callback
  if (!parameters.empty() &&
      std::find(g_knownNonParametricOps.begin(), g_knownNonParametricOps.end(),
                elemOp.to_string(false)) == g_knownNonParametricOps.end()) {
    const std::map<std::string, std::complex<double>> sortedParameters(
        parameters.begin(), parameters.end());
    auto ks = std::views::keys(sortedParameters);
    const std::vector<std::string> keys{ks.begin(), ks.end()};
    wrappedTensorCallback = wrapTensorCallback(elemOp, keys);
  }

  const auto [diags, offsets] =
      elemOp.to_diagonal_matrix(dimensions, parameters);
  const bool shouldUseDia = [&]() {
    if (diags.empty())
      return false;
    const auto dim = std::accumulate(
        subspaceExtents.begin(), subspaceExtents.end(), 1,
        std::multiplies<decltype(subspaceExtents)::value_type>());
    if (dim < m_minDimensionDiag)
      return false;
    return offsets.size() <= m_maxDiagonalsDiag;
  }();

  auto *elementaryMat_d = [&]() {
    if (shouldUseDia)
      return cudaq::dynamics::createArrayGpu(diags);
    auto flatMatrix =
        flattenMatrixColumnMajor(elemOp.to_matrix(dimensions, parameters));

    if (flatMatrix.empty())
      throw std::invalid_argument(
          "Input matrix (flat matrix) cannot be empty.");

    if (subspaceExtents.empty())
      throw std::invalid_argument("subspaceExtents cannot be empty.");

    return cudaq::dynamics::createArrayGpu(flatMatrix);
  }();

  cudensitymatElementaryOperator_t cudmElemOp = nullptr;
  if (shouldUseDia) {
    const std::vector<int32_t> diagonalOffsets(offsets.begin(), offsets.end());
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
        m_handle, static_cast<int32_t>(subspaceExtents.size()),
        subspaceExtents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_MULTIDIAGONAL,
        offsets.size(), diagonalOffsets.data(), CUDA_C_64F, elementaryMat_d,
        wrappedTensorCallback, cudensitymatTensorGradientCallbackNone,
        &cudmElemOp));
  } else {
    HANDLE_CUDM_ERROR(cudensitymatCreateElementaryOperator(
        m_handle, static_cast<int32_t>(subspaceExtents.size()),
        subspaceExtents.data(), CUDENSITYMAT_OPERATOR_SPARSITY_NONE, 0, nullptr,
        CUDA_C_64F, elementaryMat_d, wrappedTensorCallback,
        cudensitymatTensorGradientCallbackNone, &cudmElemOp));
  }

  if (!cudmElemOp) {
    std::cerr << "[ERROR] cudmElemOp is NULL in createElementaryOperator !"
              << std::endl;
    cudaq::dynamics::destroyArrayGpu(elementaryMat_d);
    throw std::runtime_error("Failed to create elementary operator.");
  }
  m_elementaryOperators.emplace(cudmElemOp);
  m_deviceBuffers.emplace(elementaryMat_d);
  return cudmElemOp;
}

cudensitymatOperatorTerm_t
cudaq::dynamics::CuDensityMatOpConverter::createProductOperatorTerm(
    const std::vector<cudensitymatElementaryOperator_t> &elemOps,
    const std::vector<int64_t> &modeExtents,
    const std::vector<std::vector<std::size_t>> &degrees,
    const std::vector<std::vector<int>> &dualModalities) {

  cudensitymatOperatorTerm_t term;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperatorTerm(
      m_handle, static_cast<int32_t>(modeExtents.size()), modeExtents.data(),
      &term));
  m_operatorTerms.emplace(term);
  if (degrees.empty())
    throw std::invalid_argument("Degrees vector cannot be empty.");

  if (elemOps.empty())
    throw std::invalid_argument("elemOps cannot be null.");

  if (degrees.size() != elemOps.size())
    throw std::invalid_argument("elemOps and degrees must have the same size.");

  const bool hasDualModalities = !dualModalities.empty();

  if (hasDualModalities && degrees.size() != dualModalities.size())
    throw std::invalid_argument(
        "degrees and dualModalities must have the same size.");

  std::vector<int32_t> allDegrees;
  std::vector<int32_t> allModeActionDuality;
  for (size_t i = 0; i < degrees.size(); i++) {
    const auto &sub_degrees = degrees[i];
    const auto &modalities = hasDualModalities
                                 ? dualModalities[i]
                                 : std::vector<int>(sub_degrees.size(), 0);

    if (sub_degrees.size() != modalities.size())
      throw std::runtime_error(
          "Mismatch between degrees and modalities sizes.");

    if (sub_degrees.size() != 1)
      throw std::runtime_error(
          "Elementary operator must act on a single degree.");

    for (size_t j = 0; j < sub_degrees.size(); j++) {
      std::size_t degree = sub_degrees[j];
      int modality = modalities[j];

      if (sub_degrees[i] < 0)
        throw std::out_of_range("Degree cannot be negative!");

      allDegrees.emplace_back(degree);
      allModeActionDuality.emplace_back(modality);
    }
  }

  assert(elemOps.size() == degrees.size());
  HANDLE_CUDM_ERROR(cudensitymatOperatorTermAppendElementaryProduct(
      m_handle, term, static_cast<int32_t>(elemOps.size()), elemOps.data(),
      allDegrees.data(), allModeActionDuality.data(),
      make_cuDoubleComplex(1.0, 0.0), cudensitymatScalarCallbackNone,
      cudensitymatScalarGradientCallbackNone));
  return term;
}

cudensitymatOperator_t
cudaq::dynamics::CuDensityMatOpConverter::convertToCudensitymatOperator(
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const sum_op<cudaq::matrix_handler> &op,
    const std::vector<int64_t> &modeExtents) {
  if (op.num_terms() == 0)
    throw std::invalid_argument("Operator sum cannot be empty.");

  cudensitymatOperator_t cudmOperator;
  HANDLE_CUDM_ERROR(cudensitymatCreateOperator(
      m_handle, static_cast<int32_t>(modeExtents.size()), modeExtents.data(),
      &cudmOperator));

  const std::map<std::string, std::complex<double>> sortedParameters(
      parameters.begin(), parameters.end());
  auto ks = std::views::keys(sortedParameters);
  const std::vector<std::string> keys{ks.begin(), ks.end()};
  for (auto &[coeff, term] :
       convertToCudensitymat(op, parameters, modeExtents)) {
    cudensitymatWrappedScalarCallback_t wrappedCallback =
        cudensitymatScalarCallbackNone;

    if (coeff.is_constant()) {
      const auto coeffVal = coeff.evaluate();
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          m_handle, cudmOperator, term, 0,
          make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
          wrappedCallback, cudensitymatScalarGradientCallbackNone));
    } else {
      wrappedCallback = wrapScalarCallback(coeff, keys);
      HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
          m_handle, cudmOperator, term, 0, make_cuDoubleComplex(1.0, 0.0),
          wrappedCallback, cudensitymatScalarGradientCallbackNone));
    }
  }

  return cudmOperator;
}

std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
cudaq::dynamics::CuDensityMatOpConverter::convertToCudensitymat(
    const sum_op<cudaq::matrix_handler> &op,
    const std::unordered_map<std::string, std::complex<double>> &parameters,
    const std::vector<int64_t> &modeExtents) {
  if (op.num_terms() == 0)
    throw std::invalid_argument("Operator sum cannot be empty.");

  std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
      result;

  for (const auto &productOp : op) {
    std::vector<cudensitymatElementaryOperator_t> elemOps;
    std::vector<std::vector<std::size_t>> allDegrees;
    for (const auto &component : productOp) {
      // No need to check type
      // just call to_matrix on it
      if (const auto *elemOp =
              dynamic_cast<const cudaq::matrix_handler *>(&component)) {
        auto cudmElemOp =
            createElementaryOperator(*elemOp, parameters, modeExtents);
        elemOps.emplace_back(cudmElemOp);
        allDegrees.emplace_back(elemOp->degrees());
      } else {
        // Catch anything that we don't know
        throw std::runtime_error("Unhandled type!");
      }
    }
    // Note: the order of operator application is the opposite of the writing:
    // i.e., ABC means C to be applied first.
    std::reverse(elemOps.begin(), elemOps.end());
    std::reverse(allDegrees.begin(), allDegrees.end());
    if (elemOps.empty()) {
      // Constant term (no operator)
      cudaq::product_op<cudaq::matrix_handler> constantTerm =
          cudaq::sum_op<cudaq::matrix_handler>::identity(0);
      cudensitymatElementaryOperator_t cudmElemOp = createElementaryOperator(
          *constantTerm.begin(), parameters, modeExtents);
      result.emplace_back(std::make_pair(
          productOp.get_coefficient(),
          createProductOperatorTerm({cudmElemOp}, modeExtents, {{0}}, {})));
    } else {
      result.emplace_back(std::make_pair(
          productOp.get_coefficient(),
          createProductOperatorTerm(elemOps, modeExtents, allDegrees, {})));
    }
  }
  return result;
}

std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
cudaq::dynamics::CuDensityMatOpConverter::computeLindbladTerms(
    const sum_op<cudaq::matrix_handler> &collapseOp,
    const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  std::vector<std::pair<cudaq::scalar_operator, cudensitymatOperatorTerm_t>>
      lindbladTerms;
  for (const product_op<matrix_handler> &l_op : collapseOp) {
    for (const product_op<matrix_handler> &r_op : collapseOp) {
      scalar_operator coeff =
          l_op.get_coefficient() * computeDagger(r_op.get_coefficient());
      auto ldag = computeDagger(r_op);
      {
        // L * rho * L_dag
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities;

        for (const auto &component : l_op) {
          if (const auto *elemOp =
                  dynamic_cast<const cudaq::matrix_handler *>(&component)) {
            auto cudmElemOp =
                createElementaryOperator(*elemOp, parameters, modeExtents);
            elemOps.emplace_back(cudmElemOp);
            allDegrees.emplace_back(elemOp->degrees());
            all_action_dual_modalities.emplace_back(
                std::vector<int>(elemOp->degrees().size(), 0));
          } else {
            // Catch anything that we don't know
            throw std::runtime_error("Unhandled type!");
          }
        }

        for (const auto &component : ldag) {
          if (const auto *elemOp =
                  dynamic_cast<const cudaq::matrix_handler *>(&component)) {
            auto cudmElemOp =
                createElementaryOperator(*elemOp, parameters, modeExtents);
            elemOps.emplace_back(cudmElemOp);
            allDegrees.emplace_back(elemOp->degrees());
            all_action_dual_modalities.emplace_back(
                std::vector<int>(elemOp->degrees().size(), 1));
          } else {
            // Catch anything that we don't know
            throw std::runtime_error("Unhandled type!");
          }
        }

        cudensitymatOperatorTerm_t D1_term = createProductOperatorTerm(
            elemOps, modeExtents, allDegrees, all_action_dual_modalities);
        lindbladTerms.emplace_back(std::make_pair(coeff, D1_term));
      }

      product_op<matrix_handler> L_daggerTimesL = -0.5 * ldag * l_op;
      {
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities_left;
        std::vector<std::vector<int>> all_action_dual_modalities_right;
        for (const auto &component : L_daggerTimesL) {
          if (const auto *elemOp =
                  dynamic_cast<const cudaq::matrix_handler *>(&component)) {
            auto cudmElemOp =
                createElementaryOperator(*elemOp, parameters, modeExtents);
            elemOps.emplace_back(cudmElemOp);
            allDegrees.emplace_back(elemOp->degrees());
            all_action_dual_modalities_left.emplace_back(
                std::vector<int>(elemOp->degrees().size(), 0));
            all_action_dual_modalities_right.emplace_back(
                std::vector<int>(elemOp->degrees().size(), 1));
          } else {
            // Catch anything that we don't know
            throw std::runtime_error("Unhandled type!");
          }
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
              std::make_pair(L_daggerTimesL.get_coefficient(), D2_term));
        }
        {
          cudensitymatOperatorTerm_t D3_term =
              createProductOperatorTerm(elemOps, modeExtents, allDegrees,
                                        all_action_dual_modalities_right);
          lindbladTerms.emplace_back(
              std::make_pair(L_daggerTimesL.get_coefficient(), D3_term));
        }
      }
    }
  }
  return lindbladTerms;
}

cudensitymatWrappedScalarCallback_t
cudaq::dynamics::CuDensityMatOpConverter::wrapScalarCallback(
    const scalar_operator &scalarOp,
    const std::vector<std::string> &paramNames) {
  if (scalarOp.is_constant())
    throw std::runtime_error(
        "scalar_operator does not have a valid generator function.");

  m_scalarCallbacks.push_back(ScalarCallBackContext(scalarOp, paramNames));
  ScalarCallBackContext *storedCallbackContext = &m_scalarCallbacks.back();
  using WrapperFuncType =
      int32_t (*)(cudensitymatScalarCallback_t, double, int64_t, int32_t,
                  const double[], cudaDataType_t, void *);

  auto wrapper = [](cudensitymatScalarCallback_t callback, double time,
                    int64_t batchSize, int32_t numParams, const double params[],
                    cudaDataType_t dataType, void *scalarStorage) -> int32_t {
    try {
      ScalarCallBackContext *context =
          reinterpret_cast<ScalarCallBackContext *>(callback);
      scalar_operator &storedOp = context->scalarOp;
      if (numParams != 2 * context->paramNames.size())
        throw std::runtime_error(
            fmt::format("[Internal Error] Invalid number of callback "
                        "parameters encountered. Expected {} double params "
                        "representing {} complex values but received {}.",
                        2 * context->paramNames.size(),
                        context->paramNames.size(), numParams));

      std::unordered_map<std::string, std::complex<double>> param_map;
      for (size_t i = 0; i < context->paramNames.size(); ++i) {
        param_map[context->paramNames[i]] =
            std::complex<double>(params[2 * i], params[2 * i + 1]);
        cudaq::debug("Callback param name {}, batch size {}, value {}",
                     context->paramNames[i], batchSize,
                     param_map[context->paramNames[i]]);
      }

      std::complex<double> result = storedOp.evaluate(param_map);
      cudaq::debug("Scalar callback evaluated result = {}", result);
      auto *tdCoef = static_cast<std::complex<double> *>(scalarStorage);
      *tdCoef = result;
      return CUDENSITYMAT_STATUS_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "Error in scalar callback: " << e.what() << std::endl;
      return CUDENSITYMAT_STATUS_INTERNAL_ERROR;
    }
  };

  cudensitymatWrappedScalarCallback_t wrappedCallback;
  wrappedCallback.callback =
      reinterpret_cast<cudensitymatScalarCallback_t>(storedCallbackContext);
  wrappedCallback.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU;
  wrappedCallback.wrapper =
      reinterpret_cast<void *>(static_cast<WrapperFuncType>(wrapper));
  return wrappedCallback;
}

cudensitymatWrappedTensorCallback_t
cudaq::dynamics::CuDensityMatOpConverter::wrapTensorCallback(
    const matrix_handler &matrixOp,
    const std::vector<std::string> &paramNames) {
  m_tensorCallbacks.push_back(TensorCallBackContext(matrixOp, paramNames));
  TensorCallBackContext *storedCallbackContext = &m_tensorCallbacks.back();
  using WrapperFuncType = int32_t (*)(
      cudensitymatTensorCallback_t, cudensitymatElementaryOperatorSparsity_t,
      int32_t, const int64_t[], const int32_t[], double, int64_t, int32_t,
      const double[], cudaDataType_t, void *, cudaStream_t);

  auto wrapper = [](cudensitymatTensorCallback_t callback,
                    cudensitymatElementaryOperatorSparsity_t sparsity,
                    int32_t num_modes, const int64_t modeExtents[],
                    const int32_t diagonal_offsets[], double time,
                    int64_t batchSize, int32_t num_params,
                    const double params[], cudaDataType_t data_type,
                    void *tensor_storage, cudaStream_t stream) -> int32_t {
    try {
      auto *context = reinterpret_cast<TensorCallBackContext *>(callback);
      matrix_handler &storedOp = context->tensorOp;

      if (num_modes <= 0) {
        std::cerr << "num_modes is invalid: " << num_modes << std::endl;
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      if (num_params != 2 * context->paramNames.size())
        throw std::runtime_error(
            fmt::format("[Internal Error] Invalid number of tensor callback "
                        "parameters. Expected {} double values "
                        "representing {} complex parameters but received "
                        "{}.",
                        std::to_string(2 * context->paramNames.size()),
                        std::to_string(context->paramNames.size()),
                        std::to_string(num_params)));

      std::unordered_map<std::string, std::complex<double>> param_map;
      for (size_t i = 0; i < context->paramNames.size(); ++i) {
        param_map[context->paramNames[i]] =
            std::complex<double>(params[2 * i], params[2 * i + 1]);
        cudaq::debug("Tensor callback param name {}, value {}",
                     context->paramNames[i], param_map[context->paramNames[i]]);
      }

      cudaq::dimension_map dimensions;
      for (std::size_t i = 0; i < num_modes; ++i) {
        dimensions[i] = modeExtents[i];
      }

      if (dimensions.empty()) {
        std::cerr << "Dimension map is empty!" << std::endl;
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }
      const std::vector<std::complex<double>> flatMatrix = [&]() {
        if (sparsity == CUDENSITYMAT_OPERATOR_SPARSITY_NONE) {
          complex_matrix matrix_data =
              storedOp.to_matrix(dimensions, param_map);
          return flattenMatrixColumnMajor(matrix_data);
        }
        auto [mDiagData, _] =
            storedOp.to_diagonal_matrix(dimensions, param_map);
        return mDiagData;
      }();

      if (data_type == CUDA_C_64F) {
        memcpy(tensor_storage, flatMatrix.data(),
               flatMatrix.size() * sizeof(cuDoubleComplex));
      } else if (data_type == CUDA_C_32F) {
        std::vector<std::complex<float>> flatMatrix_float(flatMatrix.begin(),
                                                          flatMatrix.end());

        memcpy(tensor_storage, flatMatrix_float.data(),
               flatMatrix_float.size() * sizeof(cuFloatComplex));
      } else {
        std::cerr << "Invalid CUDA data type: " << data_type << std::endl;
        return CUDENSITYMAT_STATUS_INVALID_VALUE;
      }

      return CUDENSITYMAT_STATUS_SUCCESS;
    } catch (const std::exception &e) {
      std::cerr << "Error in tensor callback: " << e.what() << std::endl;
      return CUDENSITYMAT_STATUS_INTERNAL_ERROR;
    }
  };

  cudensitymatWrappedTensorCallback_t wrappedCallback;
  wrappedCallback.callback =
      reinterpret_cast<cudensitymatTensorCallback_t>(storedCallbackContext);
  wrappedCallback.device = CUDENSITYMAT_CALLBACK_DEVICE_CPU;
  wrappedCallback.wrapper =
      reinterpret_cast<void *>(static_cast<WrapperFuncType>(wrapper));

  return wrappedCallback;
}

cudensitymatOperator_t
cudaq::dynamics::CuDensityMatOpConverter::constructLiouvillian(
    const super_op &superOp, const std::vector<int64_t> &modeExtents,
    const std::unordered_map<std::string, std::complex<double>> &parameters) {
  LOG_API_TIME();
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
  for (const auto &[leftOpOpt, rightOpOpt] : superOp) {
    if (leftOpOpt.has_value()) {
      if (rightOpOpt.has_value()) {
        scalar_operator coeff =
            leftOpOpt->get_coefficient() * rightOpOpt->get_coefficient();
        // L * rho * R
        std::vector<cudensitymatElementaryOperator_t> elemOps;
        std::vector<std::vector<std::size_t>> allDegrees;
        std::vector<std::vector<int>> all_action_dual_modalities;

        for (const auto &component : leftOpOpt.value()) {
          if (const auto *elemOp =
                  dynamic_cast<const cudaq::matrix_handler *>(&component)) {
            auto cudmElemOp =
                createElementaryOperator(*elemOp, parameters, modeExtents);
            elemOps.emplace_back(cudmElemOp);
            allDegrees.emplace_back(elemOp->degrees());
            all_action_dual_modalities.emplace_back(
                std::vector<int>(elemOp->degrees().size(), 0));
          } else {
            // Catch anything that we don't know
            throw std::runtime_error("Unhandled type!");
          }
        }

        for (const auto &component : rightOpOpt.value()) {
          if (const auto *elemOp =
                  dynamic_cast<const cudaq::matrix_handler *>(&component)) {
            auto cudmElemOp =
                createElementaryOperator(*elemOp, parameters, modeExtents);
            elemOps.emplace_back(cudmElemOp);
            allDegrees.emplace_back(elemOp->degrees());
            all_action_dual_modalities.emplace_back(
                std::vector<int>(elemOp->degrees().size(), 1));
          } else {
            // Catch anything that we don't know
            throw std::runtime_error("Unhandled type!");
          }
        }

        cudensitymatOperatorTerm_t term = createProductOperatorTerm(
            elemOps, modeExtents, allDegrees, all_action_dual_modalities);

        cudensitymatWrappedScalarCallback_t wrappedCallback =
            cudensitymatScalarCallbackNone;
        if (coeff.is_constant()) {
          const auto coeffVal = coeff.evaluate();
          HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
              m_handle, liouvillian, term, 0,
              make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
              wrappedCallback, cudensitymatScalarGradientCallbackNone));
        } else {
          wrappedCallback = wrapScalarCallback(coeff, keys);
          HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
              m_handle, liouvillian, term, 0, make_cuDoubleComplex(1.0, 0.0),
              wrappedCallback, cudensitymatScalarGradientCallbackNone));
        }
      } else {
        for (auto &[coeff, term] : convertToCudensitymat(
                 leftOpOpt.value(), parameters, modeExtents)) {
          cudensitymatWrappedScalarCallback_t wrappedCallback =
              cudensitymatScalarCallbackNone;
          if (coeff.is_constant()) {
            const auto coeffVal = coeff.evaluate();
            HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
                m_handle, liouvillian, term, 0,
                make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
                wrappedCallback, cudensitymatScalarGradientCallbackNone));
          } else {
            wrappedCallback = wrapScalarCallback(coeff, keys);
            HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
                m_handle, liouvillian, term, 0, make_cuDoubleComplex(1.0, 0.0),
                wrappedCallback, cudensitymatScalarGradientCallbackNone));
          }
        }
      }
    } else {
      if (rightOpOpt.has_value()) {
        for (auto &[coeff, term] : convertToCudensitymat(
                 rightOpOpt.value(), parameters, modeExtents)) {
          cudensitymatWrappedScalarCallback_t wrappedCallback =
              cudensitymatScalarCallbackNone;
          if (coeff.is_constant()) {
            const auto coeffVal = coeff.evaluate();
            HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
                m_handle, liouvillian, term, 1,
                make_cuDoubleComplex(coeffVal.real(), coeffVal.imag()),
                wrappedCallback, cudensitymatScalarGradientCallbackNone));
          } else {
            wrappedCallback = wrapScalarCallback(coeff, keys);
            HANDLE_CUDM_ERROR(cudensitymatOperatorAppendTerm(
                m_handle, liouvillian, term, 1, make_cuDoubleComplex(1.0, 0.0),
                wrappedCallback, cudensitymatScalarGradientCallbackNone));
          }
        }
      } else {
        throw std::runtime_error("Invalid super-operator term encountered: no "
                                 "operation action is specified.");
      }
    }
  }
  return liouvillian;
}
