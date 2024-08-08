/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/**
 * @file
 *
 * Kernel run API
 *
 * This header defines the API for kernel execution with return values
 */

#pragma once

#include <type_traits>
#include <vector>

#include "common/ExecutionContext.h"
#include "common/KernelWrapper.h"
#include "cudaq/platform.h"
#include "cudaq/platform/QuantumExecutionQueue.h"

namespace cudaq {

/// @brief Kernel result holder: this can hold either a T or an error string.
///
/// In a batched execution, there could be scenarios whereby some
/// executions failed, e.g., kernels that throw as part of their algorithm
/// design (repeat until success with a fixed max retries), or backend runtime
/// errors when a certain dynamical code path being invoked. Hence, we support
/// error propagation as part of the return type wrapper.
template <typename T>
class Result {
private:
  /// Union of result or error.
  // For efficiency, we use union, i.e., only one extra boolean is required.
  union {
    std::aligned_union_t<1, T> alignedResultStorage;
    std::aligned_union_t<1, std::string> alignedErrorStorage;
  };
  bool hasError = true;
  using ErrorTy = std::string;

public:
  /// @brief Construct a valid result
  /// @tparam OtherT The actual type of input value (convertible to the expected
  /// return type)
  /// @param val Result value to be stored
  template <typename OtherT>
  Result(OtherT &&val,
         std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr)
      : hasError(false) {
    new (getStorage()) T(std::forward<OtherT>(val));
  }

  /// @brief Construct an error result
  /// @param e The exception to be captured
  Result(const std::exception &e) : hasError(true) {
    new (getErrorStorage()) std::string(e.what());
  }

  /// @brief Move constructor
  Result(Result &&Other) { moveConstruct(std::move(Other)); }

  /// @brief Move assignment operator
  Result &operator=(Result &&Other) {
    moveAssign(std::move(Other));
    return *this;
  }

  /// @brief Destructor
  ~Result() {
    if (hasError)
      getErrorStorage()->~ErrorTy();
    else
      getStorage()->~T();
  }

  /// @brief Return true if this contains a valid result.
  bool isOk() const { return !hasError; }

  /// @brief Get the result value
  /// @return  Result value if valid. Throw otherwise.
  const T &get() const { return *getStorage(); }

  /// @brief Conversion to the result type.
  /// This will throw if this is an error result.
  operator T() const { return get(); }

  /// @brief Get the error message if any. Return an empty string otherwise.
  std::string getError() const { return hasError ? *getErrorStorage() : ""; }

  SerializeRunResult<T> toSerializable() const {
    SerializeRunResult<T> result;
    result.hasValue = !hasError;
    if (hasError)
      result.errorMessage = getError();
    else
      result.value = get();
    return result;
  }

  static Result<T>
  fromSerializable(const SerializeRunResult<T> &serializedData) {
    if (serializedData.hasValue)
      return Result<T>(serializedData.value);
    return Result<T>(std::exception(serializedData.errorMessage));
  }

private:
  template <class OtherT>
  void moveConstruct(Result<OtherT> &&Other) {
    hasError = Other.hasError;
    if (!hasError)
      new (getStorage()) T(std::move(*Other.getStorage()));
    else
      new (getErrorStorage()) ErrorTy(std::move(*Other.getErrorStorage()));
  }

  template <class OtherT>
  void moveAssign(Result<OtherT> &&Other) {
    if (this == &Other)
      return;
    this->~Result();
    new (this) Result(std::move(Other));
  }

  T *getStorage() {
    assert(!hasError && "Cannot get value when an error exists!");
    return reinterpret_cast<T *>(&alignedResultStorage);
  }

  const T *getStorage() const {
    assert(!hasError && "Cannot get value when an error exists!");
    return reinterpret_cast<const T *>(&alignedResultStorage);
  }

  const std::string *getErrorStorage() const {
    assert(hasError && "Cannot get error when no error exists!");
    return reinterpret_cast<const std::string *>(&alignedErrorStorage);
  }
  std::string *getErrorStorage() {
    assert(hasError && "Cannot get error when no error exists!");
    return reinterpret_cast<std::string *>(&alignedErrorStorage);
  }
};

namespace __internal {
// Helper function to execute the run loop.
// Make this a lambda-like function class so that it can be used with
// std::apply.
struct executeRun {
  template <class KernelTy, class... Args>
  std::vector<Result<
      std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>>
  operator()(std::size_t shots, KernelTy &&f, Args &&...args) {
    using resultTy = Result<
        std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>;

    std::vector<resultTy> results;
    results.reserve(shots);
    for (std::size_t i = 0; i < shots; ++i) {
      try {
        results.emplace_back(resultTy(f(args...)));
      } catch (std::exception &e) {
        results.emplace_back(resultTy(e));
      }
    }
    return results;
  }
};
} // namespace __internal

/// @brief Run a kernel multiple times, returning a collection of results or any
/// failures
/// @tparam KernelTy Quantum kernel type
/// @tparam ...Args Quantum kernel argument types
/// @param shots Number of shots to run
/// @param f Quantum kernel
/// @param ...args Kernel arguments
/// @return A vector of `cudaq::Result`'s encapsulating the execution results.
/// The number of elements is equal to the number of shots.
template <class KernelTy, class... Args>
std::vector<
    Result<std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>>
run(std::size_t shots, KernelTy &&f, Args &&...args) {
  if (shots < 1)
    throw std::invalid_argument("The number of shots must be greater than 0.");
  return __internal::executeRun{}(shots, std::forward<KernelTy>(f),
                                  std::forward<Args>(args)...);
}

/// @brief Run a kernel multiple times with noise, returning a collection of
/// results or any failures
/// @tparam KernelTy KernelTy Quantum kernel type
/// @tparam ...Args Quantum kernel argument types
/// @param shots Number of shots to run
/// @param noise_model Noise model to use for noisy simulation
/// @param f Quantum kernel
/// @param ...args Kernel arguments
/// @return A vector of `cudaq::Result`'s encapsulating the execution results.
/// The number of elements is equal to the number of shots.
template <class KernelTy, class... Args>
std::vector<
    Result<std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>>
run(std::size_t shots, cudaq::noise_model &noise_model, KernelTy &&f,
    Args &&...args) {
  if (shots < 1)
    throw std::invalid_argument("The number of shots must be greater than 0.");
  auto &platform = cudaq::get_platform();
  platform.set_noise(&noise_model);
  auto results = __internal::executeRun{}(shots, std::forward<KernelTy>(f),
                                          std::forward<Args>(args)...);
  platform.reset_noise();
  return results;
}

template <class T>
using async_run_result = std::future<std::vector<Result<T>>>;

/// @brief Launch a run of a kernel for multiple times on a specific
/// QPU, returning a handle to a collection of results or any failures
/// @tparam KernelTy KernelTy Quantum kernel type
/// @tparam ...Args Quantum kernel argument types
/// @param qpu_id QPU to launch
/// @param shots Number of shots to run
/// @param f Quantum kernel
/// @param ...args Kernel arguments
/// @return A handle (`std::future`) to a vector of `cudaq::Result`'s
/// encapsulating the execution results. The number of elements is equal to the
/// number of shots.
template <class KernelTy, class... Args>
async_run_result<
    std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>
run_async(std::size_t qpu_id, std::size_t shots, KernelTy &&f, Args &&...args) {
  if (shots < 1)
    throw std::invalid_argument("The number of shots must be greater than 0.");
  auto &platform = cudaq::get_platform();
  if (qpu_id >= platform.num_qpus())
    throw std::invalid_argument(
        "Provided qpu_id is invalid (must be <= to platform.num_qpus()).");
  using resultTy = Result<
      std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>;
  std::promise<std::vector<resultTy>> promise;
  auto fut = promise.get_future();
  // Wrapped it as a generic (returning void) function
  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), qpu_id, shots, &platform, &f,
       args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        std::tuple<std::size_t, KernelTy> prependArgs(shots, f);
        auto fullArgs = std::tuple_cat(prependArgs, args);
        auto results = std::apply(__internal::executeRun{}, fullArgs);
        p.set_value(std::move(results));
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return fut;
}

/// @brief Launch a run of a kernel for multiple times with noise on a specific
/// QPU, returning a handle to a collection of results or any failures
/// @tparam KernelTy KernelTy Quantum kernel type
/// @tparam ...Args Quantum kernel argument types
/// @param qpu_id QPU to launch
/// @param shots Number of shots to run
/// @param noise_model Noise model to use for noisy simulation
/// @param f Quantum kernel
/// @param ...args Kernel arguments
/// @return A handle (`std::future`) to a vector of `cudaq::Result`'s
/// encapsulating the execution results. The number of elements is equal to the
/// number of shots.
template <class KernelTy, class... Args>
async_run_result<
    std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>
run_async(std::size_t qpu_id, std::size_t shots,
          cudaq::noise_model &noise_model, KernelTy &&f, Args &&...args) {
  using resultTy = Result<
      std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>;
  if (shots < 1)
    throw std::invalid_argument("The number of shots must be greater than 0.");
  auto &platform = cudaq::get_platform();
  if (qpu_id >= platform.num_qpus())
    throw std::invalid_argument(
        "Provided qpu_id is invalid (must be <= to platform.num_qpus()).");

  std::promise<std::vector<resultTy>> promise;
  auto fut = promise.get_future();
  // Wrapped it as a generic (returning void) function
  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), qpu_id, shots, noise_model, &platform, &f,
       args = std::make_tuple(std::forward<Args>(args)...)]() mutable {
        assert(platform.get_current_qpu() == qpu_id);
        platform.set_noise(&noise_model);
        std::tuple<std::size_t, KernelTy> prependArgs(shots, f);
        auto fullArgs = std::tuple_cat(prependArgs, args);
        auto results = std::apply(__internal::executeRun{}, fullArgs);
        platform.reset_noise();
        p.set_value(std::move(results));
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return fut;
}

} // namespace cudaq
