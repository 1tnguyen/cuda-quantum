/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/ExecutionContext.h"
#include <type_traits>
#include <vector>

namespace cudaq {

/// @brief Kernel result holder: this can hold either a T or an error string.
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
  template <typename OtherT>
  Result(OtherT &&val,
         std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr)
      : hasError(false) {
    new (getStorage()) T(std::forward<OtherT>(val));
  }
  Result(const std::exception &e) : hasError(true) {
    new (getErrorStorage()) std::string(e.what());
  }

  Result(Result &&Other) { moveConstruct(std::move(Other)); }

  template <class OtherT>
  Result(Result<OtherT> &&Other,
         std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr) {
    moveConstruct(std::move(Other));
  }

  template <class OtherT>
  explicit Result(
      Result<OtherT> &&Other,
      std::enable_if_t<!std::is_convertible_v<OtherT, T>> * = nullptr) {
    moveConstruct(std::move(Other));
  }

  Result &operator=(Result &&Other) {
    moveAssign(std::move(Other));
    return *this;
  }

  ~Result() {
    if (hasError)
      getErrorStorage()->~ErrorTy();
    else
      getStorage()->~T();
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

  const T &get() const { return *getStorage(); }
  bool isOk() const { return !hasError; }
  std::string getError() const { return hasError ? *getErrorStorage() : ""; }

private:
  template <class OtherT>
  void moveConstruct(Result<OtherT> &&Other) {
    hasError = Other.hasError;
    if (!hasError)
      new (getStorage()) T(std::move(*Other.getStorage()));
    else
      new (getErrorStorage()) ErrorTy(std::move(*Other.getErrorStorage()));
  }
  template <class T1>
  static bool compareThisIfSameType(const T1 &a, const T1 &b) {
    return &a == &b;
  }

  template <class T1, class T2>
  static bool compareThisIfSameType(const T1 &, const T2 &) {
    return false;
  }
  template <class OtherT>
  void moveAssign(Result<OtherT> &&Other) {
    if (compareThisIfSameType(*this, Other))
      return;
    this->~Result();
    new (this) Result(std::move(Other));
  }
};

/// Run a kernel multiple times, returning a collection of results or any
/// failure.
template <class KernelTy, class... Args>
std::vector<
    Result<std::invoke_result_t<std::decay_t<KernelTy>, std::decay_t<Args>...>>>
run(std::size_t shots, KernelTy &&f, Args &&...args) {
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
} // namespace cudaq