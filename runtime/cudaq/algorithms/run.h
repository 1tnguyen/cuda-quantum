/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <type_traits>
#include <vector>
#include "common/ExecutionContext.h"
#include <iostream>

namespace cudaq {

template <typename T>
struct Result {
  union {
    T resultStorage;
    char *errorMsg;
  };
  bool hasError = true;

  template <typename OtherT>
  Result(OtherT &&val,
         std::enable_if_t<std::is_convertible_v<OtherT, T>> * = nullptr)
      : hasError(false), resultStorage(val) {}
  Result(const std::exception &e)
      : hasError(true), errorMsg(strdup(e.what())) {}

  Result(const Result &other) : hasError(other.hasError) {
    if (hasError)
      errorMsg = strdup(other.errorMsg);
    else
      resultStorage = other.resultStorage;
  }

  Result &operator=(Result other) {
    hasError = other.hasError;
    if (hasError)
      std::swap(errorMsg, other.errorMsg);
    else
      std::swap(resultStorage, other.resultStorage);
    return *this;
  }

  ~Result() {
    if (hasError)
      delete errorMsg;
  }

  T get() const {
    if (hasError)
      throw std::runtime_error("Attempt to access a failed result");
    return resultStorage; 
  }
  bool isOk() const { return !hasError; }
  std::string getError() const { return hasError ? std::string(errorMsg) : ""; }
};

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
