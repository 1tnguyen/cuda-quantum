/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <sstream>

namespace cudaq {
namespace qir {
class OutputRecorder {
  std::stringstream ss;

public:
  template <typename T>
  void record(const T &value) {
    const auto recordImpl = [&](const char *type, const std::string &val) {
      ss << "OUTPUT\t" << type << '\t' << val << '\n';
    };
    if constexpr (std::is_same_v<T, bool>) {
      recordImpl("BOOL", value ? "true" : "false");
    } else if constexpr (std::is_integral_v<T>) {
      recordImpl("INT", std::to_string(value));
    } else if constexpr (std::is_floating_point_v<T>) {
      const auto doubleToString = [](auto val){
        std::stringstream ss;
        ss << std::showpoint << val;
        return ss.str();
      };
      recordImpl("DOUBLE", doubleToString(value));
    } else if constexpr (requires { std::begin(value); }) {
      recordImpl("ARRAY", std::to_string(value.size()));
      for (const auto &elem : value) {
        record(elem);
      }
    } else if constexpr (requires { value.get(); }) {
      printf("Record Runresult\n");
      record(value.get());
    } else {
      throw std::runtime_error("Unsupported type for recording");
    }
  }

  std::string getOutput() const { return ss.str(); }
};
} // namespace qir
} // namespace cudaq
