/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq_internal/device_call/ArgParsing.h"

#include <cstdint>
#include <limits>
#include <string>

namespace cudaq_internal::device_call {

template <typename Config, std::string Config::*Member>
inline bool parseStringOption(Config &config, const char *value) {
  if (!value)
    return false;
  config.*Member = value;
  return true;
}

template <typename Config, int Config::*Member>
inline bool parseNonNegativeIntOption(Config &config, const char *value) {
  std::uint64_t parsed = 0;
  if (!parseUInt(value,
                 static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
                 parsed))
    return false;
  config.*Member = static_cast<int>(parsed);
  return true;
}

} // namespace cudaq_internal::device_call
