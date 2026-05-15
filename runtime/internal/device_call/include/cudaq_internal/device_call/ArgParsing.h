/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

namespace cudaq_internal::device_call {

inline const char *consumeValue(int &index, int argc, char **argv,
                                const char *current, const char *option) {
  if (!current)
    return nullptr;
  if (std::strcmp(current, option) == 0 && index + 1 < argc)
    return argv[++index];
  const std::size_t optionLen = std::strlen(option);
  if (std::strncmp(current, option, optionLen) == 0 &&
      current[optionLen] == '=')
    return current + optionLen + 1;
  return nullptr;
}

inline bool parseUInt(const char *value, std::uint64_t maxValue,
                      std::uint64_t &out) {
  if (!value || !*value)
    return false;
  char *end = nullptr;
  errno = 0;
  unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > maxValue)
    return false;
  out = static_cast<std::uint64_t>(parsed);
  return true;
}

template <typename T>
inline bool parseUIntAs(const char *value, T &out) {
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  static_assert(!std::is_signed_v<T>, "T must be unsigned");

  std::uint64_t parsed = 0;
  if (!parseUInt(value,
                 static_cast<std::uint64_t>(std::numeric_limits<T>::max()),
                 parsed))
    return false;
  out = static_cast<T>(parsed);
  return true;
}

template <typename Config>
struct CliOption {
  const char *name;
  bool (*parse)(Config &, const char *);
};

template <typename Config, std::size_t N>
inline bool parseCliOptions(int argc, char **argv,
                            const CliOption<Config> (&options)[N],
                            Config &config) {
  if (argc < 0)
    return false;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv ? argv[i] : nullptr;
    if (!arg)
      continue;

    for (const CliOption<Config> &option : options) {
      if (const char *value = consumeValue(i, argc, argv, arg, option.name)) {
        if (!option.parse(config, value))
          return false;
        break;
      }
    }
  }
  return true;
}

template <typename Config, std::string Config::*Member>
inline bool parseStringOption(Config &config, const char *value) {
  if (!value)
    return false;
  config.*Member = value;
  return true;
}

template <typename Config, typename T, T Config::*Member,
          std::uint64_t MinValue = 0>
inline bool parseUIntOption(Config &config, const char *value) {
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  static_assert(!std::is_signed_v<T>, "T must be unsigned");

  std::uint64_t parsed = 0;
  if (!parseUInt(value,
                 static_cast<std::uint64_t>(std::numeric_limits<T>::max()),
                 parsed) ||
      parsed < MinValue)
    return false;
  config.*Member = static_cast<T>(parsed);
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
