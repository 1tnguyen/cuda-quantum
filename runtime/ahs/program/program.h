/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "nlohmann/json.hpp"
#include <string>
#include <vector>

namespace cudaq {
namespace ahs {
using json = nlohmann::json;

#define TO_JSON_HELPER(field) j[#field] = p.field
#define FROM_JSON_HELPER(field) j[#field].get_to(p.field)

inline std::string doubleAsJsonString(double d) {
  json j = d;
  return j.dump();
}

inline std::vector<double>
doubleFromStr(const std::vector<std::string> &stringList) {
  std::vector<double> result;
  result.reserve(stringList.size());
  for (const auto &s : stringList) {
    result.push_back(std::stod(s));
  }
  return result;
}

inline std::vector<std::string>
strFromDouble(const std::vector<double> &doubleList) {
  std::vector<std::string> result;
  result.reserve(doubleList.size());
  for (const auto &d : doubleList) {
    result.push_back(doubleAsJsonString(d));
  }
  return result;
}

struct AtomArrangement {
  std::vector<std::vector<double>> sites;
  std::vector<int> filling;
  friend void to_json(json &j, const AtomArrangement &p) {
    TO_JSON_HELPER(filling);
    std::vector<std::vector<std::string>> floatAsStrings;
    for (const auto &site : p.sites)
      floatAsStrings.push_back(strFromDouble(site));
    j["sites"] = floatAsStrings;
  }

  friend void from_json(const json &j, AtomArrangement &p) {
    FROM_JSON_HELPER(filling);
    std::vector<std::vector<std::string>> floatAsStrings;
    j["sites"].get_to(floatAsStrings);
    for (const auto &row : floatAsStrings)
      p.sites.push_back(doubleFromStr(row));
  }
};

/// @brief  Represents the setup of neutral atom registers
struct Setup {
  AtomArrangement ahs_register;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Setup, ahs_register);
};

/// @brief Represents control signal time series
struct TimeSeries {
  TimeSeries() = default;
  TimeSeries(const std::vector<std::pair<double, double>> &data) {
    for (const auto &pair : data) {
      times.push_back(pair.first);
      values.push_back(pair.second);
    }
  }
  std::vector<double> values;
  std::vector<double> times;

  bool almostEqual(const TimeSeries &other, double tol = 1e-12) const {
    if (values.size() != other.values.size() ||
        times.size() != other.times.size()) {
      return false;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (std::abs(values[i] - other.values[i]) > tol) {
        return false;
      }
    }
    for (std::size_t i = 0; i < times.size(); ++i) {
      if (std::abs(times[i] - other.times[i]) > tol) {
        return false;
      }
    }
    return true;
  }
  friend void to_json(json &j, const TimeSeries &p) {
    j["values"] = strFromDouble(p.values);
    j["times"] = strFromDouble(p.times);
  }

  friend void from_json(const json &j, TimeSeries &p) {
    std::vector<std::string> floatAsStrings;
    j["values"].get_to(floatAsStrings);
    p.values = doubleFromStr(floatAsStrings);
    floatAsStrings.clear();
    j["times"].get_to(floatAsStrings);
    p.times = doubleFromStr(floatAsStrings);
  }
};

struct FieldPattern {
  FieldPattern() : patternStr("uniform") {}
  FieldPattern(const std::string &patternName) : patternStr(patternName) {}
  FieldPattern(const std::vector<double> &patternValues)
      : patternVals(patternValues) {}

  std::string patternStr;
  std::vector<double> patternVals;
  bool operator==(const FieldPattern &other) const {
    return patternStr == other.patternStr && patternVals == other.patternVals;
  }

  friend void to_json(json &j, const FieldPattern &p) {
    if (p.patternStr.empty())
      j = strFromDouble(p.patternVals);
    else
      j = p.patternStr;
  }

  friend void from_json(const json &j, FieldPattern &p) {
    if (j.is_array()) {
      std::vector<std::string> floatAsStrings;
      j.get_to(floatAsStrings);
      p.patternVals = doubleFromStr(floatAsStrings);
      p.patternStr.clear();
    } else {
      j.get_to(p.patternStr);
      p.patternVals.clear();
    }
  }
};

/// @brief Represents the temporal and spatial dependence of a control parameter
/// affecting the atoms
struct PhysicalField {
  TimeSeries time_series;
  FieldPattern pattern;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(PhysicalField, time_series, pattern);
};

/// @brief Represents the global driving field of neutral atom system
struct DrivingField {
  // Omega field
  PhysicalField amplitude;
  // Phi field
  PhysicalField phase;
  // Delta field
  PhysicalField detuning;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(DrivingField, amplitude, phase, detuning);
};

/// @brief Represents the local `detuning`
struct LocalDetuning {
  PhysicalField magnitude;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(LocalDetuning, magnitude);
};

/// @brief Represents the neutral atom Hamiltonian (driven parts)
struct Hamiltonian {
  std::vector<DrivingField> drivingFields;
  std::vector<LocalDetuning> localDetuning;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Hamiltonian, drivingFields, localDetuning);
};

/// @brief Represents an Analog Hamiltonian Simulation program
struct Program {
  Setup setup;
  Hamiltonian hamiltonian;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(Program, setup, hamiltonian);
};
} // namespace ahs

} // namespace cudaq