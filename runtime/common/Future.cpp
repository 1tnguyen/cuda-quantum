/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Future.h"
#include "Logger.h"
#include "ObserveResult.h"
#include "RestClient.h"
#include "ServerHelper.h"
#include <thread>

namespace cudaq::details {

sample_result future::get() {
  if (wrapsFutureSampling)
    return inFuture.get();

#ifdef CUDAQ_RESTCLIENT_AVAILABLE
  auto serverHelper = registry::get<ServerHelper>(qpuName);
  serverHelper->initialize(serverConfig);

  std::vector<ExecutionResult> results;
  for (auto &id : jobs) {
    cudaq::info("Future retrieving results for {}.", id.first);
    const auto &jobId = id.first;
    serverHelper->waitForJobCompletion(jobId);
    auto c = serverHelper->getExecutionResult(jobId);

    if (isObserve) {
      // Use the job name instead of the global register.
      results.emplace_back(c.to_map(), id.second);
      results.back().sequentialData = c.sequential_data();
    } else {
      if (c.has_expectation()) {
        // If the QPU returns the data with expectation values, just use it
        // directly.
        // This can be the case for remote emulation/simulation providers who
        // compute the expectation value for us.
        return c;
      }

      // For each register, add the results into result.
      for (auto &regName : c.register_names()) {
        results.emplace_back(c.to_map(regName), regName);
        results.back().sequentialData = c.sequential_data(regName);
      }
    }
  }

  return sample_result(results);
#else
  throw std::runtime_error("cudaq::details::future::get() requires REST Client "
                           "but CUDA-Q was built without it.");
  return sample_result();
#endif
}

future &future::operator=(future &other) {
  jobs = other.jobs;
  qpuName = other.qpuName;
  serverConfig = other.serverConfig;
  isObserve = other.isObserve;
  if (other.wrapsFutureSampling) {
    wrapsFutureSampling = true;
    inFuture = std::move(other.inFuture);
  }
  return *this;
}

future &future::operator=(future &&other) {
  jobs = other.jobs;
  qpuName = other.qpuName;
  serverConfig = other.serverConfig;
  isObserve = other.isObserve;
  if (other.wrapsFutureSampling) {
    wrapsFutureSampling = true;
    inFuture = std::move(other.inFuture);
  }
  return *this;
}

std::ostream &operator<<(std::ostream &os, future &f) {
  if (f.wrapsFutureSampling)
    throw std::runtime_error(
        "Cannot persist a cudaq::future for a local kernel execution.");

  nlohmann::json j;
  j["jobs"] = f.jobs;
  j["qpu"] = f.qpuName;
  j["config"] = f.serverConfig;
  j["isObserve"] = f.isObserve;
  os << j.dump(4);
  return os;
}

std::istream &operator>>(std::istream &is, future &f) {
  nlohmann::json j;
  try {
    is >> j;
  } catch (...) {
    throw std::runtime_error(
        "Formatting error; could not parse input as json.");
  }
  f.jobs = j["jobs"].get<std::vector<future::Job>>();
  f.qpuName = j["qpu"].get<std::string>();
  f.serverConfig = j["config"].get<std::map<std::string, std::string>>();
  f.isObserve = j["isObserve"].get<bool>();
  return is;
}

} // namespace cudaq::details
