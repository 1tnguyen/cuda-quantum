/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ServerHelper.h"
#include "Logger.h"
namespace cudaq {
void ServerHelper::parseConfigForCommonParams(const BackendConfig &config) {
  // Parse common parameters for each job and place into member variables
  for (auto &[key, val] : config) {
    // First Form a newKey with just the portion after the "." (i.e. jobId)
    auto ix = key.find_first_of('.');
    std::string newKey;
    if (ix != key.npos)
      newKey = key.substr(ix + 1);

    if (key.starts_with("output_names.")) {
      // Parse `val` into jobOutputNames.
      // Note: See `FunctionAnalysisData::resultQubitVals` of
      // LowerToQIRProfile.cpp for an example of how this was populated.
      OutputNamesType jobOutputNames;
      nlohmann::json outputNamesJSON = nlohmann::json::parse(val);
      for (const auto &el : outputNamesJSON[0]) {
        auto result = el[0].get<std::size_t>();
        auto qubitNum = el[1][0].get<std::size_t>();
        auto registerName = el[1][1].get<std::string>();
        jobOutputNames[result] = {qubitNum, registerName};
      }

      this->outputNames[newKey] = jobOutputNames;
    } else if (key.starts_with("reorderIdx.")) {
      nlohmann::json tmp = nlohmann::json::parse(val);
      this->reorderIdx[newKey] = tmp.get<std::vector<std::size_t>>();
    }
  }
}
std::string ServerHelperBase::submitJob(const KernelExecution &codeToExecute) {
  // Create the Job Payload, composed of job post path, headers,
  // and the job json messages themselves
  std::vector<KernelExecution> codesToExecute{codeToExecute};
  auto [jobPostPath, headers, jobs] = createJob(codesToExecute);
  auto &job = jobs[0];
  // Post it, get the response
  auto response = client.post(jobPostPath, "", job, headers);
  cudaq::info("Job (name={}) posted, response was {}", codeToExecute.name,
              response.dump());
  // Add the job id and the job name.
  auto task_id = extractJobId(response);
  if (task_id.empty()) {
    nlohmann::json tmp(job.at("tasks"));
    task_id = tmp[0].at("task_id");
  }
  cudaq::info("Task ID is {}", task_id);
  return task_id;
}

void ServerHelperBase::waitForJobCompletion(const std::string &jobId) {
  auto headers = getHeaders();

  auto jobGetPath = constructGetJobPath(jobId);

  auto resultResponse = client.get(jobGetPath, "", headers);
  while (!jobIsDone(resultResponse)) {
    auto polling_interval = nextResultPollingInterval(resultResponse);
    std::this_thread::sleep_for(polling_interval);
    resultResponse = client.get(jobGetPath, "", headers);
  }
}

cudaq::sample_result
ServerHelperBase::getExecutionResult(const std::string &jobId) {
  auto headers = getHeaders();

  auto jobGetPath = constructGetJobPath(jobId);
  auto resultResponse = client.get(jobGetPath, "", headers);
  if (!jobIsDone(resultResponse)) {
    throw std::runtime_error(
        fmt::format("Job {} is not done, cannot retrieve results.", jobId));
  }
  return processResults(resultResponse, jobId);
}

} // namespace cudaq

LLVM_INSTANTIATE_REGISTRY(cudaq::ServerHelper::RegistryType)
