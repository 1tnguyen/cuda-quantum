/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Executor.h"
#include "common/Logger.h"

namespace cudaq {
details::future Executor::execute(std::vector<KernelExecution> &codesToExecute,
                                  bool isObserve) {

  serverHelper->setShots(shots);

  cudaq::info("Executor creating {} jobs to execute with the {} helper.",
              codesToExecute.size(), serverHelper->name());

  // Create the Job Payload, composed of job post path, headers,
  // and the job json messages themselves

  auto config = serverHelper->getConfig();

  std::vector<details::future::Job> ids;
  for (std::size_t i = 0; auto &codeToExecute : codesToExecute) {
    cudaq::info("Job (name={}) created", codeToExecute.name);

    // Post it, get the response
    // auto response = client.post(jobPostPath, "", job, headers);
    // cudaq::info("Job (name={}) posted, response was {}", codesToExecute[i].name,
    //             response.dump());

    // Add the job id and the job name.
    auto task_id = serverHelper->submitJob(codeToExecute);
   
    cudaq::info("Task ID is {}", task_id);
    ids.emplace_back(task_id, codeToExecute.name);
    config["output_names." + task_id] = codeToExecute.output_names.dump();

    nlohmann::json jReorder = codeToExecute.mapping_reorder_idx;
    config["reorderIdx." + task_id] = jReorder.dump();

    i++;
  }

  config.insert({"shots", std::to_string(shots)});
  std::string name = serverHelper->name();
  return details::future(ids, name, config, isObserve);
}
} // namespace cudaq

LLVM_INSTANTIATE_REGISTRY(cudaq::Executor::RegistryType)
