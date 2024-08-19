/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/run.h"
#include "utils/OpaqueArguments.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <fmt/core.h>
#include <pybind11/stl.h>
#include "py_run_async.h"

namespace py = pybind11;

namespace cudaq {
py::object pyAltLaunchKernelR(const std::string &, MlirModule, MlirType,
                              OpaqueArguments &,
                              const std::vector<std::string> &);

using PyRunResult = cudaq::RunResult<py::object>;
using PyRunAsyncResult = std::future<std::vector<PyRunResult>>;

void bindRunAsync(py::module &mod) {
  py::class_<PyRunResult>(mod, "RunResult", "TODO")
      .def(py::init([](py::object obj, bool isOk = true) {
             PyRunResult result(obj);
             return result;
           }),
           py::arg("value"), py::arg("isOk") = true)
      .def(
          "isOk", [](PyRunResult &self) { return self.isOk(); },
          py::call_guard<py::gil_scoped_release>(), "TODO")
      .def(
          "get", [](PyRunResult &self) { return self.get(); },
          py::call_guard<py::gil_scoped_release>(), "TODO")
      .def(
          "getError", [](PyRunResult &self) { return self.getError(); },
          py::call_guard<py::gil_scoped_release>(), "TODO");

  py::class_<PyRunAsyncResult>(mod, "AsyncRunResult", "TODO")
      .def(
          "get", [](PyRunAsyncResult &self) { return self.get(); },
          py::call_guard<py::gil_scoped_release>(), "TODO");

  mod.def(
      "run_async",
      [&](py::object kernel, py::args args, std::size_t shots,
          std::size_t qpu_id, std::optional<noise_model> noise) {
        kernel.inc_ref();
        auto &platform = cudaq::get_platform();
        if (py::hasattr(kernel, "compile"))
          kernel.attr("compile")();

        auto kernelName = kernel.attr("name").cast<std::string>();
        auto kernelMod = kernel.attr("module").cast<MlirModule>();
        auto kernelFunc = getKernelFuncOp(kernelMod, kernelName);

        args = simplifiedValidateInputArguments(args);
        auto *argData = new cudaq::OpaqueArguments();
        cudaq::packArgs(*argData, args, kernelFunc,
                        [](OpaqueArguments &, py::object &) { return false; });

        // The function below will be executed multiple times
        // if the kernel has conditional feedback. In that case,
        // we have to be careful about deleting the `argData` and
        // only do so after the last invocation of that function.

        // Should only have C++ going on here, safe to release the GIL
        py::gil_scoped_release release;
        std::promise<std::vector<PyRunResult>> promise;
        auto fut = promise.get_future();
        auto returnTy = kernelFunc.getFunctionType().getResult(0);

        // Wrapped it as a generic (returning void) function
        QuantumTask wrapped = detail::make_copyable_function(
            [p = std::move(promise), qpu_id, shots, &platform, kernelMod,
             &kernel, &kernelName, &returnTy, argData]() mutable {
              std::vector<PyRunResult> results;
              results.reserve(shots);
              for (std::size_t i = 0; i < shots; ++i) {
                results.emplace_back(PyRunResult(pyAltLaunchKernelR(
                    kernelName, kernelMod, wrap(returnTy), *argData, {})));
              }
              p.set_value(std::move(results));
            });

        platform.enqueueAsyncTask(qpu_id, wrapped);

        return fut;
      },
      py::arg("kernel"), py::kw_only(), py::arg("shots_count") = 1,
      py::arg("qpu_id") = 0, py::arg("noise") =  py::none(),
      R"#(Asynchronously sample the state of the provided `kernel` at the 
specified number of circuit executions (`shots_count`).
When targeting a quantum platform with more than one QPU, the optional
`qpu_id` allows for control over which QPU to enable. Will return a
future whose results can be retrieved via `future.get()`.

Args:
  kernel (:class:`Kernel`): The :class:`Kernel` to execute `shots_count` 
    times on the QPU.
  *arguments (Optional[Any]): The concrete values to evaluate the kernel 
    function at. Leave empty if the kernel doesn't accept any arguments.
  shots_count (Optional[int]): The number of kernel executions on the 
    QPU. Defaults to 1000. Key-word only.
  qpu_id (Optional[int]): The optional identification for which QPU 
    on the platform to target. Defaults to zero. Key-word only.

Returns:
  :class:`AsyncSampleResult`: 
  A dictionary containing the measurement count results for the :class:`Kernel`.)#");
}
} // namespace cudaq
