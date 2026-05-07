/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

NB_MODULE(_quakeDialects, m) {
  auto quakeMod = m.def_submodule("quake");
  quakeMod.def(
      "register_dialect",
      [](bool load, MlirContext context) {
        MlirDialectHandle handle = mlirGetDialectHandle__quake__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      nanobind::arg("load") = true,
      nanobind::arg("context") = nanobind::none());

  auto ccMod = m.def_submodule("cc");
  ccMod.def(
      "register_dialect",
      [](bool load, MlirContext context) {
        MlirDialectHandle handle = mlirGetDialectHandle__cc__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      nanobind::arg("load") = true,
      nanobind::arg("context") = nanobind::none());

  m.def("register_all_dialects",
        [](MlirContext context) { cudaqRegisterAllDialects(context); },
        nanobind::arg("context") = nanobind::none());
}
