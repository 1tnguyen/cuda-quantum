/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Demangle.h"
#include "llvm\Demangle\Demangle.h"

std::string cudaq::demangle(const std::string &mangledName) {
  return llvm::demangle(mangledName);
}