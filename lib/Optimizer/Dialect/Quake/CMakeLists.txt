# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_cudaq_dialect_library(QuakeDialect
  QuakeDialect.cpp
  QuakeInterfaces.cpp
  QuakeOps.cpp
  QuakeTypes.cpp

  DEPENDS
  QuakeDialectIncGen
  QuakeOpsIncGen
  QuakeTypesIncGen

  LINK_LIBS
  CCDialect
  MLIRFuncDialect
  MLIRIR
)

