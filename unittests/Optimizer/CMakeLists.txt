# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

include(HandleLLVMOptions)

add_executable(OptimizerUnitTests HermitianTrait.cpp)

target_link_libraries(OptimizerUnitTests
  PRIVATE
    QuakeDialect
    gtest_main
)

gtest_discover_tests(OptimizerUnitTests)

add_executable(test_quake_synth QuakeSynthTester.cpp)
target_link_libraries(test_quake_synth
  PRIVATE 
  cudaq
  cudaq-operator
  cudaq-mlir-runtime
  cudaq-builder
  cudaq-common
  cudaq-platform-default
  nvqir-qpp 
  nvqir
  cudaq-em-default
  gtest_main)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
  target_link_options(test_quake_synth PRIVATE -Wl,--no-as-needed)
endif()
gtest_discover_tests(test_quake_synth)
