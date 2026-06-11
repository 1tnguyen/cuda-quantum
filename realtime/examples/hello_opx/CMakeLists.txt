# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(opnic)
if (NOT opnic_FOUND)
  message(FATAL_ERROR "opnic package not found, cannot build hello_opx examples.
  Please ensure opnic is installed and set `-Dopnic_DIR=<path_to_opnic_installation>`.")
endif()
find_package(OpenSSL COMPONENTS Crypto REQUIRED) # Required by opnic

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(bridge/cpu)
if (CUDA_FOUND)
  add_subdirectory(bridge/gpu)
endif()

# The incrementer example builds the CPU (host dispatch) variant always and the
# GPU (device dispatch) variant when CUDA is found; each links the matching
# bridge library (cudaq-realtime-opnic-cpu / -gpu).
add_subdirectory(incrementer)
