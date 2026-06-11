# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(opnic)

add_library(cudaq-realtime-opnic-cpu
  opnic_bridge_cpu.cpp
)

target_include_directories(cudaq-realtime-opnic-cpu
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cudaq-realtime-opnic-cpu
  PRIVATE
    cudaq-realtime
    Threads::Threads
    qm::opnic-cuda
)

