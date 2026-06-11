# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

if (NOT TARGET cudaq-realtime-cpu-transport)
  return()
endif()

find_package(CUDAToolkit REQUIRED)

add_library(cudaq-realtime-bridge-test-cpu-roce SHARED
  cpu_roce_bridge.cpp)

target_include_directories(cudaq-realtime-bridge-test-cpu-roce PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(cudaq-realtime-bridge-test-cpu-roce PRIVATE
  cudaq-realtime-cpu-transport
  Threads::Threads)

set_target_properties(cudaq-realtime-bridge-test-cpu-roce PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_executable(hsb_bridge_cpu_bridge
  hsb_bridge_cpu_bridge.cpp
  ${CMAKE_SOURCE_DIR}/unittests/cpu_transport/init_rpc_increment_function_table_host.cpp)

target_include_directories(hsb_bridge_cpu_bridge PRIVATE
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})

target_compile_definitions(hsb_bridge_cpu_bridge PRIVATE
  CUDAQ_REALTIME_CPU_ROCE_BRIDGE_LIB="$<TARGET_FILE:cudaq-realtime-bridge-test-cpu-roce>")

target_link_libraries(hsb_bridge_cpu_bridge PRIVATE
  cudaq-realtime
  cudaq-realtime-host-dispatch
  Threads::Threads)

set_target_properties(hsb_bridge_cpu_bridge PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/utils)

add_dependencies(hsb_bridge_cpu_bridge
  cudaq-realtime-bridge-test-cpu-roce)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hsb_test_cpu_bridge.sh
               ${CMAKE_BINARY_DIR}/unittests/utils/hsb_test_cpu_bridge.sh COPYONLY)

message(STATUS "  - hsb_bridge_cpu_bridge (CPU RoCE through bridge interface)")
