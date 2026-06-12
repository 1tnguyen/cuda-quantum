# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.   #
# ============================================================================ #
#
# Standalone build of the CUDA-QX realtime HOST_CALL shim. Builds entirely on
# top of the cuda-qx wheel (links libcudaq-qec-realtime-decoding) and the
# CUDA-Q realtime dispatch headers -- cuda-qx itself is not modified.
#
#   cmake -S . -B build \
#     -DCUDAQX_INSTALL_DIR=<cuda-qx wheel site-packages> \
#     -DCUDAQX_SOURCE_INCLUDE_DIR=<fetched cudaqx source include dir> \
#     -DCUDAQ_REALTIME_INCLUDE_DIR=<cuda-quantum realtime include dir>
#   cmake --build build
#
# Produces libcudaqx-decoder-hostcall.so exporting the cudaqx_rt_* C ABI.

cmake_minimum_required(VERSION 3.20)
project(cudaqx-decoder-hostcall LANGUAGES CXX)

# decoding_config.h uses C++20 (defaulted operator==).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CUDAQX_INSTALL_DIR)
  message(FATAL_ERROR
    "Set -DCUDAQX_INSTALL_DIR=<cuda-qx wheel site-packages dir>.")
endif()
if(NOT DEFINED CUDAQX_SOURCE_INCLUDE_DIR)
  message(FATAL_ERROR
    "Set -DCUDAQX_SOURCE_INCLUDE_DIR=<cudaqx source headers include dir>.")
endif()
if(NOT DEFINED CUDAQ_REALTIME_INCLUDE_DIR)
  message(FATAL_ERROR
    "Set -DCUDAQ_REALTIME_INCLUDE_DIR=<cuda-quantum realtime dispatch include dir>.")
endif()

# cudaq_realtime.h includes <cuda_runtime.h> for the ABI types (no CUDA language
# is compiled here -- just the header).
find_package(CUDAToolkit REQUIRED)

set(CUDAQX_QEC_LIB_DIR ${CUDAQX_INSTALL_DIR}/cudaq_qec/lib)
set(CUDAQ_LIB_DIR ${CUDAQX_INSTALL_DIR}/lib)
set(CUDAQ_AUDITWHEEL_LIB_DIR ${CUDAQX_INSTALL_DIR}/cuda_quantum_cu12.libs)

function(find_required_wheel_library out name)
  find_library(${out}
    NAMES ${name}
    HINTS
      ${CUDAQX_QEC_LIB_DIR}
      ${CUDAQ_LIB_DIR}
      ${CUDAQ_AUDITWHEEL_LIB_DIR}
    REQUIRED)
endfunction()

find_library(CUDAQX_QEC_REALTIME_DECODING
  NAMES cudaq-qec-realtime-decoding
  HINTS ${CUDAQX_QEC_LIB_DIR}
  REQUIRED)
find_required_wheel_library(CUDAQX_QEC cudaq-qec)
find_required_wheel_library(CUDAQX_QEC_REALTIME_DECODING_SIM
  cudaq-qec-realtime-decoding-simulation)
find_required_wheel_library(CUDAQ_RUNTIME cudaq)
find_required_wheel_library(CUDAQ_PLATFORM_DEFAULT cudaq-platform-default)
find_required_wheel_library(NVQIR_RUNTIME nvqir)

add_library(cudaqx-decoder-hostcall SHARED cudaqx_decoder_hostcall.cpp)

target_compile_definitions(cudaqx-decoder-hostcall PRIVATE CUDAQX_RT_BUILD)
target_compile_options(cudaqx-decoder-hostcall PRIVATE
  -fvisibility=hidden -fvisibility-inlines-hidden)

target_include_directories(cudaqx-decoder-hostcall PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CUDAQX_SOURCE_INCLUDE_DIR}
  ${CUDAQX_INSTALL_DIR}/include
  ${CUDAQ_REALTIME_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(cudaqx-decoder-hostcall PRIVATE
  ${NVQIR_RUNTIME}
  ${CUDAQ_RUNTIME}
  ${CUDAQ_PLATFORM_DEFAULT}
  ${CUDAQX_QEC}
  ${CUDAQX_QEC_REALTIME_DECODING_SIM}
  ${CUDAQX_QEC_REALTIME_DECODING})

target_link_options(cudaqx-decoder-hostcall PRIVATE
  -Wl,--no-as-needed
  -Wl,-rpath-link,${CUDAQX_QEC_LIB_DIR}
  -Wl,-rpath-link,${CUDAQ_LIB_DIR}
  -Wl,-rpath-link,${CUDAQ_AUDITWHEEL_LIB_DIR})

set_target_properties(cudaqx-decoder-hostcall PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
