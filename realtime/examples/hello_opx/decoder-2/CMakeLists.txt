# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Paths to the local CUDA-QX and CUDA-Q installs. Override on the cmake
# command line if your installs are elsewhere.
set(CUDAQX_INSTALL_DIR "$ENV{HOME}/.cudaqx" CACHE PATH
    "Root of the local CUDA-QX install (~/.cudaqx)")
set(CUDAQ_INSTALL_DIR "/usr/local/cudaq" CACHE PATH
    "Root of the local CUDA-Q install (/usr/local/cudaq)")

set(CUDAQX_LIB_DIR  "${CUDAQX_INSTALL_DIR}/lib")
set(CUDAQX_PLUGIN_DIR "${CUDAQX_INSTALL_DIR}/lib/decoder-plugins")
set(CUDAQ_LIB_DIR   "${CUDAQ_INSTALL_DIR}/lib")

# ---------------------------------------------------------------------------- #
# Locate required CUDA-QX / CUDA-Q libraries.
# ---------------------------------------------------------------------------- #

function(find_required_lib out name)
  find_library(${out}
    NAMES ${name}
    HINTS ${CUDAQX_LIB_DIR} ${CUDAQ_LIB_DIR}
    REQUIRED)
endfunction()

find_required_lib(LIB_CUDAQX_QEC_REALTIME      cudaq-qec-realtime-decoding)
find_required_lib(LIB_CUDAQX_QEC_REALTIME_SIM  cudaq-qec-realtime-decoding-simulation)
find_required_lib(LIB_CUDAQX_QEC               cudaq-qec)
find_required_lib(LIB_CUDAQ                    cudaq)
find_required_lib(LIB_CUDAQ_PLATFORM           cudaq-platform-default)
find_required_lib(LIB_NVQIR                    nvqir)

# ---------------------------------------------------------------------------- #
# CPU decoder server -- single OPNIC data stream, host-side config.
#
# Compile the CUDA-QX decoder handlers directly into the binary (no dlopen).
# The server reads --config=<yaml> at startup and calls the CUDA-QX config
# API before any QUA program runs.
# ---------------------------------------------------------------------------- #
add_executable(decoder_2_cpu
  src/decoder_server_cpu.cpp
  src/decoder_handlers.cpp
)

set_target_properties(decoder_2_cpu PROPERTIES OUTPUT_NAME "decoder-2_cpu")

target_compile_definitions(decoder_2_cpu PRIVATE)

target_include_directories(decoder_2_cpu
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}            # realtime dispatch ABI headers
    ${CUDAQX_INSTALL_DIR}/include            # cudaq/qec/* headers
    ${CUDAQ_INSTALL_DIR}/include             # cuda-qx/core/* headers
    ${CUDAToolkit_INCLUDE_DIRS}              # <cuda_runtime.h> (needed by ABI hdr)
    ${CMAKE_CURRENT_SOURCE_DIR}/src          # decoder_handlers.h, decoder_packets.h, ...
    ${CMAKE_CURRENT_SOURCE_DIR}/../bridge/cpu # opnic_bridge_cpu.hpp + opnic_direct_cpu.hpp
)

target_link_libraries(decoder_2_cpu
  PRIVATE
    cudaq-realtime-opnic-cpu
    cudaq-realtime
    Threads::Threads
    qm::opnic-cuda
    ${LIB_NVQIR}
    ${LIB_CUDAQ}
    ${LIB_CUDAQ_PLATFORM}
    ${LIB_CUDAQX_QEC}
    ${LIB_CUDAQX_QEC_REALTIME_SIM}
    ${LIB_CUDAQX_QEC_REALTIME}
)

target_link_options(decoder_2_cpu
  PRIVATE
    -Wl,--no-as-needed
    -Wl,-rpath-link,${CMAKE_BINARY_DIR}/lib
    -Wl,-rpath-link,${CUDAQX_LIB_DIR}
    -Wl,-rpath-link,${CUDAQX_PLUGIN_DIR}
    -Wl,-rpath-link,${CUDAQ_LIB_DIR}
    # Embed runtime rpaths so the binary finds libraries without LD_LIBRARY_PATH
    -Wl,-rpath,${CMAKE_BINARY_DIR}/lib
    -Wl,-rpath,${CUDAQX_LIB_DIR}
    -Wl,-rpath,${CUDAQX_PLUGIN_DIR}
    -Wl,-rpath,${CUDAQ_LIB_DIR}
)
