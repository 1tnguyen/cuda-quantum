# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------- #
# CPU decoder server -- two OPNIC streams (decode data plane + configure control
# plane) serviced by a single poll thread (see decoder_server_cpu.cpp).
#
# Serves cuda-qx decoder RPCs by dlopen-ing the out-of-tree shim
# (libcudaqx-decoder-hostcall.so, built from
# realtime/examples/cudaqx_decoder_hostcall) and injecting its 4 HOST_CALL
# entries into its own two-ring dispatch loop.  Because the handlers are loaded
# at runtime, this target has NO build dependency on cuda-qx -- it links only
# the realtime runtime, the OPNIC SDK (header-only direct transport), and libdl.
# ---------------------------------------------------------------------------- #
add_executable(decoder_server_cpu
  decoder_server_cpu.cpp
)

target_include_directories(decoder_server_cpu
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../bridge/cpu          # opnic_bridge_cpu.hpp + opnic_direct_cpu.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cudaqx_decoder_hostcall  # shim C ABI header
)

target_link_libraries(decoder_server_cpu
  PRIVATE
    cudaq-realtime-opnic-cpu
    cudaq-realtime      # realtime dispatch ABI / CPU_RELAX
    Threads::Threads
    qm::opnic-cuda      # OPNIC SDK (header-only direct transport needs it at link)
    ${CMAKE_DL_LIBS}    # dlopen/dlsym the decoder shim
)
