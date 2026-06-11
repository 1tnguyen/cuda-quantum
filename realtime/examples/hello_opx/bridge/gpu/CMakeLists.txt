# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(opnic)
find_package(CUDAToolkit REQUIRED)

# GPU OPNIC bridge (zero-copy direct path).  Compiled into a static library so
# consumers link it and include opnic_bridge.cuh, instead of #including the .cu.
add_library(cudaq-realtime-opnic-gpu STATIC
  opnic_bridge.cu
)

set_target_properties(cudaq-realtime-opnic-gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "80;86;89;90"
  CUDA_STANDARD 20
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cudaq-realtime-opnic-gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_include_directories(cudaq-realtime-opnic-gpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                    # opnic_bridge.cuh, opnic_direct.cuh, opnic_kernels.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpu             # opnic_type.h
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
)

target_link_libraries(cudaq-realtime-opnic-gpu
  PRIVATE
    cudaq-realtime
    CUDA::cudart
    qm::opnic-cuda
)
