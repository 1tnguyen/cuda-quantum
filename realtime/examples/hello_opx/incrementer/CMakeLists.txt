# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------- #
# CPU incrementer -- host dispatch.
#   default    CUDAQ_EXEC_HOST          (3-thread ring via bridge launch)
#   --unified  CUDAQ_EXEC_HOST_UNIFIED  (library generic unified loop)
# ---------------------------------------------------------------------------- #
add_executable(incrementer_cpu
  incrementer_cpu.cpp
)

target_include_directories(incrementer_cpu
  PRIVATE
    ${CUDAQ_REALTIME_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../bridge/cpu  # opnic_bridge_cpu.hpp + transport ctx
)

target_link_libraries(incrementer_cpu
  PRIVATE
    cudaq-realtime-opnic-cpu  # OPNIC CPU bridge (create_context + interface)
    cudaq-realtime            # descriptor/factory API + host dispatch
    Threads::Threads
    qm::opnic-cuda            # OPNIC SDK (static bridge lib needs it at final link)
)

# ---------------------------------------------------------------------------- #
# GPU incrementer -- device dispatch.
#   default    CUDAQ_EXEC_GPU_PERSISTENT (3-kernel: RX + dispatch + TX)
#   --unified  CUDAQ_EXEC_GPU_UNIFIED    (library generic unified kernel)
# The GPU bridge (opnic_bridge.cu) and opnic_direct.cuh are #included directly
# from ../bridge/gpu; opnic_type.h is shared from ../bridge/cpu.
# ---------------------------------------------------------------------------- #
if (CUDA_FOUND)
  add_executable(incrementer_gpu
    incrementer_gpu.cu
  )

  # The OPNIC SDK headers require C++20 (consteval / concepts / fold-exprs).
  set_target_properties(incrementer_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89;90"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
  )

  target_compile_options(incrementer_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )

  target_include_directories(incrementer_gpu
    PRIVATE
      ${CUDAQ_REALTIME_INCLUDE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/../bridge/gpu   # opnic_bridge.cuh, opnic_direct.cuh, opnic_kernels.cuh
      ${CMAKE_CURRENT_SOURCE_DIR}/../bridge/cpu   # opnic_type.h
  )

  target_link_libraries(incrementer_gpu
    PRIVATE
      cudaq-realtime-opnic-gpu   # GPU OPNIC bridge (create/connect/launch + getter)
      cudaq-realtime
      cudaq-realtime-dispatch
      CUDA::cudart
      qm::opnic-cuda
  )
endif()
