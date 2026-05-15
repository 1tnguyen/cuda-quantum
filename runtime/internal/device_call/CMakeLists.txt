# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

add_library(cudaq-device-call-runtime-headers INTERFACE)
target_include_directories(cudaq-device-call-runtime-headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if (CUDA_FOUND AND CUDAQ_ENABLE_REALTIME)
  target_link_libraries(cudaq-device-call-runtime-headers INTERFACE
    cudaq::cudaq-realtime)

  add_library(cudaq-device-call-runtime
    SHARED
      device_call_dispatch.cpp
      device_call_channel.cpp
      device_call_local_channel.cpp
      device_call_gpu_dispatch_channel.cpp
      device_call_host_dispatch_channel.cpp
      device_call_mapped_ring_buffer.cpp
      device_call_realtime.cpp
      device_call_rpc_frame.cpp
      device_call_service_utils.cpp)

  set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS
    cudaq-device-call-runtime)

  target_include_directories(cudaq-device-call-runtime
    PRIVATE
      ${CMAKE_SOURCE_DIR}/runtime)

  target_link_libraries(cudaq-device-call-runtime
    PUBLIC
      cudaq-device-call-runtime-headers
    PRIVATE
      cudaq::cudaq-realtime
      LLVMSupport
      CUDA::cudart_static
      dl)

  install(TARGETS cudaq-device-call-runtime-headers
          EXPORT cudaq-targets)

  install(TARGETS cudaq-device-call-runtime
          EXPORT cudaq-targets
          DESTINATION lib)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
          DESTINATION include
          PATTERN "ArgParsing.h" EXCLUDE)
endif()
