# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(_device_call_tcp_include_dir
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(cudaq-test-device-call-channel-tcp SHARED
  device_call_tcp_channel.cpp
  device_call_socket_utils.cpp)
target_include_directories(cudaq-test-device-call-channel-tcp
  PRIVATE
    ${_device_call_tcp_include_dir}
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CMAKE_SOURCE_DIR}/runtime
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cudaq-test-device-call-channel-tcp
  PRIVATE
    cudaq-device-call-runtime
    cudaq::cudaq-realtime
    LLVMSupport
    CUDA::cudart_static
    dl)
set_target_properties(cudaq-test-device-call-channel-tcp
  PROPERTIES
    OUTPUT_NAME cudaq-device-call-channel-tcp
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(cudaq-test-realtime-transport-tcp SHARED
  tcp_socket_provider.cpp
  device_call_socket_utils.cpp)
target_include_directories(cudaq-test-realtime-transport-tcp
  PRIVATE
    ${_device_call_tcp_include_dir}
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CMAKE_SOURCE_DIR}/runtime
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cudaq-test-realtime-transport-tcp
  PRIVATE
    cudaq-device-call-runtime
    cudaq::cudaq-realtime
    LLVMSupport
    CUDA::cudart_static)
set_target_properties(cudaq-test-realtime-transport-tcp
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_executable(cudaq-test-device-call-tcp-server
  device_call_tcp_server.cpp
  device_call_socket_utils.cpp)
target_include_directories(cudaq-test-device-call-tcp-server
  PRIVATE
    ${_device_call_tcp_include_dir}
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CMAKE_SOURCE_DIR}/runtime
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cudaq-test-device-call-tcp-server
  PRIVATE
    cudaq-device-call-runtime
    cudaq::cudaq-realtime
    LLVMSupport
    CUDA::cudart
    dl)
set_target_properties(cudaq-test-device-call-tcp-server
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
