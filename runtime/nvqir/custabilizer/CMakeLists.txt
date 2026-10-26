# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

message(STATUS "cuStabilizer_FOUND - building custabilizer NVQIR backend.")
message(STATUS "Found custabilizer version: ${cuStabilizer_VERSION}")

# Preserve native cuStabilizer lookup after installation, as for other cuQuantum
# backends. Relative paths below cover cuStabilizer installed next to the wheel.
get_target_property(_cust_loc cuStabilizer::cuStabilizer IMPORTED_LOCATION)
get_filename_component(_cust_dir "${_cust_loc}" DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${_cust_dir}")

set(LIBRARY_NAME nvqir-custabilizer)
set(INTERFACE_POSITION_INDEPENDENT_CODE ON)

add_library(${LIBRARY_NAME} SHARED CustabilizerCircuitSimulator.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/nvqir-custabilizer.map "
{
  global:
    getCircuitSimulator;
    getCircuitSimulator_custabilizer;
  local:
    *;
};
")

target_include_directories(${LIBRARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/runtime>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_SOURCE_DIR}/tpls/Stim/src)

target_link_libraries(${LIBRARY_NAME}
  PRIVATE
    libstim
    fmt::fmt-header-only
    cudaq-logger
    cudaq-common
    cuStabilizer::cuStabilizer
    CUDA::cudart_static)

target_link_options(${LIBRARY_NAME}
  PRIVATE
    "-Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/nvqir-custabilizer.map")

set_target_properties(${LIBRARY_NAME}
  PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    INSTALL_RPATH "${CMAKE_INSTALL_RPATH};$ORIGIN/../cuquantum/lib;$ORIGIN/../../cuquantum/lib")

install(TARGETS ${LIBRARY_NAME} DESTINATION lib COMPONENT Runtime)

add_target_config(custabilizer)
