# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

find_path(CUSTABILIZER_INCLUDE_DIR
  NAMES custabilizer.h
  HINTS ${CUSTABILIZER_ROOT} ENV CUSTABILIZER_ROOT
        ENV CUQUANTUM_INSTALL_PREFIX
        ENV CUSTABILIZER_INCLUDE_DIR
  PATH_SUFFIXES include)
find_library(CUSTABILIZER_LIBRARY
  NAMES custabilizer libcustabilizer.so.0
  HINTS ${CUSTABILIZER_ROOT} ENV CUSTABILIZER_ROOT
        ENV CUQUANTUM_INSTALL_PREFIX
        ENV CUSTABILIZER_LIBRARY
  PATH_SUFFIXES lib lib64 build/lib)
if(NOT CUSTABILIZER_INCLUDE_DIR OR NOT CUSTABILIZER_LIBRARY)
  if(CUSTABILIZER_ROOT OR DEFINED ENV{CUSTABILIZER_ROOT})
    message(FATAL_ERROR "CUSTABILIZER_ROOT does not contain cuStabilizer headers and library")
  endif()
  message(STATUS "cuStabilizer not found; the optional custabilizer target is unavailable")
  return()
endif()

find_package(CUDAToolkit REQUIRED)

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
    ${CUSTABILIZER_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/tpls/Stim/src)

target_link_libraries(${LIBRARY_NAME}
  PRIVATE
    libstim
    fmt::fmt-header-only
    cudaq-logger
    cudaq-common
    ${CUSTABILIZER_LIBRARY}
    CUDA::cudart)

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
