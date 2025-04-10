# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set(LIBRARY_NAME cudaq-nlopt)

add_subdirectory(nlopt-src SYSTEM)

add_library(${LIBRARY_NAME} SHARED nlopt.cpp)
set_property(GLOBAL APPEND PROPERTY CUDAQ_RUNTIME_LIBS ${LIBRARY_NAME})
target_include_directories(${LIBRARY_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/runtime)
target_include_directories(${LIBRARY_NAME} SYSTEM PRIVATE nlopt-src/ nlopt-src/src/api)
target_link_libraries(${LIBRARY_NAME} PRIVATE nlopt)

install(TARGETS ${LIBRARY_NAME} EXPORT cudaq-nlopt-targets DESTINATION lib)

install(EXPORT cudaq-nlopt-targets
        FILE CUDAQNloptTargets.cmake
        NAMESPACE cudaq::
        DESTINATION lib/cmake/cudaq)
