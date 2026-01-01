##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

# Support standalone build
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
if(NOT DEFINED CYLON_VERSION)
    set(CYLON_VERSION 0.7.0)
endif()

# Options for different environments
option(GCYLON_USE_STATIC_CUDART "Use static CUDA runtime (needed for WSL2)" OFF)

# Use conda's CUDA compiler if available instead of system nvcc
# This MUST be set before project() for CMake to use the right compiler
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
    set(GCYLON_NVCC "$ENV{CONDA_PREFIX}/bin/nvcc")
    message(STATUS "Using conda nvcc: ${CMAKE_CUDA_COMPILER}")
else()
    # Fallback to system nvcc
    find_program(GCYLON_NVCC nvcc)
    if(GCYLON_NVCC)
        set(CMAKE_CUDA_COMPILER "${GCYLON_NVCC}")
    endif()
endif()

# Detect CUDA version from nvcc to determine supported architectures
set(CUDA_VERSION_MAJOR 11)
set(CUDA_VERSION_MINOR 0)
if(GCYLON_NVCC OR CMAKE_CUDA_COMPILER)
    set(_nvcc_to_check "${CMAKE_CUDA_COMPILER}")
    if(NOT _nvcc_to_check)
        set(_nvcc_to_check "${GCYLON_NVCC}")
    endif()
    execute_process(
        COMMAND ${_nvcc_to_check} --version
        OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
        ERROR_QUIET
    )
    if(NVCC_VERSION_OUTPUT MATCHES "release ([0-9]+)\\.([0-9]+)")
        set(CUDA_VERSION_MAJOR ${CMAKE_MATCH_1})
        set(CUDA_VERSION_MINOR ${CMAKE_MATCH_2})
        message(STATUS "Detected CUDA ${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
    endif()
endif()

# CUDA architectures - MUST be set before project() or enable_language(CUDA)
# Set architectures based on CUDA version
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CUDA_VERSION_MAJOR GREATER_EQUAL 13 OR (CUDA_VERSION_MAJOR EQUAL 12 AND CUDA_VERSION_MINOR GREATER_EQUAL 8))
        # CUDA 12.8+ supports Blackwell
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 120)
        message(STATUS "Enabling Blackwell architectures (sm_100, sm_120)")
    elseif(CUDA_VERSION_MAJOR EQUAL 12)
        # CUDA 12.x supports up to Hopper
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    elseif(CUDA_VERSION_MAJOR EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 8)
        # CUDA 11.8+ supports up to Hopper
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
    else()
        # Older CUDA - use conservative architectures
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    endif()
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# CUDA runtime library selection
if(GCYLON_USE_STATIC_CUDART)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
    message(STATUS "Using static CUDA runtime")
else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    message(STATUS "Using shared CUDA runtime")
endif()

# Enable both CXX and CUDA in project declaration
project(gcylon VERSION ${CYLON_VERSION} LANGUAGES CXX CUDA)

# Find the CUDAToolkit
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_ROOT: ${CUDAToolkit_LIBRARY_ROOT}")

# Find glog (0.7.x requires CMake integration for proper macros)
find_package(glog REQUIRED)
message(STATUS "Found glog: ${glog_VERSION}")

# Find MPI
find_package(MPI REQUIRED)

# Find Arrow
find_package(Arrow REQUIRED)
message(STATUS "Found Arrow: ${Arrow_VERSION}")

# Derive CCCL include path from CUDAToolkit or conda
# CCCL (CUDA C++ Core Libraries) is needed for RMM/cuDF headers
set(CCCL_FOUND FALSE)
if(CUDAToolkit_LIBRARY_ROOT)
    # Try standard CUDA toolkit locations
    foreach(CCCL_PATH
            "${CUDAToolkit_LIBRARY_ROOT}/targets/x86_64-linux/include/cccl"
            "${CUDAToolkit_LIBRARY_ROOT}/include/cccl"
            "${CUDAToolkit_INCLUDE_DIRS}/cccl")
        if(EXISTS "${CCCL_PATH}")
            set(CCCL_INCLUDE_DIR "${CCCL_PATH}")
            set(CCCL_FOUND TRUE)
            break()
        endif()
    endforeach()
endif()
# Try conda prefix as fallback
if(NOT CCCL_FOUND AND DEFINED ENV{CONDA_PREFIX})
    foreach(CCCL_PATH
            "$ENV{CONDA_PREFIX}/include/cccl"
            "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include/cccl")
        if(EXISTS "${CCCL_PATH}")
            set(CCCL_INCLUDE_DIR "${CCCL_PATH}")
            set(CCCL_FOUND TRUE)
            break()
        endif()
    endforeach()
endif()
if(CCCL_FOUND)
    message(STATUS "Found CCCL at: ${CCCL_INCLUDE_DIR}")
else()
    message(WARNING "CCCL not found. RMM headers may fail to compile.")
endif()

# If executed on github actions, set below flag to avoid linker error for cuda
if(DEFINED ENV{GITHUB_ACTIONS})
    set(CMAKE_CXX_FLAGS "-Wl,--allow-shlib-undefined")
endif()

message(STATUS "CONDA_PREFIX: $ENV{CONDA_PREFIX}")
if((NOT DEFINED ENV{CONDA_PREFIX}) AND (NOT DEFINED ENV{CONDA_BUILD}))
    message(SEND_ERROR "Neither CONDA_PREFIX nor CONDA_BUILD is set. Activate conda environment or use conda-build")
endif()

## defaults to release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED GCYLON_SIMD_LEVEL)
    set(GCYLON_SIMD_LEVEL "SSE4_2") # options: SSE4_2 | AVX512 | AVX2
endif()

# C++ standard - cuDF 24.x+ requires C++20
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CUDA_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++20")

if(NOT HAVE_EXCLUSIVE_SCAN)
    add_definitions(-DNEED_EXCLUSIVE_SCAN)
endif()

# Required by RMM for experimental memory resource support
add_definitions(-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

# Set up include directories from conda
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{BUILD_PREFIX}/include/libcudf/libcudacxx/" "$ENV{PREFIX}/include/libcudf/libcudacxx/")
    set(CONDA_INCLUDE_DIRS "$ENV{BUILD_PREFIX}/include" "$ENV{PREFIX}/include")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx/")
    set(CONDA_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
endif()

# Put the libraries and binaries that get built into directories at the top of the build
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(TEST_DATA_DIR ${GCYLON_SOURCE_DIR}/data)

set(CYLON_INCLUDE_DIR "../")

include_directories(
        ${CYLON_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CONDA_INCLUDE_DIRS}
        ${CUDF_CUDA_INCLUDE_DIR}
        ${CCCL_INCLUDE_DIR}
)

# Add link directories for cudf, rmm from conda
if(DEFINED ENV{CONDA_PREFIX})
    link_directories("$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Using conda libs from: $ENV{CONDA_PREFIX}/lib")
endif()

# Add link directories for pre-built cylon library (must be before add_library)
if(DEFINED ENV{CYLON_LIB_DIR})
    message(STATUS "Using CYLON_LIB_DIR: $ENV{CYLON_LIB_DIR}")
    link_directories($ENV{CYLON_LIB_DIR})
else()
    # Default: look in sibling build_cylon directory (relative to parent of gcylon source)
    set(CYLON_DEFAULT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../build_cylon/lib")
    if(EXISTS "${CYLON_DEFAULT_LIB}")
        message(STATUS "Found cylon library at: ${CYLON_DEFAULT_LIB}")
        link_directories("${CYLON_DEFAULT_LIB}")
    else()
        # Also try from cpp directory
        set(CYLON_DEFAULT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../../build_cylon/lib")
        if(EXISTS "${CYLON_DEFAULT_LIB}")
            message(STATUS "Found cylon library at: ${CYLON_DEFAULT_LIB}")
            link_directories("${CYLON_DEFAULT_LIB}")
        endif()
    endif()
endif()

add_library(gcylon SHARED
        cudf_buffer.hpp
        cudf_buffer.cpp
        all2all/cudf_all_to_all.cuh
        all2all/cudf_all_to_all.cu
        all2all/cudf_all_to_all.hpp
        all2all/cudf_all_to_all.cpp
        net/cudf_net_ops.hpp
        net/cudf_bcast.cpp
        net/cudf_serialize.hpp
        net/cudf_serialize.cpp
        net/cudf_gather.cpp
        sorting/deserialize.hpp
        sorting/deserialize.cpp
        sorting/sorting.hpp
        sorting/sorting.cpp
        utils/util.hpp
        utils/util.cpp
        gtable.hpp
        gtable.cpp
        gtable_api.hpp
        gtable_api.cpp
        # Chunked operations for memory-efficient processing
        gcylon_config.hpp
        staging/memory_utils.hpp
        staging/pinned_allocator.hpp
        gtable_api_chunked.cpp
        # C API wrapper for FFI
        c_api/gcylon_c.h
        c_api/gcylon_c.cpp
        )

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

set_target_properties(gcylon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gcylon PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gcylon PROPERTIES VERSION ${CYLON_VERSION})

# Link libraries using CMake targets where available
# Note: Don't add CUDA::cudart explicitly - nvcc handles it based on CMAKE_CUDA_RUNTIME_LIBRARY
target_link_libraries(gcylon
        glog::glog
        cylon
        cudf
        rmm
        MPI::MPI_CXX
        Arrow::arrow_shared
)

# cylon_install_all_headers is defined in parent CMakeLists.txt
# For standalone builds, define a simple version
if(NOT COMMAND cylon_install_all_headers)
    function(cylon_install_all_headers DIR)
        file(GLOB_RECURSE HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
        install(FILES ${HEADERS} DESTINATION include/${DIR})
    endfunction()
endif()

cylon_install_all_headers("gcylon")

add_subdirectory(all2all)
add_subdirectory(utils)

install(TARGETS gcylon DESTINATION lib)
