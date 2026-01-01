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

add_definitions(-DEXAMPLE_CMAKE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

set(CMAKE_CUDA_COMPILER $ENV{CUDACXX})

# project(GCYLON LANGUAGES CXX CUDA)
enable_language(CUDA)

set("GCYLON_SIMD_LEVEL" "SSE4_2") # options: SSE4_2| AVX512 | AVX2
# CUDA architectures for CUDA 13.x (dropped older arch support)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# C++ standard
# Newer cuDF (24.x+) requires C++20 for 'requires' clauses
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_REQUIRED_FLAGS "-std=c++20")

# Required by RMM for experimental memory resource support
add_definitions(-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

# find CUDA
find_package(CUDAToolkit REQUIRED)

# If executed on github actions, set below flag to avoid linker error for cuda
# ref: https://github.com/NVIDIA/nvidia-docker/issues/775
if(DEFINED ENV{GITHUB_ACTIONS})
    set(CMAKE_CXX_FLAGS "-Wl,--allow-shlib-undefined")
endif()

# if conda build is used
if("$ENV{CONDA_BUILD}" STREQUAL "1")
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{BUILD_PREFIX}/include/libcudf/libcudacxx/" "$ENV{PREFIX}/include/libcudf/libcudacxx/")
elseif(DEFINED ENV{CONDA_PREFIX})
    set(CUDF_CUDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include/libcudf/libcudacxx/")
endif("$ENV{CONDA_BUILD}" STREQUAL "1")

# Derive CCCL include path from CUDAToolkit (CUDA 12+ includes CCCL)
if(CUDAToolkit_LIBRARY_ROOT)
    set(CCCL_INCLUDE_DIR "${CUDAToolkit_LIBRARY_ROOT}/targets/x86_64-linux/include/cccl")
    if(NOT EXISTS ${CCCL_INCLUDE_DIR})
        set(CCCL_INCLUDE_DIR "${CUDAToolkit_LIBRARY_ROOT}/include/cccl")
    endif()
endif()

include_directories(../../
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CONDA_INCLUDE_DIRS}
        ${CUDF_CUDA_INCLUDE_DIR}
        ${CCCL_INCLUDE_DIR}
        )

# Add link directories for pre-built cylon library
if(DEFINED ENV{CYLON_LIB_DIR})
    link_directories($ENV{CYLON_LIB_DIR})
else()
    set(CYLON_DEFAULT_LIB "${CMAKE_SOURCE_DIR}/build_cylon/lib")
    if(EXISTS "${CYLON_DEFAULT_LIB}")
        link_directories("${CYLON_DEFAULT_LIB}")
    endif()
endif()

#build a binary with cuda
macro(cylon_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp ../../gcylon/utils/construct.hpp print.hpp)
    target_link_libraries(${EXENAME} gcylon cudf rmm)
    install(TARGETS ${EXENAME} RUNTIME DESTINATION gcylon/examples)
endmacro(cylon_add_exe)

cylon_add_exe(gjoin)
cylon_add_exe(gjoin_csv)
cylon_add_exe(gshuffle)
cylon_add_exe(gsort)
