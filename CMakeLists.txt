# Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.18)

# Set HIP compiler before project() on Linux; on Windows the caller sets the
# all-clang toolchain via CMAKE_CXX_COMPILER before invoking add_subdirectory.
if(NOT WIN32 AND NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
endif()

project(
	tiny-rocm-nn
	VERSION 1.0
	DESCRIPTION "Lightning fast & tiny C++/ROCm neural network framework"
	LANGUAGES CXX HIP
)

option(TCNN_BUILD_BENCHMARK "Build tiny-rocm-nn example benchmark?" OFF)
option(TCNN_BUILD_EXAMPLES "Build tiny-rocm-nn example applications?" ON)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# Setup ROCm dependencies
###############################################################################

# Find ROCm packages; on Windows the HIP SDK lives in the TheRock devel tree
# found via CMAKE_PREFIX_PATH -- HIP_INCLUDE_DIRS is set by find_package(hip).
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)
find_package(rocblas REQUIRED)
find_package(hipblas REQUIRED)

if(WIN32)
    # On Windows, use HIP_INCLUDE_DIRS (set by find_package(hip)) instead of
    # the hardcoded Linux path.
    list(GET HIP_INCLUDE_DIRS 0 ROCM_INCLUDE_DIR)
    set(ROCM_INCLUDE_DIR "${ROCM_INCLUDE_DIR}/")
    get_filename_component(ROCM_ROOT_DIR "${ROCM_INCLUDE_DIR}/.." ABSOLUTE)
else()
    set(ROCM_ROOT_DIR "/opt/rocm/")
    set(ROCM_INCLUDE_DIR "/opt/rocm/include/")
endif()

# Include directories - add all ROCm include paths
include_directories(${ROCM_INCLUDE_DIR})
include_directories(${ROCM_INCLUDE_DIR}/hipblas)
include_directories(${ROCM_INCLUDE_DIR}/rocblas)
include_directories(${ROCM_INCLUDE_DIR}/rocwmma)
include_directories(${HIP_INCLUDE_DIRS})

###############################################################################
# Dependencies
###############################################################################

if (NOT MSVC)
	set(ROCM_TOOLKIT_ROOT_DIR /opt/rocm)
endif()

set(BUILD_SHARED_LIBS OFF)
add_subdirectory("dependencies/fmt")

###############################################################################
# tiny-rocm-nn library, samples, and benchmarks
###############################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

set(TCNN_SOURCES
	src/common_host.cpp
	src/cpp_api.cpp
	src/encoding.cpp
	src/fully_fused_mlp.cpp
	src/loss.cpp
	src/network.cpp
	src/object.cpp
	src/optimizer.cpp
	src/reduce_sum.cpp
)

###############################################################################
# Linker / library
###############################################################################

# Define TCNN_MIN_GPU_ARCH for ROCm (use a value that enables all features)
# For AMD GPUs, we use 75 to enable FullyFusedMLP compilation (architecture check is bypassed in code)
# Note: AMD GPUs use different architecture numbering (gfx908, gfx90a, etc.), but we use
# NVIDIA's numbering for compatibility with preprocessor checks in the codebase
list(APPEND TCNN_DEFINITIONS -DTCNN_MIN_GPU_ARCH=75)

add_library(tiny-rocm-nn STATIC ${TCNN_SOURCES})
target_compile_definitions(tiny-rocm-nn PUBLIC ${TCNN_DEFINITIONS})
target_include_directories(tiny-rocm-nn PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies"
	"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/fmt/include"
	${ROCM_INCLUDE_DIR}
	${ROCM_INCLUDE_DIR}/hipblas
	${ROCM_INCLUDE_DIR}/rocblas
	${ROCM_INCLUDE_DIR}/rocwmma
)
target_link_libraries(tiny-rocm-nn PUBLIC hip::device hip::host roc::rocblas roc::hipblas fmt)

# Propagate HIP architecture to this target so rocWMMA selects the correct
# backend (MFMA on gfx9, WMMA on RDNA). CMAKE_HIP_ARCHITECTURES is set by the
# top-level DDN-SLAM build or instant-ngp-kf CMakeLists before this subdir runs.
# Note: HIP_ARCHITECTURES target property only applies to .cu files; the .cpp
# sources compiled with -x hip need an explicit --offload-arch flag instead.
if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set_target_properties(tiny-rocm-nn PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    # Build the arch list as --offload-arch=<arch> flags (one list element per arch)
    set(_TCNN_OFFLOAD_ARCH_FLAGS)
    foreach(_arch ${CMAKE_HIP_ARCHITECTURES})
        list(APPEND _TCNN_OFFLOAD_ARCH_FLAGS "--offload-arch=${_arch}")
    endforeach()
    target_compile_options(tiny-rocm-nn PRIVATE ${_TCNN_OFFLOAD_ARCH_FLAGS})
endif()

# instant-ngp-kf links the target name "tiny-cuda-nn"; alias it to this HIP port so the
# vendored swap is transparent to the consumer's CMake.
if (NOT TARGET tiny-cuda-nn)
	add_library(tiny-cuda-nn ALIAS tiny-rocm-nn)
endif()

if (TCNN_BUILD_EXAMPLES)
	add_subdirectory("samples")
endif()

# test/ + benchmarks are standalone harnesses needing local data; skip when vendored.
if (TCNN_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
	add_subdirectory("test")
endif()
if (TCNN_BUILD_BENCHMARK)
	add_subdirectory("benchmarks/image")
endif()
