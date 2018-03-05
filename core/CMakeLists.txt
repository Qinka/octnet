# Copyright (c) 2017, The OctNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.3)
set(CMAKE_MACOSX_RPATH 1)

set(CMAKE_CXX_STANDARD 11)

option(ENABLE_CUDA"Enable CUDA" OFF)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
if(MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAG}   /arch:SSE /arch:SSE2 /arch:SSE3 /arch:SSE4.2 ")
  set(CMAKE_CXX_FLAGS "${CMAKE_XXC_FLAG} /arch:SSE /arch:SSE2 /arch:SSE3 /arch:SSE4.2 ")
else(MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -msse -msse2 -msse3 -msse4.2 -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -msse4.2 -fPIC")
endif(MSVC)


find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories("include/")
if(ENABLE_CUDA)
  find_package(CUDA 8.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -Wall")
endif(ENABLE_CUDA)


set(OCTREE_CPU_SRCS 
  src/core.cc
  src/io.cc
  src/o2d.cc
  src/d2o.cc
  src/conv.cc
  src/pool.cc
  src/gridpool.cc
  src/gridunpool.cc
  src/math.cc
  src/misc.cc
  src/concat.cc
  src/combine.cc
  src/activations.cc
  src/split.cc
  src/loss.cc
  src/volumetric_upsampling.cc
)

set(OCTREE_GPU_SRCS 
  src/core.cu
  src/o2d.cu
  src/d2o.cu
  src/conv.cu
  src/oc2col.cu
  src/col2oc.cu
  src/conv_mm.cu
  src/pool.cu
  src/gridpool.cu
  src/gridunpool.cu
  src/activations.cu
  src/split.cu
  src/math.cu
  src/misc.cu
  src/concat.cu
  src/combine.cu
  src/loss.cu
  src/volumetric_upsampling.cu
)

if(ENABLE_CUDA)
cuda_add_library(octnet_core SHARED ${OCTREE_GPU_SRCS} ${OCTREE_CPU_SRCS})
CUDA_ADD_CUBLAS_TO_TARGET(octnet_core)
else(ENABLE_CUDA)
add_library(octnet_core SHARED ${OCTREE_CPU_SRCS})
endif(ENABLE_CUDA)

if(MSVC)
target_compile_definitions(octnet_core PUBLIC /D OCTREE_EXPORT)
endif(MSVC)

add_executable(test_octree test/test_octree.cc)
target_link_libraries(test_octree octnet_core)


if(ENABLE_CUDA)
  # performance tests
  # cuda_add_executable(test_data_idx_speed test/test_data_idx_speed.cu)
  # target_link_libraries(test_data_idx_speed ${OctNetCore_LIBRARY} octnet_core)
  # cuda_add_executable(test_n_leafs_speed test/test_n_leafs_speed.cu)
  # target_link_libraries(test_n_leafs_speed ${OctNetCore_LIBRARY} octnet_core_gpu)
  # cuda_add_executable(test_leaf_idx_to_grid_idx test/test_leaf_idx_to_grid_idx.cu)
  # target_link_libraries(test_leaf_idx_to_grid_idx ${OctNetCore_LIBRARY} octnet_core_gpu)
  # cuda_add_executable(test_data_idx_to_bit_idx_speed test/test_data_idx_to_bit_idx_speed.cu)
  # target_link_libraries(test_data_idx_to_bit_idx_speed ${OctNetCore_LIBRARY} octnet_core_gpu)
  # cuda_add_executable(test_conv_mm test/test_conv_mm.cu)
  # target_link_libraries(test_conv_mm ${OctNetCore_LIBRARY} octnet_core_gpu)
  # cuda_add_cublas_to_target(test_conv_mm)
endif(ENABLE_CUDA)