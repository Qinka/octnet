# Copyright (c) 2017, The OctNet authors
#               2018, Johann Lee
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

## CMAKE
cmake_minimum_required(VERSION 3.3)
project(octnet)


###
## Setting
###

set(CMAKE_MACOSX_RPATH 1)

## C/C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD   11)

## SSE arch
if(MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAG}   /arch:SSE /arch:SSE2 /arch:SSE3 /arch:SSE4.2 ")
  set(CMAKE_CXX_FLAGS "${CMAKE_XXC_FLAG} /arch:SSE /arch:SSE2 /arch:SSE3 /arch:SSE4.2 ")
else(MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -msse -msse2 -msse3 -msse4.2 -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -msse4.2 -fPIC")
endif(MSVC)

## Open MP
find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


###
## Options
###
option(
  USING_CUDA
  "Enable support of CUDA"
  OFF
  )
option(
  USING_TORCH
  "Enable support of torch"
  ON
  )
option(
  USING_CREATER
  "ENABLE tools of create datas"
  ON
  )

  variable_watch(USING_CUDA)
  variable_watch(USING_TORCH)



###
## Variables
###
#PATH
set(OCTNET_DIR       ${PROJECT_SOURCE_DIR})
set(OCTNET_SRC_DIR   ${OCTNET_DIR}/src)
set(CORE_DIR         ${OCTNET_SRC_DIR}/core)
set(CORE_CPU_DIR     ${CORE_DIR}/cpu)
set(CORE_GPU_DIR     ${CORE_DIR}/gpu)
set(CREATER_DIR      ${OCTNET_SRC_DIR}/create)
set(GEOMETRY_DIR      ${OCTNET_SRC_DIR}/geometry)
set(TH_DIR           ${OCTNET_SRC_DIR}/th)
set(TH_CPU_DIR       ${TH_DIR}/cpu)
set(TH_GPU_DIR       ${TH_DIR}/gpu)
set(OCTNET_INC_DIR   ${OCTNET_DIR}/include)
set(OCTNET_TEST_DIR  ${OCTNET_DIR}/test)

# SRC
aux_source_directory(${CORE_CPU_DIR} CORE_CPU_SRC)
aux_source_directory(${CORE_GPU_DIR} CORE_GPU_SRC)
aux_source_directory(${CREATER_DIR}  CREATER_SRC)
aux_source_directory(${GEOMETRY_DIR} GEOMETRY_SRC)
aux_source_directory(${TH_CPU_DIR}   TH_CPU_SRC)
aux_source_directory(${TH_GPU_DIR}   TH_GPU_SRC)
file(GLOB CORE_GPU_SRC_F "${CORE_GPU_DIR}/*.cu")
file(GLOB TH_GPU_SRC_F   "${TH_GPU_DIR}/*.cu")

set(OCTNET_SRC ${CORE_CPU_SRC})
if (USING_CUDA)
  set(OCTNET_SRC ${OCTNET_SRC} ${CORE_GPU_SRC_F} ${CORE_GPU_SRC})
endif(USING_CUDA)
if (USING_CREATER)
  set(OCTNET_SRC ${OCTNET_SRC} ${CREATER_SRC} ${GEOMETRY_SRC})
endif(USING_CREATER)
if (USING_TORCH)
  set(OCTNET_SRC ${OCTNET_SRC} ${TH_CPU_SRC})
  if(USING_CUDA)
    set(OCTNET_SRC ${OCTNET_SRC} ${TH_GPU_SRC} ${TH_GPU_SRC_F})
  endif(USING_CUDA)
endif(USING_TORCH)

# find cuda
if(USING_CUDA)
  find_package(CUDA 8.0 REQUIRED)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -Wall")
endif(USING_CUDA)

# find torch
if (USING_TORCH)
  find_package(Torch REQUIRED)
  include_directories(${TH_INCLUDE_DIR})
  include_directories("${Torch_INSTALL_INCLUDE}/THC")
  link_directories("${Torch_INSTALL_LIB}")
  set(TORCH_LIBS TH THC)
endif(USING_TORCH)

# include Path
include_directories(${OCTNET_INC_DIR})

if(USING_CUDA)
  cuda_add_library(octnet SHARED ${OCTNET_SRC})
  CUDA_ADD_CUBLAS_TO_TARGET(octnet)
else(USING_CUDA)
  add_library(octnet SHARED ${OCTNET_SRC})
endif(USING_CUDA)
#define
if(MSVC)
  target_compile_definitions(octnet PUBLIC /D OCTREE_EXPORT)
endif(MSVC)
# torch
if (USING_TORCH)
  target_link_libraries(octnet ${TORCH_LIBS})
endif(USING_TORCH)

ENABLE_testing()

add_executable(test_octree ${OCTNET_TEST_DIR}/test_octree.cc)
target_link_libraries(test_octree octnet)

add_executable(test_create ${OCTNET_TEST_DIR}/test_create.cpp)
target_link_libraries(test_create octnet ${OctNetCore_LIBRARY})



add_test(OCTNET_TEST test_octnet)
add_test(OCTNET_TEST test_create)


if(USING_CUDA)
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
endif(USING_CUDA)
