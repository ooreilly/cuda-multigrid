cmake_minimum_required(VERSION 3.10)
project(POISSON_MULTIGRID VERSION 1.0 LANGUAGES CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if (DEFINED ENV{ARCH})
        set(ARCH $ENV{ARCH})
else()
        set(ARCH sm_70)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O4 -g -use_fast_math -Xcompiler -fopenmp -std=c++11 -arch=${ARCH} -Xptxas=-v -lineinfo")
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=gnu11 -D${ARCH}")

#find_package(MPI REQUIRED)
include_directories(src)
include_directories(test)
include(CTest)
enable_testing()

add_subdirectory(test)


