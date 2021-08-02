if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(RTAM VERSION 1.0)

cmake_minimum_required(VERSION 3.2)
if (NOT WIN32)
set (CMAKE_CXX_FLAGS "--std=c++11")
set (CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "")
include(CMake/configure_build_type.cmake)
include(CMake/configure_optix.cmake)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

include_directories(common)

set(optix_LIBRARY "")

add_subdirectory(test)
