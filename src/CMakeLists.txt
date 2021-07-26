project(RTAM)

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
)

find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED)

if (WIN32)
  add_definitions(-DNOMINMAX)
endif()

include_directories(
  "${CUDA_TOOLKIT_INCLUDE}"
  "${OptiX_INCLUDE}"
)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

add_subdirectory(test)

add_subdirectory(sutil)
