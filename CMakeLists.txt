cmake_minimum_required(VERSION 3.10)

project(ray_tracer LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

option(BUILD_DOCUMENTATION "Generate doxygen documentation" OFF)
option(BUILD_TESTS "Build unit test (requires GTest)" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

include(CMakeModules/ClangTidy.cmake)
include(docs/Doxygen.cmake)
include(CTest)

add_subdirectory(src)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
