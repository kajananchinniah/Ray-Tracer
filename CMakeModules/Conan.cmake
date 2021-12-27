if(NOT EXISTS "${CMAKE_BINARY_DIRS}/conan.cmake")
  message(STATUS "Downloading conan cmake")
  file(
    DOWNLOAD
    "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
    "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(
  REQUIRES
  opencv/4.5.3
  gtest/1.11.0
  BASIC_SETUP
  NO_OUTPUT_DIRS
  CMAKE_TARGETS
  OPTIONS
  BUILD
  missing
  GENERATORS
  cmake_find_package)
