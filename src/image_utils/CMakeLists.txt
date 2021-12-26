find_package(OpenCV REQUIRED)

add_library(
  raytracer_image_utils
  image_utils.cpp image.cpp cuda_image_utils.cu include/image_utils/image.hpp
  include/image_utils/image_utils.hpp)
target_link_libraries(
  raytracer_image_utils
  PUBLIC raytracer_common raytracer_vector3
  PRIVATE ${OpenCV_LIBS})
target_include_directories(
  raytracer_image_utils
  PUBLIC include
  PRIVATE ${OpenCV_INCLUDE_DIRS})
