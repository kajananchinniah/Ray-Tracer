find_package(OpenCV REQUIRED)

add_library(raytracer_image image_utils.cpp image.cpp cuda_image_utils.cu)
target_link_libraries(
  raytracer_image
  PUBLIC raytracer_common raytracer_vector3
  PRIVATE ${OpenCV_LIBS})
target_include_directories(
  raytracer_image
  PUBLIC include
  PRIVATE ${OpenCV_INCLUDE_DIRS})
