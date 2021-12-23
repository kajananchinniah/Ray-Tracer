set(RAYTRACER_IMAGE_UTILS_SOURCE image_utils.cpp)

find_package(OpenCV REQUIRED)

add_library(raytracer_image_utils ${RAYTRACER_IMAGE_UTILS_SOURCE}) 
target_link_libraries(raytracer_image_utils PUBLIC raytracer_common PRIVATE ${OpenCV_LIBS})
target_include_directories(raytracer_image_utils PUBLIC include PRIVATE ${OpenCV_INCLUDE_DIRS})
