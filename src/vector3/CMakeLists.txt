set(RAYTRACER_VECTOR3_SOURCE vector3.cpp)

add_library(raytracer_vector3 ${RAYTRACER_VECTOR3_SOURCE})
target_include_directories(raytracer_vector3 PUBLIC include)
