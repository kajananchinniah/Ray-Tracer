add_library(raytracer_surface sphere.cu cuda_sphere_array_utils.cu)
target_link_libraries(raytracer_surface PUBLIC raytracer_common
                                               raytracer_vector3 raytracer_ray)
target_include_directories(raytracer_surface PUBLIC include)
