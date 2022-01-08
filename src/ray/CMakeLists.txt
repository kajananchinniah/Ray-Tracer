add_library(raytracer_ray ray_utils.cu)
target_include_directories(raytracer_ray PUBLIC include)
target_link_libraries(raytracer_ray PUBLIC raytracer_common raytracer_vector3
                                           raytracer_surface)
set_property(TARGET raytracer_ray PROPERTY CUDA_SEPARABLE_COMPILATION ON)
