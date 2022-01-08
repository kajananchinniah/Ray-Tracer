add_library(raytracer_render render.cu)
target_include_directories(raytracer_render PUBLIC include)
target_link_libraries(
  raytracer_render
  PUBLIC raytracer_common
         raytracer_ray
         raytracer_material
         raytracer_vector3
         raytracer_surface
         raytracer_camera
         raytracer_image)
set_property(TARGET raytracer_render PROPERTY CUDA_SEPARABLE_COMPILATION ON)
