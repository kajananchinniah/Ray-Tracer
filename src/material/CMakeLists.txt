add_library(raytracer_material lambertian.cu metal.cu)
target_include_directories(raytracer_material PUBLIC include)
target_link_libraries(
  raytracer_material
  PUBLIC raytracer_ray raytracer_vector3
  PRIVATE raytracer_surface)
set_property(TARGET raytracer_material PROPERTY CUDA_SEPARABLE_COMPILATION ON)
