add_subdirectory(common)
add_subdirectory(vector3)
add_subdirectory(ray)
add_subdirectory(material)
add_subdirectory(surface)
add_subdirectory(camera)
add_subdirectory(image)
add_subdirectory(render)

add_executable(ray_tracer main.cu)
target_link_libraries(
  ray_tracer
  PRIVATE raytracer_common
          raytracer_vector3
          raytracer_ray
          raytracer_material
          raytracer_surface
          raytracer_camera
          raytracer_image
          raytracer_render)
