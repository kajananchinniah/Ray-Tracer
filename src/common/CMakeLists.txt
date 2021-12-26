add_library(raytracer_common cuda_memory_utils.cu)
target_include_directories(raytracer_common PUBLIC include)
set_target_properties(raytracer_common PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS
                                                  ON)
