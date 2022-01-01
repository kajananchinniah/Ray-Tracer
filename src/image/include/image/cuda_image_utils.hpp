#ifndef RAY_TRACER_IMAGE_CUDA_IMAGE_UTILS_HPP_
#define RAY_TRACER_IMAGE_CUDA_IMAGE_UTILS_HPP_

#include "image.hpp"

#include "common/common_types.hpp"
#include "vector3/vector3.hpp"

#include "curand_kernel.h"

namespace RayTracer
{
namespace cuda
{

__device__ void writeColourAt(u8 *image_buffer,
                              const ImageProperties &properties,
                              const Colour &colour, s64 u, s64 v,
                              int samples_per_pixel = 1);

void initializeImageRandomState(curandState *random_state,
                                const ImageProperties &properties);
} // namespace cuda

} // namespace RayTracer

#endif // RAY_TRACER_IMAGE_CUDA_IMAGE_UTILS_HPP_
