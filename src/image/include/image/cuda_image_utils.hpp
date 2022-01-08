/// @file cuda_image_utils.hpp
/// @brief Contains useful functions for working with images
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

/// @brief Writes a colour at a specified (u, v) coordinate
///
/// @param image_buffer The image buffer to write to
/// @param properties Contains properties describing the above image buffer
/// @param colour The colour to write
/// @param u The u coordinate to write to
/// @param v The v coordinate to write to
/// @param samples_per_pixel The number of samples per pixel that was usede
__device__ void writeColourAt(u8 *image_buffer,
                              const ImageProperties &properties,
                              const Colour &colour, s64 u, s64 v,
                              int samples_per_pixel = 1);

/// @brief Writes a gamma corrected colour at a specified (u, v) coordinate
///
/// @param image_buffer The image buffer to write to
/// @param properties Contains properties describing the above image buffer
/// @param colour The colour to write
/// @param u The u coordinate to write to
/// @param v The v coordinate to write to
/// @param samples_per_pixel The number of samples per pixel that was usede
__device__ void writeGammaCorrectedColourAt(u8 *image_buffer,
                                            const ImageProperties &properties,
                                            const Colour &colour, s64 u, s64 v,
                                            int samples_per_pixel = 1);

/// @brief Initializes an array of random states for each pixel in an image
///
/// @param random_state The random state buffer
/// @param properties Contains properties describing the image
void initializeImageRandomState(curandState *random_state,
                                const ImageProperties &properties);
} // namespace cuda

} // namespace RayTracer

#endif // RAY_TRACER_IMAGE_CUDA_IMAGE_UTILS_HPP_
