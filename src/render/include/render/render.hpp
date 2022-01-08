/// @file render.hpp
/// @brief Contains functions to render
#ifndef RAY_TRACER_RENDER_RENDER_HPP_
#define RAY_TRACER_RENDER_RENDER_HPP_

#include "camera/camera.hpp"
#include "common/common_types.hpp"
#include "image/image.hpp"
#include "surface/sphere_array.hpp"

namespace RayTracer
{
namespace cuda
{
/// @brief render
///
/// @param camera The camera
/// @param sphere_array The array of spheres
/// @param samples_per_pixel The number of samples per pixel to take. THis is
///        really computational expensive
/// @param max_depth The max depth to bounce rays to
/// @param[out] output_image The image that will be modified
void render(const Camera camera, SphereArray &sphere_array,
            s32 samples_per_pixel, s32 max_depth, Image &output_image);
} // namespace cuda
} // namespace RayTracer

#endif // RAY_TRACER_RENDER_RENDER_HPP_
