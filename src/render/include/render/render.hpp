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
void render(const Camera camera, SphereArray &sphere_array,
            s32 samples_per_pixel, s32 max_depth, Image &output_image);
} // namespace cuda
} // namespace RayTracer

#endif // RAY_TRACER_RENDER_RENDER_HPP_
