#ifndef RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_
#define RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_

#include "common/common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "ray.hpp"
#include "vector3/vector3f.hpp"

namespace RayTracer
{

namespace cuda
{

Colour getRayColour(const Ray &ray)
{
    Vector3f unit_direction = normalize_device(ray.direction());
    f32 t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * Colour{1.0f, 1.0f, 1.0f} + t * Colour{0.5f, 0.7f, 1.0f};
}

} // namespace cuda

Colour getRayColour(const Ray &ray)
{
    Vector3f unit_direction = normalize_host(ray.direction());
    f32 t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * Colour{1.0f, 1.0f, 1.0f} + t * Colour{0.5f, 0.7f, 1.0f};
}

} // namespace RayTracer

#endif // RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_
