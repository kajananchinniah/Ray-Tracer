#ifndef RAY_TRACER_SURFACE_CUDA_SPHERE_ARRAY_UTILS_HPP_
#define RAY_TRACER_SURFACE_CUDA_SPHERE_ARRAY_UTILS_HPP_

#include "common/common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "hit_record.hpp"
#include "ray/ray.hpp"
#include "sphere.hpp"
#include "sphere_array.hpp"

namespace RayTracer
{
namespace cuda
{

__device__ bool hitSphereArray(Sphere *sphere_array,
                               SphereArrayProperties sphere_array_properties,
                               const Ray &ray, f32 t_min, f32 t_max,
                               HitRecord &record);

} // namespace cuda
} // namespace RayTracer

#endif // RAY_TRACER_SURFACE_CUDA_SPHERE_ARRAY_UTILS_HPP_
