#ifndef RAY_TRACER_MATERIAL_METAL_HPP_
#define RAY_TRACER_MATERIAL_METAL_HPP_

#include "common/common_types.hpp"
#include "ray/ray.hpp"
#include "vector3/vector3.hpp"

#include "curand_kernel.h"

namespace RayTracer
{

struct HitRecord;

struct Metal {
    __device__ __host__ Metal(const Colour &a, f32 f)
        : albedo{a}, fuzz{f < 1 ? f : 1}
    {
    }

    Colour albedo;
    f32 fuzz;

    __device__ bool scatter(const Ray &input_ray, const HitRecord &record,
                            Colour &attenuation, Ray &scattered_ray,
                            curandState &random_state) const;
};

} // namespace RayTracer

#endif // RAY_TRACER_MATERIAL_METAL_HPP_
