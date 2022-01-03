#ifndef RAY_TRACER_MATERIAL_LAMBERTIAN_HPP_
#define RAY_TRACER_MATERIAL_LAMBERTIAN_HPP_

#include "curand_kernel.h"
#include "ray/ray.hpp"
#include "vector3/cuda_vector3_utils.hpp"
#include "vector3/vector3.hpp"

namespace RayTracer
{

struct HitRecord;

struct Lambertian {
    __device__ __host__ Lambertian(const Colour &a) : albedo{a}
    {
    }

    Colour albedo;

    __device__ bool scatter(const Ray &input_ray, const HitRecord &record,
                            Colour &attenuation, Ray &scattered_ray,
                            curandState &random_state) const;
};

} // namespace RayTracer

#endif // RAY_TRACER_MATERIAL_LAMBERTIAN_HPP_
