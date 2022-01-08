#ifndef RAY_TRACER_MATERIAL_LAMBERTIAN_HPP_
#define RAY_TRACER_MATERIAL_LAMBERTIAN_HPP_

#include "curand_kernel.h"
#include "ray/ray.hpp"
#include "vector3/cuda_vector3_utils.hpp"
#include "vector3/vector3.hpp"

namespace RayTracer
{

struct HitRecord;

/// A Lambertian material
struct Lambertian {
    /// @brief Constructs a Lambertian material
    ///
    /// @param a The albedo of the material
    __device__ __host__ Lambertian(const Colour &a) : albedo{a}
    {
    }

    /// The albedo of the material
    Colour albedo;

    /// @brief Tries to produce a scattered ray from an incident ray
    /// and returns whether or not it succeeded (e.g. it might be fully
    /// absorbed)
    ///
    /// @param input_ray The incident ray
    /// @param record Contains useful information
    /// @param attenuation Gives the attenuation of the new ray
    /// @param scattered_ray The ray that has been scattered
    /// @param random_state The random state
    __device__ bool scatter(const Ray &input_ray, const HitRecord &record,
                            Colour &attenuation, Ray &scattered_ray,
                            curandState &random_state) const;
};

} // namespace RayTracer

#endif // RAY_TRACER_MATERIAL_LAMBERTIAN_HPP_
