#ifndef RAY_TRACER_MATERIAL_METAL_HPP_
#define RAY_TRACER_MATERIAL_METAL_HPP_

#include "common/common_types.hpp"
#include "ray/ray.hpp"
#include "vector3/vector3.hpp"

#include "curand_kernel.h"

namespace RayTracer
{

struct HitRecord;

/// A metal material
struct Metal {
    /// @brief Constructs a metal object
    ///
    /// @param a The albedo of the metal
    /// @param f The fuzz of the metal
    __device__ __host__ Metal(const Colour &a, f32 f)
        : albedo{a}, fuzz{f < 1 ? f : 1}
    {
    }

    /// The albedo of the metal
    Colour albedo;

    /// The fuzz of the metal
    f32 fuzz;

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

#endif // RAY_TRACER_MATERIAL_METAL_HPP_
