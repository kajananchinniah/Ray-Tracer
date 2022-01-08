#ifndef RAY_TRACER_MATERIAL_DIELECTRIC_HPP_
#define RAY_TRACER_MATERIAL_DIELECTRIC_HPP_

#include "ray/ray.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

namespace RayTracer
{

// forward declaration (needed to get this to compile)
struct HitRecord;

/// A dielectric material
class Dielectric
{
public:
    /// @brief Construct a Dielectric material
    ///
    /// @param index_of_refraction The index of refraction of the material
    __device__ __host__ explicit Dielectric(f32 index_of_refraction)
        : index_of_refraction_{index_of_refraction}
    {
    }

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

private:
    f32 index_of_refraction_{};
};

} // namespace RayTracer

#endif // RAY_TRACER_MATERIAL_DIELECTRIC_HPP_
