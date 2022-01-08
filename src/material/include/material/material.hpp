#ifndef RAY_TRACER_MATERIAL_MATERIAL_HPP_
#define RAY_TRACER_MATERIAL_MATERIAL_HPP_

#include "material/dielectric.hpp"
#include "material/lambertian.hpp"
#include "material/metal.hpp"
#include "ray/ray.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

namespace RayTracer
{

struct HitRecord;

/// The base class for all materials.
class Material
{
public:
    /// @brief Construct an empty material
    __device__ __host__ Material()
    {
    }

    /// @brief Construct a lambertian material
    ///
    /// @param material The lambertian material
    __device__ __host__ explicit Material(Lambertian material)
    {
        material_.type = MaterialTypes::kLambertian;
        material_.data.lambertian = material;
    }

    /// @brief Construct a metal material
    ///
    /// @param material The metal material
    __device__ __host__ explicit Material(Metal material)
    {
        material_.type = MaterialTypes::kMetal;
        material_.data.metal = material;
    }

    /// @brief Construct a dielectricmaterial
    ///
    /// @param material The dielectric material
    __device__ __host__ explicit Material(Dielectric material)
    {
        material_.type = MaterialTypes::kDielectric;
        material_.data.dielectric = material;
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
                            curandState &random_state) const
    {
        switch (material_.type) {
        case MaterialTypes::kLambertian:
            return material_.data.lambertian.scatter(
                input_ray, record, attenuation, scattered_ray, random_state);
        case MaterialTypes::kMetal:
            return material_.data.metal.scatter(input_ray, record, attenuation,
                                                scattered_ray, random_state);
        case MaterialTypes::kDielectric:
            return material_.data.dielectric.scatter(
                input_ray, record, attenuation, scattered_ray, random_state);
        }
        return false;
    }

private:
    /// Contains a union of potential material types and the actual data
    struct MaterialTypes {

        /// The enum of potential types of materials
        enum { kLambertian, kMetal, kDielectric } type;

        /// The actual data of the material
        union Materials {
            Lambertian lambertian;
            Metal metal;
            Dielectric dielectric;
            __device__ __host__ Materials()
            {
            }
            __device__ __host__ ~Materials()
            {
            }
        } data;
    };

    /// The material
    MaterialTypes material_;
};

} // namespace RayTracer

#endif // RAY_TRACER_MATERIAL_MATERIAL_HPP_
