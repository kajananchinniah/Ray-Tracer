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

class Material
{
public:
    __device__ __host__ Material()
    {
    }

    __device__ __host__ explicit Material(Lambertian material)
    {
        material_.type = MaterialTypes::kLambertian;
        material_.data.lambertian = material;
    }

    __device__ __host__ explicit Material(Metal material)
    {
        material_.type = MaterialTypes::kMetal;
        material_.data.metal = material;
    }

    __device__ __host__ explicit Material(Dielectric material)
    {
        material_.type = MaterialTypes::kDielectric;
        material_.data.dielectric = material;
    }

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
    struct MaterialTypes {
        enum { kLambertian, kMetal, kDielectric } type;
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

    MaterialTypes material_;
};

} // namespace RayTracer

#endif // RAY_TRACER_MATERIAL_MATERIAL_HPP_
