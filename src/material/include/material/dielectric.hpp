#ifndef RAY_TRACER_MATERIAL_DIELECTRIC_HPP_
#define RAY_TRACER_MATERIAL_DIELECTRIC_HPP_

#include "ray/ray.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

namespace RayTracer
{

struct HitRecord;

class Dielectric
{
public:
    __device__ __host__ explicit Dielectric(f32 index_of_refraction)
        : index_of_refraction_{index_of_refraction}
    {
    }

    __device__ bool scatter(const Ray &input_ray, const HitRecord &record,
                            Colour &attenuation, Ray &scattered_ray,
                            curandState &random_state) const;

private:
    f32 index_of_refraction_{};
};

} // namespace RayTracer

#endif // RAY_TRACER_MATERIAL_DIELECTRIC_HPP_
