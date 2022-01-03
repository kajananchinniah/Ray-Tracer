#include "material/dielectric.hpp"
#include "surface/hit_record.hpp"

#include "curand_kernel.h"

namespace
{

using namespace RayTracer;
__device__ inline f32 reflectance(f32 cosine, f32 index_of_refraction)
{
    f32 r0{(1.0f - index_of_refraction) / (1 + index_of_refraction)};
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf(1 - cosine, 5);
}

} // namespace

namespace RayTracer
{

__device__ bool Dielectric::scatter(const Ray &input_ray,
                                    const HitRecord &record,
                                    Colour &attenuation, Ray &scattered_ray,
                                    curandState &random_state) const
{
    attenuation = Colour{1.0f, 1.0f, 1.0f};
    f32 refraction_ratio = record.front_face ? (1.0f / index_of_refraction_)
                                             : index_of_refraction_;
    Vector3f unit_direction = normalize_device(input_ray.direction());
    f32 cos_theta = fminf(dot(-unit_direction, record.normal), 1.0f);
    f32 sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
    Vector3f direction{};

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) >
                              curand_uniform(&random_state)) {
        direction = reflect(unit_direction, record.normal);
    } else {
        direction =
            refract_device(unit_direction, record.normal, refraction_ratio);
    }

    scattered_ray = Ray{record.point, direction};
    return true;
}

} // namespace RayTracer
