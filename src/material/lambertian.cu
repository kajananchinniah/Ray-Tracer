#include "material/lambertian.hpp"
#include "surface/hit_record.hpp"

namespace RayTracer
{

__device__ bool Lambertian::scatter(const Ray &input_ray,
                                    const HitRecord &record,
                                    Colour &attenuation, Ray &scattered_ray,
                                    curandState &random_state) const
{
    Vector3f scatter_direction{record.normal +
                               cuda::randomUnitVector(random_state)};
    if (scatter_direction.nearZero_device()) {
        scatter_direction = record.normal;
    }
    scattered_ray = Ray{record.point, scatter_direction};
    attenuation = albedo;
    return true;
}

} // namespace RayTracer
