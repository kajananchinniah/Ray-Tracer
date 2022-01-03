#include "material/metal.hpp"
#include "surface/hit_record.hpp"

namespace RayTracer
{

__device__ bool Metal::scatter(const Ray &input_ray, const HitRecord &record,
                               Colour &attenuation, Ray &scattered_ray,
                               curandState &random_state) const
{
    Vector3f reflected_ray =
        reflect(normalize_device(input_ray.direction()), record.normal);
    scattered_ray =
        Ray{record.point,
            reflected_ray + fuzz * cuda::randomInUnitSphere(random_state)};
    attenuation = albedo;
    return (dot(scattered_ray.direction(), record.normal) > 0);
}

} // namespace RayTracer
