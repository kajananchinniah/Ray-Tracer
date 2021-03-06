#include "ray/ray_utils.hpp"

namespace RayTracer
{

namespace cuda
{

__device__ f32 hitSphere(const Point3f &center, f32 radius, const Ray &ray)
{
    Vector3f vec_origin_center = ray.origin() - center;
    f32 a = ray.direction().magnitude_squared();
    f32 b_over_2 = dot(vec_origin_center, ray.direction());
    f32 c = vec_origin_center.magnitude_squared() - radius * radius;
    f32 discriminant = b_over_2 * b_over_2 - a * c;
    if (discriminant < 0) {
        return -1.0f;
    } else {
        return (-b_over_2 - sqrt(discriminant)) / a;
    }
}
__device__ Colour getRayColourBasic(const Ray &ray)
{
    Vector3f unit_direction = normalize_device(ray.direction());
    f32 t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * Colour{1.0f, 1.0f, 1.0f} + t * Colour{0.5f, 0.7f, 1.0f};
}

__device__ Colour getRayColourWithRedSphere(const Ray &ray)
{
    f32 t = hitSphere(Point3f{0.0f, 0.0f, -1.0f}, 0.5f, ray);
    if (t > 0.0f) {
        Vector3f N = normalize_device(ray.at(t) - Vector3f{0.0f, 0.0f, -1.0f});
        return 0.5f * Colour{N.x() + 1.0f, N.y() + 1.0f, N.z() + 1.0f};
    } else {
        return getRayColourBasic(ray);
    }
}

__device__ Colour
getRayColourWithSphereArray(const Ray &ray, Sphere *sphere_array,
                            SphereArrayProperties sphere_array_properties)
{
    HitRecord record;
    if (hitSphereArray(sphere_array, sphere_array_properties, ray, 0, infinity,
                       record)) {
        return 0.5f * (record.normal + Colour{1.0f, 1.0f, 1.0f});
    }

    Vector3f unit_direction = normalize_device(ray.direction());
    f32 t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * Colour{1.0f, 1.0f, 1.0f} + t * Colour{0.5f, 0.7f, 1.0f};
}

__device__ Colour
getRayColourWithDiffuse(const Ray &ray, Sphere *sphere_array,
                        SphereArrayProperties sphere_array_properties,
                        s64 max_depth, curandState &random_state)
{
    Ray current_ray = ray;
    Colour current_colour{1.0f, 1.0f, 1.0f};
    while (max_depth > 0) {
        HitRecord record;

        if (hitSphereArray(sphere_array, sphere_array_properties, current_ray,
                           0.001f, infinity, record)) {
            Point3f target{record.point + record.normal +
                           randomUnitVector(random_state)};
            current_ray = Ray(record.point, target - record.point);
            current_colour = current_colour * 0.5f;
        } else {
            return getRayColourBasic(current_ray) * current_colour;
        }
        max_depth--;
    }
    return Colour{0.0f, 0.0f, 0.0f};
}

__device__ Colour
getRayColourWithMaterial(const Ray &ray, Sphere *sphere_array,
                         SphereArrayProperties sphere_array_properties,
                         s64 max_depth, curandState &random_state)
{
    Ray current_ray = ray;
    Colour current_attenuation{1.0f, 1.0f, 1.0f};
    while (max_depth > 0) {
        HitRecord record;

        if (hitSphereArray(sphere_array, sphere_array_properties, current_ray,
                           0.001f, infinity, record)) {
            Ray scattered_ray;
            Colour attenuation;
            if (record.material.scatter(current_ray, record, attenuation,
                                        scattered_ray, random_state)) {
                current_attenuation = current_attenuation * attenuation;
                current_ray = scattered_ray;
            } else {
                return Colour{0.0f, 0.0f, 0.0f};
            }
        } else {
            return getRayColourBasic(current_ray) * current_attenuation;
        }
        max_depth--;
    }
    return Colour{0.0f, 0.0f, 0.0f};
}

// TODO: we don't need this redirection
__device__ Colour getRayColour(const Ray &ray, Sphere *sphere_array,
                               SphereArrayProperties sphere_array_properties,
                               s64 max_depth, curandState &random_state)
{
    return getRayColourWithMaterial(ray, sphere_array, sphere_array_properties,
                                    max_depth, random_state);
}

} // namespace cuda
} // namespace RayTracer
