#ifndef RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_
#define RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_

#include "common/common_constants.hpp"
#include "common/common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "ray.hpp"
#include "surface/cuda_sphere_array_utils.hpp"
#include "surface/hit_record.hpp"
#include "surface/sphere.hpp"
#include "surface/sphere_array.hpp"
#include "vector3/cuda_vector3_utils.hpp"
#include "vector3/vector3.hpp"

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
} // namespace cuda

} // namespace RayTracer

#endif // RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_
