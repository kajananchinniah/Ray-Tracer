#ifndef RAY_TRACER_SURFACE_HITTABLE_HPP_
#define RAY_TRACER_SURFACE_HITTABLE_HPP_

#include "common/common_types.hpp"
#include "material/material.hpp"
#include "ray/ray.hpp"
#include "vector3/vector3.hpp"

namespace RayTracer
{

struct HitRecord {
    Point3f point{};
    Vector3f normal{};
    f32 t{};
    Material material{};
    bool front_face{};

    __device__ __host__ inline void
    setFaceNormal(const Ray &ray, const Vector3f &outward_normal)
    {
        front_face = dot(ray.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

} // namespace RayTracer

#endif // RAY_TRACER_SURFACE_HITTABLE_HPP_
