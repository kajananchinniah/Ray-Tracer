#ifndef RAY_TRACER_SURFACE_HITTABLE_HPP_
#define RAY_TRACER_SURFACE_HITTABLE_HPP_

#include "common/common_types.hpp"
#include "vector3/vector3.hpp"

namespace RayTracer
{

struct HitRecord {
    Point3f point;
    Vector3f normal;
    f32 t;
};

} // namespace RayTracer

#endif // RAY_TRACER_SURFACE_HITTABLE_HPP_
