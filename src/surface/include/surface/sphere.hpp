#ifndef RAY_TRACER_SURFACE_SPHERE_HPP_
#define RAY_TRACER_SURFACE_SPHERE_HPP_

#include "common/common_types.hpp"
#include "hit_record.hpp"
#include "ray/ray.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{
class Sphere
{
public:
    __device__ __host__ Sphere()
    {
    }
    __device__ __host__ Sphere(Point3f center, f32 radius)
        : center_{center}, radius_{radius}
    {
    }

    __device__ __host__ bool hit(const Ray &ray, f32 t_min, f32 t_max,
                                 HitRecord &record) const;

private:
    Point3f center_;
    f32 radius_;
};
} // namespace RayTracer

#endif // RAY_TRACER_SURFACE_SPHERE_HPP_
