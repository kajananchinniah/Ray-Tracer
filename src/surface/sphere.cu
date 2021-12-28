#include "surface/sphere.hpp"

namespace RayTracer
{

__device__ __host__ bool Sphere::hit(const Ray &ray, f32 t_min, f32 t_max,
                                     HitRecord &record) const
{
    return true;
}

} // namespace RayTracer
