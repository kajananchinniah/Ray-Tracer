#include "surface/cuda_sphere_array_utils.hpp"

namespace RayTracer
{
namespace cuda
{

__device__ bool hitSphereArray(Sphere *sphere_array,
                               SphereArrayProperties sphere_array_properties,
                               const Ray &ray, f32 t_min, f32 t_max,
                               HitRecord &record)
{
    HitRecord tmp_record;
    bool hit_anything{false};
    f32 closest_so_far{t_max};

    for (s64 i = 0; i < sphere_array_properties.size; ++i) {
        if (sphere_array[i].hit(ray, t_min, closest_so_far, tmp_record)) {
            hit_anything = true;
            closest_so_far = tmp_record.t;
            record = tmp_record;
        }
    }

    return hit_anything;
}

} // namespace cuda

} // namespace RayTracer
