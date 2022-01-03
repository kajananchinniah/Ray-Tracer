#include "surface/sphere.hpp"

namespace RayTracer
{

__device__ __host__ bool Sphere::hit(const Ray &ray, f32 t_min, f32 t_max,
                                     HitRecord &record) const
{
    Vector3f vec_origin_center{ray.origin() - center_};
    f32 a{ray.direction().magnitude_squared()};
    f32 b_over_2 = dot(vec_origin_center, ray.direction());
    f32 c = vec_origin_center.magnitude_squared() - radius_ * radius_;
    f32 discriminant{b_over_2 * b_over_2 - a * c};
    if (discriminant < 0) {
        return false;
    }
    f32 sqrtd{sqrt(discriminant)};

    f32 root{(-b_over_2 - sqrtd) / a};
    if (root < t_min || t_max < root) {
        root = (-b_over_2 + sqrtd) / a;
        if (root < t_min || t_max < root) {
            return false;
        }
    }

    record.t = root;
    record.point = ray.at(record.t);
    Vector3f outward_normal = (record.point - center_) / radius_;
    record.setFaceNormal(ray, outward_normal);
    record.material = material_;
    return true;
}

} // namespace RayTracer
