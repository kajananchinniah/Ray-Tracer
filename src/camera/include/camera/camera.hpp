#ifndef RAY_TRACER_CAMERA_CAMERA_HPP_
#define RAY_TRACER_CAMERA_CAMERA_HPP_

#include <cmath>

#include "common/common_functions.hpp"
#include "common/common_types.hpp"
#include "ray/ray.hpp"
#include "vector3/cuda_vector3_utils.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

namespace RayTracer
{

class Camera
{
public:
    __device__ __host__ Camera()
    {
        f32 aspect_ratio{16.0 / 9.0};
        f32 viewport_height{2.0};
        f32 viewport_width{aspect_ratio * viewport_height};
        f32 focal_length{1.0};

        origin_ = Point3f{0.0, 0.0, 0.0};
        horizontal_ = Vector3f{viewport_width, 0.0, 0.0};
        vertical_ = Vector3f{0.0, viewport_height, 0.0};
        lower_left_corner_ = origin_ - horizontal_ / 2.0 - vertical_ / 2.0 -
                             Vector3f{0.0, 0.0, focal_length};
    }

    __host__ Camera(Point3f look_from, Point3f look_at, Vector3f v_up,
                    f32 vertical_FOV, f32 aspect_ratio)
    {
        f32 theta = degreesToRadians(vertical_FOV);
        f32 h = std::tan(theta / 2);
        f32 viewport_height{2.0f * h};
        f32 viewport_width{aspect_ratio * viewport_height};

        w_ = normalize_host(look_from - look_at);
        u_ = normalize_host(cross(v_up, w_));
        v_ = cross(w_, u_);

        origin_ = look_from;
        horizontal_ = viewport_width * u_;
        vertical_ = viewport_height * v_;
        lower_left_corner_ = origin_ - horizontal_ / 2.0 - vertical_ / 2.0 - w_;
        lens_radius_ = 0.0f;
    }

    __host__ Camera(Point3f look_from, Point3f look_at, Vector3f v_up,
                    f32 vertical_FOV, f32 aspect_ratio, f32 aperture,
                    f32 focus_dist)
    {
        f32 theta = degreesToRadians(vertical_FOV);
        f32 h = std::tan(theta / 2);
        f32 viewport_height{2.0f * h};
        f32 viewport_width{aspect_ratio * viewport_height};

        w_ = normalize_host(look_from - look_at);
        u_ = normalize_host(cross(v_up, w_));
        v_ = cross(w_, u_);

        origin_ = look_from;
        horizontal_ = focus_dist * viewport_width * u_;
        vertical_ = focus_dist * viewport_height * v_;
        lower_left_corner_ =
            origin_ - horizontal_ / 2.0 - vertical_ / 2.0 - focus_dist * w_;
        lens_radius_ = aperture / 2.0f;
    }

    __device__ __host__ Ray getRay(f32 s, f32 t) const
    {
        return Ray{origin_, lower_left_corner_ + s * horizontal_ +
                                t * vertical_ - origin_};
    }

    __device__ Ray getRay(f32 s, f32 t, curandState &random_state) const
    {
        Vector3f rd{lens_radius_ * cuda::randomInUnitDisk(random_state)};
        Vector3f offset{u_ * rd.x() + v_ * rd.y()};
        return Ray{origin_ + offset, lower_left_corner_ + s * horizontal_ +
                                         t * vertical_ - origin_ - offset};
    }

private:
    Point3f origin_{};
    Vector3f horizontal_{};
    Vector3f vertical_{};
    Point3f lower_left_corner_{};
    Vector3f u_{};
    Vector3f v_{};
    Vector3f w_{};
    f32 lens_radius_{};
};

} // namespace RayTracer

#endif // RAY_TRACER_CAMERA_CAMERA_HPP_
