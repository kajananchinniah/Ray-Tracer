#ifndef RAY_TRACER_CAMERA_CAMERA_HPP_
#define RAY_TRACER_CAMERA_CAMERA_HPP_

#include <cmath>

#include "common/common_functions.hpp"
#include "common/common_types.hpp"
#include "ray/ray.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

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

    __device__ __host__ Camera(f32 vertical_FOV, f32 aspect_ratio)
    {
        f32 theta = degreesToRadians(vertical_FOV);
        f32 h = std::tan(theta / 2);
        f32 viewport_height{2.0f * h};
        f32 viewport_width{aspect_ratio * viewport_height};
        f32 focal_length{1.0f};

        origin_ = Point3f{0.0, 0.0, 0.0};
        horizontal_ = Vector3f{viewport_width, 0.0, 0.0};
        vertical_ = Vector3f{0.0, viewport_height, 0.0};
        lower_left_corner_ = origin_ - horizontal_ / 2.0 - vertical_ / 2.0 -
                             Vector3f{0.0, 0.0, focal_length};
    }

    __device__ __host__ Ray getRay(f32 u_scaled, f32 v_scaled) const
    {
        return Ray{origin_, lower_left_corner_ + u_scaled * horizontal_ +
                                v_scaled * vertical_ - origin_};
    }

private:
    Point3f origin_{};
    Vector3f horizontal_{};
    Vector3f vertical_{};
    Point3f lower_left_corner_{};
};

} // namespace RayTracer

#endif // RAY_TRACER_CAMERA_CAMERA_HPP_
