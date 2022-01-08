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

/// Describes the camera for raytracing
class Camera
{
public:
    /// @brief Construct a camera with random default parameters. This is mainly
    ///        for compatability with previous tests.
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

    /// @brief Construct a camera object. Note: only callable in CPU
    ///
    /// @param look_from The position to place the camera at
    /// @param look_at The point to look at
    /// @param v_up The up vector for the camera
    /// @param vertical_FOV The vertical field-of-view (in degrees)
    /// @param aspect_ratio The aspect ratio that will be used
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

    /// @brief Construct a camera object. Note: only callable in CPU
    ///
    /// @param look_from The position to place the camera at
    /// @param look_at The point to look at
    /// @param v_up The up vector for the camera
    /// @param vertical_FOV The vertical field-of-view (in degrees)
    /// @param aspect_ratio The aspect ratio that will be used
    /// @param aperture The size of the camera lens' opening
    /// @param focus_dist The distance to the focus plane
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

    /// @brief Get the camera's ray at a specific (s, t)
    ///
    /// @param s The normalized u coordinate
    /// @param t The normalized v coordinate
    /// @return The ray
    __device__ __host__ Ray getRay(f32 s, f32 t) const
    {
        return Ray{origin_, lower_left_corner_ + s * horizontal_ +
                                t * vertical_ - origin_};
    }

    /// @brief Get the camera's ray at a specific (s, t). This uses a randomness
    ///        to allow for defocus blur
    ///
    /// @param s The normalized u coordinate
    /// @param t The normalized v coordinate
    /// @param random_state The random state
    /// @return The ray
    __device__ Ray getRay(f32 s, f32 t, curandState &random_state) const
    {
        Vector3f rd{lens_radius_ * cuda::randomInUnitDisk(random_state)};
        Vector3f offset{u_ * rd.x() + v_ * rd.y()};
        return Ray{origin_ + offset, lower_left_corner_ + s * horizontal_ +
                                         t * vertical_ - origin_ - offset};
    }

private:
    /// The origin of the camera
    Point3f origin_{};

    /// The horizontal of the camera
    Vector3f horizontal_{};

    /// The vertical of the camera
    Vector3f vertical_{};

    /// The lower left corner of the camera
    Point3f lower_left_corner_{};

    /// One of the axes describing the camera's orientation
    Vector3f u_{};

    /// One of the axes describing the camera's orientation
    Vector3f v_{};

    /// One of the axeV describing the camera's orientation
    Vector3f w_{};

    /// The radius of the lens
    f32 lens_radius_{};
};

} // namespace RayTracer

#endif // RAY_TRACER_CAMERA_CAMERA_HPP_
