#ifndef RAY_TRACER_RAY_RAY_HPP_
#define RAY_TRACER_RAY_RAY_HPP_

#include "common/common_types.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

/// The definition of a ray
class Ray
{
public:
    /// @brief Construct an empty ray
    __device__ __host__ Ray()
    {
    }

    /// @brief Construct a ray
    ///
    /// @param origin The origin of the ray
    /// @param direction The direction of the ray
    __device__ __host__ Ray(const Point3f &origin, const Vector3f &direction)
        : origin_{origin}, direction_{direction}
    {
    }

    /// @brief Returns the origin of a ray
    ///
    /// @return The origin
    __device__ __host__ Point3f origin() const
    {
        return origin_;
    }

    /// @brief Returns the direction of a ray
    ///
    /// @return The direction
    __device__ __host__ Vector3f direction() const
    {
        return direction_;
    }

    /// @brief Linearly interpolates the point of a ray at a specified t value
    ///
    /// @param t The t value to interpolate
    /// @return The interporlated value
    __device__ __host__ Point3f at(f32 t) const
    {
        return origin_ + t * direction_;
    }

private:
    Point3f origin_{};
    Vector3f direction_{};
};

} // namespace RayTracer
#endif // RAY_TRACER_RAY_RAY_HPP_
