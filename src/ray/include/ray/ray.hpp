#ifndef RAY_TRACER_RAY_RAY_HPP_
#define RAY_TRACER_RAY_RAY_HPP_

#include "common/common_types.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

class Ray
{
public:
    __device__ __host__ Ray()
    {
    }

    __device__ __host__ Ray(const Point3f &origin, const Vector3f &direction)
        : origin_{origin}, direction_{direction}
    {
    }

    __device__ __host__ Point3f origin() const
    {
        return origin_;
    }

    __device__ __host__ Vector3f direction() const
    {
        return direction_;
    }

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
