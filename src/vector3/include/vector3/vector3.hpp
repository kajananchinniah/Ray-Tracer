#ifndef RAY_TRACER_VECTOR3_VECTOR3_HPP_
#define RAY_TRACER_VECTOR3_VECTOR3_HPP_

#include "common/common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cmath>
#include <iostream>

namespace RayTracer
{

class Vector3f
{
public:
    __device__ __host__ Vector3f() : x_{0}, y_{0}, z_{0}
    {
    }

    __device__ __host__ Vector3f(f32 x, f32 y, f32 z) : x_{x}, y_{y}, z_{z}
    {
    }

    __device__ __host__ f32 &x()
    {
        return x_;
    }

    __device__ __host__ f32 &y()
    {
        return y_;
    }

    __device__ __host__ f32 &z()
    {
        return z_;
    }

    __device__ __host__ const f32 &x() const
    {
        return x_;
    }

    __device__ __host__ const f32 &y() const
    {
        return y_;
    }

    __device__ __host__ const f32 &z() const
    {
        return z_;
    }

    __device__ __host__ Vector3f operator-() const
    {
        return Vector3f{-x_, -y_, -z_};
    }

    __device__ __host__ Vector3f &operator+=(const Vector3f &other)
    {
        x_ += other.x();
        y_ += other.y();
        z_ += other.z();
        return *this;
    }

    __device__ __host__ Vector3f &operator-=(const Vector3f &other)
    {
        x_ -= other.x();
        y_ -= other.y();
        z_ -= other.z();
        return *this;
    }

    __device__ __host__ Vector3f &operator*=(const f32 t)
    {
        x_ = x_ * t;
        y_ = y_ * t;
        z_ = z_ * t;
        return *this;
    }

    __device__ __host__ Vector3f &operator/=(const f32 t)
    {
        f32 factor = 1 / t;
        x_ = x_ * factor;
        y_ = y_ * factor;
        z_ = z_ * factor;
        return *this;
    }

    __device__ __host__ f32 magnitude_squared() const
    {
        return x_ * x_ + y_ * y_ + z_ * z_;
    }

    __device__ f32 magnitude_device() const
    {
        return sqrt(magnitude_squared());
    }

    __host__ f32 magnitude_host() const
    {
        return std::sqrt(magnitude_squared());
    }

    __device__ void normalize_device()
    {
        f64 length = magnitude_device();
        x_ = x_ / length;
        y_ = y_ / length;
        z_ = z_ / length;
    }

    __host__ void normalize_host()
    {
        f64 length = magnitude_host();
        x_ = x_ / length;
        y_ = y_ / length;
        z_ = z_ / length;
    }

    __host__ bool nearZero_host() const
    {
        const f32 s{1e-8};
        return (std::fabs(x_) < s) && (std::fabs(y_) < s) &&
               (std::fabs(z_) < s);
    }

    __device__ bool nearZero_device() const
    {

        const f32 s{1e-8};
        return (fabsf(x_) < s) && (fabsf(y_) < s) && (fabsf(z_) < s);
    }

private:
    f32 x_;
    f32 y_;
    f32 z_;
};

inline std::ostream &operator<<(std::ostream &out, const Vector3f &vec)
{
    return out << vec.x() << ' ' << vec.y() << ' ' << vec.z();
}

__device__ __host__ inline Vector3f operator+(const Vector3f &left,
                                              const Vector3f &right)
{
    return Vector3f{left.x() + right.x(), left.y() + right.y(),
                    left.z() + right.z()};
}

__device__ __host__ inline Vector3f operator-(const Vector3f &left,
                                              const Vector3f &right)
{
    return Vector3f{left.x() - right.x(), left.y() - right.y(),
                    left.z() - right.z()};
}

__device__ __host__ inline Vector3f operator*(const Vector3f &left,
                                              const Vector3f &right)
{
    return Vector3f{left.x() * right.x(), left.y() * right.y(),
                    left.z() * right.z()};
}

__device__ __host__ inline Vector3f operator*(f32 t, const Vector3f &vec)
{
    return Vector3f{vec.x() * t, vec.y() * t, vec.z() * t};
}

__device__ __host__ inline Vector3f operator*(const Vector3f &vec, f32 t)
{
    return t * vec;
}

__device__ __host__ inline Vector3f operator/(const Vector3f &vec, f32 t)
{
    return (1.0f / t) * vec;
}

__device__ __host__ inline f32 dot(const Vector3f &left, const Vector3f &right)
{
    return left.x() * right.x() + left.y() * right.y() + left.z() * right.z();
}

__device__ __host__ inline Vector3f cross(const Vector3f &left,
                                          const Vector3f &right)
{

    return Vector3f{left.y() * right.z() - left.z() * right.y(),
                    left.z() * right.x() - left.x() * right.z(),
                    left.x() * right.y() - left.y() * right.x()};
}

__device__ inline Vector3f normalize_device(const Vector3f &vec)
{
    return vec / vec.magnitude_device();
}

__host__ inline Vector3f normalize_host(const Vector3f &vec)
{
    return vec / vec.magnitude_host();
}

__device__ __host__ inline Vector3f reflect(const Vector3f &left,
                                            const Vector3f &right)
{
    return left - 2 * dot(left, right) * right;
}

__device__ inline Vector3f refract_device(const Vector3f &left,
                                          const Vector3f &right,
                                          f32 index_of_refraction_ratio)
{
    f32 cos_theta = fminf(dot(-left, right), 1.0f);
    Vector3f r_out_perp =
        index_of_refraction_ratio * (left + cos_theta * right);
    Vector3f r_out_parallel =
        -sqrt(fabsf(1.0f - r_out_perp.magnitude_squared())) * right;
    return r_out_perp + r_out_parallel;
}

using Colour = Vector3f;
using Point3f = Vector3f;

} // namespace RayTracer

#endif // RAY_TRACER_VECTOR3_VECTOR3_HPP_
