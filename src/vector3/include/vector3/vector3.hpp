#ifndef RAY_TRACER_VECTOR3_VECTOR3_HPP_
#define RAY_TRACER_VECTOR3_VECTOR3_HPP_

#include "common/common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cmath>
#include <iostream>

namespace RayTracer
{

template <typename T> class vector3
{
public:
    __device__ __host__ vector3() : x_{0}, y_{0}, z_{0}
    {
    }

    __device__ __host__ vector3(T x, T y, T z) : x_{x}, y_{y}, z_{z}
    {
    }

    __device__ __host__ T &x()
    {
        return x_;
    }

    __device__ __host__ T &y()
    {
        return y_;
    }

    __device__ __host__ T &z()
    {
        return z_;
    }

    __device__ __host__ const T &x() const
    {
        return x_;
    }

    __device__ __host__ const T &y() const
    {
        return y_;
    }

    __device__ __host__ const T &z() const
    {
        return z_;
    }

    __device__ __host__ vector3<T> operator-() const
    {
        return vector3<T>{-x_, -y_, -z_};
    }

    __device__ __host__ vector3<T> &operator+=(const vector3<T> &other)
    {
        x_ += other.x();
        y_ += other.y();
        z_ += other.z();
        return *this;
    }

    __device__ __host__ vector3<T> &operator*=(const T t)
    {
        x_ = x_ * t;
        y_ = y_ * t;
        z_ = z_ * t;
        return *this;
    }

    __device__ __host__ vector3<T> &operator/=(const T t)
    {
        auto factor = 1 / t;
        x_ = x_ * factor;
        y_ = y_ * factor;
        z_ = z_ * factor;
        return *this;
    }

    __device__ __host__ f64 magnitude_squared() const
    {
        return x_ * x_ + y_ * y_ + z_ * z_;
    }

    __device__ f64 magnitude_device() const
    {
        return sqrt(magnitude_squared());
    }

    __host__ f64 magnitude_host() const
    {
        return std::sqrt(magnitude_squared());
    }

    __device__ f64 normalize_device()
    {
        return *this /= magnitude_device();
    }

    __host__ f64 normalize_host()
    {
        return *this /= magnitude_host();
    }

private:
    T x_;
    T y_;
    T z_;
};

template <typename T>
inline std::ostream &operator<<(std::ostream &out, const vector3<T> &vec)
{
    return out << vec.x() << ' ' << vec.y() << ' ' << vec.z();
}

template <typename T>
__device__ __host__ inline vector3<T> operator+(const vector3<T> &left,
                                                const vector3<T> &right)
{
    return vector3<T>{left.x() + right.x(), left.y() + right.y(),
                      left.z() + right.z()};
}

template <typename T>
__device__ __host__ inline vector3<T> operator-(const vector3<T> &left,
                                                const vector3<T> &right)
{
    return vector3<T>{left.x() - right.x(), left.y() - right.y(), left.z(),
                      right.z()};
}

template <typename T>
__device__ __host__ inline vector3<T> operator*(const vector3<T> &left,
                                                const vector3<T> &right)
{
    return vector3<T>{left.x() * right.x(), left.y() * right.y(),
                      left.z() * right.z()};
}

template <typename T>
__device__ __host__ inline vector3<T> operator*(T t, const vector3<T> &vec)
{
    return vector3<T>{vec.x() * t, vec.y() * t, vec.z() * t};
}

template <typename T>
__device__ __host__ inline vector3<T> operator*(const vector3<T> &vec, T t)
{
    return t * vec;
}

template <typename T>
__device__ __host__ inline vector3<T> operator/(const vector3<T> &vec, T t)
{
    return (1.0 / t) * vec;
}

template <typename T>
__device__ __host__ inline T dot(const vector3<T> &left,
                                 const vector3<T> &right)
{
    return left.x() * right.x() + left.y() * right.y() + left.z() * right.z();
}

template <typename T>
__device__ __host__ inline T cross(const vector3<T> &left,
                                   const vector3<T> &right)
{

    return vector3<T>{left.y() * right.z() - left.z() * right.u(),
                      left.z() * right.x() - left.x() * right.z(),
                      left.x() * right.y() - left.y() * right.x()};
}

template <typename T>
__device__ inline T normalize_device(const vector3<T> &vec)
{
    return vec / vec.magnitude_device();
}

template <typename T> __host__ inline T normalize_host(const vector3<T> &vec)
{
    return vec / vec.magnitude_host();
}

using vector3f32 = vector3<f32>;
using vector3f64 = vector3<f64>;

using Color = vector3f32;
using Point3f = vector3f32;

} // namespace RayTracer

#endif // RAY_TRACER_VECTOR3_VECTOR3_HPP_
