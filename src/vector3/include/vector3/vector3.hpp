#ifndef RAY_TRACER_VECTOR3_VECTOR3_HPP_
#define RAY_TRACER_VECTOR3_VECTOR3_HPP_

#include "common/common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

template <typename T> class vector3
{
public:
    vector3()
    {
    }

    vector3(T x, T y, T z) : x_{x}, y_{y}, z_{z}
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

private:
    T x_;
    T y_;
    T z_;
};

} // namespace RayTracer

#endif // RAY_TRACER_VECTOR3_VECTOR3_HPP_
