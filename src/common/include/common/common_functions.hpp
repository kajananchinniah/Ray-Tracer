#ifndef RAY_TRACER_COMMON_COMMON_FUNCTIONS_HPP_
#define RAY_TRACER_COMMON_COMMON_FUNCTIONS_HPP_

#include "common/common_constants.hpp"
#include "common/common_types.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

__device__ __host__ inline f32 degreesToRadians(f32 degrees)
{
    return degrees * pi / 180.0f;
}
} // namespace RayTracer

#endif // RAY_TRACER_COMMON_COMMON_FUNCTIONS_HPP_
