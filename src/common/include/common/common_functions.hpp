/// @file common_functions.hpp
/// @brief Contains useful functions
#ifndef RAY_TRACER_COMMON_COMMON_FUNCTIONS_HPP_
#define RAY_TRACER_COMMON_COMMON_FUNCTIONS_HPP_

#include "common/common_constants.hpp"
#include "common/common_types.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

/// @brief Converts degrees to radians
///
/// @param degrees An angle in degrees
/// @return An angle in radians
__device__ __host__ inline f32 degreesToRadians(f32 degrees)
{
    return degrees * (3.1415926535897932385f / 180.0f);
}
} // namespace RayTracer

#endif // RAY_TRACER_COMMON_COMMON_FUNCTIONS_HPP_
