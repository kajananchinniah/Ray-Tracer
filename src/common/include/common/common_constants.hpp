#ifndef RAY_TRACER_COMMON_COMMON_CONSTANTS_HPP_
#define RAY_TRACER_COMMON_COMMON_CONSTANTS_HPP_

#include "common/common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include <limits>

namespace RayTracer
{

namespace cuda
{
constexpr f32 infinity = std::numeric_limits<f32>::infinity();

} // namespace cuda
} // namespace RayTracer

#endif // RAY_TRACER_COMMON_COMMON_CONSTANTS_HPP_
