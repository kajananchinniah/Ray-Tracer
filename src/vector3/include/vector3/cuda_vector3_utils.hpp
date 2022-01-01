#ifndef RAY_TRACER_VECTOR3_CUDA_VECTOR3_UTILS_HPP_
#define RAY_TRACER_VECTOR3_CUDA_VECTOR3_UTILS_HPP_

#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"

namespace RayTracer
{
namespace cuda
{

__device__ Vector3f randomInUnitSphere(curandState &random_state);
__device__ Vector3f randomUnitVector(curandState &random_state);

} // namespace cuda
} // namespace RayTracer

#endif // RAY_TRACER_VECTOR3_CUDA_VECTOR3_UTILS_HPP_
