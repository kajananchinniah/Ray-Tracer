#include "vector3/cuda_vector3_utils.hpp"
#include "vector3/vector3.hpp"

#include "curand_kernel.h"

namespace
{

using namespace RayTracer;

__device__ f32 changeRange(f32 value, f32 min, f32 max)
{
    return min + (max - min) * value;
}

} // namespace

namespace RayTracer
{
namespace cuda
{

__device__ Vector3f randomInUnitSphere(curandState &random_state)
{
    while (true) {
        f32 x{changeRange(curand_uniform(&random_state), -1, 1)};
        f32 y{changeRange(curand_uniform(&random_state), -1, 1)};
        f32 z{changeRange(curand_uniform(&random_state), -1, 1)};
        Vector3f rval = Vector3f{x, y, z};
        if (rval.magnitude_squared() >= 1) {
            continue;
        }
        return rval;
    }
}

__device__ Vector3f randomUnitVector(curandState &random_state)
{
    return normalize_device(randomInUnitSphere(random_state));
}

} // namespace cuda

} // namespace RayTracer
