#include "common/common_types.hpp"
#include "image/cuda_image_utils.hpp"

#include "image/image.hpp"
#include "vector3/vector3.hpp"

#include "curand_kernel.h"

namespace
{

using namespace RayTracer;
__constant__ u64 seed = 2021;

__global__ void initializeImageRandomStateCuda(curandState *random_state,
                                               const ImageProperties properties)
{

    s64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    s64 u_stride = gridDim.x * blockDim.x;

    s64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    s64 v_stride = gridDim.y * blockDim.y;

    for (s64 v = v_idx; v < properties.height; v += v_stride) {
        for (s64 u = u_idx; u < properties.width; u += u_stride) {
            u64 idx = properties.randomStateIndex(u, v);
            curand_init(seed, idx, 0, &random_state[idx]);
        }
    }
}

__device__ inline f32 clamp(f32 value, f32 min, f32 max)
{
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
}

} // namespace

namespace RayTracer
{
namespace cuda
{

__device__ void writeColourAt(u8 *image_buffer,
                              const ImageProperties &properties,
                              const Colour &colour, s64 u, s64 v,
                              int samples_per_pixel)
{
    f32 red = colour.x();
    f32 green = colour.y();
    f32 blue = colour.z();

    f32 scale = 1.0 / samples_per_pixel;
    red = red * scale;
    green = green * scale;
    blue = blue * scale;

    image_buffer[properties.redIndex(u, v)] =
        static_cast<u8>(256 * clamp(red, 0.0, 0.999));
    image_buffer[properties.greenIndex(u, v)] =
        static_cast<u8>(256 * clamp(green, 0.0, 0.999));
    image_buffer[properties.blueIndex(u, v)] =
        static_cast<u8>(256 * clamp(blue, 0.0, 0.999));
}

__device__ void writeGammaCorrectedColourAt(u8 *image_buffer,
                                            const ImageProperties &properties,
                                            const Colour &colour, s64 u, s64 v,
                                            int samples_per_pixel)
{
    f32 red = colour.x();
    f32 green = colour.y();
    f32 blue = colour.z();

    f32 scale = 1.0 / samples_per_pixel;
    red = sqrt(red * scale);
    green = sqrt(green * scale);
    blue = sqrt(blue * scale);

    image_buffer[properties.redIndex(u, v)] =
        static_cast<u8>(256 * clamp(red, 0.0, 0.999));
    image_buffer[properties.greenIndex(u, v)] =
        static_cast<u8>(256 * clamp(green, 0.0, 0.999));
    image_buffer[properties.blueIndex(u, v)] =
        static_cast<u8>(256 * clamp(blue, 0.0, 0.999));
}

void initializeImageRandomState(curandState *random_state,
                                const ImageProperties &properties)
{
    const dim3 threads{16, 16, 1};
    const u32 block_x = (properties.width + threads.x - 1) / threads.x;
    const u32 block_y = (properties.height + threads.y - 1) / threads.y;
    const dim3 blocks{block_x, block_y, 1};
    cuda::prefetchToGpu(random_state, properties.area());
    initializeImageRandomStateCuda<<<blocks, threads>>>(random_state,
                                                        properties);
    cuda::waitForCuda();
}

} // namespace cuda

} // namespace RayTracer
