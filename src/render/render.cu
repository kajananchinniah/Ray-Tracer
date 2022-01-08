#include "render/render.hpp"

#include "camera/camera.hpp"
#include "common/common_types.hpp"
#include "curand_kernel.h"
#include "image/cuda_image_utils.hpp"
#include "image/image.hpp"
#include "ray/ray.hpp"
#include "ray/ray_utils.hpp"
#include "surface/sphere_array.hpp"
#include "vector3/vector3.hpp"

namespace RayTracer
{
namespace cuda
{
namespace
{
__device__ f32 scaleUCoordinateRandomized(u64 u, s64 width, curandState &state)
{
    return static_cast<f32>(u + curand_uniform(&state)) /
           static_cast<f32>(width - 1);
}

__device__ f32 scaleVCoordinateRandomized(u64 v, s64 height, curandState &state)
{
    return static_cast<f32>(height - v - 1 + curand_uniform(&state)) /
           static_cast<f32>(height - 1);
}

__global__ void renderCuda(u8 *image_buffer, ImageProperties image_properties,
                           Camera camera, Sphere *sphere_array,
                           SphereArrayProperties sphere_array_properties,
                           curandState *random_state, s32 samples_per_pixel,
                           s32 max_depth)
{
    u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    u64 u_stride = gridDim.x * blockDim.x;

    u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    u64 v_stride = gridDim.y * blockDim.y;

    for (u64 v = v_idx; v < image_properties.height; v += v_stride) {
        for (u64 u = u_idx; u < image_properties.width; u += u_stride) {
            Colour colour{0.0, 0.0, 0.0};
            curandState pixel_random_state =
                random_state[image_properties.randomStateIndex(u, v)];
            for (s32 s = 0; s < samples_per_pixel; ++s) {
                f32 scaled_u{scaleUCoordinateRandomized(
                    u, image_properties.width, pixel_random_state)};
                f32 scaled_v{scaleVCoordinateRandomized(
                    v, image_properties.height, pixel_random_state)};
                Ray ray = camera.getRay(scaled_u, scaled_v, pixel_random_state);
                colour += cuda::getRayColour(ray, sphere_array,
                                             sphere_array_properties, max_depth,
                                             pixel_random_state);
            }
            cuda::writeGammaCorrectedColourAt(image_buffer, image_properties,
                                              colour, u, v, samples_per_pixel);
        }
    }
}

} // namespace

void render(const Camera camera, SphereArray &sphere_array,
            s32 samples_per_pixel, s32 max_depth, Image &output_image)
{
    const dim3 threads{8, 8, 1};
    const dim3 blocks(
        (output_image.properties.width + threads.x - 1) / threads.x,
        (output_image.properties.height + threads.y - 1) / threads.y, 1);
    cuda::prefetchToGpu(output_image.data_buffer.get(),
                        output_image.properties.size());
    cuda::prefetchToGpu(sphere_array.data_buffer.get(),
                        sphere_array.properties.size);
    renderCuda<<<blocks, threads>>>(
        output_image.data_buffer.get(), output_image.properties, camera,
        sphere_array.data_buffer.get(), sphere_array.properties,
        output_image.random_state.get(), samples_per_pixel, max_depth);
    cuda::waitForCuda();
}
} // namespace cuda
} // namespace RayTracer
