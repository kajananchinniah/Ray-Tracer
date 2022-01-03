#include "camera/camera.hpp"
#include "common/common_types.hpp"
#include "image/image.hpp"
#include "material/material.hpp"
#include "ray/ray.hpp"
#include "surface/cuda_sphere_array_utils.hpp"
#include "surface/sphere.hpp"
#include "surface/sphere_array.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

__device__ f32 scaleUCoordinate(u64 u, s64 width)
{
    return static_cast<f32>(u) / static_cast<f32>(width - 1);
}

__device__ f32 scaleVCoordinate(u64 v, s64 height)
{
    return static_cast<f32>(height - v - 1) / static_cast<f32>(height - 1);
}

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

__global__ void testBasicRenderCuda(u8 *image_buffer,
                                    ImageProperties properties)
{
    u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    u64 u_stride = gridDim.x * blockDim.x;

    u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    u64 v_stride = gridDim.y * blockDim.y;

    for (u64 v = v_idx; v < properties.height; v += v_stride) {
        for (u64 u = u_idx; u < properties.width; u += u_stride) {
            f32 red{scaleUCoordinate(u, properties.width)};
            f32 green{scaleVCoordinate(v, properties.height)};
            f32 blue{0.25};
            Colour colour{red, green, blue};
            cuda::writeColourAt(image_buffer, properties, colour, u, v);
        }
    }
}

__global__ void testBasicRenderWithRayCuda(u8 *image_buffer,
                                           ImageProperties properties,
                                           Camera camera)
{
    u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    u64 u_stride = gridDim.x * blockDim.x;

    u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    u64 v_stride = gridDim.y * blockDim.y;

    for (u64 v = v_idx; v < properties.height; v += v_stride) {
        for (u64 u = u_idx; u < properties.width; u += u_stride) {
            f32 scaled_u{scaleUCoordinate(u, properties.width)};
            f32 scaled_v{scaleVCoordinate(v, properties.height)};
            Ray ray = camera.getRay(scaled_u, scaled_v);
            Colour colour = cuda::getRayColourBasic(ray);
            cuda::writeColourAt(image_buffer, properties, colour, u, v);
        }
    }
}

__global__ void testBasicRenderWithSphereCuda(u8 *image_buffer,
                                              ImageProperties properties,
                                              Camera camera)
{

    u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    u64 u_stride = gridDim.x * blockDim.x;

    u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    u64 v_stride = gridDim.y * blockDim.y;

    for (u64 v = v_idx; v < properties.height; v += v_stride) {
        for (u64 u = u_idx; u < properties.width; u += u_stride) {
            f32 scaled_u{scaleUCoordinate(u, properties.width)};
            f32 scaled_v{scaleVCoordinate(v, properties.height)};
            Ray ray = camera.getRay(scaled_u, scaled_v);
            Colour colour = cuda::getRayColourWithRedSphere(ray);
            cuda::writeColourAt(image_buffer, properties, colour, u, v);
        }
    }
}

__global__ void testRenderBasicRenderWithWorld(
    u8 *image_buffer, ImageProperties image_properties, Camera camera,
    Sphere *sphere_array, SphereArrayProperties sphere_array_properties)
{
    u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    u64 u_stride = gridDim.x * blockDim.x;

    u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    u64 v_stride = gridDim.y * blockDim.y;

    for (u64 v = v_idx; v < image_properties.height; v += v_stride) {
        for (u64 u = u_idx; u < image_properties.width; u += u_stride) {
            f32 scaled_u{scaleUCoordinate(u, image_properties.width)};
            f32 scaled_v{scaleVCoordinate(v, image_properties.height)};
            Ray ray = camera.getRay(scaled_u, scaled_v);
            Colour colour = cuda::getRayColourWithSphereArray(
                ray, sphere_array, sphere_array_properties);
            cuda::writeColourAt(image_buffer, image_properties, colour, u, v);
        }
    }
}

__global__ void testRenderBasicWithAntiAliasing(
    u8 *image_buffer, ImageProperties image_properties, Camera camera,
    Sphere *sphere_array, SphereArrayProperties sphere_array_properties,
    curandState *random_state, s32 samples_per_pixel)
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
                Ray ray = camera.getRay(scaled_u, scaled_v);
                colour += cuda::getRayColourWithSphereArray(
                    ray, sphere_array, sphere_array_properties);
            }
            cuda::writeColourAt(image_buffer, image_properties, colour, u, v,
                                samples_per_pixel);
        }
    }
}

__global__ void testRenderBasicWithDiffuse(
    u8 *image_buffer, ImageProperties image_properties, Camera camera,
    Sphere *sphere_array, SphereArrayProperties sphere_array_properties,
    curandState *random_state, s32 samples_per_pixel, s32 max_depth = 50)
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
                Ray ray = camera.getRay(scaled_u, scaled_v);
                colour += cuda::getRayColourWithDiffuse(
                    ray, sphere_array, sphere_array_properties, max_depth,
                    pixel_random_state);
            }
            cuda::writeGammaCorrectedColourAt(image_buffer, image_properties,
                                              colour, u, v, samples_per_pixel);
        }
    }
}

__global__ void testRenderBasicWithMaterial(
    u8 *image_buffer, ImageProperties image_properties, Camera camera,
    Sphere *sphere_array, SphereArrayProperties sphere_array_properties,
    curandState *random_state, s32 samples_per_pixel, s32 max_depth = 50)
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
                Ray ray = camera.getRay(scaled_u, scaled_v);
                colour += cuda::getRayColourWithMaterial(
                    ray, sphere_array, sphere_array_properties, max_depth,
                    pixel_random_state);
            }
            cuda::writeGammaCorrectedColourAt(image_buffer, image_properties,
                                              colour, u, v, samples_per_pixel);
        }
    }
}

} // namespace RayTracer
