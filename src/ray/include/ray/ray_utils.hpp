#ifndef RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_
#define RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_

#include "common/common_constants.hpp"
#include "common/common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "ray.hpp"
#include "surface/cuda_sphere_array_utils.hpp"
#include "surface/hit_record.hpp"
#include "surface/sphere.hpp"
#include "surface/sphere_array.hpp"
#include "vector3/cuda_vector3_utils.hpp"
#include "vector3/vector3.hpp"

namespace RayTracer
{

namespace cuda
{

__device__ f32 hitSphere(const Point3f &center, f32 radius, const Ray &ray);

__device__ Colour getRayColourBasic(const Ray &ray);

__device__ Colour getRayColourWithRedSphere(const Ray &ray);

__device__ Colour
getRayColourWithSphereArray(const Ray &ray, Sphere *sphere_array,
                            SphereArrayProperties sphere_array_properties);
__device__ Colour
getRayColourWithDiffuse(const Ray &ray, Sphere *sphere_array,
                        SphereArrayProperties sphere_array_properties,
                        s64 max_depth, curandState &random_state);

__device__ Colour
getRayColourWithMaterial(const Ray &ray, Sphere *sphere_array,
                         SphereArrayProperties sphere_array_properties,
                         s64 max_depth, curandState &random_state);

// TODO: we don't need this redirection
__device__ Colour getRayColour(const Ray &ray, Sphere *sphere_array,
                               SphereArrayProperties sphere_array_properties,
                               s64 max_depth, curandState &random_state);
} // namespace cuda

} // namespace RayTracer

#endif // RAY_TRACER_RAY_CUDA_RAY_UTILS_HPP_
