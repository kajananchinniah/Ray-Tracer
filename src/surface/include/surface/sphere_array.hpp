#ifndef RAY_TRACER_SURFACE_SPHERE_ARRAY_HPP_
#define RAY_TRACER_SURFACE_SPHERE_ARRAY_HPP_

#include "common/common_types.hpp"
#include "common/cuda_memory_utils.hpp"
#include <memory>

#include "sphere.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

struct SphereArrayProperties {
    explicit SphereArrayProperties(s64 max_capacity)
        : size{0}, capacity{max_capacity}
    {
    }
    s64 size{};
    const s64 capacity{};
};

struct SphereArray {
    explicit SphereArray(s64 max_capacity)
        : properties{max_capacity},
          data_buffer{cuda::createCudaUniquePtrArray<Sphere>(max_capacity)}
    {
    }

    void clear()
    {
        properties.size = 0;
    }

    bool add(const Sphere &sphere)
    {
        if (properties.size >= properties.capacity) {
            return false;
        }

        data_buffer[properties.size] = sphere;
        properties.size++;
        return true;
    }

    SphereArrayProperties properties;
    std::unique_ptr<Sphere[], decltype(&cudaFree)> data_buffer{nullptr,
                                                               cudaFree};
};

} // namespace RayTracer

#endif // RAY_TRACER_SURFACE_SPHERE_ARRAY_HPP_
