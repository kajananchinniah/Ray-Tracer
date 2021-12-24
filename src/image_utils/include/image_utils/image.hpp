#ifndef RAYTRACER_IMAGE_UTILS_IMAGE_HPP_
#define RAYTRACER_IMAGE_UTILS_IMAGE_HPP_

#include "common/common_types.hpp"
#include <memory>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"

#include "common/cuda_memory_utils.hpp"

namespace RayTracer
{

/// Holds relevant properties to work with images.
/// The helper functions in image_utils.hpp should be used to access the data.
struct Image {
    Image()
    {
    }
    Image(u64 w, u64 h, ImageEncodings e)
        : width{w}, height{h}, channels{3}, encoding{e},
          data_buffer{cuda::createCudaUniquePtrArray<u8>(size())}
    {
    }
    /// The width of the image
    u64 width{};

    /// The height of the image
    u64 height{};

    /// The number of channels of the image
    u64 channels{};

    /// The encoding of the image
    ImageEncodings encoding{};

    /// A data buffer holding the pixels of an image
    std::unique_ptr<u8[], decltype(&cudaFree)> data_buffer{nullptr, cudaFree};

    /// @brief Calculates the size of the image's data buffer
    ///
    /// @return The size of the image data buffer
    __device__ __host__ u64 size() const
    {
        return width * height * channels * sizeof(u8);
    }

    /// @brief Calculates the pitch of the image
    ///
    /// @return The pitch of the image
    __device__ __host__ u64 pitch() const
    {
        return width * channels * sizeof(u8);
    }

    /// @brief Calculates the colour step of the image
    ///
    /// @return The colour step of the image
    __device__ __host__ u64 colourStep() const
    {
        return channels * sizeof(u8);
    }

    /// @brief Calculates the flattened index to access an element
    ///
    /// @param u The u coordinate (e.g. along the width) of interest
    /// @param v The v coordinate (e.g. along the height) of interest
    /// @param c The c coordinate (e.g. along the channel) of interest
    /// @return The flattened index
    __device__ __host__ u64 flattenedIndex(u64 u, u64 v, u64 c) const
    {
        return v * pitch() + u * colourStep() + c;
    }

    /// @brief Gets the element at (u, v, c) in the image
    ///
    /// @param u The u coordinate (e.g. along the width) of interest
    /// @param v The v coordinate (e.g. along the height) of interest
    /// @param c The c coordinate (e.g. along the channel) of interest
    /// @return The data element
    __device__ __host__ u8 at(u64 u, u64 v, u64 c) const
    {
        return data_buffer[flattenedIndex(u, v, c)];
    }
};

} // namespace RayTracer

#endif // RAYTRACER_IMAGE_UTILS_IMAGE_HPP_
