///@file image.hpp
/// @brief Contains the definition of an image struct
#ifndef RAYTRACER_IMAGE_UTILS_IMAGE_HPP_
#define RAYTRACER_IMAGE_UTILS_IMAGE_HPP_

#include "common/common_types.hpp"
#include <memory>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"

#include "common/cuda_memory_utils.hpp"
#include <iostream>

#include "vector3/vector3.hpp"

namespace RayTracer
{

/// Holds image properties and relevant functions to work with images.
/// This struct can be used in cuda kernels.
/// Ideally, this should be passed along side an image buffer.
struct ImageProperties {
    ImageProperties(s64 w, s64 h, ImageEncodings e = ImageEncodings::kBGR8);

    /// The width of the image
    const s64 width{};

    /// The height of the image
    const s64 height{};

    /// The number of channels of the image
    const s64 channels{};

    /// The encoding of the image
    const ImageEncodings encoding{};

    const s64 red_offset{};

    const s64 green_offset{};

    const s64 blue_offset{};

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
    __device__ __host__ s64 pitch() const
    {
        return width * channels * sizeof(u8);
    }

    /// @brief Calculates the colour step of the image
    ///
    /// @return The colour step of the image
    __device__ __host__ s64 colourStep() const
    {
        return channels * sizeof(u8);
    }

    /// @brief Calculates the flattened index to access an element
    ///
    /// @param u The u coordinate (e.g. along the width) of interest
    /// @param v The v coordinate (e.g. along the height) of interest
    /// @param c The c coordinate (e.g. along the channel) of interest
    /// @return The flattened index
    __device__ __host__ s64 flattenedIndex(s64 u, s64 v, s64 c) const
    {
        return v * pitch() + u * colourStep() + c;
    }

    __device__ __host__ s64 redIndex(s64 u, s64 v) const
    {
        return flattenedIndex(u, v, red_offset);
    }

    __device__ __host__ s64 greenIndex(s64 u, s64 v) const
    {
        return flattenedIndex(u, v, green_offset);
    }

    __device__ __host__ s64 blueIndex(s64 u, s64 v) const
    {
        return flattenedIndex(u, v, blue_offset);
    }
};

/// Holds relevant properties to work with images and a pointer to an image
// buffer. Note, when using with cuda, use access the pointers raw memory.
/// Also, pass in the properties member as another argument.
struct Image {
    Image(s64 w, s64 h, ImageEncodings e = ImageEncodings::kBGR8)
        : properties{w, h, e}, data_buffer{cuda::createCudaUniquePtrArray<u8>(
                                   properties.size())}
    {
    }

    /// Holds relevant image properties
    const ImageProperties properties;

    /// A data buffer holding the pixels of an image
    std::unique_ptr<u8[], decltype(&cudaFree)> data_buffer{nullptr, cudaFree};

    /// @brief Gets the element at (u, v, c) in the image
    ///
    /// @param u The u coordinate (e.g. along the width) of interest
    /// @param v The v coordinate (e.g. along the height) of interest
    /// @param c The c coordinate (e.g. along the channel) of interest
    /// @return The data element
    __host__ u8 &at(s64 u, s64 v, s64 c)
    {
        return data_buffer[properties.flattenedIndex(u, v, c)];
    }

    __host__ u8 &atRed(s64 u, s64 v)
    {
        return data_buffer[properties.redIndex(u, v)];
    }

    __host__ u8 &atGreen(s64 u, s64 v)
    {
        return data_buffer[properties.greenIndex(u, v)];
    }

    __host__ u8 &atBlue(s64 u, s64 v)
    {
        return data_buffer[properties.blueIndex(u, v)];
    }

    /// @brief Gets the element at (u, v, c) in the image
    ///
    /// @param u The u coordinate (e.g. along the width) of interest
    /// @param v The v coordinate (e.g. along the height) of interest
    /// @param c The c coordinate (e.g. along the channel) of interest
    /// @return The data element
    __host__ const u8 &at(s64 u, s64 v, s64 c) const
    {
        return data_buffer[properties.flattenedIndex(u, v, c)];
    }

    __host__ const u8 &atRed(s64 u, s64 v) const
    {
        return data_buffer[properties.redIndex(u, v)];
    }

    __host__ const u8 &atGreen(s64 u, s64 v) const
    {
        return data_buffer[properties.greenIndex(u, v)];
    }

    __host__ const u8 &atBlue(s64 u, s64 v) const
    {
        return data_buffer[properties.blueIndex(u, v)];
    }

    __host__ void writeColourAt(Colour colour, s64 u, s64 v)
    {
        constexpr f32 kF32ToU8{255.999};
        atRed(u, v) = static_cast<u8>(kF32ToU8 * colour.x());
        atGreen(u, v) = static_cast<u8>(kF32ToU8 * colour.y());
        atBlue(u, v) = static_cast<u8>(kF32ToU8 * colour.z());
    }
};

} // namespace RayTracer

#endif // RAYTRACER_IMAGE_UTILS_IMAGE_HPP_
