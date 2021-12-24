#ifndef RAYTRACER_IMAGE_UTILS_IMAGE_HPP_
#define RAYTRACER_IMAGE_UTILS_IMAGE_HPP_

#include "common/common_types.hpp"
#include <memory>
#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

/// Holds relevant properties to work with images.
/// The helper functions in image_utils.hpp should be used to access the data.
struct Image {
    /// A data buffer holding the pixels of an image
    std::unique_ptr<u8[], decltype(&cudaFree)> data_buffer{nullptr, cudaFree};

    /// The size of the image in the type of the image
    u64 size{};

    /// The size of the image in bytes
    u64 byte_size{};

    /// The height of the image
    u64 height{};

    /// The width of the image
    u64 width{};

    /// The number of channels of the image
    u64 channels{};

    /// The pitch of the image
    u64 pitch{};

    /// The encoding of the image
    ImageEncodings encoding{};
};

} // namespace RayTracer

#endif // RAYTRACER_IMAGE_UTILS_IMAGE_HPP_
