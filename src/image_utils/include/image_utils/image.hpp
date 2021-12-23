#ifndef RAYTRACER_IMAGE_UTILS_IMAGE_HPP_
#define RAYTRACER_IMAGE_UTILS_IMAGE_HPP_

#include "common/common_types.hpp"
#include <vector>

namespace RayTracer
{

/// Holds relevant properties to work with images.
/// The helper functions in image_utils.hpp should be used to access the data.
struct Image {
    /// A data buffer holding the pixels of an image
    std::vector<u8> data_buffer{};

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
