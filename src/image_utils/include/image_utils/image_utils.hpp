#ifndef RAYTRACER_IMAGE_UTILS_IMAGE_UTILS_HPP_
#define RAYTRACER_IMAGE_UTILS_IMAGE_UTILS_HPP_

#include "image.hpp"
#include <optional>

namespace RayTracer
{

/// Contains utility functions to work with images.
/// For the definition of an image, see image.hpp
class ImageUtils
{
public:
    /// @brief Saves an image to disk
    ///
    /// @param filename The filename to save the image to
    /// @param image The image to save
    /// @return Whether the save was successful or not
    static bool saveImage(const char *filename, const Image &image);

    /// @brief Reads an image to disk
    ///
    /// @param filepath The filepath to the image that should be read
    /// @param requested_encoding The desired encoding of the image
    /// @return Returns the image if successful, or std::nullopt otherwise
    static std::optional<Image>
    readImage(const char *filepath,
              ImageEncodings requested_encoding = ImageEncodings::kBGR8);

    /// @brief Calculates the pitch of an image
    ///
    /// @param width The width to use
    /// @param channels The number of channels
    /// @return The pitch of the image
    static u64 calculatePitch(u64 width, u64 channels);

    /// @brief Calculates the flattened index of an image
    ///
    /// @param image The image whose properties will be used
    /// @param u The u coordinate (e.g. along the width) of interest
    /// @param v The v coordinate (e.g. along the height) of interest
    static u64 calculateFlattenedIndex(const Image &image, u64 u, u64 v);

    /// @brief Calculates the pitch of an image
    ///
    /// @param image The image whose properties will be used
    static u64 calculatePitch(const Image &image);

    /// @brief Calculates the colour step of an image
    ///
    /// @param image The image whose properties will be used
    static u64 calculateColourStep(const Image &image);
};

} // namespace RayTracer

#endif // RAYTRACER_IMAGE_UTILS_IMAGE_UTILS_HPP_
