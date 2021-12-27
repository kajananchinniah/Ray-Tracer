/// @file image_utils.hpp
/// @brief Contains utility functions to use with images
#ifndef RAYTRACER_IMAGE_IMAGE_UTILS_HPP_
#define RAYTRACER_IMAGE_IMAGE_UTILS_HPP_

#include "image.hpp"
#include <optional>

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

namespace ImageUtils
{

/// @brief Saves an image to disk
///
/// @param filename The filename to save the image to
/// @param image The image to save
/// @return Whether the save was successful or not
bool saveImage(const char *filename, const Image &image);

/// @brief Reads an image to disk
///
/// @param filepath The filepath to the image that should be read
/// @param requested_encoding The desired encoding of the image
/// @return Returns the image if successful, or std::nullopt otherwise
std::optional<Image>
readImage(const char *filepath,
          ImageEncodings requested_encoding = ImageEncodings::kBGR8);

} // namespace ImageUtils
} // namespace RayTracer

#endif // RAYTRACER_IMAGE_UTILS_IMAGE_UTILS_HPP_
