#ifndef RAYTRACER_IMAGE_UTILS_IMAGE_UTILS_HPP_
#define RAYTRACER_IMAGE_UTILS_IMAGE_UTILS_HPP_

#include "image.hpp"
#include <optional>

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

namespace ImageUtils
{

/// @brief Creates and allocates space for an empty image
///
/// @param height The height of the empty image
/// @param width The width of the empty image
/// @param encoding The encoding of empty image
/// @return The newly created image
Image createEmptyImage(u64 height, u64 width, ImageEncodings encoding);

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

/// @brief Calculates the pitch of an image
///
/// @param width The width to use
/// @param channels The number of channels
/// @return The pitch of the image
__device__ __host__ u64 calculatePitch(u64 width, u64 channels);

/// @brief Calculates the flattened index of an image
///
/// @param image The image whose properties will be used
/// @param u The u coordinate (e.g. along the width) of interest
/// @param v The v coordinate (e.g. along the height) of interest
__device__ __host__ u64 calculateFlattenedIndex(const Image &image, u64 u,
                                                u64 v);

/// @brief Calculates the pitch of an image
///
/// @param image The image whose properties will be used
__device__ __host__ u64 calculatePitch(const Image &image);

/// @brief Calculates the colour step of an image
///
/// @param image The image whose properties will be used
__device__ __host__ u64 calculateColourStep(const Image &image);

/// @brief Calculates the size of an image
///
/// @param width The width of the image
/// @param height The height of the image
/// @param channels The number of channels of the image
__device__ __host__ u64 calculateSize(u64 width, u64 height, u64 channels);

/// @brief Calculates the byte size of an image
///
/// @param width The width of the image
/// @param height The height of the image
/// @param channels The number of channels of the image
__device__ __host__ u64 calculateByteSize(u64 width, u64 height, u64 channels);

} // namespace ImageUtils
} // namespace RayTracer

#endif // RAYTRACER_IMAGE_UTILS_IMAGE_UTILS_HPP_
