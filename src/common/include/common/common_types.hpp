/// @file common_types.hpp
/// @brief Contains definitions for common types
#ifndef RAYTRACER_COMMON_COMMON_TYPES_HPP_
#define RAYTRACER_COMMON_COMMON_TYPES_HPP_

#include <cstdint>

namespace RayTracer
{

/// 8 bit unsigned integer
using u8 = uint8_t;

/// 16 bit unsigned integer
using u16 = uint16_t;

/// 32 bit unsigned integer
using u32 = uint32_t;

/// 64 bit unsigned integer
using u64 = uint64_t;

/// 8 bit signed integer
using s8 = int8_t;

/// 16 bit signed integer
using s16 = int16_t;

/// 32 bit signed integer
using s32 = int32_t;

/// 64 bit signed integer
using s64 = int64_t;

/// 32 bit floating point
using f32 = float;

/// 64 bit floating point
using f64 = double;

static_assert(sizeof(f32) == sizeof(u32), "Size of float is not 32 bit");
static_assert(sizeof(f64) == sizeof(u64), "Size of double is not 64 bit");

/// Supported image encodings
enum class ImageEncodings {
    /// RGB8 image
    kRGB8,

    /// BGR8 image
    kBGR8
};

} // namespace RayTracer

#endif // RAYTRACER_COMMON_COMMON_TYPES_HPP_
