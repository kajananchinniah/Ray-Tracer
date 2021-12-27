#include "image_utils/image.hpp"
#include "common/common_types.hpp"
#include <iostream>

namespace
{

RayTracer::s64 getRedIndex(RayTracer::ImageEncodings e)
{
    switch (e) {
    case RayTracer::ImageEncodings::kBGR8:
        return 2;
    case RayTracer::ImageEncodings::kRGB8:
        return 0;
    default:
        std::cout << "Received unexpected encoding!";
        std::exit(1);
    }
}

RayTracer::s64 getGreenIndex(RayTracer::ImageEncodings e)
{
    switch (e) {
    case RayTracer::ImageEncodings::kBGR8:
    case RayTracer::ImageEncodings::kRGB8:
        return 1;
    default:
        std::cout << "Received unexpected encoding!";
        std::exit(1);
    }
}

RayTracer::s64 getBlueIndex(RayTracer::ImageEncodings e)
{
    switch (e) {
    case RayTracer::ImageEncodings::kBGR8:
        return 0;
    case RayTracer::ImageEncodings::kRGB8:
        return 2;
    default:
        std::cout << "Received unexpected encoding!";
        std::exit(1);
    }
}

} // namespace
namespace RayTracer
{

ImageProperties::ImageProperties(s64 w, s64 h, ImageEncodings e)
    : width{w}, height{h}, channels{3}, encoding{e}, red_offset{getRedIndex(e)},
      green_offset{getGreenIndex(e)}, blue_offset{getBlueIndex(e)}
{
}

} // namespace RayTracer
