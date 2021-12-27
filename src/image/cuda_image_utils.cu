#include "common/common_types.hpp"
#include "image/cuda_image_utils.hpp"

#include "image/image.hpp"
#include "vector3/vector3.hpp"

namespace RayTracer
{
namespace cuda
{

__device__ void writeColourAt(u8 *image_buffer,
                              const ImageProperties &properties,
                              const Colour &colour, s64 u, s64 v)
{
    image_buffer[properties.redIndex(u, v)] =
        static_cast<u8>(255.999 * colour.x());
    image_buffer[properties.greenIndex(u, v)] =
        static_cast<u8>(255.999 * colour.y());
    image_buffer[properties.blueIndex(u, v)] =
        static_cast<u8>(255.999 * colour.z());
}

} // namespace cuda

} // namespace RayTracer
