#include "common/common_types.hpp"
#include "image.hpp"

#include "vector3/vector3.hpp"

namespace RayTracer
{
namespace cuda
{

__device__ void writeColourAt(u8 *image_buffer,
                              const ImageProperties &properties,
                              const Colour &colour, s64 u, s64 v);
} // namespace cuda

} // namespace RayTracer
