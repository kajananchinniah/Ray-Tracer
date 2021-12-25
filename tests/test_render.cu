#include "common/common_types.hpp"
#include "image_utils/image.hpp"
#include "image_utils/image_utils.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace
{

__global__ void test_render_cuda(RayTracer::u8 *image_buffer,
                                 RayTracer::ImageProperties properties)
{
    RayTracer::u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RayTracer::u64 u_stride = gridDim.x * blockDim.x;

    RayTracer::u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    RayTracer::u64 v_stride = gridDim.y * blockDim.y;

    for (RayTracer::u64 v = v_idx; v < properties.height; v += v_stride) {
        for (RayTracer::u64 u = u_idx; u < properties.width; u += u_stride) {
            RayTracer::Color color(double(u) / (properties.width - 1),
                                   double(properties.height - v - 1) /
                                       (properties.height - 1),
                                   0.25);
            image_buffer[properties.flattenedIndex(u, v, 0)] =
                static_cast<RayTracer::u8>(255.999 * color.z());
            image_buffer[properties.flattenedIndex(u, v, 1)] =
                static_cast<RayTracer::u8>(255.999 * color.y());
            image_buffer[properties.flattenedIndex(u, v, 2)] =
                static_cast<RayTracer::u8>(255.999 * color.x());
        }
    }
}

} // namespace

namespace RayTracer
{

void test_render()
{
    int image_width = 256;
    int image_height = 256;

    Image image(image_width, image_height);
    dim3 blocks(1, 1, 1);
    dim3 threads(1, 1, 1);
    test_render_cuda<<<blocks, threads>>>(image.data_buffer.get(),
                                          image.properties);
    auto result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cout << "Error: " << std::string(cudaGetErrorString(result))
                  << "\n";
        exit(0);
    }

    ImageUtils::saveImage("test_render.png", image);
}
} // namespace RayTracer

int main(void)
{
    RayTracer::test_render();
    return 0;
}
