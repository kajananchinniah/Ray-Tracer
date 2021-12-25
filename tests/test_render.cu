#include "gtest/gtest.h"

#include "common/common_types.hpp"
#include "common/cuda_memory_utils.hpp"
#include "image_utils/image.hpp"
#include "image_utils/image_utils.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace
{

const char *g_image_file_path{};

__global__ void test_render_cuda(RayTracer::u8 *image_buffer,
                                 RayTracer::ImageProperties properties)
{
    RayTracer::u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RayTracer::u64 u_stride = gridDim.x * blockDim.x;

    RayTracer::u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    RayTracer::u64 v_stride = gridDim.y * blockDim.y;

    for (RayTracer::u64 v = v_idx; v < properties.height; v += v_stride) {
        for (RayTracer::u64 u = u_idx; u < properties.width; u += u_stride) {
            RayTracer::Colour colour(double(u) / (properties.width - 1),
                                     double(properties.height - v - 1) /
                                         (properties.height - 1),
                                     0.25);
            image_buffer[properties.redIndex(u, v)] =
                static_cast<RayTracer::u8>(255.999 * colour.z());
            image_buffer[properties.greenIndex(u, v)] =
                static_cast<RayTracer::u8>(255.999 * colour.y());
            image_buffer[properties.blueIndex(u, v)] =
                static_cast<RayTracer::u8>(255.999 * colour.x());
        }
    }
}

} // namespace

namespace RayTracer
{

TEST(Render, BasicRender)
{
    constexpr int image_width{256};
    constexpr int image_height{256};

    Image image{image_width, image_height};

    constexpr dim3 threads(16, 16, 1);

    constexpr int blocks_x = (image_width + threads.x - 1) / threads.x;
    constexpr int blocks_y = (image_height - threads.y - 1) / threads.y;

    constexpr dim3 blocks(blocks_x, blocks_y, 1);

    test_render_cuda<<<blocks, threads>>>(image.data_buffer.get(),
                                          image.properties);
    cuda::waitForCuda();
    ImageUtils::saveImage("test_render_basic.png", image);
}

} // namespace RayTracer

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    // assert(argc == 2);
    // g_image_file_path = argv[1];
    return RUN_ALL_TESTS();
}
