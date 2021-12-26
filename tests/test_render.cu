#include "gtest/gtest.h"

#include "common/common_types.hpp"
#include "common/cuda_memory_utils.hpp"
#include "image_utils/cuda_image_utils.hpp"
#include "image_utils/image.hpp"
#include "image_utils/image_utils.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace
{

const char *g_image_file_path{};

__global__ void testBasicRenderCuda(RayTracer::u8 *image_buffer,
                                    RayTracer::ImageProperties properties)
{
    RayTracer::u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RayTracer::u64 u_stride = gridDim.x * blockDim.x;

    RayTracer::u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    RayTracer::u64 v_stride = gridDim.y * blockDim.y;

    for (RayTracer::u64 v = v_idx; v < properties.height; v += v_stride) {
        for (RayTracer::u64 u = u_idx; u < properties.width; u += u_stride) {
            float red{static_cast<RayTracer::f32>(u) /
                      static_cast<RayTracer::f32>(properties.width - 1)};
            float green{static_cast<RayTracer::f32>(properties.height - v - 1) /
                        static_cast<RayTracer::f32>(properties.height - 1)};
            float blue{0.25};
            RayTracer::Colour colour{red, green, blue};
            RayTracer::cuda::writeColourAt(image_buffer, properties, colour, u,
                                           v);
        }
    }
}

} // namespace

namespace RayTracer
{

void checkIfImagesAreEqual(const Image &left, const Image &right)
{
    EXPECT_EQ(left.properties.height, right.properties.height);
    EXPECT_EQ(left.properties.width, right.properties.height);
    EXPECT_EQ(left.properties.channels, right.properties.channels);
    EXPECT_EQ(left.properties.encoding, right.properties.encoding);
    for (s64 v = 0; v < left.properties.height; ++v) {
        for (s64 u = 0; u < left.properties.width; ++u) {
            ASSERT_EQ(left.atRed(u, v), right.atRed(u, v));
            ASSERT_EQ(left.atGreen(u, v), right.atGreen(u, v));
            ASSERT_EQ(left.atBlue(u, v), right.atBlue(u, v));
        }
    }
}

TEST(Render, BasicRender)
{
    constexpr int image_width{256};
    constexpr int image_height{256};

    Image image{image_width, image_height};

    constexpr dim3 threads(16, 16, 1);

    constexpr int blocks_x = (image_width + threads.x - 1) / threads.x;
    constexpr int blocks_y = (image_height - threads.y - 1) / threads.y;

    constexpr dim3 blocks(blocks_x, blocks_y, 1);

    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testBasicRenderCuda<<<blocks, threads>>>(image.data_buffer.get(),
                                             image.properties);
    cuda::waitForCuda();

    auto maybe_image = ImageUtils::readImage(g_image_file_path);
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    checkIfImagesAreEqual(image, ground_truth);
}

} // namespace RayTracer

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    assert(argc == 2);
    g_image_file_path = argv[1];
    return RUN_ALL_TESTS();
}
