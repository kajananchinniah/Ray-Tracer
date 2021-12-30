#include "gtest/gtest.h"

#include "common/common_types.hpp"
#include "common/cuda_memory_utils.hpp"
#include "image/cuda_image_utils.hpp"
#include "image/image.hpp"
#include "image/image_utils.hpp"
#include "ray/ray.hpp"
#include "ray/ray_utils.hpp"
#include "vector3/vector3.hpp"

#include "cuda.h"
#include "cuda_runtime.h"
#include "test_render_kernels.hpp"

#include <string>

namespace
{

std::string g_base_absolute_path{};

} // namespace

namespace RayTracer
{

void checkIfImagesAreEqual(const Image &left, const Image &right)
{
    EXPECT_EQ(left.properties.height, right.properties.height);
    EXPECT_EQ(left.properties.width, right.properties.width);
    EXPECT_EQ(left.properties.channels, right.properties.channels);
    EXPECT_EQ(left.properties.encoding, right.properties.encoding);
    for (s64 v = 0; v < left.properties.height; ++v) {
        for (s64 u = 0; u < left.properties.width; ++u) {
            EXPECT_NEAR(left.atRed(u, v), right.atRed(u, v), 1);
            EXPECT_NEAR(left.atGreen(u, v), right.atGreen(u, v), 1);
            EXPECT_NEAR(left.atBlue(u, v), right.atBlue(u, v), 1);
        }
    }
}

TEST(Render, BasicRender)
{
    constexpr s32 image_width{256};
    constexpr s32 image_height{256};

    Image image{image_width, image_height};

    constexpr dim3 threads(16, 16, 1);

    constexpr s32 blocks_x = (image_width + threads.x - 1) / threads.x;
    constexpr s32 blocks_y = (image_height - threads.y - 1) / threads.y;

    constexpr dim3 blocks(blocks_x, blocks_y, 1);

    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testBasicRenderCuda<<<blocks, threads>>>(image.data_buffer.get(),
                                             image.properties);
    cuda::waitForCuda();

    std::string file_path = g_base_absolute_path + "/test_basic_render.png";
    auto maybe_image = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    checkIfImagesAreEqual(image, ground_truth);
}

TEST(Render, BasicRenderWithRay)
{
    constexpr f32 aspect_ratio{16.0f / 9.0f};
    constexpr s32 image_width{400};
    constexpr s32 image_height{static_cast<int>(image_width / aspect_ratio)};

    constexpr f32 viewport_height{2.0f};
    constexpr f32 viewport_width{aspect_ratio * viewport_height};
    constexpr f32 focal_length{1.0f};

    const Point3f origin{0.0f, 0.0f, 0.0f};
    const Vector3f horizontal{viewport_width, 0.0f, 0.0f};
    const Vector3f vertical{0.0f, viewport_height, 0.0f};
    const auto lower_left_corner{origin - horizontal / 2 - vertical / 2 -
                                 Vector3f{0.0f, 0.0f, focal_length}};

    constexpr dim3 threads{16, 16, 1};

    constexpr s32 blocks_x = (image_width + threads.x - 1) / threads.x;
    constexpr s32 blocks_y = (image_height - threads.y - 1) / threads.y;

    constexpr dim3 blocks(blocks_x, blocks_y, 1);

    Image image{image_width, image_height};
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testBasicRenderWithRayCuda<<<blocks, threads>>>(
        image.data_buffer.get(), image.properties, origin, lower_left_corner,
        horizontal, vertical);
    cuda::waitForCuda();

    std::string file_path =
        g_base_absolute_path + "/test_basic_render_with_ray.png";
    auto maybe_image = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    checkIfImagesAreEqual(image, ground_truth);
}

TEST(Render, BasicRenderWithSphere)
{
    constexpr f32 aspect_ratio{16.0f / 9.0f};
    constexpr s32 image_width{400};
    constexpr s32 image_height{static_cast<int>(image_width / aspect_ratio)};

    constexpr f32 viewport_height{2.0f};
    constexpr f32 viewport_width{aspect_ratio * viewport_height};
    constexpr f32 focal_length{1.0f};

    const Point3f origin{0.0f, 0.0f, 0.0f};
    const Vector3f horizontal{viewport_width, 0.0f, 0.0f};
    const Vector3f vertical{0.0f, viewport_height, 0.0f};
    const auto lower_left_corner{origin - horizontal / 2 - vertical / 2 -
                                 Vector3f{0.0f, 0.0f, focal_length}};

    constexpr dim3 threads{16, 16, 1};

    constexpr s32 blocks_x = (image_width + threads.x - 1) / threads.x;
    constexpr s32 blocks_y = (image_height - threads.y - 1) / threads.y;

    constexpr dim3 blocks(blocks_x, blocks_y, 1);

    Image image{image_width, image_height};
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testBasicRenderWithSphereCuda<<<blocks, threads>>>(
        image.data_buffer.get(), image.properties, origin, lower_left_corner,
        horizontal, vertical);
    cuda::waitForCuda();
    ImageUtils::saveImage("test_basic_with_sphere.png", image);
}

} // namespace RayTracer

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    assert(argc == 2);
    g_base_absolute_path = std::string{argv[1]};
    return RUN_ALL_TESTS();
}
