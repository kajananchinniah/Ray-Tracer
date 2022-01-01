#include "gtest/gtest.h"

#include "camera/camera.hpp"
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
constexpr dim3 kThreads{16, 16, 1};

constexpr RayTracer::s32 kTestBasicImageWidth{256};
constexpr RayTracer::s32 kTestBasicImageHeight{256};
constexpr dim3 kTestBasicBlocks{
    (kTestBasicImageWidth + kThreads.x - 1) / kThreads.x,
    (kTestBasicImageWidth + kThreads.y - 1) / kThreads.y, 1};

constexpr RayTracer::f32 kAspectRatio{16.0 / 9.0};
constexpr RayTracer::s32 kImageWidth{400};
constexpr RayTracer::s32 kImageHeight{
    static_cast<RayTracer::s32>(kImageWidth / kAspectRatio)};
constexpr dim3 kBlocks{(kImageWidth + kThreads.x - 1) / kThreads.x,
                       (kImageHeight + kThreads.y - 1) / kThreads.y, 1};

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
    Image image{kTestBasicImageWidth, kTestBasicImageHeight};

    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testBasicRenderCuda<<<kTestBasicBlocks, kThreads>>>(image.data_buffer.get(),
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
    Image image{kImageWidth, kImageHeight};
    Camera camera;
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testBasicRenderWithRayCuda<<<kBlocks, kThreads>>>(image.data_buffer.get(),
                                                      image.properties, camera);
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
    Image image{kImageWidth, kImageHeight};
    Camera camera;
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testBasicRenderWithSphereCuda<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera);
    cuda::waitForCuda();
    ImageUtils::saveImage("test_basic_with_sphere.png", image);
}

TEST(Render, BasicRenderWithWorld)
{
    Image image{kImageWidth, kImageHeight};
    Camera camera;
    SphereArray world{2};
    ASSERT_TRUE(world.add(Sphere{Point3f(0.0f, 0.0f, -1.0f), 0.5f}));
    ASSERT_TRUE(world.add(Sphere{Point3f(0.0f, -100.5f, -1.0f), 100.0f}));
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testRenderBasicRenderWithWorld<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera,
        world.data_buffer.get(), world.properties);
    cuda::waitForCuda();
    ImageUtils::saveImage("test_basic_with_world.png", image);
}

TEST(Render, BasicRenderWithAntiAliasing)
{
    Image image{kImageWidth, kImageHeight};
    Camera camera;
    SphereArray world{2};
    ASSERT_TRUE(world.add(Sphere{Point3f(0.0f, 0.0f, -1.0f), 0.5f}));
    ASSERT_TRUE(world.add(Sphere{Point3f(0.0f, -100.5f, -1.0f), 100.0f}));
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testRenderBasicWithAntiAliasing<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera,
        world.data_buffer.get(), world.properties, image.random_state.get(),
        100);
    cuda::waitForCuda();
    ImageUtils::saveImage("test_basic_with_antialiasing.png", image);
}

} // namespace RayTracer

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    assert(argc == 2);
    g_base_absolute_path = std::string{argv[1]};
    return RUN_ALL_TESTS();
}
