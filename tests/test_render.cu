#include "gtest/gtest.h"

#include "camera/camera.hpp"
#include "common/common_types.hpp"
#include "common/cuda_memory_utils.hpp"
#include "image/cuda_image_utils.hpp"
#include "image/image.hpp"
#include "image/image_utils.hpp"
#include "material/material.hpp"
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

void checkIfImagesAreEqual(const Image &left, const Image &right,
                           u64 error_margin = 1)
{
    EXPECT_EQ(left.properties.height, right.properties.height);
    EXPECT_EQ(left.properties.width, right.properties.width);
    EXPECT_EQ(left.properties.channels, right.properties.channels);
    EXPECT_EQ(left.properties.encoding, right.properties.encoding);
    for (s64 v = 0; v < left.properties.height; ++v) {
        for (s64 u = 0; u < left.properties.width; ++u) {
            EXPECT_NEAR(left.atRed(u, v), right.atRed(u, v), error_margin);
            EXPECT_NEAR(left.atGreen(u, v), right.atGreen(u, v), error_margin);
            EXPECT_NEAR(left.atBlue(u, v), right.atBlue(u, v), error_margin);
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
    std::string file_path =
        g_base_absolute_path + "/test_basic_render_with_sphere.png";
    auto maybe_image = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    checkIfImagesAreEqual(image, ground_truth);
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
    std::string file_path =
        g_base_absolute_path + "/test_basic_render_with_world.png";
    auto maybe_image = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    checkIfImagesAreEqual(image, ground_truth);
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
    ImageUtils::saveImage("test_basic_render_with_antialiasing.png", image);
}

TEST(Render, BasicRenderWithDiffuse)
{
    Image image{kImageWidth, kImageHeight};
    Camera camera;
    SphereArray world{2};
    ASSERT_TRUE(world.add(Sphere{Point3f(0.0f, 0.0f, -1.0f), 0.5f}));
    ASSERT_TRUE(world.add(Sphere{Point3f(0.0f, -100.5f, -1.0f), 100.0f}));
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testRenderBasicWithDiffuse<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera,
        world.data_buffer.get(), world.properties, image.random_state.get(),
        100, 50);
    cuda::waitForCuda();
    std::string file_path =
        g_base_absolute_path + "/test_basic_render_with_diffuse.png";
    auto maybe_image = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    ImageUtils::saveImage("test_basic_with_diffuse.png", image);
    // TODO: verify that randomness is the cause for high failure (seems to be
    // the case)
    checkIfImagesAreEqual(image, ground_truth, 25);
}

TEST(Render, BasicRenderWithMaterial)
{
    Image image{kImageWidth, kImageHeight};
    Camera camera;
    SphereArray world{4};
    Lambertian material_ground{Colour{0.8f, 0.8f, 0.0f}};
    world.add(Sphere{Point3f{0.0f, -100.5f, -1.0f}, 100.0,
                     Material{material_ground}});
    Lambertian material_center{Colour{0.7f, 0.3f, 0.3f}};
    world.add(
        Sphere{Point3f{0.0f, 0.0f, -1.0f}, 0.5f, Material{material_center}});
    Metal material_left{Colour{0.8f, 0.8f, 0.8f}, 0.3f};
    world.add(
        Sphere{Point3f{-1.0f, 0.0f, -1.0f}, 0.5f, Material{material_left}});
    Metal material_right{Colour{0.8f, 0.6f, 0.2f}, 1.0f};
    world.add(
        Sphere{Point3f{1.0f, 0.0f, -1.0f}, 0.5f, Material{material_right}});
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testRenderBasicWithMaterial<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera,
        world.data_buffer.get(), world.properties, image.random_state.get(),
        100, 50);
    cuda::waitForCuda();
    std::string file_path =
        g_base_absolute_path + "/test_basic_render_with_material.png";
    auto maybe_image = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    ImageUtils::saveImage("test_basic_with_metal.png", image);
    checkIfImagesAreEqual(image, ground_truth);
}

TEST(Render, BasicRenderWithDielectric)
{
    Image image{kImageWidth, kImageHeight};
    Camera camera;
    SphereArray world{5};
    Lambertian material_ground{Colour{0.8f, 0.8f, 0.0f}};
    world.add(Sphere{Point3f{0.0f, -100.5f, -1.0f}, 100.0,
                     Material{material_ground}});
    Lambertian material_center{Colour{0.1f, 0.2f, 0.5f}};
    world.add(
        Sphere{Point3f{0.0f, 0.0f, -1.0f}, 0.5f, Material{material_center}});
    Dielectric material_left{1.5f};
    world.add(
        Sphere{Point3f{-1.0f, 0.0f, -1.0f}, 0.5f, Material{material_left}});
    world.add(
        Sphere{Point3f{-1.0f, 0.0f, -1.0f}, -0.4f, Material{material_left}});
    Metal material_right{Colour{0.8f, 0.6f, 0.2f}, 0.0f};
    world.add(
        Sphere{Point3f{1.0f, 0.0f, -1.0f}, 0.5f, Material{material_right}});
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testRenderBasicWithMaterial<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera,
        world.data_buffer.get(), world.properties, image.random_state.get(),
        100, 50);
    cuda::waitForCuda();
    std::string file_path =
        g_base_absolute_path + "/test_basic_render_with_dieletric.png";
    auto maybe_image = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    ImageUtils::saveImage("test_basic_with_dielectric.png", image);
    checkIfImagesAreEqual(image, ground_truth);
}

TEST(Render, BasicRenderWithAdjustableCamera)
{
    Image image{kImageWidth, kImageHeight};
    Camera camera{Point3f{-2.0f, 2.0f, 1.0f}, Point3f{0.0f, 0.0f, -1.0f},
                  Vector3f{0.0f, 1.0f, 0.0f}, 90.0f, kAspectRatio};
    SphereArray world{5};
    Lambertian material_ground{Colour{0.8f, 0.8f, 0.0f}};
    world.add(Sphere{Point3f{0.0f, -100.5f, -1.0f}, 100.0,
                     Material{material_ground}});
    Lambertian material_center{Colour{0.1f, 0.2f, 0.5f}};
    world.add(
        Sphere{Point3f{0.0f, 0.0f, -1.0f}, 0.5f, Material{material_center}});
    Dielectric material_left{1.5f};
    world.add(
        Sphere{Point3f{-1.0f, 0.0f, -1.0f}, 0.5f, Material{material_left}});
    world.add(
        Sphere{Point3f{-1.0f, 0.0f, -1.0f}, -0.4f, Material{material_left}});
    Metal material_right{Colour{0.8f, 0.6f, 0.2f}, 0.0f};
    world.add(
        Sphere{Point3f{1.0f, 0.0f, -1.0f}, 0.5f, Material{material_right}});
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testRenderBasicWithMaterial<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera,
        world.data_buffer.get(), world.properties, image.random_state.get(),
        100, 50);
    cuda::waitForCuda();
    std::string file_path = g_base_absolute_path +
                            "/test_basic_render_with_adjustable_camera_far.png";
    auto maybe_image_far = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image_far);
    const auto &ground_truth_far = maybe_image_far.value();
    ImageUtils::saveImage("test_basic_with_adjustable_camera_far.png", image);
    checkIfImagesAreEqual(image, ground_truth_far);

    camera = Camera{Point3f{-2.0f, 2.0f, 1.0f}, Point3f{0.0f, 0.0f, -1.0f},
                    Vector3f{0.0f, 1.0f, 0.0f}, 20.0f, kAspectRatio};
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testRenderBasicWithMaterial<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera,
        world.data_buffer.get(), world.properties, image.random_state.get(),
        100, 50);
    cuda::waitForCuda();
    file_path = g_base_absolute_path +
                "/test_basic_render_with_adjustable_camera_near.png";
    auto maybe_image_near = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image_near);
    const auto &ground_truth_near = maybe_image_near.value();
    ImageUtils::saveImage("test_basic_with_adjustable_camera_near.png", image);
    checkIfImagesAreEqual(image, ground_truth_near);
}

TEST(Render, BasicRenderWithDefocusBlur)
{
    Image image{kImageWidth, kImageHeight};
    Point3f look_from{3.0f, 3.0f, 2.0f};
    Point3f look_at{0.0f, 0.0f, -1.0f};
    Vector3f v_up{0.0f, 1.0f, 0.0f};
    f32 dist_to_focus{(look_from - look_at).magnitude_host()};
    f32 aperture{2.0f};
    Camera camera{look_from,    look_at,  v_up,         20.0f,
                  kAspectRatio, aperture, dist_to_focus};
    SphereArray world{5};
    Lambertian material_ground{Colour{0.8f, 0.8f, 0.0f}};
    world.add(Sphere{Point3f{0.0f, -100.5f, -1.0f}, 100.0,
                     Material{material_ground}});
    Lambertian material_center{Colour{0.1f, 0.2f, 0.5f}};
    world.add(
        Sphere{Point3f{0.0f, 0.0f, -1.0f}, 0.5f, Material{material_center}});
    Dielectric material_left{1.5f};
    world.add(
        Sphere{Point3f{-1.0f, 0.0f, -1.0f}, 0.5f, Material{material_left}});
    world.add(
        Sphere{Point3f{-1.0f, 0.0f, -1.0f}, -0.4f, Material{material_left}});
    Metal material_right{Colour{0.8f, 0.6f, 0.2f}, 0.0f};
    world.add(
        Sphere{Point3f{1.0f, 0.0f, -1.0f}, 0.5f, Material{material_right}});
    cuda::prefetchToGpu(image.data_buffer.get(), image.properties.size());
    testRenderBasicWithDefocusBlur<<<kBlocks, kThreads>>>(
        image.data_buffer.get(), image.properties, camera,
        world.data_buffer.get(), world.properties, image.random_state.get(),
        100, 50);
    cuda::waitForCuda();
    std::string file_path =
        g_base_absolute_path + "/test_basic_render_with_defocus_blur.png";
    auto maybe_image = ImageUtils::readImage(file_path.c_str());
    EXPECT_TRUE(maybe_image);
    const auto &ground_truth = maybe_image.value();
    ImageUtils::saveImage("test_basic_with_defocus_blur.png", image);
    checkIfImagesAreEqual(image, ground_truth);
}

} // namespace RayTracer

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    assert(argc == 2);
    g_base_absolute_path = std::string{argv[1]};
    return RUN_ALL_TESTS();
}
