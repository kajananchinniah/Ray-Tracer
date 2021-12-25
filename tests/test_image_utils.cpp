#include <filesystem>

#include "gtest/gtest.h"

#include "common/common_types.hpp"
#include "image_utils/image.hpp"
#include "image_utils/image_utils.hpp"
#include <iostream>

namespace
{

const char *g_image_file_path{};

} // namespace

namespace RayTracer
{

constexpr u64 kTestImageHeight{280};
constexpr u64 kTestImageWidth{628};
constexpr u64 kTestImageChannels{3};

static void readImageSuccessBase(const char *file_path,
                                 ImageEncodings test_encoding)
{
    u64 red_idx{};
    u64 green_idx{};
    u64 blue_idx{};

    switch (test_encoding) {
    case ImageEncodings::kRGB8:
        red_idx = 0;
        green_idx = 1;
        blue_idx = 2;
        break;
    case ImageEncodings::kBGR8:
        blue_idx = 0;
        green_idx = 1;
        red_idx = 2;
        break;
    default:
        EXPECT_TRUE(false);
    }

    auto maybe_image = ImageUtils::readImage(file_path, test_encoding);
    EXPECT_TRUE(maybe_image);
    auto &image = maybe_image.value();
    EXPECT_EQ(image.properties.height, kTestImageHeight);
    EXPECT_EQ(image.properties.width, kTestImageWidth);
    EXPECT_EQ(image.properties.channels, kTestImageChannels);
    EXPECT_EQ(image.properties.encoding, test_encoding);
    for (s64 v = 0; v < image.properties.height; ++v) {
        for (s64 u = 0; u < image.properties.width; ++u) {
            ASSERT_EQ(image.at(u, v, red_idx), 0x12);
            ASSERT_EQ(image.at(u, v, green_idx), 0x34);
            ASSERT_EQ(image.at(u, v, blue_idx), 0x56);
        }
    }
}

TEST(ImageUtils, ReadImageSuccessRGB8)
{
    readImageSuccessBase(g_image_file_path, ImageEncodings::kRGB8);
}

TEST(ImageUtils, ReadImageSuccessBGR8)
{
    readImageSuccessBase(g_image_file_path, ImageEncodings::kBGR8);
}

TEST(ImageUtils, ReadImageFail)
{
    auto maybe_image = ImageUtils::readImage("fail.png", ImageEncodings::kRGB8);
    EXPECT_TRUE(!maybe_image);
}

TEST(ImageUtils, SaveImage)
{
    auto maybe_image =
        ImageUtils::readImage(g_image_file_path, ImageEncodings::kBGR8);
    EXPECT_TRUE(maybe_image);
    auto &image = maybe_image.value();
    bool result = ImageUtils::saveImage("test.png", image);
    EXPECT_TRUE(result);
    readImageSuccessBase("test.png", ImageEncodings::kBGR8);
}

} // namespace RayTracer

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    assert(argc == 2);
    g_image_file_path = argv[1];
    return RUN_ALL_TESTS();
}
