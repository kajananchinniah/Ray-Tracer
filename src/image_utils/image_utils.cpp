#include "image_utils/image_utils.hpp"

#include "common/common_types.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <optional>

namespace RayTracer
{

bool ImageUtils::saveImage(const char *filename, const Image &image)
{
    if (image.channels != 3) {
        std::cout << "Warning: received an unsupported number of channels. Not "
                     "saving\n";
        return false;
    }

    cv::Mat output_image(image.height, image.width, CV_8UC3);

    if (image.data_buffer.size() !=
        output_image.total() * output_image.elemSize()) {
        std::cout << "Warning: image is not the correct size. Not saving\n";
        return false;
    }

    std::memcpy(output_image.data, image.data_buffer.data(),
                image.data_buffer.size());

    return cv::imwrite(filename, output_image);
}

std::optional<Image> ImageUtils::readImage(const char *filename,
                                           ImageEncodings requested_encoding)
{
    cv::Mat cv_image = cv::imread(filename);
    if (!cv_image.data) {
        return std::nullopt;
    }

    if (requested_encoding == ImageEncodings::kRGB8) {
        cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
    }

    Image image;
    image.width = cv_image.size().width;
    image.height = cv_image.size().height;
    image.channels = 3;
    image.pitch = ImageUtils::calculatePitch(image.width, image.channels);
    image.encoding = requested_encoding;
    image.data_buffer.resize(cv_image.total() * cv_image.elemSize());
    std::memcpy(image.data_buffer.data(), cv_image.data,
                image.data_buffer.size());
    return image;
}

u64 ImageUtils::calculatePitch(u64 width, u64 channels)
{
    return width * channels * sizeof(u8);
}

u64 ImageUtils::calculatePitch(const Image &image)
{
    return image.width * image.channels * sizeof(u8);
}

u64 ImageUtils::calculateColourStep(const Image &image)
{
    return image.channels * sizeof(u8);
}

u64 ImageUtils::calculateFlattenedIndex(const Image &image, u64 u, u64 v)
{
    return v * ImageUtils::calculatePitch(image) +
           u * ImageUtils::calculateColourStep(image);
}

} // namespace RayTracer
