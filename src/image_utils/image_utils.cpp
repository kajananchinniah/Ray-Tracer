#include "image_utils/image_utils.hpp"

#include "common/common_types.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <optional>

#include "common/cuda_memory_utils.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

namespace RayTracer
{

namespace ImageUtils
{
Image createEmptyImage(u64 height, u64 width, ImageEncodings encoding)
{
    Image image;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.encoding = encoding;
    image.pitch = calculatePitch(image.width, image.channels);
    image.size = calculateSize(image.width, image.height, image.channels);
    image.byte_size =
        calculateByteSize(image.width, image.height, image.channels);
    image.pitch = calculatePitch(image.width, image.channels);
    image.data_buffer = cuda::createCudaUniquePtrArray<u8>(image.size);
    return image;
}

bool saveImage(const char *filename, const Image &image)
{
    if (image.channels != 3) {
        std::cout << "Warning: received an unsupported number of channels. Not "
                     "saving\n";
        return false;
    }

    cv::Mat output_image(image.height, image.width, CV_8UC3);

    if (image.size != output_image.total() * output_image.elemSize()) {
        std::cout << "Warning: image is not the correct size. Not saving\n";
        return false;
    }

    cuda::copyCudaMemory(output_image.data, image.data_buffer.get(),
                         image.size);

    return cv::imwrite(filename, output_image);
}

std::optional<Image> readImage(const char *filename,
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
    image.pitch = calculatePitch(image.width, image.channels);
    image.encoding = requested_encoding;
    image.size = cv_image.total() * cv_image.elemSize();
    image.byte_size = cv_image.total() * cv_image.elemSize();
    image.data_buffer = cuda::createCudaUniquePtrArray<u8>(image.size);
    cuda::copyCudaMemory(image.data_buffer.get(), cv_image.data, image.size);
    return image;
}

__device__ __host__ u64 calculatePitch(u64 width, u64 channels)
{
    return width * channels * sizeof(u8);
}

__device__ __host__ u64 calculatePitch(const Image &image)
{
    return image.width * image.channels * sizeof(u8);
}

__device__ __host__ u64 calculateColourStep(const Image &image)
{
    return image.channels * sizeof(u8);
}

__device__ __host__ u64 calculateFlattenedIndex(const Image &image, u64 u,
                                                u64 v)
{
    return v * calculatePitch(image) + u * calculateColourStep(image);
}

__device__ __host__ u64 calculateSize(u64 width, u64 height, u64 channels)
{
    return width * height * channels;
}

__device__ __host__ u64 calculateByteSize(u64 width, u64 height, u64 channels)
{
    return calculateSize(width, height, channels) * sizeof(u8);
}

} // namespace ImageUtils
} // namespace RayTracer
