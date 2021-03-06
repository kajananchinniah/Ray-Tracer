#include "image/image_utils.hpp"

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

bool saveImage(const char *filename, const Image &image)
{
    if (image.properties.channels != 3) {
        std::cout << "Warning: received an unsupported number of channels. Not "
                     "saving\n";
        return false;
    }

    cuda::waitForCuda();
    cuda::prefetchToCpu(image.data_buffer.get(), image.properties.size());
    cv::Mat output_image(image.properties.height, image.properties.width,
                         CV_8UC3, image.data_buffer.get());

    return cv::imwrite(filename, output_image);
}

std::optional<Image> readImage(const char *filepath,
                               ImageEncodings requested_encoding)
{
    cv::Mat cv_image = cv::imread(filepath);
    if (cv_image.data == nullptr) {
        return std::nullopt;
    }

    if (requested_encoding == ImageEncodings::kRGB8) {
        cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
    }

    Image image(cv_image.size().width, cv_image.size().height,
                requested_encoding);
    cuda::copyCudaMemory(image.data_buffer.get(), cv_image.data,
                         image.properties.size());
    return image;
}

} // namespace ImageUtils
} // namespace RayTracer
