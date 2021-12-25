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

bool saveImage(const char *filename, const Image &image)
{
    if (image.properties.channels != 3) {
        std::cout << "Warning: received an unsupported number of channels. Not "
                     "saving\n";
        return false;
    }

    cv::Mat output_image(image.properties.height, image.properties.width,
                         CV_8UC3);

    if (image.properties.size() !=
        output_image.total() * output_image.elemSize()) {
        std::cout << "Warning: image is not the correct size. Not saving\n";
        return false;
    }

    cuda::prefetchToCpu(image.data_buffer.get(), image.properties.size());
    cuda::copyCudaMemory(output_image.data, image.data_buffer.get(),
                         image.properties.size());

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

    Image image(cv_image.size().width, cv_image.size().height,
                requested_encoding);
    cuda::copyCudaMemory(image.data_buffer.get(), cv_image.data,
                         image.properties.size());
    return image;
}

} // namespace ImageUtils
} // namespace RayTracer
