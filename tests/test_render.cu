#include "test_render.hpp"

namespace
{

__global__ void test_render_cuda(RayTracer::Image image)
{
    RayTracer::u64 u_idx = blockIdx.x * blockDim.x + threadIdx.x;
    RayTracer::u64 u_stride = gridDim.x * blockDim.x;

    RayTracer::u64 v_idx = blockIdx.y * blockDim.y + threadIdx.y;
    RayTracer::u64 v_stride = gridDim.y * blockDim.y;

    for (RayTracer::u64 v = v_idx; v < image.height; v += v_stride) {
        for (RayTracer::u64 u = u_idx; u < image.width; u += u_stride) {
            RayTracer::Color color(double(u) / (image.width - 1),
                                   double(v) / (image.height - 1), 0.25);
            image.writeColorAt(color, u, v);
        }
    }
}

} // namespace

namespace RayTracer
{

void test_render()
{
    int image_width = 256;
    int image_height = 256;

    Image image(image_width, image_height);
    dim3 blocks(1, 1, 1);
    dim3 threads(1, 1, 1);
    test_render_cuda<<<blocks, threads>>>(image);
    auto result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cout << "Error: " << std::string(cudaGetErrorString(result))
                  << "\n";
        exit(0);
    }
    ImageUtils::saveImage("test_render.png", image);
}
} // namespace RayTracer

int main(void)
{
    RayTracer::test_render();
    return 0;
}
