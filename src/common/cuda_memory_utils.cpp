#include "common/check_cuda_errors.hpp"
#include "common/cuda_memory_utils.hpp"

namespace RayTracer
{
namespace cuda
{
namespace lowlevel
{
void *allocateCudaMemory(u64 byte_size)
{
    void *ptr;
    CHECK_CUDA_ERRORS(cudaMallocManaged(&ptr, byte_size));
    return ptr;
}

void prefetchToGpu(const void *ptr, u64 byte_size)
{
    int device{-1};
    cudaGetDevice(&device);
    CHECK_CUDA_ERRORS(cudaMemPrefetchAsync(ptr, byte_size, device, NULL));
}

void prefetchToCpu(const void *ptr, u64 byte_size)
{
    CHECK_CUDA_ERRORS(cudaMemPrefetchAsync(ptr, byte_size, cudaCpuDeviceId));
}

void copyCudaMemory(void *dst, const void *src, u64 byte_size,
                    cudaMemcpyKind kind)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(dst, src, byte_size, kind));
}

void deallocateCudaMemory(void *ptr)
{
    CHECK_CUDA_ERRORS(cudaFree(ptr));
}

} // namespace lowlevel

void waitForCuda()
{
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}
} // namespace cuda

} // namespace RayTracer
