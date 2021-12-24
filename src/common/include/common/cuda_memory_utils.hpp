#ifndef RAY_TRACER_COMMON_CUDA_MEMORY_UTILS_HPP_
#define RAY_TRACER_COMMON_CUDA_MEMORY_UTILS_HPP_

#include "common_types.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

#include <memory>
#include <type_traits>

namespace RayTracer
{

namespace cuda
{

namespace lowlevel
{

void *allocateCudaMemory(u64 byte_size);
void prefetchToGpu(const void *ptr, u64 byte_size);
void prefetchToCpu(const void *ptr, u64 byte_size);
void transferCudaMemory(void *dst, const void *src, u64 byte_size,
                        cudaMemcpyKind kind = cudaMemcpyDefault);

} // namespace lowlevel

template <typename T>
std::unique_ptr<T, decltype(&cudaFree)> createCudaUniquePtr(u64 size)
{
    u64 byte_size{sizeof(T) * size};
    std::unique_ptr<T, decltype(&cudaFree)> ptr{
        static_cast<T *>(lowlevel::allocateCudaMemory(byte_size)), cudaFree};
    return ptr;
}

template <typename T>
std::unique_ptr<T[], decltype(&cudaFree)> createCudaUniquePtrArray(u64 size)
{
    u64 byte_size{sizeof(T) * size};
    std::unique_ptr<T[], decltype(&cudaFree)> ptr{
        static_cast<T *>(lowlevel::allocateCudaMemory(byte_size)), cudaFree};
    return ptr;
}
template <typename T> void prefetchToGpu(const T *ptr, u64 size)
{
    lowlevel::prefetchToGpu(static_cast<const void *>(ptr), sizeof(T) * size);
}

template <typename T> void prefetchToCpu(const T *ptr, u64 size)
{
    lowlevel::prefetchToCpu(static_cast<const void *>(ptr), sizeof(T) * size);
}

template <typename T>
void transferCudaMemory(T *dst, const T *src, u64 size,
                        cudaMemcpyKind kind = cudaMemcpyDefault)
{
    lowlevel::transferCudaMemory(static_cast<void *>(dst),
                                 static_cast<const void *>(src),
                                 sizeof(T) * size, kind);
}

} // namespace cuda

} // namespace RayTracer

#endif // RAY_TRACER_COMMON_CUDA_MEMORY_UTILS_HPP_
