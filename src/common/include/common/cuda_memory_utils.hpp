/// @file cuda_memory_utils.hpp
/// @brief Contains useful functions to handle cuda memory
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

/// @brief Allocates unified cuda memory
///
/// @param byte_size The number of bytes that should be allocated
/// @return The newly allocated pointer. This should be type casted
void *allocateCudaMemory(u64 byte_size);

/// @brief Prefetches unified cuda memory to device
///
/// @param ptr The pointer to cuda allocated memory
/// @param byte_size The number of bytes to prefetch
void prefetchToGpu(const void *ptr, u64 byte_size);

/// @brief Prefetches unified cuda memory to host
///
/// @param ptr The pointer to cuda allocated memory
/// @param byte_size The number of bytes to prefetch
void prefetchToCpu(const void *ptr, u64 byte_size);

/// @brief Copies cuda memory from source into destiation
///
/// @param dst The destination pointer
/// @param src The source pointer
/// @param byte_size The number of bytes to copy
/// @param cudaMemcpyKind the type of copy that should occur
void copyCudaMemory(void *dst, const void *src, u64 byte_size,
                    cudaMemcpyKind kind = cudaMemcpyDefault);

} // namespace lowlevel

/// @brief Allocates a cuda allocated array and wraps it around a unique pointer
///
/// @param T the type of the array
/// @param size The requested size to allocate
template <typename T>
std::unique_ptr<T[], decltype(&cudaFree)> createCudaUniquePtrArray(u64 size)
{
    u64 byte_size{sizeof(T) * size};
    std::unique_ptr<T[], decltype(&cudaFree)> ptr{
        static_cast<T *>(lowlevel::allocateCudaMemory(byte_size)), cudaFree};
    return ptr;
}

/// @brief Prefetches unified cuda memory to device
///
/// @param T the type of the pointer
/// @param ptr The pointer to cuda allocated memory
/// @param size The number of elements to prefetch
template <typename T> void prefetchToGpu(const T *ptr, u64 size)
{
    lowlevel::prefetchToGpu(static_cast<const void *>(ptr), sizeof(T) * size);
}

/// @brief Prefetches unified cuda memory to host
///
/// @param T the type of the pointer
/// @param ptr The pointer to cuda allocated memory
/// @param size The number of elements to prefetch
template <typename T> void prefetchToCpu(const T *ptr, u64 size)
{
    lowlevel::prefetchToCpu(static_cast<const void *>(ptr), sizeof(T) * size);
}

/// @brief Copies cuda memory from source into destiation
///
/// @param T the type of the pointer
/// @param dst The destination pointer
/// @param src The source pointer
/// @param size The number of elements to copy
/// @param cudaMemcpyKind the type of copy that should occur
template <typename T>
void copyCudaMemory(T *dst, const T *src, u64 size,
                    cudaMemcpyKind kind = cudaMemcpyDefault)
{
    lowlevel::copyCudaMemory(static_cast<void *>(dst),
                             static_cast<const void *>(src), sizeof(T) * size,
                             kind);
}

/// @brief Deallocates a cuda allocated pointer. Note that the deallocated
/// pointer will be invalid, but will not be set to null.
///
/// @param ptr The pointer that will be deallocated.
void deallocateCudaMemory(void *ptr);

} // namespace cuda

} // namespace RayTracer

#endif // RAY_TRACER_COMMON_CUDA_MEMORY_UTILS_HPP_
