#include "cuda.h"
#include "cuda_runtime.h"

#include <iostream>
#include <string>

namespace RayTracer
{
namespace cuda
{

#define CHECK_CUDA_ERRORS(result) \
    { \
        checkCudaErrors(result, __FILE__, __LINE__); \
    }

/// @brief Checks CUDA errors and terminates upon an error. The wrapper macro
/// CHECK_CUDA_ERRORS(cudaError_t
// result) should be used instead.
///
/// @param result The result of a cuda call
/// @param filename The filename that the error occurred in
/// @param line_number The line number that the error occurred in
inline void checkCudaErrors(cudaError_t result, const char *filename,
                            int line_number)
{
    if (result != cudaSuccess) {
        std::cerr << "Exiting due to CUDA error: "
                  << std::string(cudaGetErrorString(result))
                  << " (error code: " << result << ") at "
                  << std::string(filename) << " in line number " << line_number;
        std::exit(1);
    }
}

} // namespace cuda

} // namespace RayTracer
