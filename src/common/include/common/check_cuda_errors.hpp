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
