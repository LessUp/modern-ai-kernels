#pragma once
/**
 * @file cuda_check.hpp
 * @brief CUDA error checking utilities
 * 
 * Provides macros for checking CUDA API calls and reporting errors
 * with file, line, and error description.
 */

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace tensorcraft {

/**
 * @brief Exception class for CUDA errors
 */
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& file, int line, cudaError_t error)
        : std::runtime_error(format_message(file, line, error))
        , error_(error)
        , file_(file)
        , line_(line) {}
    
    cudaError_t error() const noexcept { return error_; }
    const std::string& file() const noexcept { return file_; }
    int line() const noexcept { return line_; }

private:
    static std::string format_message(const std::string& file, int line, cudaError_t error) {
        std::ostringstream oss;
        oss << file << ":" << line << " CUDA error: " 
            << cudaGetErrorString(error) << " (" << static_cast<int>(error) << ")";
        return oss.str();
    }
    
    cudaError_t error_;
    std::string file_;
    int line_;
};

/**
 * @brief Check CUDA API call and throw exception on error
 */
inline void cuda_check(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw CudaException(file, line, error);
    }
}

/**
 * @brief Macro to check CUDA API calls
 * 
 * Usage:
 *   TC_CUDA_CHECK(cudaMalloc(&ptr, size));
 */
#define TC_CUDA_CHECK(call)                                     \
    do {                                                        \
        cudaError_t err = (call);                               \
        if (err != cudaSuccess) {                               \
            throw ::tensorcraft::CudaException(__FILE__, __LINE__, err); \
        }                                                       \
    } while (0)

/**
 * @brief Macro to check last CUDA error (for kernel launches)
 * 
 * Usage:
 *   my_kernel<<<grid, block>>>(...);
 *   TC_CUDA_CHECK_LAST();
 */
#define TC_CUDA_CHECK_LAST()                                    \
    do {                                                        \
        cudaError_t err = cudaGetLastError();                   \
        if (err != cudaSuccess) {                               \
            throw ::tensorcraft::CudaException(__FILE__, __LINE__, err); \
        }                                                       \
    } while (0)

/**
 * @brief Macro to synchronize and check for errors
 */
#define TC_CUDA_SYNC_CHECK()                                    \
    do {                                                        \
        TC_CUDA_CHECK(cudaDeviceSynchronize());                 \
        TC_CUDA_CHECK_LAST();                                   \
    } while (0)

} // namespace tensorcraft
