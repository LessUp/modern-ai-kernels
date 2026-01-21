#pragma once
/**
 * @file features.hpp
 * @brief Compile-time feature detection for C++ and CUDA versions
 * 
 * This header provides macros for detecting available C++ and CUDA features
 * at compile time, enabling conditional compilation of optimized code paths.
 */

#include <cuda_runtime.h>
#include <utility>

// ============================================================================
// C++ Version Detection
// ============================================================================

#if __cplusplus >= 202302L
    #define TC_CPP23 1
    #define TC_CPP20 1
    #define TC_CPP17 1
#elif __cplusplus >= 202002L
    #define TC_CPP20 1
    #define TC_CPP17 1
#elif __cplusplus >= 201703L
    #define TC_CPP17 1
#else
    #error "TensorCraft requires C++17 or later"
#endif

// ============================================================================
// CUDA Version Detection (compile-time)
// ============================================================================

#if defined(__CUDACC__)
    // CUDA compiler version
    #define TC_CUDA_VERSION (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10)
    
    #if __CUDACC_VER_MAJOR__ >= 13
        #define TC_CUDA_13 1
        #define TC_CUDA_12 1
        #define TC_CUDA_11 1
    #elif __CUDACC_VER_MAJOR__ >= 12
        #define TC_CUDA_12 1
        #define TC_CUDA_11 1
    #elif __CUDACC_VER_MAJOR__ >= 11
        #define TC_CUDA_11 1
    #else
        #error "TensorCraft requires CUDA 11.0 or later"
    #endif
    
    // Feature availability based on CUDA version
    #if defined(TC_CUDA_12)
        #define TC_HAS_TMA 1      // Tensor Memory Accelerator (Hopper+)
        #define TC_HAS_WGMMA 1    // Warp Group MMA (Hopper+)
        #define TC_HAS_FP8 1      // FP8 data types
    #endif
    
    #if defined(TC_CUDA_11)
        #define TC_HAS_WMMA 1     // Warp Matrix Multiply-Accumulate (Volta+)
        #define TC_HAS_BF16 1     // BFloat16 support
    #endif
#endif

// ============================================================================
// GPU Architecture Detection (device code only)
// ============================================================================

#if defined(__CUDA_ARCH__)
    // Blackwell (SM 100+)
    #if __CUDA_ARCH__ >= 1000
        #define TC_ARCH_BLACKWELL 1
        #define TC_ARCH_HOPPER 1
        #define TC_ARCH_AMPERE 1
        #define TC_ARCH_VOLTA 1
    // Hopper (SM 90)
    #elif __CUDA_ARCH__ >= 900
        #define TC_ARCH_HOPPER 1
        #define TC_ARCH_AMPERE 1
        #define TC_ARCH_VOLTA 1
    // Ada Lovelace (SM 89)
    #elif __CUDA_ARCH__ >= 890
        #define TC_ARCH_ADA 1
        #define TC_ARCH_AMPERE 1
        #define TC_ARCH_VOLTA 1
    // Ampere (SM 80-86)
    #elif __CUDA_ARCH__ >= 800
        #define TC_ARCH_AMPERE 1
        #define TC_ARCH_VOLTA 1
    // Turing (SM 75)
    #elif __CUDA_ARCH__ >= 750
        #define TC_ARCH_TURING 1
        #define TC_ARCH_VOLTA 1
    // Volta (SM 70)
    #elif __CUDA_ARCH__ >= 700
        #define TC_ARCH_VOLTA 1
    #endif
    
    // Tensor Core availability
    #if __CUDA_ARCH__ >= 700
        #define TC_HAS_TENSOR_CORE 1
    #endif
    
    // Warp size
    #define TC_WARP_SIZE 32
    
    // Maximum threads per block
    #define TC_MAX_THREADS_PER_BLOCK 1024
    
    // Shared memory size (48KB default, can be configured up to 164KB on Hopper)
    #if defined(TC_ARCH_HOPPER)
        #define TC_MAX_SHARED_MEMORY (164 * 1024)
    #elif defined(TC_ARCH_AMPERE)
        #define TC_MAX_SHARED_MEMORY (164 * 1024)
    #else
        #define TC_MAX_SHARED_MEMORY (48 * 1024)
    #endif
#endif

// ============================================================================
// Utility Macros
// ============================================================================

// Force inline for device functions
#if defined(__CUDACC__)
    #define TC_DEVICE __device__
    #define TC_HOST __host__
    #define TC_HOST_DEVICE __host__ __device__
    #define TC_GLOBAL __global__
    #define TC_FORCEINLINE __forceinline__
    #define TC_DEVICE_INLINE __device__ __forceinline__
    #define TC_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
    #define TC_DEVICE
    #define TC_HOST
    #define TC_HOST_DEVICE
    #define TC_GLOBAL
    #define TC_FORCEINLINE inline
    #define TC_DEVICE_INLINE inline
    #define TC_HOST_DEVICE_INLINE inline
#endif

// Restrict pointer hint
#define TC_RESTRICT __restrict__

// Likely/unlikely branch hints
#if defined(TC_CPP20)
    #define TC_LIKELY [[likely]]
    #define TC_UNLIKELY [[unlikely]]
#else
    #define TC_LIKELY
    #define TC_UNLIKELY
#endif

// Nodiscard attribute
#if defined(TC_CPP17)
    #define TC_NODISCARD [[nodiscard]]
#else
    #define TC_NODISCARD
#endif

namespace tensorcraft {

/**
 * @brief Runtime CUDA version check
 */
inline int get_cuda_runtime_version() {
    int version = 0;
    cudaRuntimeGetVersion(&version);
    return version;
}

/**
 * @brief Runtime CUDA driver version check
 */
inline int get_cuda_driver_version() {
    int version = 0;
    cudaDriverGetVersion(&version);
    return version;
}

/**
 * @brief Get compute capability of current device
 */
inline std::pair<int, int> get_compute_capability(int device = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return {prop.major, prop.minor};
}

/**
 * @brief Check if Tensor Cores are available on current device
 */
inline bool has_tensor_cores(int device = 0) {
    auto [major, minor] = get_compute_capability(device);
    return major >= 7;  // Volta and later
}

/**
 * @brief Check if TMA is available on current device
 */
inline bool has_tma(int device = 0) {
    auto [major, minor] = get_compute_capability(device);
    return major >= 9;  // Hopper and later
}

} // namespace tensorcraft
