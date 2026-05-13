#pragma once
/**
 * @file memory_ops.hpp
 * @brief Memory operation kernels for GPU tensors
 *
 * Provides optimized memory operations like fill, copy, etc.
 * This module belongs to the kernels layer in the three-layer architecture.
 */

#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"
#include "../memory/aligned_vector.hpp"

namespace tensorcraft {
namespace kernels {

namespace detail {

/// GPU kernel to fill memory with a typed value (vectorized version)
template <typename T, int VecSize>
__global__ void fill_kernel_vectorized(T* __restrict__ data, T value, size_t n) {
    using Vec = AlignedVector<T, VecSize>;

    const size_t vec_n = n / VecSize;
    const size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_idx < vec_n) {
        // Vectorized fill
        Vec vec;
        for (int i = 0; i < VecSize; ++i) {
            vec[i] = value;
        }
        *reinterpret_cast<Vec*>(data + vec_idx * VecSize) = vec;
    }

    // Handle remaining elements
    const size_t remaining_start = vec_n * VecSize;
    const size_t remaining_idx = remaining_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (remaining_idx < n) {
        data[remaining_idx] = value;
    }
}

/// GPU kernel to fill memory with a typed value (scalar version for fallback)
template <typename T>
__global__ void fill_kernel_scalar(T* __restrict__ data, T value, size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

}  // namespace detail

/**
 * @brief Fill GPU memory with a typed value
 *
 * @tparam T Element type
 * @param data Pointer to GPU memory
 * @param value Value to fill
 * @param n Number of elements
 * @param stream CUDA stream (optional, default stream if nullptr)
 *
 * Optimization strategies:
 * - int8_t: Uses cudaMemsetAsync (hardware optimized)
 * - float/half etc.: Uses vectorized kernel (2-4x bandwidth improvement)
 * - Non-aligned memory: Falls back to scalar kernel
 */
template <typename T>
void fill(T* data, T value, size_t n, cudaStream_t stream = nullptr) {
    if (n == 0 || !data) {
        return;
    }

    // Special case: int8_t uses cudaMemset (most efficient for single byte)
    if constexpr (sizeof(T) == 1) {
        TC_CUDA_CHECK(cudaMemsetAsync(data, static_cast<int>(value), n, stream));
        return;
    }

    // Check alignment for vectorized access
    const uintptr_t ptr_val = reinterpret_cast<uintptr_t>(data);
    const bool is_aligned = (ptr_val % 16 == 0);  // 16-byte alignment for vectorized loads

    constexpr int block = 256;

    // Use vectorized kernel if aligned and enough elements
    if (is_aligned && n >= 4) {
        // Choose vector size based on type size and alignment
        constexpr int vec_size = (sizeof(T) == 4) ? 4 : (sizeof(T) == 2) ? 8 : 2;
        const size_t vec_n = (n + vec_size - 1) / vec_size;
        const int grid = static_cast<int>((vec_n + block - 1) / block);

        detail::fill_kernel_vectorized<T, vec_size>
            <<<grid, block, 0, stream>>>(data, value, n);
    } else {
        // Fallback to scalar kernel
        const int grid = static_cast<int>((n + block - 1) / block);
        detail::fill_kernel_scalar<<<grid, block, 0, stream>>>(data, value, n);
    }

    TC_CUDA_CHECK_LAST();
}

/**
 * @brief Copy data between device memory locations
 *
 * @tparam T Element type
 * @param dst Destination pointer
 * @param src Source pointer
 * @param n Number of elements
 * @param stream CUDA stream (optional)
 */
template <typename T>
void copy_d2d(T* dst, const T* src, size_t n, cudaStream_t stream = nullptr) {
    if (n == 0 || !dst || !src) {
        return;
    }

    TC_CUDA_CHECK(
        cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice, stream));
}

}  // namespace kernels
}  // namespace tensorcraft
