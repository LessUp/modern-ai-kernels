#pragma once
/**
 * @file warp_utils.hpp
 * @brief Warp-level reduction and shuffle utilities
 * 
 * Provides efficient warp-level primitives using CUDA shuffle intrinsics
 * for use across all kernel implementations.
 */

#include "features.hpp"

namespace tensorcraft {

// ============================================================================
// Warp-Level Reductions
// ============================================================================

/**
 * @brief Warp-level max reduction using shuffle
 * 
 * Reduces across all 32 lanes of a warp to find the maximum value.
 * Result is valid only in lane 0 after the reduction.
 */
template<typename T>
TC_DEVICE_INLINE T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

/**
 * @brief Warp-level sum reduction using shuffle
 * 
 * Reduces across all 32 lanes of a warp to compute the sum.
 * Result is valid only in lane 0 after the reduction.
 */
template<typename T>
TC_DEVICE_INLINE T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Warp-level min reduction using shuffle
 */
template<typename T>
TC_DEVICE_INLINE T warp_reduce_min(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = val < other ? val : other;
    }
    return val;
}

/**
 * @brief Broadcast value from lane 0 to all lanes in a warp
 */
template<typename T>
TC_DEVICE_INLINE T warp_broadcast(T val, int src_lane = 0) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

// ============================================================================
// Block-Level Reduction Helpers
// ============================================================================

/**
 * @brief Block-level sum reduction using warp shuffles + shared memory
 * 
 * @tparam T Value type
 * @tparam BLOCK_SIZE Number of threads per block
 * @param val Per-thread value
 * @param shared Shared memory array of size [BLOCK_SIZE / 32]
 * @return Sum across all threads (valid in thread 0)
 */
template<typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_sum(T val, T* shared) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = BLOCK_SIZE / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : T(0);
        val = warp_reduce_sum(val);
    }
    return val;
}

/**
 * @brief Block-level max reduction using warp shuffles + shared memory
 */
template<typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_max(T val, T* shared) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = BLOCK_SIZE / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : T(-1e30);
        val = warp_reduce_max(val);
    }
    return val;
}

} // namespace tensorcraft
