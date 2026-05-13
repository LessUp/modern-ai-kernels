#pragma once
/**
 * @file warp_utils.hpp
 * @brief Warp-level reduction and shuffle utilities
 *
 * Provides efficient warp-level primitives using CUDA shuffle intrinsics
 * for use across all kernel implementations.
 *
 * Features:
 * - Warp-level reductions (max, sum, min)
 * - Warp-level predicates (all, any, ballot)
 * - Warp-level scans (inclusive/exclusive prefix sum)
 * - Block-level reductions
 */

#include <limits>

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
template <typename T>
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
template <typename T>
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
template <typename T>
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
template <typename T>
TC_DEVICE_INLINE T warp_broadcast(T val, int src_lane = 0) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

// ============================================================================
// Warp-Level Predicates
// ============================================================================

/**
 * @brief Check if predicate is true for all lanes in a warp
 * @param predicate Condition to check
 * @return true if all lanes have predicate == true
 */
TC_DEVICE_INLINE bool warp_all(bool predicate) {
    return __all_sync(0xffffffff, predicate);
}

/**
 * @brief Check if predicate is true for any lane in a warp
 * @param predicate Condition to check
 * @return true if any lane has predicate == true
 */
TC_DEVICE_INLINE bool warp_any(bool predicate) {
    return __any_sync(0xffffffff, predicate);
}

/**
 * @brief Collect predicate bits from all lanes in a warp
 * @param predicate Condition to collect
 * @return 32-bit mask where bit i indicates predicate result from lane i
 */
TC_DEVICE_INLINE unsigned warp_ballot(bool predicate) {
    return __ballot_sync(0xffffffff, predicate);
}

// ============================================================================
// Warp-Level Scan (Prefix Sum)
// ============================================================================

/**
 * @brief Inclusive scan (prefix sum) across a warp
 *
 * Computes inclusive prefix sum where each lane contains the sum
 * of all values from lane 0 to itself.
 *
 * Example: input [1, 2, 3, 4, ...] -> output [1, 3, 6, 10, ...]
 *
 * @param val Per-lane value
 * @return Inclusive prefix sum (valid in all lanes)
 */
template <typename T>
TC_DEVICE_INLINE T warp_scan_sum_inclusive(T val) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        T other = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += other;
        }
    }
    return val;
}

/**
 * @brief Exclusive scan (prefix sum) across a warp
 *
 * Computes exclusive prefix sum where each lane contains the sum
 * of all values from lane 0 to the previous lane.
 *
 * Example: input [1, 2, 3, 4, ...] -> output [0, 1, 3, 6, ...]
 *
 * @param val Per-lane value
 * @return Exclusive prefix sum (valid in all lanes, lane 0 returns 0)
 */
template <typename T>
TC_DEVICE_INLINE T warp_scan_sum_exclusive(T val) {
    T inclusive = warp_scan_sum_inclusive(val);
    T exclusive = __shfl_up_sync(0xffffffff, inclusive, 1);
    return (threadIdx.x % 32 == 0) ? T(0) : exclusive;
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
template <typename T, int BLOCK_SIZE>
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
template <typename T, int BLOCK_SIZE>
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
        // Use lowest representable value for proper max reduction semantics
        // This handles all negative value domains correctly
        val = (lane < num_warps) ? shared[lane] : std::numeric_limits<T>::lowest();
        val = warp_reduce_max(val);
    }
    return val;
}

/**
 * @brief Block-level min reduction using warp shuffles + shared memory
 *
 * @tparam T Value type
 * @tparam BLOCK_SIZE Number of threads per block
 * @param val Per-thread value
 * @param shared Shared memory array of size [BLOCK_SIZE / 32]
 * @return Min across all threads (valid in thread 0)
 */
template <typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_min(T val, T* shared) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = BLOCK_SIZE / 32;

    val = warp_reduce_min(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        // Use highest representable value for proper min reduction semantics
        val = (lane < num_warps) ? shared[lane] : std::numeric_limits<T>::max();
        val = warp_reduce_min(val);
    }
    return val;
}

// ============================================================================
// Block-Level Convenience Interfaces
// ============================================================================

/**
 * @brief Block-level sum reduction with automatic shared memory
 *
 * Convenience wrapper that manages shared memory internally and
 * broadcasts result to all threads.
 *
 * @tparam BLOCK_SIZE Number of threads per block (must be power of 2)
 * @param val Per-thread value
 * @return Sum across all threads (broadcast to all threads)
 */
template <int BLOCK_SIZE>
TC_DEVICE_INLINE float block_sum(float val) {
    __shared__ float shared[BLOCK_SIZE / 32];

    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = BLOCK_SIZE / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    // Broadcast result to all threads
    __shared__ float result;
    if (threadIdx.x == 0) {
        result = val;
    }
    __syncthreads();

    return result;
}

/**
 * @brief Block-level max reduction with automatic shared memory
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param val Per-thread value
 * @return Max across all threads (broadcast to all threads)
 */
template <int BLOCK_SIZE>
TC_DEVICE_INLINE float block_max(float val) {
    __shared__ float shared[BLOCK_SIZE / 32];

    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = BLOCK_SIZE / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : std::numeric_limits<float>::lowest();
        val = warp_reduce_max(val);
    }

    __shared__ float result;
    if (threadIdx.x == 0) {
        result = val;
    }
    __syncthreads();

    return result;
}

/**
 * @brief Block-level min reduction with automatic shared memory
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param val Per-thread value
 * @return Min across all threads (broadcast to all threads)
 */
template <int BLOCK_SIZE>
TC_DEVICE_INLINE float block_min(float val) {
    __shared__ float shared[BLOCK_SIZE / 32];

    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = BLOCK_SIZE / 32;

    val = warp_reduce_min(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : std::numeric_limits<float>::max();
        val = warp_reduce_min(val);
    }

    __shared__ float result;
    if (threadIdx.x == 0) {
        result = val;
    }
    __syncthreads();

    return result;
}

/**
 * @brief Block-level mean with automatic shared memory
 *
 * Computes sum and divides by divisor. Useful for normalization kernels.
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param val Per-thread value
 * @param divisor Value to divide sum by
 * @return Mean (sum / divisor) broadcast to all threads
 */
template <int BLOCK_SIZE>
TC_DEVICE_INLINE float block_mean(float val, int divisor) {
    __shared__ float shared[BLOCK_SIZE / 32];

    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = BLOCK_SIZE / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    __shared__ float result;
    if (threadIdx.x == 0) {
        result = val / static_cast<float>(divisor);
    }
    __syncthreads();

    return result;
}

// ============================================================================
// Half Precision Optimizations
// ============================================================================

#if defined(__CUDACC__)
#include <cuda_fp16.h>

/**
 * @brief Half2 vectorized warp sum reduction
 *
 * Processes two half values at once for improved throughput.
 * Result is valid only in lane 0.
 *
 * @param val Half2 value (contains 2 half elements)
 * @return Sum across all lanes (valid in lane 0)
 */
TC_DEVICE_INLINE __half2 warp_reduce_sum_half2(__half2 val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        __half2 other = __shfl_down_sync(0xffffffff, val, offset);
        val = __hadd2(val, other);
    }
    return val;
}

/**
 * @brief Mixed-precision half warp sum reduction
 *
 * Converts half to float for accumulation to maintain precision,
 * then converts back to half. Use when precision is critical.
 *
 * @param val Half value
 * @return Sum across all lanes (valid in lane 0)
 */
TC_DEVICE_INLINE __half warp_reduce_sum_mixed(__half val) {
    // Convert to float for high-precision accumulation
    float sum = __half2float(val);
    sum = warp_reduce_sum(sum);
    return __float2half(sum);
}

/**
 * @brief Block-level half2 vectorized sum reduction
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param val Half2 value per thread
 * @param shared Shared memory array of size [BLOCK_SIZE / 32]
 * @return Half2 sum across all threads (valid in thread 0)
 */
template <int BLOCK_SIZE>
TC_DEVICE_INLINE __half2 block_reduce_sum_half2(__half2 val, __half2* shared) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    constexpr int num_warps = BLOCK_SIZE / 32;

    val = warp_reduce_sum_half2(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : __half2zero;
        val = warp_reduce_sum_half2(val);
    }
    return val;
}

#endif  // __CUDACC__

}  // namespace tensorcraft
