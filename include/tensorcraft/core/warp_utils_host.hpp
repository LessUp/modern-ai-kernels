#pragma once
/**
 * @file warp_utils_host.hpp
 * @brief CPU host-side simulation of warp primitives with unified interface
 *
 * Provides CPU implementations of warp-level primitives for:
 * - Offline testing without GPU
 * - Fuzzing and property-based testing
 * - Debugging warp-level algorithms
 *
 * This creates a Seam between warp algorithm logic and CUDA hardware,
 * enabling the same code to run on CPU for verification.
 *
 * ## Unified Interface (WarpOps)
 *
 * Use `WarpOps<HostImpl>` for CPU testing:
 * ```cpp
 * WarpOps<HostImpl>::reduce_max(ctx, value);  // CPU simulation
 * WarpOps<GpuImpl>::reduce_max(value);        // GPU (in warp_utils.hpp)
 * ```
 */

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>

namespace tensorcraft {
namespace host {

// ============================================================================
// Simulated Warp Size
// ============================================================================

constexpr int WARP_SIZE = 32;

// ============================================================================
// Implementation Tags
// ============================================================================

/// Tag for CPU host implementation
struct HostImpl {};

/// Tag for GPU device implementation (defined in warp_utils.hpp)
struct GpuImpl {};

// ============================================================================
// Warp-Level Reduction Simulations
// ============================================================================

/**
 * @brief Simulate warp max reduction
 *
 * @param values Array of WARP_SIZE values representing warp lanes
 * @return Maximum value (same for all "lanes")
 */
template <typename T>
T warp_reduce_max_sim(const std::array<T, WARP_SIZE>& values) {
    T result = values[0];
    for (int i = 1; i < WARP_SIZE; ++i) {
        result = std::max(result, values[i]);
    }
    return result;
}

/**
 * @brief Simulate warp sum reduction
 */
template <typename T>
T warp_reduce_sum_sim(const std::array<T, WARP_SIZE>& values) {
    T result = T(0);
    for (int i = 0; i < WARP_SIZE; ++i) {
        result += values[i];
    }
    return result;
}

/**
 * @brief Simulate warp min reduction
 */
template <typename T>
T warp_reduce_min_sim(const std::array<T, WARP_SIZE>& values) {
    T result = values[0];
    for (int i = 1; i < WARP_SIZE; ++i) {
        result = std::min(result, values[i]);
    }
    return result;
}

// ============================================================================
// Warp-Level Predicate Simulations
// ============================================================================

/**
 * @brief Simulate __all_sync predicate
 */
inline bool warp_all_sim(const std::array<bool, WARP_SIZE>& predicates) {
    for (bool p : predicates) {
        if (!p)
            return false;
    }
    return true;
}

/**
 * @brief Simulate __any_sync predicate
 */
inline bool warp_any_sim(const std::array<bool, WARP_SIZE>& predicates) {
    for (bool p : predicates) {
        if (p)
            return true;
    }
    return false;
}

/**
 * @brief Simulate __ballot_sync predicate
 */
inline unsigned warp_ballot_sim(const std::array<bool, WARP_SIZE>& predicates) {
    unsigned mask = 0;
    for (int i = 0; i < WARP_SIZE; ++i) {
        if (predicates[i]) {
            mask |= (1u << i);
        }
    }
    return mask;
}

// ============================================================================
// Warp-Level Scan Simulations
// ============================================================================

/**
 * @brief Simulate inclusive prefix sum
 */
template <typename T>
std::array<T, WARP_SIZE> warp_scan_sum_inclusive_sim(const std::array<T, WARP_SIZE>& values) {
    std::array<T, WARP_SIZE> result;
    result[0] = values[0];
    for (int i = 1; i < WARP_SIZE; ++i) {
        result[i] = result[i - 1] + values[i];
    }
    return result;
}

/**
 * @brief Simulate exclusive prefix sum
 */
template <typename T>
std::array<T, WARP_SIZE> warp_scan_sum_exclusive_sim(const std::array<T, WARP_SIZE>& values) {
    std::array<T, WARP_SIZE> result;
    result[0] = T(0);
    for (int i = 1; i < WARP_SIZE; ++i) {
        result[i] = result[i - 1] + values[i - 1];
    }
    return result;
}

// ============================================================================
// Block-Level Reduction Simulations
// ============================================================================

/**
 * @brief Simulate block-level sum reduction
 *
 * @tparam BLOCK_SIZE Number of threads per block
 * @param values Array of BLOCK_SIZE values
 * @return Sum of all values
 */
template <int BLOCK_SIZE, typename T>
T block_reduce_sum_sim(const std::array<T, BLOCK_SIZE>& values) {
    T result = T(0);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        result += values[i];
    }
    return result;
}

/**
 * @brief Simulate block-level max reduction
 */
template <int BLOCK_SIZE, typename T>
T block_reduce_max_sim(const std::array<T, BLOCK_SIZE>& values) {
    T result = values[0];
    for (int i = 1; i < BLOCK_SIZE; ++i) {
        result = std::max(result, values[i]);
    }
    return result;
}

/**
 * @brief Simulate block-level min reduction
 */
template <int BLOCK_SIZE, typename T>
T block_reduce_min_sim(const std::array<T, BLOCK_SIZE>& values) {
    T result = values[0];
    for (int i = 1; i < BLOCK_SIZE; ++i) {
        result = std::min(result, values[i]);
    }
    return result;
}

// ============================================================================
// WarpContext - For Testing Warp-Level Algorithms
// ============================================================================

/**
 * @brief Simulated warp context for testing algorithms
 *
 * Provides a test-friendly interface for verifying warp-level algorithms:
 * - Each "thread" can be simulated independently
 * - Results can be verified against expected outcomes
 *
 * Example usage:
 * ```cpp
 * WarpContext<32> ctx;
 * ctx.set_lane_values({1, 2, 3, ...});
 * float max_val = ctx.reduce_max<float>();
 * EXPECT_EQ(max_val, 32.0f);
 * ```
 */
template <int BLOCK_SIZE = 256>
class WarpContext {
public:
    static constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    /// Set value for a specific thread
    void set_thread_value(int tid, float val) {
        thread_values_[tid] = val;
    }

    /// Set values for all threads in a warp
    void set_warp_values(int warp_id, const std::array<float, WARP_SIZE>& values) {
        for (int i = 0; i < WARP_SIZE; ++i) {
            thread_values_[warp_id * WARP_SIZE + i] = values[i];
        }
    }

    /// Get warp sum reduction result
    float warp_sum(int warp_id) const {
        std::array<float, WARP_SIZE> values;
        for (int i = 0; i < WARP_SIZE; ++i) {
            values[i] = thread_values_[warp_id * WARP_SIZE + i];
        }
        return warp_reduce_sum_sim(values);
    }

    /// Get warp max reduction result
    float warp_max(int warp_id) const {
        std::array<float, WARP_SIZE> values;
        for (int i = 0; i < WARP_SIZE; ++i) {
            values[i] = thread_values_[warp_id * WARP_SIZE + i];
        }
        return warp_reduce_max_sim(values);
    }

    /// Get block sum reduction result
    float block_sum() const {
        return block_reduce_sum_sim<BLOCK_SIZE>(thread_values_);
    }

    /// Get block max reduction result
    float block_max() const {
        return block_reduce_max_sim<BLOCK_SIZE>(thread_values_);
    }

    /// Get raw thread values (for direct inspection)
    const std::array<float, BLOCK_SIZE>& values() const { return thread_values_; }

private:
    std::array<float, BLOCK_SIZE> thread_values_{};
};

// ============================================================================
// Property-Based Testing Helpers
// ============================================================================

/**
 * @brief Generate random test values for property-based testing
 *
 * @tparam T Value type
 * @tparam N Array size
 * @param gen Random number generator
 * @param min Minimum value
 * @param max Maximum value
 * @return Array of random values
 */
template <typename T, int N, typename RNG>
std::array<T, N> generate_random_values(RNG& gen, T min, T max) {
    std::array<T, N> result;
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(min, max);
        for (int i = 0; i < N; ++i) {
            result[i] = dist(gen);
        }
    } else {
        std::uniform_real_distribution<T> dist(min, max);
        for (int i = 0; i < N; ++i) {
            result[i] = dist(gen);
        }
    }
    return result;
}

// ============================================================================
// Unified WarpOps Interface
// ============================================================================

/**
 * @brief Unified warp operations interface
 *
 * Template-specialized for HostImpl (CPU) and GpuImpl (GPU).
 * This allows algorithms to be written generically and tested on CPU.
 *
 * @tparam Impl Implementation tag (HostImpl or GpuImpl)
 */
template <typename Impl>
struct WarpOps;

/// CPU host specialization
template <>
struct WarpOps<HostImpl> {
    static constexpr int WARP_SIZE = 32;

    /// Context for CPU simulation (holds per-warp state)
    template <int BLOCK_SIZE = 256>
    using Context = WarpContext<BLOCK_SIZE>;

    /// Warp max reduction (takes warp context and lane index)
    template <int BLOCK_SIZE = 256>
    static float reduce_max(WarpContext<BLOCK_SIZE>& ctx, int warp_id) {
        return ctx.warp_max(warp_id);
    }

    /// Warp sum reduction
    template <int BLOCK_SIZE = 256>
    static float reduce_sum(WarpContext<BLOCK_SIZE>& ctx, int warp_id) {
        return ctx.warp_sum(warp_id);
    }

    /// Block max reduction
    template <int BLOCK_SIZE = 256>
    static float block_reduce_max(WarpContext<BLOCK_SIZE>& ctx) {
        return ctx.block_max();
    }

    /// Block sum reduction
    template <int BLOCK_SIZE = 256>
    static float block_reduce_sum(WarpContext<BLOCK_SIZE>& ctx) {
        return ctx.block_sum();
    }
};

/// Type alias for convenient use
using HostWarpOps = WarpOps<HostImpl>;

}  // namespace host
}  // namespace tensorcraft
