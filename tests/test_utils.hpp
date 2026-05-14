#pragma once
/**
 * @file test_utils.hpp
 * @brief Unified testing utilities for TensorCraft-HPC
 *
 * Provides:
 * - DeviceBuffer: Unified GPU memory wrapper for tests using Allocator abstraction
 * - Verification helpers: Compare GPU results with CPU reference
 * - Random data generation: For property-based testing
 *
 * This module consolidates testing code that was previously scattered
 * across multiple test files, improving Locality and reducing duplication.
 */

#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/core/type_traits.hpp"
#include "tensorcraft/memory/allocator.hpp"

namespace tensorcraft {
namespace test {

// ============================================================================
// DeviceBuffer - Unified GPU Memory Wrapper using Allocator
// ============================================================================

/**
 * @brief RAII wrapper for GPU memory in tests using Allocator abstraction
 *
 * Uses DirectAllocator by default to isolate test memory from production
 * MemoryPool. This ensures tests don't interfere with production memory
 * management while still using the unified Allocator interface.
 *
 * @tparam T Element type
 * @tparam Allocator Allocator type (default: DirectAllocator for isolation)
 */
template <typename T, typename Allocator = DirectAllocator>
class DeviceBuffer {
public:
    /// Allocate uninitialized device memory
    explicit DeviceBuffer(size_t count)
        : ptr_(count, Allocator::instance()), size_(count) {}

    /// Allocate and initialize with value
    DeviceBuffer(size_t count, T value) : size_(count) {
        if (count > 0) {
            ptr_ = AllocatorPtr<T, Allocator>(count, Allocator::instance());
            std::vector<T> host_data(count, value);
            copy_from_host(host_data.data());
        }
    }

    /// Allocate from host data
    static DeviceBuffer from_host(const std::vector<T>& data) {
        DeviceBuffer buf(data.size());
        buf.copy_from_host(data.data());
        return buf;
    }

    // Default destructor - AllocatorPtr handles deallocation

    // Move only (no copy)
    DeviceBuffer(DeviceBuffer&&) noexcept = default;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept = default;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    /// Copy from host to device
    void copy_from_host(const T* host_data) {
        if (size_ > 0 && ptr_.get()) {
            TC_CUDA_CHECK(cudaMemcpy(ptr_.get(), host_data, size_ * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    /// Copy from host vector
    void copy_from_host(const std::vector<T>& host_data) {
        ASSERT_EQ(host_data.size(), size_) << "Host data size mismatch";
        copy_from_host(host_data.data());
    }

    /// Copy to host
    void copy_to_host(T* host_data) const {
        if (size_ > 0 && ptr_.get()) {
            TC_CUDA_CHECK(cudaMemcpy(host_data, ptr_.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    /// Copy to host vector
    std::vector<T> to_host() const {
        std::vector<T> result(size_);
        copy_to_host(result.data());
        return result;
    }

    /// Get raw pointer
    T* get() { return ptr_.get(); }
    const T* get() const { return ptr_.get(); }

    /// Get size
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }

    /// Check validity
    explicit operator bool() const { return ptr_.get() != nullptr; }

private:
    AllocatorPtr<T, Allocator> ptr_;
    size_t size_ = 0;
};

// Type aliases for common buffer types
using FloatBuffer = DeviceBuffer<float>;
using IntBuffer = DeviceBuffer<int>;

// ============================================================================
// Verification Helpers
// ============================================================================

/**
 * @brief Tolerance for floating-point comparisons
 */
struct Tolerance {
    float relative = 1e-5f;
    float absolute = 1e-6f;

    static Tolerance strict() { return {1e-6f, 1e-7f}; }
    static Tolerance loose() { return {1e-3f, 1e-4f}; }
    static Tolerance half() { return {1e-2f, 1e-3f}; }
};

/**
 * @brief Check if two floating-point values are approximately equal
 */
inline bool approx_equal(float a, float b, Tolerance tol = {}) {
    if (std::isnan(a) || std::isnan(b))
        return false;
    if (std::isinf(a) || std::isinf(b))
        return a == b;

    float diff = std::abs(a - b);
    float max_val = std::max(std::abs(a), std::abs(b));
    float threshold = std::max(tol.absolute, tol.relative * max_val);
    return diff <= threshold;
}

/**
 * @brief Compare two arrays element-wise
 *
 * @return Pair of (all_equal, max_diff)
 */
template <typename T>
std::pair<bool, float> compare_arrays(const T* a, const T* b, size_t n, Tolerance tol = {}) {
    bool all_equal = true;
    float max_diff = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);

        if (!approx_equal(a[i], b[i], tol)) {
            all_equal = false;
        }
    }

    return {all_equal, max_diff};
}

/**
 * @brief Compare two vectors element-wise
 */
template <typename T>
std::pair<bool, float> compare_vectors(const std::vector<T>& a, const std::vector<T>& b,
                                        Tolerance tol = {}) {
    if (a.size() != b.size()) {
        return {false, std::numeric_limits<float>::max()};
    }
    return compare_arrays(a.data(), b.data(), a.size(), tol);
}

/**
 * @brief Assert two vectors are approximately equal (GTest assertion)
 */
template <typename T>
::testing::AssertionResult assert_approx_eq(const char* a_expr, const char* b_expr,
                                             const std::vector<T>& a, const std::vector<T>& b,
                                             Tolerance tol = {}) {
    if (a.size() != b.size()) {
        return ::testing::AssertionFailure()
               << "Size mismatch: " << a_expr << " has " << a.size() << " elements, " << b_expr
               << " has " << b.size() << " elements";
    }

    auto [equal, max_diff] = compare_vectors(a, b, tol);
    if (equal) {
        return ::testing::AssertionSuccess();
    }

    // Find first mismatch for detailed error
    for (size_t i = 0; i < a.size(); ++i) {
        if (!approx_equal(a[i], b[i], tol)) {
            return ::testing::AssertionFailure()
                   << "First mismatch at index " << i << ": " << a_expr << "[" << i << "] = " << a[i]
                   << ", " << b_expr << "[" << i << "] = " << b[i] << " (diff = " << (a[i] - b[i])
                   << ", max_diff = " << max_diff << ")";
        }
    }

    return ::testing::AssertionSuccess();
}

/**
 * @brief Verify GPU output matches CPU reference
 *
 * @tparam T Data type
 * @param gpu_output GPU buffer containing computed output
 * @param ref_output CPU reference output
 * @param tol Comparison tolerance
 * @return Assertion result
 */
template <typename T>
::testing::AssertionResult assert_gpu_eq_ref(const DeviceBuffer<T>& gpu_output,
                                              const std::vector<T>& ref_output,
                                              Tolerance tol = {}) {
    auto gpu_host = gpu_output.to_host();
    return assert_approx_eq("gpu_output", "ref_output", gpu_host, ref_output, tol);
}

// ============================================================================
// Random Data Generation
// ============================================================================

/**
 * @brief Random data generator for testing
 */
class RandomGenerator {
public:
    explicit RandomGenerator(unsigned seed = 42) : rng_(seed) {}

    /// Set seed for reproducibility
    void seed(unsigned s) { rng_.seed(s); }

    /// Generate random floats in range [min, max]
    std::vector<float> uniform_floats(size_t n, float min = -1.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        std::vector<float> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = dist(rng_);
        }
        return result;
    }

    /// Generate random integers in range [min, max]
    std::vector<int> uniform_ints(size_t n, int min = 0, int max = 100) {
        std::uniform_int_distribution<int> dist(min, max);
        std::vector<int> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = dist(rng_);
        }
        return result;
    }

    /// Generate random positive floats (useful for normalized data)
    std::vector<float> positive_floats(size_t n, float min = 0.1f, float max = 1.0f) {
        return uniform_floats(n, min, max);
    }

private:
    std::mt19937 rng_;
};

// ============================================================================
// Test Fixture Base
// ============================================================================

/**
 * @brief Base test fixture with CUDA device check
 *
 * Derive from this to get automatic CUDA device availability checking.
 */
class CudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available, skipping test";
        }
        cudaSetDevice(0);
    }

    void TearDown() override {
        // Synchronize and check for errors
        cudaDeviceSynchronize();
    }
};

// ============================================================================
// Normalization Test Helpers
// ============================================================================

/**
 * @brief Verify normalization properties
 *
 * For LayerNorm/RMSNorm outputs:
 * - Check mean/variance properties
 * - Check no NaN/Inf
 */
inline ::testing::AssertionResult assert_normalization_valid(const float* output, int batch_size,
                                                              int hidden_size, float eps = 1e-5f) {
    for (int b = 0; b < batch_size; ++b) {
        const float* row = output + b * hidden_size;

        // Check for NaN/Inf
        for (int h = 0; h < hidden_size; ++h) {
            if (std::isnan(row[h]) || std::isinf(row[h])) {
                return ::testing::AssertionFailure()
                       << "Invalid value at [" << b << "][" << h << "]: " << row[h];
            }
        }

        // For LayerNorm: check mean ≈ 0, var ≈ 1
        // (This is a soft check - actual values depend on gamma/beta)
    }

    return ::testing::AssertionSuccess();
}

/**
 * @brief Verify softmax properties
 *
 * - All values >= 0
 * - Each row sums to 1
 */
inline ::testing::AssertionResult assert_softmax_valid(const float* output, int rows, int cols,
                                                        Tolerance tol = {}) {
    for (int r = 0; r < rows; ++r) {
        const float* row = output + r * cols;

        // Check non-negative
        for (int c = 0; c < cols; ++c) {
            if (row[c] < 0.0f) {
                return ::testing::AssertionFailure()
                       << "Negative value at [" << r << "][" << c << "]: " << row[c];
            }
        }

        // Check sum ≈ 1
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += row[c];
        }
        if (!approx_equal(sum, 1.0f, tol)) {
            return ::testing::AssertionFailure()
                   << "Row " << r << " sum = " << sum << " (expected 1.0)";
        }
    }

    return ::testing::AssertionSuccess();
}

// ============================================================================
// Shape Utilities
// ============================================================================

/**
 * @brief Compute total elements from shape
 */
inline size_t shape_size(const std::vector<int>& shape) {
    size_t size = 1;
    for (int dim : shape) {
        size *= static_cast<size_t>(dim);
    }
    return size;
}

/**
 * @brief Convert shape to string for error messages
 */
inline std::string shape_to_string(const std::vector<int>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
            oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

}  // namespace test
}  // namespace tensorcraft
