/**
 * @file test_warp_utils.cpp
 * @brief Tests for warp-level utilities
 */

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/core/warp_utils.hpp"

using namespace tensorcraft;

// ============================================================================
// Host-side validation helpers (for testing device functions)
// ============================================================================

// Note: These tests verify the warp operations work correctly.
// Since warp operations require CUDA device code, we test them
// indirectly through kernel launchers or use small test kernels.

namespace {

// Test kernel for warp_all
__global__ void test_warp_all_kernel(bool* results, int* test_values) {
    int idx = threadIdx.x;
    bool predicate = test_values[idx] > 0;
    results[idx] = warp_all(predicate);
}

// Test kernel for warp_any
__global__ void test_warp_any_kernel(bool* results, int* test_values) {
    int idx = threadIdx.x;
    bool predicate = test_values[idx] > 0;
    results[idx] = warp_any(predicate);
}

// Test kernel for warp_ballot
__global__ void test_warp_ballot_kernel(unsigned* results, int* test_values) {
    int idx = threadIdx.x;
    bool predicate = test_values[idx] > 0;
    unsigned mask = warp_ballot(predicate);
    if (idx == 0) {
        results[0] = mask;
    }
}

// Test kernel for warp_scan_sum_inclusive
__global__ void test_warp_scan_kernel(int* input, int* output) {
    int idx = threadIdx.x;
    int val = input[idx];
    output[idx] = warp_scan_sum_inclusive(val);
}

// Test kernel for block_sum
template <int BLOCK_SIZE>
__global__ void test_block_sum_kernel(float* input, float* output, int n) {
    float val = (threadIdx.x < n) ? input[threadIdx.x] : 0.0f;
    float sum = block_sum<BLOCK_SIZE>(val);
    if (threadIdx.x == 0) {
        output[0] = sum;
    }
}

// Test kernel for block_mean
template <int BLOCK_SIZE>
__global__ void test_block_mean_kernel(float* input, float* output, int n, int divisor) {
    float val = (threadIdx.x < n) ? input[threadIdx.x] : 0.0f;
    float mean = block_mean<BLOCK_SIZE>(val, divisor);
    if (threadIdx.x == 0) {
        output[0] = mean;
    }
}

// Test kernel for block_max
template <int BLOCK_SIZE>
__global__ void test_block_max_kernel(float* input, float* output, int n) {
    float val = (threadIdx.x < n) ? input[threadIdx.x] : std::numeric_limits<float>::lowest();
    float max_val = block_max<BLOCK_SIZE>(val);
    if (threadIdx.x == 0) {
        output[0] = max_val;
    }
}

// Test kernel for block_min
template <int BLOCK_SIZE>
__global__ void test_block_min_kernel(float* input, float* output, int n) {
    float val = (threadIdx.x < n) ? input[threadIdx.x] : std::numeric_limits<float>::max();
    float min_val = block_min<BLOCK_SIZE>(val);
    if (threadIdx.x == 0) {
        output[0] = min_val;
    }
}

}  // namespace

// ============================================================================
// Warp Predicate Tests
// ============================================================================

class WarpUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        TC_CUDA_CHECK(cudaMalloc(&d_results, 32 * sizeof(bool)));
        TC_CUDA_CHECK(cudaMalloc(&d_test_values, 32 * sizeof(int)));
        TC_CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(int)));
        TC_CUDA_CHECK(cudaMalloc(&d_float_output, sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_float_input, 256 * sizeof(float)));
        TC_CUDA_CHECK(cudaMalloc(&d_unsigned_output, sizeof(unsigned)));
    }

    void TearDown() override {
        cudaFree(d_results);
        cudaFree(d_test_values);
        cudaFree(d_output);
        cudaFree(d_float_output);
        cudaFree(d_float_input);
        cudaFree(d_unsigned_output);
    }

    bool* d_results = nullptr;
    int* d_test_values = nullptr;
    int* d_output = nullptr;
    float* d_float_output = nullptr;
    float* d_float_input = nullptr;
    unsigned* d_unsigned_output = nullptr;
};

TEST_F(WarpUtilsTest, WarpAllAllTrue) {
    // All lanes have predicate = true
    std::vector<int> h_test_values(32, 1);  // All positive
    std::vector<bool> h_results(32);

    TC_CUDA_CHECK(cudaMemcpy(d_test_values, h_test_values.data(), 32 * sizeof(int), cudaMemcpyHostToDevice));
    test_warp_all_kernel<<<1, 32>>>(d_results, d_test_values);
    TC_CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 32 * sizeof(bool), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 32; ++i) {
        EXPECT_TRUE(h_results[i]) << "warp_all should return true when all predicates are true";
    }
}

TEST_F(WarpUtilsTest, WarpAllSomeFalse) {
    // Some lanes have predicate = false
    std::vector<int> h_test_values(32, 1);
    h_test_values[15] = -1;  // Make lane 15 predicate false
    std::vector<bool> h_results(32);

    TC_CUDA_CHECK(cudaMemcpy(d_test_values, h_test_values.data(), 32 * sizeof(int), cudaMemcpyHostToDevice));
    test_warp_all_kernel<<<1, 32>>>(d_results, d_test_values);
    TC_CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 32 * sizeof(bool), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 32; ++i) {
        EXPECT_FALSE(h_results[i]) << "warp_all should return false when any predicate is false";
    }
}

TEST_F(WarpUtilsTest, WarpAnyAllTrue) {
    std::vector<int> h_test_values(32, 1);
    std::vector<bool> h_results(32);

    TC_CUDA_CHECK(cudaMemcpy(d_test_values, h_test_values.data(), 32 * sizeof(int), cudaMemcpyHostToDevice));
    test_warp_any_kernel<<<1, 32>>>(d_results, d_test_values);
    TC_CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 32 * sizeof(bool), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 32; ++i) {
        EXPECT_TRUE(h_results[i]) << "warp_any should return true when all predicates are true";
    }
}

TEST_F(WarpUtilsTest, WarpAnyOneTrue) {
    // Only one lane has predicate = true
    std::vector<int> h_test_values(32, -1);  // All negative
    h_test_values[0] = 1;  // Only lane 0 is positive
    std::vector<bool> h_results(32);

    TC_CUDA_CHECK(cudaMemcpy(d_test_values, h_test_values.data(), 32 * sizeof(int), cudaMemcpyHostToDevice));
    test_warp_any_kernel<<<1, 32>>>(d_results, d_test_values);
    TC_CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 32 * sizeof(bool), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 32; ++i) {
        EXPECT_TRUE(h_results[i]) << "warp_any should return true when any predicate is true";
    }
}

TEST_F(WarpUtilsTest, WarpAnyAllFalse) {
    std::vector<int> h_test_values(32, -1);  // All negative
    std::vector<bool> h_results(32);

    TC_CUDA_CHECK(cudaMemcpy(d_test_values, h_test_values.data(), 32 * sizeof(int), cudaMemcpyHostToDevice));
    test_warp_any_kernel<<<1, 32>>>(d_results, d_test_values);
    TC_CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, 32 * sizeof(bool), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 32; ++i) {
        EXPECT_FALSE(h_results[i]) << "warp_any should return false when all predicates are false";
    }
}

TEST_F(WarpUtilsTest, WarpBallot) {
    // Test ballot: should produce correct mask
    std::vector<int> h_test_values(32, -1);
    h_test_values[0] = 1;   // Lane 0 true
    h_test_values[2] = 1;   // Lane 2 true
    h_test_values[31] = 1;  // Lane 31 true
    unsigned h_mask = 0;

    TC_CUDA_CHECK(cudaMemcpy(d_test_values, h_test_values.data(), 32 * sizeof(int), cudaMemcpyHostToDevice));
    test_warp_ballot_kernel<<<1, 32>>>(d_unsigned_output, d_test_values);
    TC_CUDA_CHECK(cudaMemcpy(&h_mask, d_unsigned_output, sizeof(unsigned), cudaMemcpyDeviceToHost));

    // Expected mask: bit 0, 2, 31 set = 0x80000005
    unsigned expected = (1u << 0) | (1u << 2) | (1u << 31);
    EXPECT_EQ(h_mask, expected);
}

// ============================================================================
// Warp Scan Tests
// ============================================================================

TEST_F(WarpUtilsTest, WarpScanInclusive) {
    // Test inclusive scan: input [1, 2, 3, 4, ...] -> output [1, 3, 6, 10, ...]
    std::vector<int> h_input(32);
    std::vector<int> h_output(32);

    for (int i = 0; i < 32; ++i) {
        h_input[i] = i + 1;
    }

    TC_CUDA_CHECK(cudaMemcpy(d_test_values, h_input.data(), 32 * sizeof(int), cudaMemcpyHostToDevice));
    test_warp_scan_kernel<<<1, 32>>>(d_test_values, d_output);
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, 32 * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify inclusive scan results
    int running_sum = 0;
    for (int i = 0; i < 32; ++i) {
        running_sum += h_input[i];
        EXPECT_EQ(h_output[i], running_sum) << "Inclusive scan failed at index " << i;
    }
}

// ============================================================================
// Block Reduction Tests
// ============================================================================

TEST_F(WarpUtilsTest, BlockSum) {
    // Test block_sum with 256 threads
    std::vector<float> h_input(256);
    float expected_sum = 0.0f;

    for (int i = 0; i < 256; ++i) {
        h_input[i] = static_cast<float>(i);
        expected_sum += h_input[i];
    }

    float h_output = 0.0f;

    TC_CUDA_CHECK(cudaMemcpy(d_float_input, h_input.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
    test_block_sum_kernel<256><<<1, 256>>>(d_float_input, d_float_output, 256);
    TC_CUDA_CHECK(cudaMemcpy(&h_output, d_float_output, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_output, expected_sum, 1e-3f);
}

TEST_F(WarpUtilsTest, BlockMean) {
    // Test block_mean
    std::vector<float> h_input(256);
    float sum = 0.0f;

    for (int i = 0; i < 256; ++i) {
        h_input[i] = static_cast<float>(i);
        sum += h_input[i];
    }

    int divisor = 256;
    float expected_mean = sum / divisor;
    float h_output = 0.0f;

    TC_CUDA_CHECK(cudaMemcpy(d_float_input, h_input.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
    test_block_mean_kernel<256><<<1, 256>>>(d_float_input, d_float_output, 256, divisor);
    TC_CUDA_CHECK(cudaMemcpy(&h_output, d_float_output, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_output, expected_mean, 1e-3f);
}

TEST_F(WarpUtilsTest, BlockMax) {
    // Test block_max
    std::vector<float> h_input(256);
    float expected_max = -1000.0f;

    for (int i = 0; i < 256; ++i) {
        h_input[i] = static_cast<float>(i) - 128.0f;  // Range: -128 to 127
        expected_max = std::max(expected_max, h_input[i]);
    }

    float h_output = 0.0f;

    TC_CUDA_CHECK(cudaMemcpy(d_float_input, h_input.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
    test_block_max_kernel<256><<<1, 256>>>(d_float_input, d_float_output, 256);
    TC_CUDA_CHECK(cudaMemcpy(&h_output, d_float_output, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_output, expected_max, 1e-3f);
}

TEST_F(WarpUtilsTest, BlockMin) {
    // Test block_min
    std::vector<float> h_input(256);
    float expected_min = 1000.0f;

    for (int i = 0; i < 256; ++i) {
        h_input[i] = static_cast<float>(i) - 128.0f;  // Range: -128 to 127
        expected_min = std::min(expected_min, h_input[i]);
    }

    float h_output = 0.0f;

    TC_CUDA_CHECK(cudaMemcpy(d_float_input, h_input.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
    test_block_min_kernel<256><<<1, 256>>>(d_float_input, d_float_output, 256);
    TC_CUDA_CHECK(cudaMemcpy(&h_output, d_float_output, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_output, expected_min, 1e-3f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(WarpUtilsTest, BlockSumEmpty) {
    // Test with all zeros
    std::vector<float> h_input(256, 0.0f);
    float h_output = -1.0f;

    TC_CUDA_CHECK(cudaMemcpy(d_float_input, h_input.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
    test_block_sum_kernel<256><<<1, 256>>>(d_float_input, d_float_output, 256);
    TC_CUDA_CHECK(cudaMemcpy(&h_output, d_float_output, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_output, 0.0f, 1e-6f);
}

TEST_F(WarpUtilsTest, BlockSumNegative) {
    // Test with negative values
    std::vector<float> h_input(256);
    float expected_sum = 0.0f;

    for (int i = 0; i < 256; ++i) {
        h_input[i] = static_cast<float>(i) - 128.0f;  // Range: -128 to 127
        expected_sum += h_input[i];
    }

    float h_output = 0.0f;

    TC_CUDA_CHECK(cudaMemcpy(d_float_input, h_input.data(), 256 * sizeof(float), cudaMemcpyHostToDevice));
    test_block_sum_kernel<256><<<1, 256>>>(d_float_input, d_float_output, 256);
    TC_CUDA_CHECK(cudaMemcpy(&h_output, d_float_output, sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_NEAR(h_output, expected_sum, 1e-2f);  // Allow more tolerance for negative sums
}
