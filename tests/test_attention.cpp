/**
 * @file test_attention.cpp
 * @brief Unit tests for FlashAttention, RoPE, and MoE Router kernels
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/attention.hpp"

using namespace tensorcraft::kernels;

// Helper to allocate and copy data to GPU
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : size_(count) {
        TC_CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (ptr_)
            cudaFree(ptr_);
    }

    void copy_from_host(const T* host_data) {
        TC_CUDA_CHECK(cudaMemcpy(ptr_, host_data, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_data) const {
        TC_CUDA_CHECK(cudaMemcpy(host_data, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

// ============================================================================
// FlashAttention Tests
// ============================================================================

class FlashAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA device availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available, skipping FlashAttention tests";
        }
        cudaSetDevice(0);
    }

    // Reference implementation of FlashAttention for validation
    void reference_attention(const float* Q, const float* K, const float* V, float* O, int batch,
                             int heads, int seq, int head_dim) {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < heads; ++h) {
                for (int s = 0; s < seq; ++s) {
                    // Compute attention scores for this query position
                    std::vector<float> scores(seq);
                    float max_score = -1e30f;

                    for (int t = 0; t < seq; ++t) {
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            int q_idx = ((b * heads + h) * seq + s) * head_dim + d;
                            int k_idx = ((b * heads + h) * seq + t) * head_dim + d;
                            score += Q[q_idx] * K[k_idx];
                        }
                        scores[t] = score * scale;
                        max_score = std::max(max_score, scores[t]);
                    }

                    // Softmax
                    float sum_exp = 0.0f;
                    for (int t = 0; t < seq; ++t) {
                        scores[t] = std::exp(scores[t] - max_score);
                        sum_exp += scores[t];
                    }

                    // Weighted sum of values
                    for (int d = 0; d < head_dim; ++d) {
                        float out = 0.0f;
                        for (int t = 0; t < seq; ++t) {
                            int v_idx = ((b * heads + h) * seq + t) * head_dim + d;
                            out += scores[t] / sum_exp * V[v_idx];
                        }
                        int o_idx = ((b * heads + h) * seq + s) * head_dim + d;
                        O[o_idx] = out;
                    }
                }
            }
        }
    }
};

// Basic functionality test
TEST_F(FlashAttentionTest, BasicForward) {
    const int batch = 2;
    const int heads = 4;
    const int seq = 64;
    const int head_dim = 64;

    const size_t total_size = batch * heads * seq * head_dim;

    // Create random input data
    std::vector<float> h_Q(total_size), h_K(total_size), h_V(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f - 0.05f;
    }

    // Allocate device buffers
    DeviceBuffer<float> d_Q(total_size), d_K(total_size), d_V(total_size), d_O(total_size);

    d_Q.copy_from_host(h_Q.data());
    d_K.copy_from_host(h_K.data());
    d_V.copy_from_host(h_V.data());

    // Launch FlashAttention
    flash_attention(d_Q.get(), d_K.get(), d_V.get(), d_O.get(), batch, heads, seq, head_dim);

    // Copy output back
    std::vector<float> h_O(total_size);
    d_O.copy_to_host(h_O.data());

    // Verify output is not all zeros
    bool has_nonzero = false;
    for (size_t i = 0; i < total_size; ++i) {
        if (std::abs(h_O[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "FlashAttention output should not be all zeros";
}

// Test with small dimensions for detailed comparison
TEST_F(FlashAttentionTest, SmallDimensionsCorrectness) {
    const int batch = 1;
    const int heads = 1;
    const int seq = 8;
    const int head_dim = 64;

    const size_t total_size = batch * heads * seq * head_dim;

    // Create simple input data
    std::vector<float> h_Q(total_size), h_K(total_size), h_V(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        h_Q[i] = static_cast<float>(i % head_dim) / head_dim;
        h_K[i] = static_cast<float>((i + 1) % head_dim) / head_dim;
        h_V[i] = static_cast<float>((i + 2) % head_dim) / head_dim;
    }

    // Compute reference output
    std::vector<float> h_O_ref(total_size);
    reference_attention(h_Q.data(), h_K.data(), h_V.data(), h_O_ref.data(), batch, heads, seq,
                        head_dim);

    // Compute GPU output
    DeviceBuffer<float> d_Q(total_size), d_K(total_size), d_V(total_size), d_O(total_size);
    d_Q.copy_from_host(h_Q.data());
    d_K.copy_from_host(h_K.data());
    d_V.copy_from_host(h_V.data());

    flash_attention(d_Q.get(), d_K.get(), d_V.get(), d_O.get(), batch, heads, seq, head_dim);

    std::vector<float> h_O(total_size);
    d_O.copy_to_host(h_O.data());

    // Compare outputs with tolerance
    const float tolerance = 1e-3f;  // Allow some numerical difference due to different
                                    // computation order
    for (size_t i = 0; i < total_size; ++i) {
        EXPECT_NEAR(h_O[i], h_O_ref[i], tolerance)
            << "Mismatch at index " << i << ": GPU=" << h_O[i] << ", Ref=" << h_O_ref[i];
    }
}

// Test invalid head_dim (should throw)
TEST_F(FlashAttentionTest, InvalidHeadDim) {
    const int batch = 1, heads = 1, seq = 8, head_dim = 128;  // head_dim != 64

    DeviceBuffer<float> d_Q(1), d_K(1), d_V(1), d_O(1);

    EXPECT_THROW(flash_attention(d_Q.get(), d_K.get(), d_V.get(), d_O.get(), batch, heads, seq,
                                 head_dim),
                 std::invalid_argument);
}

// Test with empty input
TEST_F(FlashAttentionTest, EmptyInput) {
    // seq_len = 0 should be handled gracefully
    EXPECT_NO_THROW({
        DeviceBuffer<float> d_Q(0), d_K(0), d_V(0), d_O(0);
        flash_attention(d_Q.get(), d_K.get(), d_V.get(), d_O.get(), 0, 4, 0, 64);
    });
}

// ============================================================================
// RoPE Tests
// ============================================================================

class RoPETest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available, skipping RoPE tests";
        }
        cudaSetDevice(0);
    }
};

TEST_F(RoPETest, PrecomputeCache) {
    const int max_seq = 64;
    const int head_dim = 64;

    DeviceBuffer<float> d_cos(max_seq * head_dim / 2);
    DeviceBuffer<float> d_sin(max_seq * head_dim / 2);

    EXPECT_NO_THROW(precompute_rope_cache(d_cos.get(), d_sin.get(), max_seq, head_dim));

    // Verify cache is computed (not all zeros)
    std::vector<float> h_cos(max_seq * head_dim / 2);
    d_cos.copy_to_host(h_cos.data());

    bool has_nonzero = false;
    for (float v : h_cos) {
        if (std::abs(v) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "RoPE cos cache should have non-zero values";
}

TEST_F(RoPETest, InvalidHeadDim) {
    DeviceBuffer<float> d_x(64), d_cos(32), d_sin(32);

    EXPECT_THROW(
        rope(d_x.get(), d_cos.get(), d_sin.get(), 1, 1, 1, 63),  // odd head_dim
        std::invalid_argument);
}

TEST_F(RoPETest, InvalidStartPos) {
    DeviceBuffer<float> d_x(64), d_cos(32), d_sin(32);

    EXPECT_THROW(
        rope(d_x.get(), d_cos.get(), d_sin.get(), 1, 1, 1, 64, -1),  // negative start_pos
        std::invalid_argument);
}

// ============================================================================
// MoE Router Tests
// ============================================================================

class MoERouterTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available, skipping MoE Router tests";
        }
        cudaSetDevice(0);
    }
};

TEST_F(MoERouterTest, TopKSelection) {
    const int batch = 4;
    const int num_experts = 4;
    const int top_k = 2;

    // Create gate logits with known ordering
    std::vector<float> h_logits(batch * num_experts);
    for (int b = 0; b < batch; ++b) {
        // Expert 2 has highest score, Expert 1 has second highest
        h_logits[b * num_experts + 0] = 0.1f;
        h_logits[b * num_experts + 1] = 0.5f;
        h_logits[b * num_experts + 2] = 1.0f;
        h_logits[b * num_experts + 3] = 0.2f;
    }

    DeviceBuffer<float> d_logits(batch * num_experts);
    DeviceBuffer<int> d_indices(batch * top_k);
    DeviceBuffer<float> d_weights(batch * top_k);

    d_logits.copy_from_host(h_logits.data());

    launch_moe_router(d_logits.get(), d_indices.get(), d_weights.get(), batch, num_experts, top_k);

    std::vector<int> h_indices(batch * top_k);
    std::vector<float> h_weights(batch * top_k);
    d_indices.copy_to_host(h_indices.data());
    d_weights.copy_to_host(h_weights.data());

    // Verify top-k selection
    for (int b = 0; b < batch; ++b) {
        // First selected expert should be expert 2 (score 1.0)
        EXPECT_EQ(h_indices[b * top_k + 0], 2) << "Batch " << b << " should select expert 2 first";

        // Second selected expert should be expert 1 (score 0.5)
        EXPECT_EQ(h_indices[b * top_k + 1], 1) << "Batch " << b << " should select expert 1 second";

        // Weights should sum to 1.0
        float weight_sum = h_weights[b * top_k + 0] + h_weights[b * top_k + 1];
        EXPECT_NEAR(weight_sum, 1.0f, 1e-5f) << "Weights should sum to 1.0 for batch " << b;
    }
}

TEST_F(MoERouterTest, InvalidNumExperts) {
    DeviceBuffer<float> d_logits(8 * 16);  // 8 experts
    DeviceBuffer<int> d_indices(8);
    DeviceBuffer<float> d_weights(8);

    EXPECT_THROW(
        launch_moe_router(d_logits.get(), d_indices.get(), d_weights.get(), 1, 16, 2),  // 16 > 8
        std::invalid_argument);
}

TEST_F(MoERouterTest, InvalidTopK) {
    DeviceBuffer<float> d_logits(4);
    DeviceBuffer<int> d_indices(4);
    DeviceBuffer<float> d_weights(4);

    EXPECT_THROW(
        launch_moe_router(d_logits.get(), d_indices.get(), d_weights.get(), 1, 4, 5),  // top_k >
                                                                                       // num_experts
        std::invalid_argument);
}

TEST_F(MoERouterTest, EmptyBatch) {
    DeviceBuffer<float> d_logits(0);
    DeviceBuffer<int> d_indices(0);
    DeviceBuffer<float> d_weights(0);

    EXPECT_NO_THROW(launch_moe_router(d_logits.get(), d_indices.get(), d_weights.get(), 0, 4, 2));
}
