/**
 * @file test_fusion.cpp
 * @brief Tests for fused operators and quantization kernels
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <cfloat>

#include "tensorcraft/core/cuda_check.hpp"
#include "cuda_test_ops.hpp"

using namespace tensorcraft;
using namespace tensorcraft::tests;

class FusionTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen = std::mt19937(42);
        dist = std::uniform_real_distribution<float>(-2.0f, 2.0f);
    }

    std::vector<float> random_vec(size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = dist(gen);
        return v;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

// ── INT8 Quantization Tests ─────────────────────────────────────

TEST_F(FusionTest, QuantizeDeQuantizeRoundTrip) {
    const size_t N = 1024;
    float scale = 0.05f;
    int zero_point = 0;

    auto h_input = random_vec(N);

    float *d_input, *d_output;
    int8_t *d_q;
    TC_CUDA_CHECK(cudaMalloc(&d_input,  N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_q,      N * sizeof(int8_t)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    quantize_int8(d_input, d_q, scale, zero_point, N);
    dequantize_int8(d_q, d_output, scale, zero_point, N);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output(N);
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Round-trip error should be bounded by scale
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(h_output[i], h_input[i], scale + 1e-5f)
            << "index " << i << " input=" << h_input[i] << " output=" << h_output[i];
    }

    cudaFree(d_input); cudaFree(d_q); cudaFree(d_output);
}

TEST_F(FusionTest, QuantizeInt8Clamps) {
    // Values outside [-128*scale, 127*scale] should be clamped
    const size_t N = 4;
    float scale = 1.0f;
    int zero_point = 0;

    std::vector<float> h_input = {200.0f, -200.0f, 50.0f, -50.0f};

    float *d_input;
    int8_t *d_q;
    TC_CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_q,     N * sizeof(int8_t)));

    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    quantize_int8(d_input, d_q, scale, zero_point, N);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int8_t> h_q(N);
    TC_CUDA_CHECK(cudaMemcpy(h_q.data(), d_q, N * sizeof(int8_t), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_q[0], 127);   // clamped
    EXPECT_EQ(h_q[1], -128);  // clamped
    EXPECT_EQ(h_q[2], 50);
    EXPECT_EQ(h_q[3], -50);

    cudaFree(d_input); cudaFree(d_q);
}

// ── Fused GEMM Tests ────────────────────────────────────────────

TEST_F(FusionTest, GemmBiasReLU) {
    const int M = 32, N = 32, K = 32;

    auto h_A = random_vec(M * K);
    auto h_B = random_vec(K * N);
    auto h_bias = random_vec(N);

    // CPU reference: C = ReLU(A @ B + bias)
    std::vector<float> h_ref(M * N);
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += h_A[m * K + k] * h_B[k * N + n];
        sum += h_bias[n];
        h_ref[m * N + n] = sum > 0.0f ? sum : 0.0f;
    }

    float *d_A, *d_B, *d_bias, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A,    M * K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_B,    K * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C,    M * N * sizeof(float)));

    TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    gemm_bias_relu(d_A, d_B, d_bias, d_C, (size_t)M, (size_t)N, (size_t)K);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_C(M * N);
    TC_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(h_C[i], h_ref[i], 1e-2f) << "index " << i;
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_bias); cudaFree(d_C);
}
