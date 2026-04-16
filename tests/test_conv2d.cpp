/**
 * @file test_conv2d.cpp
 * @brief Tests for Conv2D kernels
 */

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"

#include "cuda_test_ops.hpp"

using namespace tensorcraft;
using namespace tensorcraft::tests;

class Conv2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        gen = std::mt19937(42);
        dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }

    std::vector<float> random_vec(size_t n) {
        std::vector<float> v(n);
        for (auto& x : v)
            x = dist(gen);
        return v;
    }

    // CPU reference for naive conv2d
    std::vector<float> reference_conv2d(const std::vector<float>& input,
                                        const std::vector<float>& weight,
                                        const std::vector<float>& bias, int N, int C, int H, int W,
                                        int K, int R, int S, int stride, int pad) {
        int OH = (H + 2 * pad - R) / stride + 1;
        int OW = (W + 2 * pad - S) / stride + 1;
        std::vector<float> output(N * K * OH * OW, 0.0f);

        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                for (int oh = 0; oh < OH; ++oh)
                    for (int ow = 0; ow < OW; ++ow) {
                        float sum = 0.0f;
                        for (int c = 0; c < C; ++c)
                            for (int r = 0; r < R; ++r)
                                for (int s = 0; s < S; ++s) {
                                    int ih = oh * stride - pad + r;
                                    int iw = ow * stride - pad + s;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        sum += input[((n * C + c) * H + ih) * W + iw] *
                                               weight[((k * C + c) * R + r) * S + s];
                                    }
                                }
                        if (!bias.empty())
                            sum += bias[k];
                        output[((n * K + k) * OH + oh) * OW + ow] = sum;
                    }
        return output;
    }

    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_F(Conv2DTest, NaiveCorrectness) {
    const int N = 1, C = 3, H = 8, W = 8;
    const int K = 4, R = 3, S = 3;
    const int stride = 1, pad = 1;
    const int OH = (H + 2 * pad - R) / stride + 1;
    const int OW = (W + 2 * pad - S) / stride + 1;

    auto h_input = random_vec(N * C * H * W);
    auto h_weight = random_vec(K * C * R * S);
    auto h_bias = random_vec(K);
    auto h_ref = reference_conv2d(h_input, h_weight, h_bias, N, C, H, W, K, R, S, stride, pad);

    float *d_in, *d_w, *d_b, *d_out;
    TC_CUDA_CHECK(cudaMalloc(&d_in, N * C * H * W * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_w, K * C * R * S * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_b, K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_out, N * K * OH * OW * sizeof(float)));

    TC_CUDA_CHECK(
        cudaMemcpy(d_in, h_input.data(), N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(
        cudaMemcpy(d_w, h_weight.data(), K * C * R * S * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_b, h_bias.data(), K * sizeof(float), cudaMemcpyHostToDevice));

    conv2d(d_in, d_w, d_b, d_out, N, C, H, W, K, R, S, stride, pad);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(N * K * OH * OW);
    TC_CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_ref.size(); ++i) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-3f) << "index " << i;
    }

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_out);
}

TEST_F(Conv2DTest, DepthwiseCorrectness) {
    const int N = 1, C = 4, H = 8, W = 8;
    const int R = 3, S = 3;
    const int stride = 1, pad = 1;
    const int OH = (H + 2 * pad - R) / stride + 1;
    const int OW = (W + 2 * pad - S) / stride + 1;

    auto h_input = random_vec(N * C * H * W);
    auto h_weight = random_vec(C * R * S);
    std::vector<float> h_bias;  // no bias

    // CPU reference for depthwise
    std::vector<float> h_ref(N * C * OH * OW, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float sum = 0.0f;
                    for (int r = 0; r < R; ++r)
                        for (int s = 0; s < S; ++s) {
                            int ih = oh * stride - pad + r;
                            int iw = ow * stride - pad + s;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += h_input[((n * C + c) * H + ih) * W + iw] *
                                       h_weight[(c * R + r) * S + s];
                            }
                        }
                    h_ref[((n * C + c) * OH + oh) * OW + ow] = sum;
                }

    float *d_in, *d_w, *d_out;
    TC_CUDA_CHECK(cudaMalloc(&d_in, N * C * H * W * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_w, C * R * S * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_out, N * C * OH * OW * sizeof(float)));

    TC_CUDA_CHECK(
        cudaMemcpy(d_in, h_input.data(), N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(
        cudaMemcpy(d_w, h_weight.data(), C * R * S * sizeof(float), cudaMemcpyHostToDevice));

    conv2d_depthwise(d_in, d_w, static_cast<const float*>(nullptr), d_out, N, C, H, W, R, S, stride,
                     pad);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(N * C * OH * OW);
    TC_CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_ref.size(); ++i) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-3f) << "index " << i;
    }

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_out);
}

TEST_F(Conv2DTest, NoBias) {
    const int N = 1, C = 1, H = 4, W = 4;
    const int K = 1, R = 3, S = 3;
    const int stride = 1, pad = 0;
    const int OH = (H - R) / stride + 1;
    const int OW = (W - S) / stride + 1;

    auto h_input = random_vec(N * C * H * W);
    auto h_weight = random_vec(K * C * R * S);
    auto h_ref = reference_conv2d(h_input, h_weight, {}, N, C, H, W, K, R, S, stride, pad);

    float *d_in, *d_w, *d_out;
    TC_CUDA_CHECK(cudaMalloc(&d_in, N * C * H * W * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_w, K * C * R * S * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_out, N * K * OH * OW * sizeof(float)));

    TC_CUDA_CHECK(
        cudaMemcpy(d_in, h_input.data(), N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(
        cudaMemcpy(d_w, h_weight.data(), K * C * R * S * sizeof(float), cudaMemcpyHostToDevice));

    conv2d(d_in, d_w, static_cast<const float*>(nullptr), d_out, N, C, H, W, K, R, S, stride, pad);
    TC_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(N * K * OH * OW);
    TC_CUDA_CHECK(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_ref.size(); ++i) {
        EXPECT_NEAR(h_out[i], h_ref[i], 1e-3f) << "index " << i;
    }

    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_out);
}
