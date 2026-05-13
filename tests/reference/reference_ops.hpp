#pragma once
/**
 * @file reference_ops.hpp
 * @brief Reference implementations for kernel correctness verification
 *
 * This module provides CPU-based reference implementations for all
 * compute kernels. These are used for testing and verification purposes.
 *
 * All implementations are header-only for easy inclusion in test files.
 */

#include <cassert>
#include <cmath>
#include <vector>

namespace tensorcraft {
namespace reference {

// ============================================================================
// GEMM Reference
// ============================================================================

/**
 * @brief Reference GEMM implementation
 *
 * Computes C = alpha * A @ B + beta * C
 * A: [M, K], B: [K, N], C: [M, N]
 */
inline std::vector<float> gemm(const std::vector<float>& A, const std::vector<float>& B,
                               int M, int N, int K, float alpha = 1.0f, float beta = 0.0f,
                               const std::vector<float>& C = {}) {
    // Validate input sizes to prevent out-of-bounds access
    assert(A.size() >= static_cast<size_t>(M * K) && "A must have at least M*K elements");
    assert(B.size() >= static_cast<size_t>(K * N) && "B must have at least K*N elements");
    assert(C.empty() || C.size() >= static_cast<size_t>(M * N) && "C must have at least M*N elements if provided");

    std::vector<float> result(M * N, 0.0f);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            float c_val = C.empty() ? 0.0f : C[m * N + n];
            result[m * N + n] = alpha * sum + beta * c_val;
        }
    }

    return result;
}

// ============================================================================
// Softmax Reference
// ============================================================================

/**
 * @brief Reference softmax implementation
 *
 * Computes numerically stable softmax along the last dimension.
 */
inline std::vector<float> softmax(const std::vector<float>& input, int rows, int cols) {
    std::vector<float> output(rows * cols);

    for (int r = 0; r < rows; ++r) {
        // Find max for numerical stability
        float max_val = input[r * cols];
        for (int c = 1; c < cols; ++c) {
            max_val = std::max(max_val, input[r * cols + c]);
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            output[r * cols + c] = std::exp(input[r * cols + c] - max_val);
            sum += output[r * cols + c];
        }

        // Normalize (handle edge case where sum == 0)
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (int c = 0; c < cols; ++c) {
            output[r * cols + c] *= inv_sum;
        }
    }

    return output;
}

// ============================================================================
// LayerNorm Reference
// ============================================================================

/**
 * @brief Reference LayerNorm implementation
 *
 * Computes: output = gamma * (x - mean) / sqrt(var + eps) + beta
 */
inline void layernorm(const float* input, const float* gamma, const float* beta,
                      float* output, int batch_size, int hidden_size, float eps = 1e-5f) {
    for (int b = 0; b < batch_size; ++b) {
        // Compute mean
        float mean = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            mean += input[b * hidden_size + h];
        }
        mean /= hidden_size;

        // Compute variance
        float var = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            float diff = input[b * hidden_size + h] - mean;
            var += diff * diff;
        }
        var /= hidden_size;

        // Normalize and apply affine transform
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int h = 0; h < hidden_size; ++h) {
            float normalized = (input[b * hidden_size + h] - mean) * inv_std;
            float g = gamma ? gamma[h] : 1.0f;
            float bt = beta ? beta[h] : 0.0f;
            output[b * hidden_size + h] = normalized * g + bt;
        }
    }
}

inline std::vector<float> layernorm(const std::vector<float>& input,
                                    const std::vector<float>& gamma,
                                    const std::vector<float>& beta,
                                    int batch_size, int hidden_size, float eps = 1e-5f) {
    std::vector<float> output(batch_size * hidden_size);
    // Allow empty gamma/beta vectors (will use default values: gamma=1, beta=0)
    const float* gamma_ptr = gamma.empty() ? nullptr : gamma.data();
    const float* beta_ptr = beta.empty() ? nullptr : beta.data();
    layernorm(input.data(), gamma_ptr, beta_ptr,
              output.data(), batch_size, hidden_size, eps);
    return output;
}

// ============================================================================
// RMSNorm Reference
// ============================================================================

/**
 * @brief Reference RMSNorm implementation
 *
 * Computes: output = x / rms(x) * weight
 * where rms(x) = sqrt(mean(x^2) + eps)
 */
inline void rmsnorm(const float* input, const float* weight,
                    float* output, int batch_size, int hidden_size, float eps = 1e-6f) {
    for (int b = 0; b < batch_size; ++b) {
        // Compute RMS
        float sum_sq = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            float val = input[b * hidden_size + h];
            sum_sq += val * val;
        }
        float rms_inv = 1.0f / std::sqrt(sum_sq / hidden_size + eps);

        // Apply normalization and weight
        for (int h = 0; h < hidden_size; ++h) {
            float w = weight ? weight[h] : 1.0f;
            output[b * hidden_size + h] = input[b * hidden_size + h] * rms_inv * w;
        }
    }
}

inline std::vector<float> rmsnorm(const std::vector<float>& input,
                                  const std::vector<float>& weight,
                                  int batch_size, int hidden_size, float eps = 1e-6f) {
    std::vector<float> output(batch_size * hidden_size);
    // Allow empty weight vector (will use default value: weight=1)
    const float* weight_ptr = weight.empty() ? nullptr : weight.data();
    rmsnorm(input.data(), weight_ptr, output.data(), batch_size, hidden_size, eps);
    return output;
}

// ============================================================================
// BatchNorm Reference (Inference Mode)
// ============================================================================

/**
 * @brief Reference BatchNorm inference implementation
 */
inline void batchnorm_inference(const float* input, const float* gamma, const float* beta,
                                const float* running_mean, const float* running_var,
                                float* output, int N, int C, int H, int W, float eps = 1e-5f) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float inv_std = 1.0f / std::sqrt(running_var[c] + eps);
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    float normalized = (input[idx] - running_mean[c]) * inv_std;
                    output[idx] = normalized * gamma[c] + beta[c];
                }
            }
        }
    }
}

// ============================================================================
// Activation References
// ============================================================================

inline float relu(float x) { return std::max(0.0f, x); }

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

inline float gelu(float x) {
    // Approximate GELU
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// ============================================================================
// Transpose Reference
// ============================================================================

inline std::vector<float> transpose(const std::vector<float>& input, int rows, int cols) {
    std::vector<float> output(rows * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output[c * rows + r] = input[r * cols + c];
        }
    }
    return output;
}

}  // namespace reference
}  // namespace tensorcraft
