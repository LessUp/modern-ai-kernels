#include "tensorcraft/kernels/attention.hpp"
#include "tensorcraft/kernels/conv2d.hpp"
#include "tensorcraft/kernels/elementwise.hpp"
#include "tensorcraft/kernels/fusion.hpp"
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/kernels/normalization.hpp"
#include "tensorcraft/kernels/softmax.hpp"

#include "cuda_test_ops.hpp"

namespace tensorcraft {
namespace tests {

namespace {

kernels::GemmVersion to_kernel_gemm_version(GemmVersion version) {
    switch (version) {
        case GemmVersion::Naive:
            return kernels::GemmVersion::Naive;
        case GemmVersion::Tiled:
            return kernels::GemmVersion::Tiled;
        case GemmVersion::DoubleBuffer:
            return kernels::GemmVersion::DoubleBuffer;
        case GemmVersion::TensorCore:
            return kernels::GemmVersion::TensorCore;
        default:
            return kernels::GemmVersion::Tiled;
    }
}

}  // namespace

void vector_add(const float* lhs, const float* rhs, float* output, std::size_t n) {
    kernels::vector_add(lhs, rhs, output, n);
}

void relu(const float* input, float* output, std::size_t n) {
    kernels::relu(input, output, n);
}

void silu(const float* input, float* output, std::size_t n) {
    kernels::silu(input, output, n);
}

void gelu(const float* input, float* output, std::size_t n) {
    kernels::gelu(input, output, n);
}

void softmax(const float* input, float* output, int rows, int cols) {
    kernels::softmax(input, output, rows, cols);
}

void layernorm(const float* input, const float* gamma, const float* beta, float* output,
               int batch_size, int hidden_size, float eps) {
    kernels::layernorm(input, gamma, beta, output, batch_size, hidden_size, eps);
}

void rmsnorm(const float* input, const float* weight, float* output, int batch_size,
             int hidden_size, float eps) {
    kernels::rmsnorm(input, weight, output, batch_size, hidden_size, eps);
}

void launch_batchnorm(const float* input, const float* gamma, const float* beta,
                      const float* running_mean, const float* running_var, float* output, int n,
                      int c, int h, int w, float eps, bool training) {
    kernels::launch_batchnorm(input, gamma, beta, running_mean, running_var, output, n, c, h, w,
                              eps, training);
}

void transpose(const float* input, float* output, int rows, int cols) {
    kernels::transpose(input, output, rows, cols);
}

void launch_gemm(const float* lhs, const float* rhs, float* output, int m, int n, int k,
                 float alpha, float beta, GemmVersion version) {
    kernels::launch_gemm(lhs, rhs, output, m, n, k, alpha, beta, to_kernel_gemm_version(version));
}

void conv2d(const float* input, const float* weight, const float* bias, float* output, int batch,
            int channels, int height, int width, int filters, int kernel_h, int kernel_w,
            int stride, int pad) {
    kernels::conv2d(input, weight, bias, output, batch, channels, height, width, filters, kernel_h,
                    kernel_w, stride, pad);
}

void conv2d_depthwise(const float* input, const float* weight, const float* bias, float* output,
                      int batch, int channels, int height, int width, int kernel_h, int kernel_w,
                      int stride, int pad) {
    kernels::conv2d_depthwise(input, weight, bias, output, batch, channels, height, width, kernel_h,
                              kernel_w, stride, pad);
}

void quantize_int8(const float* input, int8_t* output, float scale, int zero_point, std::size_t n) {
    kernels::quantize_int8(input, output, scale, zero_point, n);
}

void dequantize_int8(const int8_t* input, float* output, float scale, int zero_point,
                     std::size_t n) {
    kernels::dequantize_int8(input, output, scale, zero_point, n);
}

void gemm_bias_relu(const float* lhs, const float* rhs, const float* bias, float* output,
                    std::size_t m, std::size_t n, std::size_t k) {
    kernels::gemm_bias_relu(lhs, rhs, bias, output, m, n, k);
}

void launch_flash_attention(const float* q, const float* k, const float* v, float* output,
                            int batch_size, int num_heads, int seq_len, int head_dim, float scale) {
    kernels::launch_flash_attention(q, k, v, output, batch_size, num_heads, seq_len, head_dim,
                                    scale);
}

void launch_rope(float* input, const float* cos_table, const float* sin_table, int batch_size,
                 int seq_len, int num_heads, int head_dim, int start_pos) {
    kernels::launch_rope(input, cos_table, sin_table, batch_size, seq_len, num_heads, head_dim,
                         start_pos);
}

void launch_moe_router(const float* logits, int* topk_indices, float* topk_weights, int num_tokens,
                       int num_experts, int k) {
    kernels::launch_moe_router(logits, topk_indices, topk_weights, num_tokens, num_experts, k);
}

void precompute_rope_cache(float* cos_table, float* sin_table, int seq_len, int head_dim) {
    kernels::precompute_rope_cache(cos_table, sin_table, seq_len, head_dim);
}

}  // namespace tests
}  // namespace tensorcraft
