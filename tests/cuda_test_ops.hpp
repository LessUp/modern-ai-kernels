#pragma once

#include <cstddef>
#include <cstdint>

namespace tensorcraft {
namespace tests {

enum class GemmVersion {
    Naive,
    Tiled,
    DoubleBuffer,
    TensorCore
};


void vector_add(const float* lhs, const float* rhs, float* output, std::size_t n);
void relu(const float* input, float* output, std::size_t n);
void silu(const float* input, float* output, std::size_t n);
void gelu(const float* input, float* output, std::size_t n);
void softmax(const float* input, float* output, int rows, int cols);
void layernorm(const float* input,
               const float* gamma,
               const float* beta,
               float* output,
               int batch_size,
               int hidden_size,
               float eps);
void rmsnorm(const float* input,
             const float* weight,
             float* output,
             int batch_size,
             int hidden_size,
             float eps);
void launch_batchnorm(const float* input,
                      const float* gamma,
                      const float* beta,
                      const float* running_mean,
                      const float* running_var,
                      float* output,
                      int n,
                      int c,
                      int h,
                      int w,
                      float eps,
                      bool training);
void transpose(const float* input, float* output, int rows, int cols);
void launch_gemm(const float* lhs,
                 const float* rhs,
                 float* output,
                 int m,
                 int n,
                 int k,
                 float alpha,
                 float beta,
                 GemmVersion version);
void conv2d(const float* input,
            const float* weight,
            const float* bias,
            float* output,
            int batch,
            int channels,
            int height,
            int width,
            int filters,
            int kernel_h,
            int kernel_w,
            int stride,
            int pad);
void conv2d_depthwise(const float* input,
                      const float* weight,
                      const float* bias,
                      float* output,
                      int batch,
                      int channels,
                      int height,
                      int width,
                      int kernel_h,
                      int kernel_w,
                      int stride,
                      int pad);
void quantize_int8(const float* input, int8_t* output, float scale, int zero_point, std::size_t n);
void dequantize_int8(const int8_t* input, float* output, float scale, int zero_point, std::size_t n);
void gemm_bias_relu(const float* lhs,
                    const float* rhs,
                    const float* bias,
                    float* output,
                    std::size_t m,
                    std::size_t n,
                    std::size_t k);
void launch_flash_attention(const float* q,
                            const float* k,
                            const float* v,
                            float* output,
                            int batch_size,
                            int num_heads,
                            int seq_len,
                            int head_dim,
                            float scale);
void launch_rope(float* input,
                 const float* cos_table,
                 const float* sin_table,
                 int batch_size,
                 int seq_len,
                 int num_heads,
                 int head_dim,
                 int start_pos);
void launch_moe_router(const float* logits,
                       int* topk_indices,
                       float* topk_weights,
                       int num_tokens,
                       int num_experts,
                       int k);
void precompute_rope_cache(float* cos_table, float* sin_table, int seq_len, int head_dim);

} // namespace tests
} // namespace tensorcraft
