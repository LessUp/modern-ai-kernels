#pragma once

#include <cstddef>

namespace tensorcraft {
namespace python_ops {

void relu(const float* input, float* output, std::size_t n);
void silu(const float* input, float* output, std::size_t n);
void gelu(const float* input, float* output, std::size_t n);
void sigmoid(const float* input, float* output, std::size_t n);
void vector_add(const float* lhs, const float* rhs, float* output, std::size_t n);
void softmax(const float* input, float* output, int rows, int cols);
void layernorm(const float* input, const float* gamma, const float* beta, float* output,
               int batch_size, int hidden_size, float eps);
void rmsnorm(const float* input, const float* weight, float* output, int batch_size,
             int hidden_size, float eps);
void gemm(const float* lhs, const float* rhs, float* output, int m, int n, int k, float alpha,
          float beta, const char* version);
void transpose(const float* input, float* output, int rows, int cols);

}  // namespace python_ops
}  // namespace tensorcraft
