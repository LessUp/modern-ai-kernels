#include <stdexcept>
#include <string>

#include "tensorcraft/kernels/elementwise.hpp"
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/kernels/normalization.hpp"
#include "tensorcraft/kernels/softmax.hpp"

#include "cuda_ops.hpp"

namespace tensorcraft {
namespace python_ops {
namespace {

kernels::GemmVersion parse_gemm_version(const char* version) {
    const std::string name = version == nullptr ? "tiled" : version;
    if (name == "naive")
        return kernels::GemmVersion::Naive;
    if (name == "tiled")
        return kernels::GemmVersion::Tiled;
    if (name == "double_buffer")
        return kernels::GemmVersion::DoubleBuffer;
    throw std::invalid_argument("Unsupported GEMM version: " + name);
}

}  // namespace

void relu(const float* input, float* output, std::size_t n) {
    kernels::relu(input, output, n);
}

void silu(const float* input, float* output, std::size_t n) {
    kernels::silu(input, output, n);
}

void gelu(const float* input, float* output, std::size_t n) {
    kernels::gelu(input, output, n);
}

void sigmoid(const float* input, float* output, std::size_t n) {
    kernels::sigmoid(input, output, n);
}

void vector_add(const float* lhs, const float* rhs, float* output, std::size_t n) {
    kernels::vector_add(lhs, rhs, output, n);
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

void gemm(const float* lhs, const float* rhs, float* output, int m, int n, int k, float alpha,
          float beta, const char* version) {
    kernels::launch_gemm(lhs, rhs, output, m, n, k, alpha, beta, parse_gemm_version(version));
}

void transpose(const float* input, float* output, int rows, int cols) {
    kernels::transpose(input, output, rows, cols);
}

}  // namespace python_ops
}  // namespace tensorcraft
