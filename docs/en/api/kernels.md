# Kernels Module API

The Kernels module contains all compute kernels including elementwise operations, softmax, normalization, GEMM, attention, convolution, sparse operations, and fusion/quantization.

## Headers

| Header | Description |
|--------|-------------|
| `elementwise.hpp` | Elementwise operations and activation functions |
| `softmax.hpp` | Numerically stable softmax |
| `normalization.hpp` | LayerNorm, RMSNorm, BatchNorm |
| `gemm.hpp` | Matrix multiplication (naive to Tensor Core) |
| `attention.hpp` | FlashAttention, RoPE, MoE router |
| `conv2d.hpp` | 2D convolution operations |
| `sparse.hpp` | Sparse matrix operations (CSR/CSC) |
| `fusion.hpp` | Fused operators and quantization |

---

## elementwise.hpp

Generic elementwise kernel framework with vectorized access.

### Activation Functors

```cpp
namespace tensorcraft::kernels;

struct ReLU {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const;  // max(0, x)
};

struct SiLU {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const;  // x * sigmoid(x)
};

struct GeLU {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const;  // GELU approximation
};

struct GeLUExact {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const;  // GELU using erf
};

struct Sigmoid {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const;  // 1 / (1 + exp(-x))
};

struct Tanh {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const;  // tanh(x)
};

struct Softplus {
    template<typename T>
    TC_DEVICE_INLINE T operator()(T x) const;  // log(1 + exp(x))
};

template<typename T = float>
struct LeakyReLU {
    T alpha;
    TC_HOST_DEVICE LeakyReLU(T alpha = T(0.01));
    TC_DEVICE_INLINE T operator()(T x) const;  // x if x > 0, else alpha * x
};

template<typename T = float>
struct ELU {
    T alpha;
    TC_HOST_DEVICE ELU(T alpha = T(1.0));
    TC_DEVICE_INLINE T operator()(T x) const;
};

template<typename T = float>
struct Swish {
    T beta;
    TC_HOST_DEVICE Swish(T beta = T(1.0));
    TC_DEVICE_INLINE T operator()(T x) const;  // x * sigmoid(beta * x)
};
```

### Binary Operation Functors

```cpp
struct Add { template<typename T> TC_DEVICE_INLINE T operator()(T a, T b) const; };
struct Sub { template<typename T> TC_DEVICE_INLINE T operator()(T a, T b) const; };
struct Mul { template<typename T> TC_DEVICE_INLINE T operator()(T a, T b) const; };
struct Div { template<typename T> TC_DEVICE_INLINE T operator()(T a, T b) const; };
struct Max { template<typename T> TC_DEVICE_INLINE T operator()(T a, T b) const; };
struct Min { template<typename T> TC_DEVICE_INLINE T operator()(T a, T b) const; };
```

### Launcher Functions

```cpp
// Generic elementwise launcher
template<typename T, typename Func>
void launch_elementwise(const T* input, T* output, size_t n, Func func,
                        cudaStream_t stream = nullptr, bool use_vectorized = true);

// Binary elementwise launcher
template<typename T, typename Func>
void launch_elementwise_binary(const T* input1, const T* input2, T* output, size_t n, Func func,
                               cudaStream_t stream = nullptr);
```

### Convenience Functions

```cpp
template<typename T> void relu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr);
template<typename T> void silu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr);
template<typename T> void gelu(const T* input, T* output, size_t n, cudaStream_t stream = nullptr);
template<typename T> void sigmoid(const T* input, T* output, size_t n, cudaStream_t stream = nullptr);
template<typename T> void tanh_activation(const T* input, T* output, size_t n, cudaStream_t stream = nullptr);
template<typename T> void vector_add(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = nullptr);
template<typename T> void vector_mul(const T* a, const T* b, T* c, size_t n, cudaStream_t stream = nullptr);
```

---

## softmax.hpp

Numerically stable softmax using online algorithm.

### Launcher Functions

```cpp
namespace tensorcraft::kernels;

template<typename T>
void launch_softmax(const T* input, T* output, int rows, int cols,
                    cudaStream_t stream = nullptr);

template<typename T>
void softmax(const T* input, T* output, size_t batch_size, size_t dim,
             cudaStream_t stream = nullptr);
```

### Example

```cpp
#include "tensorcraft/kernels/softmax.hpp"

// Softmax along last dimension
float *d_input, *d_output;
// ... allocate and initialize ...

tensorcraft::kernels::softmax(d_input, d_output, batch_size, dim);
```

---

## normalization.hpp

Normalization kernels with warp-level reductions.

### LayerNorm

```cpp
// y = gamma * (x - mean) / sqrt(var + eps) + beta
template<typename T>
void layernorm(const T* input, const T* gamma, const T* beta, T* output,
               size_t batch_size, size_t hidden_size, float eps = 1e-5f,
               cudaStream_t stream = nullptr);

template<typename T>
void launch_layernorm(const T* input, const T* gamma, const T* beta, T* output,
                      int batch_size, int hidden_size, float eps = 1e-5f,
                      cudaStream_t stream = nullptr);
```

### RMSNorm

```cpp
// y = x / RMS(x) * weight
template<typename T>
void rmsnorm(const T* input, const T* weight, T* output,
             size_t batch_size, size_t hidden_size, float eps = 1e-6f,
             cudaStream_t stream = nullptr);

template<typename T>
void launch_rmsnorm(const T* input, const T* weight, T* output,
                    int batch_size, int hidden_size, float eps = 1e-6f,
                    cudaStream_t stream = nullptr);
```

### BatchNorm

```cpp
template<typename T>
void launch_batchnorm(const T* input, const T* gamma, const T* beta,
                      const float* running_mean, const float* running_var, T* output,
                      int N, int C, int H, int W, float eps = 1e-5f,
                      bool fuse_relu = false, cudaStream_t stream = nullptr);
```

### Example

```cpp
#include "tensorcraft/kernels/normalization.hpp"

// LayerNorm
tensorcraft::kernels::layernorm(input, gamma, beta, output, 
                                batch_size, hidden_size, 1e-5f);

// RMSNorm
tensorcraft::kernels::rmsnorm(input, weight, output,
                              batch_size, hidden_size, 1e-6f);

// BatchNorm (inference mode)
tensorcraft::kernels::launch_batchnorm(input, gamma, beta, 
                                       running_mean, running_var, output,
                                       N, C, H, W, 1e-5f, false);
```

---

## gemm.hpp

GEMM kernels with progressive optimization levels.

### GemmVersion Enum

```cpp
enum class GemmVersion {
    Naive,          // Basic implementation
    Tiled,          // Shared memory tiling
    DoubleBuffer,   // Double buffering
    TensorCore,     // WMMA tensor core (use launch_gemm_wmma)
    Auto            // Automatic selection
};
```

### GEMM Functions

```cpp
// Main launcher with version selection
template<typename T>
void launch_gemm(const T* A, const T* B, T* C, int M, int N, int K,
                 T alpha = T(1), T beta = T(0),
                 GemmVersion version = GemmVersion::Tiled,
                 cudaStream_t stream = nullptr);

// Convenience function (uses Tiled by default)
template<typename T>
void gemm(const T* A, const T* B, T* C, size_t M, size_t N, size_t K,
          T alpha = T(1), T beta = T(0), cudaStream_t stream = nullptr);

// Tensor Core GEMM (half precision input, float accumulation)
#ifdef TC_HAS_WMMA
void launch_gemm_wmma(const half* A, const half* B, float* C,
                      int M, int N, int K, float alpha = 1.0f, float beta = 0.0f,
                      cudaStream_t stream = nullptr);
#endif
```

### Matrix Transpose

```cpp
template<typename T>
void launch_transpose(const T* input, T* output, int rows, int cols,
                      bool use_shared = true, cudaStream_t stream = nullptr);

template<typename T>
void transpose(const T* input, T* output, size_t rows, size_t cols,
               cudaStream_t stream = nullptr);
```

### Example

```cpp
#include "tensorcraft/kernels/gemm.hpp"

using namespace tensorcraft::kernels;

// Basic GEMM
gemm(A, B, C, M, N, K);

// With specific version
launch_gemm(A, B, C, M, N, K, 1.0f, 0.0f, GemmVersion::DoubleBuffer);

// Tensor Core GEMM (if available)
#ifdef TC_HAS_WMMA
launch_gemm_wmma(A_half, B_half, C_float, M, N, K);
#endif
```

---

## attention.hpp

Attention kernels including FlashAttention, RoPE, and MoE router.

### FlashAttention

```cpp
namespace tensorcraft::kernels;

// Launch FlashAttention (currently only supports head_dim == 64)
template<typename T>
void launch_flash_attention(const T* Q, const T* K, const T* V, T* O,
                            int batch_size, int num_heads, int seq_len, int head_dim,
                            float scale, cudaStream_t stream = nullptr);

// Convenience wrapper (computes scale = 1/sqrt(head_dim))
template<typename T>
void flash_attention(const T* Q, const T* K, const T* V, T* O,
                     size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
                     cudaStream_t stream = nullptr);
```

### RoPE (Rotary Position Embedding)

```cpp
// Apply RoPE to input
template<typename T>
void launch_rope(T* x, const float* cos_cache, const float* sin_cache,
                 int batch_size, int seq_len, int num_heads, int head_dim,
                 int start_pos = 0, cudaStream_t stream = nullptr);

// Precompute RoPE cos/sin cache
void precompute_rope_cache(float* cos_cache, float* sin_cache,
                           int max_seq_len, int head_dim, float base = 10000.0f,
                           cudaStream_t stream = nullptr);

// Convenience wrapper
template<typename T>
void rope(T* x, const float* cos_cache, const float* sin_cache,
          size_t batch_size, size_t seq_len, size_t num_heads, size_t head_dim,
          int start_pos = 0, cudaStream_t stream = nullptr);
```

### MoE Router

```cpp
// Top-k expert routing (supports up to 8 experts)
template<typename T>
void launch_moe_router(const T* gate_logits, int* expert_indices, float* expert_weights,
                       int batch_size, int num_experts, int top_k,
                       cudaStream_t stream = nullptr);
```

### Example

```cpp
#include "tensorcraft/kernels/attention.hpp"

using namespace tensorcraft::kernels;

// FlashAttention (requires head_dim == 64)
float scale = 1.0f / sqrtf(64.0f);
launch_flash_attention(Q, K, V, O, batch_size, num_heads, seq_len, 64, scale);

// RoPE
float *cos_cache, *sin_cache;
// Allocate caches...
precompute_rope_cache(cos_cache, sin_cache, max_seq_len, head_dim);
launch_rope(x, cos_cache, sin_cache, batch_size, seq_len, num_heads, head_dim);

// MoE Router
launch_moe_router(logits, indices, weights, num_tokens, num_experts, top_k);
```

---

## conv2d.hpp

2D convolution kernels.

### Standard Conv2D

```cpp
namespace tensorcraft::kernels;

template<typename T>
void conv2d(const T* input, const T* weight, const T* bias, T* output,
            size_t N, size_t C, size_t H, size_t W, size_t K, size_t R, size_t S,
            int stride = 1, int padding = 0, cudaStream_t stream = nullptr);

template<typename T>
void launch_conv2d_naive(const T* input, const T* weight, const T* bias, T* output,
                         int N, int C, int H, int W, int K, int R, int S,
                         int stride_h, int stride_w, int pad_h, int pad_w,
                         cudaStream_t stream = nullptr);
```

### Depthwise Conv2D

```cpp
template<typename T>
void conv2d_depthwise(const T* input, const T* weight, const T* bias, T* output,
                      size_t N, size_t C, size_t H, size_t W, size_t R, size_t S,
                      int stride = 1, int padding = 0, cudaStream_t stream = nullptr);
```

### Pointwise Conv2D (1x1)

```cpp
template<typename T>
void launch_conv2d_pointwise(const T* input, const T* weight, const T* bias, T* output,
                             int N, int C, int H, int W, int K,
                             cudaStream_t stream = nullptr);
```

### Im2Col

```cpp
template<typename T>
void launch_im2col(const T* input, T* col, int N, int C, int H, int W, int R, int S,
                   int stride_h, int stride_w, int pad_h, int pad_w,
                   cudaStream_t stream = nullptr);
```

---

## sparse.hpp

Sparse matrix operations.

### Sparse Matrix Formats

```cpp
namespace tensorcraft::kernels;

template<typename T>
struct CSRMatrix {
    T* values;           // Non-zero values [nnz]
    int* col_indices;    // Column indices [nnz]
    int* row_ptrs;       // Row pointers [rows + 1]
    int rows, cols, nnz;
};

template<typename T>
struct CSCMatrix {
    T* values;           // Non-zero values [nnz]
    int* row_indices;    // Row indices [nnz]
    int* col_ptrs;       // Column pointers [cols + 1]
    int rows, cols, nnz;
};
```

### SpMV (Sparse Matrix-Vector)

```cpp
template<typename T>
void launch_spmv_csr(const T* values, const int* col_indices, const int* row_ptrs,
                     const T* x, T* y, int rows, bool use_vector = true,
                     cudaStream_t stream = nullptr);

template<typename T>
void spmv(const CSRMatrix<T>& A, const T* x, T* y, cudaStream_t stream = nullptr);
```

### SpMM (Sparse Matrix-Matrix)

```cpp
template<typename T>
void launch_spmm_csr(const T* A_values, const int* A_col_indices, const int* A_row_ptrs,
                     const T* B, T* C, int M, int K, int N,
                     cudaStream_t stream = nullptr);

template<typename T>
void spmm(const CSRMatrix<T>& A, const T* B, T* C, int N, cudaStream_t stream = nullptr);
```

### Format Conversion

```cpp
template<typename T>
void launch_csr_to_dense(const T* values, const int* col_indices, const int* row_ptrs,
                         T* dense, int rows, int cols, cudaStream_t stream = nullptr);
```

---

## fusion.hpp

Fused operators and quantization support.

### Fused GEMM

```cpp
namespace tensorcraft::kernels;

// GEMM + Bias + ReLU
template<typename T>
void gemm_bias_relu(const T* A, const T* B, const T* bias, T* C,
                    size_t M, size_t N, size_t K, cudaStream_t stream = nullptr);

// GEMM + Bias + GeLU
template<typename T>
void gemm_bias_gelu(const T* A, const T* B, const T* bias, T* C,
                    size_t M, size_t N, size_t K, cudaStream_t stream = nullptr);

// Generic fused GEMM with custom epilogue
template<typename T, typename Epilogue>
void launch_gemm_fused(const T* A, const T* B, T* C, int M, int N, int K,
                       T alpha, Epilogue epilogue, cudaStream_t stream = nullptr);
```

### Epilogue Functors

```cpp
struct EpilogueIdentity;                    // No-op
template<typename T> struct EpilogueBias;   // Add bias
template<typename T> struct EpilogueBiasReLU;   // Add bias + ReLU
template<typename T> struct EpilogueBiasGeLU;   // Add bias + GeLU
template<typename T> struct EpilogueBiasSiLU;   // Add bias + SiLU
```

### INT8 Quantization

```cpp
struct QuantParams {
    float scale;
    int zero_point;
};

template<typename T>
void quantize_int8(const T* input, int8_t* output, float scale, int zero_point, size_t n,
                   cudaStream_t stream = nullptr);

template<typename T>
void dequantize_int8(const int8_t* input, T* output, float scale, int zero_point, size_t n,
                     cudaStream_t stream = nullptr);
```

### FP8 Quantization (CUDA 12.0+)

```cpp
#ifdef TC_HAS_FP8
template<typename T>
void quantize_fp8_e4m3(const T* input, __nv_fp8_e4m3* output, float scale, size_t n,
                       cudaStream_t stream = nullptr);

template<typename T>
void dequantize_fp8_e4m3(const __nv_fp8_e4m3* input, T* output, float scale, size_t n,
                         cudaStream_t stream = nullptr);
#endif
```

### Example

```cpp
#include "tensorcraft/kernels/fusion.hpp"

using namespace tensorcraft::kernels;

// Fused GEMM
gemm_bias_relu(A, B, bias, C, M, N, K);

// Quantization
quantize_int8(input_fp32, output_int8, scale, zero_point, n);
dequantize_int8(input_int8, output_fp32, scale, zero_point, n);
```
