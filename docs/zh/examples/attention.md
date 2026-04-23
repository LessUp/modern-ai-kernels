---
title: Attention Examples
lang: zh
---

# Attention Examples

This guide demonstrates FlashAttention, RoPE (Rotary Position Embedding), and MoE Router usage.

## Overview

| Component | Description |
|-----------|-------------|
| FlashAttention | Memory-efficient attention computation |
| RoPE | Rotary position embeddings for transformers |
| MoE Router | Top-k expert routing for mixture-of-experts |

## FlashAttention

### Basic Usage

```cpp
#include "tensorcraft/kernels/attention.hpp"
#include "tensorcraft/core/cuda_check.hpp"
#include <vector>

int main() {
    using namespace tensorcraft::kernels;
    
    // Parameters
    const int batch_size = 4;
    const int num_heads = 8;
    const int seq_len = 128;
    const int head_dim = 64;  // Currently only head_dim=64 is supported
    
    const int total_elements = batch_size * num_heads * seq_len * head_dim;
    
    // Allocate host memory
    std::vector<float> h_Q(total_elements);
    std::vector<float> h_K(total_elements);
    std::vector<float> h_V(total_elements);
    std::vector<float> h_O(total_elements);
    
    // Initialize with random values...
    for (int i = 0; i < total_elements; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O;
    TC_CUDA_CHECK(cudaMalloc(&d_Q, total_elements * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_K, total_elements * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_V, total_elements * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_O, total_elements * sizeof(float)));
    
    // Copy to device
    TC_CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute scale factor
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Run FlashAttention
    launch_flash_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim, scale);
    
    // Copy result back
    TC_CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, total_elements * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    
    return 0;
}
```

### Using Tensor Wrapper

```cpp
#include "tensorcraft/memory/tensor.hpp"
#include "tensorcraft/kernels/attention.hpp"

void run_attention() {
    using namespace tensorcraft;
    
    const size_t batch = 4, heads = 8, seq = 128, dim = 64;
    const size_t total = batch * heads * seq * dim;
    
    // Create tensors
    FloatTensor Q({batch, heads, seq, dim});
    FloatTensor K({batch, heads, seq, dim});
    FloatTensor V({batch, heads, seq, dim});
    FloatTensor O({batch, heads, seq, dim});
    
    // Initialize (example: random values)
    std::vector<float> h_data(total);
    for (auto& x : h_data) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    
    Q.copy_from_host(h_data);
    K.copy_from_host(h_data);
    V.copy_from_host(h_data);
    
    // Run attention
    kernels::flash_attention(Q.data(), K.data(), V.data(), O.data(),
                             batch, heads, seq, dim);
    
    // Get output
    auto result = O.to_host();
}
```

### Limitations

- Currently only supports `head_dim == 64`
- Input layout: `[batch, heads, seq_len, head_dim]`

---

## RoPE (Rotary Position Embedding)

### Basic Usage

```cpp
#include "tensorcraft/kernels/attention.hpp"
#include <cmath>

int main() {
    using namespace tensorcraft::kernels;
    
    const int batch_size = 4;
    const int seq_len = 128;
    const int num_heads = 8;
    const int head_dim = 64;  // Must be even
    
    // Allocate input/output
    int total = batch_size * seq_len * num_heads * head_dim;
    float *d_x;
    TC_CUDA_CHECK(cudaMalloc(&d_x, total * sizeof(float)));
    
    // Precompute RoPE cache
    float *d_cos_cache, *d_sin_cache;
    int cache_size = seq_len * (head_dim / 2);
    TC_CUDA_CHECK(cudaMalloc(&d_cos_cache, cache_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_sin_cache, cache_size * sizeof(float)));
    
    precompute_rope_cache(d_cos_cache, d_sin_cache, seq_len, head_dim);
    
    // Apply RoPE (in-place modification)
    launch_rope(d_x, d_cos_cache, d_sin_cache, 
                batch_size, seq_len, num_heads, head_dim, 
                0);  // start_pos = 0
    
    // For continuation (e.g., generation), use start_pos
    // launch_rope(d_x, d_cos_cache, d_sin_cache, 
    //             batch_size, new_seq_len, num_heads, head_dim,
    //             previous_seq_len);  // start_pos = previous length
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_cos_cache);
    cudaFree(d_sin_cache);
    
    return 0;
}
```

### Custom RoPE Base

```cpp
// Use different base frequency (default is 10000.0f)
precompute_rope_cache(d_cos_cache, d_sin_cache, max_seq_len, head_dim, 500000.0f);
```

---

## MoE Router

### Basic Usage

```cpp
#include "tensorcraft/kernels/attention.hpp"
#include <vector>

int main() {
    using namespace tensorcraft::kernels;
    
    const int num_tokens = 256;
    const int num_experts = 8;   // Maximum supported: 8
    const int top_k = 2;         // Select top-2 experts
    
    // Gate logits: [num_tokens, num_experts]
    std::vector<float> h_logits(num_tokens * num_experts);
    for (auto& x : h_logits) x = static_cast<float>(rand()) / RAND_MAX;
    
    // Allocate device memory
    float *d_logits;
    int *d_indices;       // [num_tokens, top_k]
    float *d_weights;     // [num_tokens, top_k]
    
    TC_CUDA_CHECK(cudaMalloc(&d_logits, num_tokens * num_experts * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_indices, num_tokens * top_k * sizeof(int)));
    TC_CUDA_CHECK(cudaMalloc(&d_weights, num_tokens * top_k * sizeof(float)));
    
    // Copy logits to device
    TC_CUDA_CHECK(cudaMemcpy(d_logits, h_logits.data(), 
                             num_tokens * num_experts * sizeof(float), 
                             cudaMemcpyHostToDevice));
    
    // Run MoE router
    launch_moe_router(d_logits, d_indices, d_weights, 
                      num_tokens, num_experts, top_k);
    
    // Copy results back
    std::vector<int> h_indices(num_tokens * top_k);
    std::vector<float> h_weights(num_tokens * top_k);
    
    TC_CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, 
                             num_tokens * top_k * sizeof(int), 
                             cudaMemcpyDeviceToHost));
    TC_CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights,
                             num_tokens * top_k * sizeof(float),
                             cudaMemcpyDeviceToHost));
    
    // Verify weights sum to ~1.0
    for (int t = 0; t < num_tokens; ++t) {
        float sum = h_weights[t * top_k] + h_weights[t * top_k + 1];
        std::cout << "Token " << t << " weight sum: " << sum << std::endl;
    }
    
    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_indices);
    cudaFree(d_weights);
    
    return 0;
}
```

### Limitations

- Maximum `num_experts`: 8
- `top_k` must be in range `[1, num_experts]`

---

## Complete Attention Layer Example

```cpp
#include "tensorcraft/kernels/attention.hpp"
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

// Simplified attention layer
void attention_layer(
    const float* input,      // [batch, seq, hidden]
    const float* Wq,         // [hidden, hidden]
    const float* Wk,
    const float* Wv,
    const float* Wo,
    const float* cos_cache,
    const float* sin_cache,
    float* output,
    int batch, int seq, int heads, int dim, int hidden) {
    
    using namespace tensorcraft::kernels;
    
    int total = batch * seq * hidden;
    
    // Allocate temporary buffers
    float *d_Q, *d_K, *d_V, *d_attn_out;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_attn_out, total * sizeof(float));
    
    // Project Q, K, V
    gemm(input, Wq, d_Q, batch * seq, hidden, hidden);
    gemm(input, Wk, d_K, batch * seq, hidden, hidden);
    gemm(input, Wv, d_V, batch * seq, hidden, hidden);
    
    // Apply RoPE to Q and K
    launch_rope(d_Q, cos_cache, sin_cache, batch, seq, heads, dim, 0);
    launch_rope(d_K, cos_cache, sin_cache, batch, seq, heads, dim, 0);
    
    // Compute attention
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    launch_flash_attention(d_Q, d_K, d_V, d_attn_out, batch, heads, seq, dim, scale);
    
    // Output projection
    gemm(d_attn_out, Wo, output, batch * seq, hidden, hidden);
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_attn_out);
}
```
