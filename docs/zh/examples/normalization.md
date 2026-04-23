---
title: Normalization Examples
lang: zh
---

# Normalization Examples

This guide demonstrates LayerNorm, RMSNorm, and BatchNorm usage.

## Overview

| Operation | Formula | Use Case |
|-----------|---------|----------|
| LayerNorm | `y = γ * (x - μ) / √(σ² + ε) + β` | Transformer layers |
| RMSNorm | `y = x / RMS(x) * w` | LLaMA-style models |
| BatchNorm | `y = γ * (x - μ) / √(σ² + ε) + β` | CNN inference |

---

## LayerNorm

### Basic Usage

```cpp
#include "tensorcraft/kernels/normalization.hpp"
#include "tensorcraft/core/cuda_check.hpp"
#include <vector>

int main() {
    using namespace tensorcraft::kernels;
    
    const int batch_size = 32;
    const int hidden_size = 256;
    
    // Allocate host memory
    std::vector<float> h_input(batch_size * hidden_size);
    std::vector<float> h_gamma(hidden_size, 1.0f);  // Scale
    std::vector<float> h_beta(hidden_size, 0.0f);   // Shift
    std::vector<float> h_output(batch_size * hidden_size);
    
    // Initialize input with random values
    for (auto& x : h_input) {
        x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    
    // Allocate device memory
    float *d_input, *d_gamma, *d_beta, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_gamma, hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_beta, hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, batch_size * hidden_size * sizeof(float)));
    
    // Copy to device
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run LayerNorm
    layernorm(d_input, d_gamma, d_beta, d_output, batch_size, hidden_size);
    
    // Copy result back
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify: each row should have mean ≈ 0 and std ≈ 1
    for (int b = 0; b < 2; ++b) {  // Check first 2 batches
        float mean = 0.0f, var = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            mean += h_output[b * hidden_size + h];
        }
        mean /= hidden_size;
        for (int h = 0; h < hidden_size; ++h) {
            var += (h_output[b * hidden_size + h] - mean) * (h_output[b * hidden_size + h] - mean);
        }
        var /= hidden_size;
        std::cout << "Batch " << b << ": mean=" << mean << ", std=" << std::sqrt(var) << std::endl;
    }
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);
    
    return 0;
}
```

### Multi-dimensional Input

```cpp
#include "tensorcraft/kernels/normalization.hpp"

// LayerNorm on 3D input [batch, seq, hidden]
void layernorm_3d() {
    using namespace tensorcraft::kernels;
    
    const int batch = 4;
    const int seq = 128;
    const int hidden = 256;
    
    const int total = batch * seq * hidden;
    
    float *d_input, *d_gamma, *d_beta, *d_output;
    // Allocate...
    
    // Treat as 2D: (batch * seq) x hidden
    layernorm(d_input, d_gamma, d_beta, d_output, batch * seq, hidden);
}
```

### Custom Epsilon

```cpp
#include "tensorcraft/kernels/normalization.hpp"

// Use custom epsilon for numerical stability
tensorcraft::kernels::layernorm(d_input, d_gamma, d_beta, d_output, 
                                batch_size, hidden_size, 1e-6f);  // Default: 1e-5f
```

---

## RMSNorm

### Basic Usage

```cpp
#include "tensorcraft/kernels/normalization.hpp"
#include <vector>

int main() {
    using namespace tensorcraft::kernels;
    
    const int batch_size = 32;
    const int hidden_size = 256;
    
    // Allocate host memory
    std::vector<float> h_input(batch_size * hidden_size);
    std::vector<float> h_weight(hidden_size, 1.0f);  // Scale weights
    std::vector<float> h_output(batch_size * hidden_size);
    
    // Initialize input
    for (auto& x : h_input) {
        x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    
    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_weight, hidden_size * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, batch_size * hidden_size * sizeof(float)));
    
    // Copy to device
    TC_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run RMSNorm
    rmsnorm(d_input, d_weight, d_output, batch_size, hidden_size);
    
    // Copy result back
    TC_CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    
    return 0;
}
```

### RMSNorm vs LayerNorm

```cpp
// RMSNorm is simpler and faster than LayerNorm
// - No mean subtraction
// - No bias parameter
// - Used in LLaMA, Gopher, etc.

// LayerNorm: y = γ * (x - μ) / √(σ² + ε) + β
// RMSNorm:   y = x / RMS(x) * w, where RMS(x) = √(mean(x²) + ε)
```

---

## BatchNorm

### Inference Mode

```cpp
#include "tensorcraft/kernels/normalization.hpp"
#include <vector>

int main() {
    using namespace tensorcraft::kernels;
    
    // Image dimensions: NCHW format
    const int N = 8;    // Batch size
    const int C = 64;   // Channels
    const int H = 32;   // Height
    const int W = 32;   // Width
    
    const int total = N * C * H * W;
    
    // Allocate host memory
    std::vector<float> h_input(total);
    std::vector<float> h_gamma(C, 1.0f);          // Scale per channel
    std::vector<float> h_beta(C, 0.0f);           // Shift per channel
    std::vector<float> h_running_mean(C, 0.0f);   // Running mean
    std::vector<float> h_running_var(C, 1.0f);    // Running variance
    std::vector<float> h_output(total);
    
    // Initialize input
    for (auto& x : h_input) {
        x = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_gamma, *d_beta, *d_running_mean, *d_running_var, *d_output;
    TC_CUDA_CHECK(cudaMalloc(&d_input, total * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_beta, C * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_running_mean, C * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_running_var, C * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_output, total * sizeof(float)));
    
    // Copy to device...
    
    // Run BatchNorm (inference mode)
    launch_batchnorm(d_input, d_gamma, d_beta, d_running_mean, d_running_var, d_output,
                     N, C, H, W, 1e-5f, false);  // fuse_relu = false
    
    // Copy result back...
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_running_mean);
    cudaFree(d_running_var);
    cudaFree(d_output);
    
    return 0;
}
```

### Fused BatchNorm + ReLU

```cpp
#include "tensorcraft/kernels/normalization.hpp"

// BatchNorm followed by ReLU (fused for better performance)
tensorcraft::kernels::launch_batchnorm(d_input, d_gamma, d_beta, 
                                       d_running_mean, d_running_var, d_output,
                                       N, C, H, W, 1e-5f, 
                                       true);  // fuse_relu = true
```

---

## Using Tensor Wrapper

```cpp
#include "tensorcraft/memory/tensor.hpp"
#include "tensorcraft/kernels/normalization.hpp"

void normalization_with_tensor() {
    using namespace tensorcraft;
    
    const size_t batch = 32;
    const size_t hidden = 256;
    
    // Create tensors
    FloatTensor input({batch, hidden});
    FloatTensor gamma({hidden});
    FloatTensor beta({hidden});
    FloatTensor output({batch, hidden});
    
    // Initialize
    std::vector<float> h_data(batch * hidden);
    for (auto& x : h_data) x = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    
    input.copy_from_host(h_data);
    gamma.fill(1.0f);
    beta.zero();
    
    // Run LayerNorm
    kernels::layernorm(input.data(), gamma.data(), beta.data(), output.data(),
                       batch, hidden);
    
    // Get result
    auto result = output.to_host();
}
```

---

## Performance Tips

1. **Batch multiple operations**: Process multiple sequences together

   ```cpp
   // Instead of: layernorm for each sequence
   // Do: layernorm for batch * sequences at once
   layernorm(d_input, d_gamma, d_beta, d_output, batch * seq, hidden);
   ```

2. **Use RMSNorm when possible**: Simpler computation, faster

3. **Fused operations**: Use `fuse_relu` in BatchNorm when applicable

4. **Memory layout**: Ensure contiguous memory for best performance
