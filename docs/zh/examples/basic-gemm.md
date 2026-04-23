---
title: GEMM Examples
lang: zh
---

# GEMM Examples

This guide walks through GEMM optimization from naive to Tensor Core implementations.

## Overview

TensorCraft-HPC provides four GEMM implementations:

| Version | Description | Performance |
|---------|-------------|-------------|
| `Naive` | Basic row-column dot product | Baseline |
| `Tiled` | Shared memory tiling | ~10x faster |
| `DoubleBuffer` | Overlapped memory/compute | ~15% faster than Tiled |
| `TensorCore` | WMMA hardware acceleration | ~2-4x faster than Tiled |

## Basic Usage

### C++ Example

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/core/cuda_check.hpp"
#include <vector>
#include <iostream>

int main() {
    const int M = 256, N = 512, K = 128;
    
    // Allocate host memory
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C(M * N, 0.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    TC_CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    TC_CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy to device
    TC_CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    TC_CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run GEMM with different versions
    using namespace tensorcraft::kernels;
    
    // Tiled version (recommended for most cases)
    launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::Tiled);
    
    // Or use convenience function
    gemm(d_A, d_B, d_C, M, N, K);
    
    // Copy result back
    TC_CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify result
    std::cout << "C[0] = " << h_C[0] << " (expected: " << K * 2.0f << ")" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
```

### Python Example

```python
import tensorcraft_ops as tc
import numpy as np

# Create matrices
M, N, K = 256, 512, 128
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

# Run GEMM with different versions
C_tiled = tc.gemm(A, B, version='tiled')
C_double = tc.gemm(A, B, version='double_buffer')
C_naive = tc.gemm(A, B, version='naive')

# Compare results
print(f"Tiled vs Double buffer diff: {np.abs(C_tiled - C_double).max()}")
print(f"Tiled vs Naive diff: {np.abs(C_tiled - C_naive).max()}")
```

## Using Tensor RAII Wrapper

```cpp
#include "tensorcraft/memory/tensor.hpp"
#include "tensorcraft/kernels/gemm.hpp"

int main() {
    using namespace tensorcraft;
    
    const size_t M = 256, N = 512, K = 128;
    
    // Create tensors with RAII memory management
    FloatTensor A({M, K});
    FloatTensor B({K, N});
    FloatTensor C({M, N});
    
    // Initialize from host
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    A.copy_from_host(h_A);
    B.copy_from_host(h_B);
    C.zero();
    
    // Run GEMM
    kernels::gemm(A.data(), B.data(), C.data(), M, N, K);
    
    // Get result
    auto h_C = C.to_host();
    
    std::cout << "Result: " << h_C[0] << std::endl;
    
    return 0;
}
```

## Version Comparison

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include <chrono>

void benchmark_gemm(int M, int N, int K, int iterations = 100) {
    // Allocate memory...
    
    auto benchmark = [&](auto version, const char* name) {
        // Warmup
        for (int i = 0; i < 5; ++i) {
            tensorcraft::kernels::launch_gemm(d_A, d_B, d_C, M, N, K, 
                                              1.0f, 0.0f, version);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            tensorcraft::kernels::launch_gemm(d_A, d_B, d_C, M, N, K,
                                              1.0f, 0.0f, version);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        std::cout << name << ": " << ms << " ms, " << gflops << " GFLOPS\n";
    };
    
    using tensorcraft::kernels::GemmVersion;
    benchmark(GemmVersion::Naive, "Naive");
    benchmark(GemmVersion::Tiled, "Tiled");
    benchmark(GemmVersion::DoubleBuffer, "DoubleBuffer");
}
```

## Tensor Core GEMM

For Tensor Core support, use the dedicated `launch_gemm_wmma` function:

```cpp
#include "tensorcraft/kernels/gemm.hpp"

// Requires half-precision input
__half *d_A_half, *d_B_half;
float *d_C_float;

// Allocate and initialize half-precision data...
// Note: A and B must be half, C is float for accumulation

// Run Tensor Core GEMM
#ifdef TC_HAS_WMMA
tensorcraft::kernels::launch_gemm_wmma(d_A_half, d_B_half, d_C_float,
                                        M, N, K, 1.0f, 0.0f);
#else
std::cout << "Tensor Core not available on this device" << std::endl;
#endif
```

## Matrix Transpose

```cpp
#include "tensorcraft/kernels/gemm.hpp"

// Transpose with shared memory optimization
float *d_input, *d_output;
int rows = 256, cols = 512;

// Allocate and initialize...

// Optimized transpose (recommended)
tensorcraft::kernels::transpose(d_input, d_output, rows, cols);

// Or explicitly control optimization
tensorcraft::kernels::launch_transpose(d_input, d_output, rows, cols, true);  // with shared memory
tensorcraft::kernels::launch_transpose(d_input, d_output, rows, cols, false); // naive
```

## Non-Square Matrices

```cpp
#include "tensorcraft/kernels/gemm.hpp"

// GEMM works with non-square matrices
int M = 128, N = 512, K = 64;  // A: 128x64, B: 64x512, C: 128x512

float *d_A, *d_B, *d_C;
// Allocate...

tensorcraft::kernels::gemm(d_A, d_B, d_C, M, N, K);
```

## Streaming Multiple GEMMs

```cpp
#include "tensorcraft/kernels/gemm.hpp"

// Use CUDA streams for overlapping GEMMs
cudaStream_t streams[4];
for (int i = 0; i < 4; ++i) {
    cudaStreamCreate(&streams[i]);
}

// Launch multiple GEMMs concurrently
for (int i = 0; i < 4; ++i) {
    tensorcraft::kernels::gemm(d_A[i], d_B[i], d_C[i], M, N, K, 
                               1.0f, 0.0f, streams[i]);
}

// Synchronize all streams
for (int i = 0; i < 4; ++i) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
}
```

## Common Pitfalls

1. **Wrong dimensions**: Ensure A is M×K and B is K×N
2. **Memory alignment**: For best performance, ensure 256-byte alignment
3. **Large K**: Use tiled/double-buffer versions for better memory access
4. **Small matrices**: Naive may be faster for very small matrices due to lower overhead
