---
title: TensorCraft-HPC Architecture Design
lang: en
---

# TensorCraft-HPC Architecture Design

This document describes the overall architecture and design decisions of TensorCraft-HPC.

## Design Principles

### 1. Header-Only Design

The core library adopts a pure header-only design with the following benefits:

- **Zero-configuration integration**: Simply `#include` to use
- **Compile-time optimization**: Template code can be fully inlined
- **Cross-platform compatibility**: No pre-compiled libraries needed

```cpp
// Usage example
#include "tensorcraft/kernels/gemm.hpp"
tensorcraft::kernels::gemm(A, B, C, M, N, K);
```

### 2. Progressive Optimization

Each operator provides multiple optimization levels for learning and comparison:

```
Naive → Tiled → Double Buffer → Tensor Core
  ↓        ↓          ↓              ↓
Basic   Shared     Latency       Hardware
        Memory     Hiding       Acceleration
```

### 3. Modern C++ First

Fully leverages C++17/20/23 features:

- **Concepts** (C++20): Type constraints
- **constexpr if** (C++17): Compile-time branching
- **Structured Bindings** (C++17): Multiple return values
- **RAII**: Automatic resource management

### 4. Backward Compatibility

Supports CUDA 11.0-13.1 with conditional compilation for new features:

```cpp
#if TC_CUDA_VERSION >= 12000
    // FP8 support
#endif

#if TC_CUDA_VERSION >= 11080
    // WGMMA support
#endif
```

---

## Module Architecture

```
tensorcraft/
├── core/           # Core utilities layer
├── memory/         # Memory management layer
└── kernels/        # Operator implementation layer
```

### Core Layer

Provides fundamental infrastructure:

```
core/
├── cuda_check.hpp    # Error checking
├── features.hpp      # Feature detection
└── type_traits.hpp   # Type system
```

**cuda_check.hpp**: CUDA error checking macros

```cpp
#define TC_CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(e)); \
    } \
} while(0)
```

**features.hpp**: Compile-time feature detection

```cpp
// C++ version detection
#if __cplusplus >= 202002L
    #define TC_CPP20 1
#endif

// CUDA feature detection
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    #define TC_HAS_WMMA 1
#endif
```

**type_traits.hpp**: Type traits and Concepts

```cpp
template<typename T>
concept Numeric = std::is_arithmetic_v<T> || is_half_v<T>;
```

### Memory Layer

Provides memory abstractions:

```
memory/
├── aligned_vector.hpp  # Vectorized loading
├── tensor.hpp          # Tensor wrapper
└── memory_pool.hpp     # Memory pool
```

**AlignedVector**: Supports vectorized memory access

```cpp
template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
};

// 128-bit load
using Vec4 = AlignedVector<float, 4>;
Vec4 data = *reinterpret_cast<const Vec4*>(&input[idx]);
```

**Tensor**: RAII-style GPU tensor

```cpp
template<typename T>
class Tensor {
    T* data_ = nullptr;
    std::vector<size_t> shape_;
    
public:
    Tensor(const std::vector<size_t>& shape);
    ~Tensor() { if (data_) cudaFree(data_); }
    
    // Move-only
    Tensor(Tensor&&) noexcept;
    Tensor(const Tensor&) = delete;
};
```

**MemoryPool**: Reduces allocation overhead

```cpp
class MemoryPool {
    std::map<size_t, std::vector<void*>> free_blocks_;
    
public:
    void* allocate(size_t bytes);
    void deallocate(void* ptr);
};
```

### Kernels Layer

Operator implementations:

```
kernels/
├── elementwise.hpp    # Element-wise operations
├── softmax.hpp        # Softmax
├── normalization.hpp  # Normalization
├── gemm.hpp           # Matrix multiplication
├── attention.hpp      # Attention mechanism
├── conv2d.hpp         # Convolution
├── sparse.hpp         # Sparse matrices
└── fusion.hpp         # Fusion and quantization
```

---

## Kernel Design Patterns

### 1. Functor Pattern

Using Functors for composable operations:

```cpp
struct ReLU {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        return x > T(0) ? x : T(0);
    }
};

template<typename T, typename Func>
__global__ void elementwise_kernel(const T* in, T* out, size_t n, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = func(in[idx]);
}
```

### 2. Version Selection Pattern

Selecting optimization level via enumeration:

```cpp
enum class GemmVersion { Naive, Tiled, DoubleBuffer, TensorCore, Auto };

template<typename T>
void launch_gemm(const T* A, const T* B, T* C, int M, int N, int K,
                 GemmVersion version) {
    switch (version) {
        case GemmVersion::Naive:
            gemm_naive<<<grid, block>>>(A, B, C, M, N, K);
            break;
        case GemmVersion::Tiled:
            gemm_tiled<<<grid, block, smem>>>(A, B, C, M, N, K);
            break;
        case GemmVersion::TensorCore:
            // Currently uses Tensor Core via dedicated WMMA entry launch_gemm_wmma
            break;
        case GemmVersion::Auto:
            // Default fallback to stable implementation
            break;
    }
}
```

### 3. Epilogue Pattern

Extensible design for GEMM post-processing:

```cpp
struct EpilogueBiasReLU {
    const float* bias;
    
    __device__ __forceinline__ float operator()(float acc, int col) const {
        float result = acc + bias[col];
        return result > 0.0f ? result : 0.0f;
    }
};

template<typename T, typename Epilogue>
__global__ void gemm_with_epilogue(/* ... */, Epilogue epilogue) {
    // ... GEMM computation ...
    C[idx] = epilogue(acc, col);
}
```

### 4. Compile-time Configuration

Using template parameters for compile-time configuration:

```cpp
template<typename T, int TILE_M = 128, int TILE_N = 128, int TILE_K = 32>
__global__ void gemm_optimized(/* ... */) {
    __shared__ T As[TILE_M][TILE_K];
    __shared__ T Bs[TILE_K][TILE_N];
    // ...
}
```

---

## Memory Access Optimization

### Vectorized Loading

```cpp
// Scalar load: 4 memory transactions
float a = input[idx];
float b = input[idx+1];
float c = input[idx+2];
float d = input[idx+3];

// Vectorized load: 1 memory transaction
float4 vec = *reinterpret_cast<const float4*>(&input[idx]);
```

### Coalesced Access

```cpp
// Good: Adjacent threads access adjacent memory
output[threadIdx.x] = input[threadIdx.x];

// Bad: Strided access
output[threadIdx.x * stride] = input[threadIdx.x * stride];
```

### Bank Conflict Avoidance

```cpp
// With bank conflict
__shared__ float tile[32][32];

// Without bank conflict (add padding)
__shared__ float tile[32][33];
```

---

## Numerical Stability

### Softmax Stability

```cpp
// Unstable: exp may overflow
for (int i = 0; i < n; ++i)
    output[i] = exp(input[i]) / sum;

// Stable: subtract maximum value
float max_val = *max_element(input, input + n);
for (int i = 0; i < n; ++i)
    output[i] = exp(input[i] - max_val) / sum;
```

### LayerNorm Stability

```cpp
// Welford's algorithm: Single-pass mean and variance computation
float mean = 0.0f, M2 = 0.0f;
for (int i = 0; i < n; ++i) {
    float delta = x[i] - mean;
    mean += delta / (i + 1);
    M2 += delta * (x[i] - mean);
}
float var = M2 / n;
```

---

## Extension Guide

### Adding New Operators

1. Create header file in `include/tensorcraft/kernels/`
2. Implement kernel function and launcher
3. Add tests to `tests/`
4. Add benchmarks to `benchmarks/`
5. Update documentation

### Adding New Optimization Levels

1. Add new kernel implementation in existing header
2. Update version enumeration
3. Add branch in launcher
4. Add comparative tests

### Supporting New Data Types

1. Add type detection in `type_traits.hpp`
2. Add feature detection in `features.hpp`
3. Specialize related kernel templates
4. Add type conversion utilities

---

## Performance Considerations

### Occupancy vs Registers

```cpp
// High occupancy: More parallelism
__launch_bounds__(256, 4)  // 256 threads, 4 blocks/SM

// More registers: Less spilling
__launch_bounds__(128, 2)  // 128 threads, 2 blocks/SM
```

### Shared Memory vs L1 Cache

```cpp
// Prefer shared memory
cudaFuncSetAttribute(kernel, 
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

// Prefer L1 Cache
cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout, 0);
```

### Asynchronous Execution

```cpp
// Use multiple streams for overlapped execution
cudaStream_t streams[4];
for (int i = 0; i < 4; ++i) {
    cudaStreamCreate(&streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(data[i]);
}
```
