# Design Document: TensorCraft-HPC

## Overview

TensorCraft-HPC 是一个模块化的高性能 AI 算子优化库，采用分层架构设计，支持从基础 CUDA 11.0 到最新 CUDA 13.1 的多版本兼容。项目核心设计原则：

1. **渐进式优化**: 每个算子提供从 naive 到极致优化的多个版本
2. **编译期特性检测**: 通过宏和模板在编译期选择最优实现路径
3. **零成本抽象**: 使用现代 C++ 模板技术实现高性能泛型编程
4. **教学友好**: 代码结构清晰，注释详尽，便于学习理解

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python Bindings (pybind11)                 │
├─────────────────────────────────────────────────────────────────┤
│                         Kernel Launchers                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │Elementwise│ │Reduction │ │  GEMM    │ │Attention │ │  Conv  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                        Core Utilities                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Tensor  │ │  Memory  │ │   Math   │ │  Config  │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                     CUDA Runtime / Driver API                    │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Core Utilities (include/tensorcraft/core/)

#### 1.1 CUDA Error Handling

```cpp
// include/tensorcraft/core/cuda_check.hpp
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tensorcraft {

#define TC_CUDA_CHECK(call)                                           \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            throw std::runtime_error(                                 \
                std::string(__FILE__) + ":" + std::to_string(__LINE__)\
                + " CUDA error: " + cudaGetErrorString(err));         \
        }                                                             \
    } while (0)

#define TC_CUDA_CHECK_LAST() TC_CUDA_CHECK(cudaGetLastError())

} // namespace tensorcraft
```

#### 1.2 Feature Detection

```cpp
// include/tensorcraft/core/features.hpp
#pragma once

// C++ version detection
#if __cplusplus >= 202302L
    #define TC_CPP23 1
#endif
#if __cplusplus >= 202002L
    #define TC_CPP20 1
#endif
#if __cplusplus >= 201703L
    #define TC_CPP17 1
#endif

// CUDA version detection
#if defined(__CUDACC__)
    #if __CUDACC_VER_MAJOR__ >= 13
        #define TC_CUDA_13 1
    #endif
    #if __CUDACC_VER_MAJOR__ >= 12
        #define TC_CUDA_12 1
        #define TC_HAS_TMA 1
        #define TC_HAS_WGMMA 1
        #define TC_HAS_FP8 1
    #endif
    #if __CUDACC_VER_MAJOR__ >= 11
        #define TC_CUDA_11 1
        #define TC_HAS_WMMA 1
    #endif
#endif

// Architecture detection
#if defined(__CUDA_ARCH__)
    #if __CUDA_ARCH__ >= 900
        #define TC_ARCH_HOPPER 1
    #endif
    #if __CUDA_ARCH__ >= 800
        #define TC_ARCH_AMPERE 1
    #endif
    #if __CUDA_ARCH__ >= 700
        #define TC_ARCH_VOLTA 1
    #endif
#endif
```

#### 1.3 Type Traits and Concepts

```cpp
// include/tensorcraft/core/type_traits.hpp
#pragma once
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tensorcraft {

// Half precision type traits
template<typename T>
struct is_half : std::false_type {};

template<>
struct is_half<__half> : std::true_type {};

template<>
struct is_half<__nv_bfloat16> : std::true_type {};

template<typename T>
inline constexpr bool is_half_v = is_half<T>::value;

// Numeric concept (C++20) or SFINAE fallback (C++17)
#ifdef TC_CPP20
template<typename T>
concept Numeric = std::is_arithmetic_v<T> || is_half_v<T>;

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T> || is_half_v<T>;
#else
// C++17 SFINAE version
template<typename T, typename = void>
struct is_numeric : std::false_type {};

template<typename T>
struct is_numeric<T, std::enable_if_t<
    std::is_arithmetic_v<T> || is_half_v<T>>> : std::true_type {};

template<typename T>
inline constexpr bool is_numeric_v = is_numeric<T>::value;
#endif

} // namespace tensorcraft
```

### 2. Memory Management (include/tensorcraft/memory/)

#### 2.1 Aligned Vector for Vectorized Access

```cpp
// include/tensorcraft/memory/aligned_vector.hpp
#pragma once

namespace tensorcraft {

template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
    
    __host__ __device__ T& operator[](int i) { return val[i]; }
    __host__ __device__ const T& operator[](int i) const { return val[i]; }
};

// Common type aliases
template<typename T>
using Vec2 = AlignedVector<T, 2>;

template<typename T>
using Vec4 = AlignedVector<T, 4>;

template<typename T>
using Vec8 = AlignedVector<T, 8>;

} // namespace tensorcraft
```

#### 2.2 Tensor Wrapper

```cpp
// include/tensorcraft/memory/tensor.hpp
#pragma once
#include "tensorcraft/core/cuda_check.hpp"
#include <memory>
#include <vector>

namespace tensorcraft {

template<typename T>
class Tensor {
public:
    Tensor() = default;
    
    explicit Tensor(const std::vector<size_t>& shape) 
        : shape_(shape), size_(compute_size(shape)) {
        TC_CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(T)));
    }
    
    ~Tensor() {
        if (data_) {
            cudaFree(data_);
        }
    }
    
    // Move semantics
    Tensor(Tensor&& other) noexcept 
        : data_(other.data_), shape_(std::move(other.shape_)), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Disable copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    const std::vector<size_t>& shape() const { return shape_; }
    
    void copy_from_host(const T* host_data) {
        TC_CUDA_CHECK(cudaMemcpy(data_, host_data, size_ * sizeof(T), 
                                  cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* host_data) const {
        TC_CUDA_CHECK(cudaMemcpy(host_data, data_, size_ * sizeof(T), 
                                  cudaMemcpyDeviceToHost));
    }

private:
    static size_t compute_size(const std::vector<size_t>& shape) {
        size_t s = 1;
        for (auto d : shape) s *= d;
        return s;
    }
    
    T* data_ = nullptr;
    std::vector<size_t> shape_;
    size_t size_ = 0;
};

} // namespace tensorcraft
```

#### 2.3 Memory Pool

```cpp
// include/tensorcraft/memory/memory_pool.hpp
#pragma once
#include "tensorcraft/core/cuda_check.hpp"
#include <unordered_map>
#include <vector>
#include <mutex>

namespace tensorcraft {

class MemoryPool {
public:
    static MemoryPool& instance() {
        static MemoryPool pool;
        return pool;
    }
    
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Round up to alignment
        size = (size + 255) & ~255;
        
        auto& free_list = free_blocks_[size];
        if (!free_list.empty()) {
            void* ptr = free_list.back();
            free_list.pop_back();
            return ptr;
        }
        
        void* ptr;
        TC_CUDA_CHECK(cudaMalloc(&ptr, size));
        allocated_sizes_[ptr] = size;
        return ptr;
    }
    
    void deallocate(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocated_sizes_.find(ptr);
        if (it != allocated_sizes_.end()) {
            free_blocks_[it->second].push_back(ptr);
        }
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [size, blocks] : free_blocks_) {
            for (void* ptr : blocks) {
                cudaFree(ptr);
            }
        }
        free_blocks_.clear();
        allocated_sizes_.clear();
    }
    
    ~MemoryPool() { clear(); }

private:
    MemoryPool() = default;
    std::mutex mutex_;
    std::unordered_map<size_t, std::vector<void*>> free_blocks_;
    std::unordered_map<void*, size_t> allocated_sizes_;
};

} // namespace tensorcraft
```

### 3. Kernel Implementations

#### 3.1 Elementwise Kernels

```cpp
// include/tensorcraft/kernels/elementwise.hpp
#pragma once
#include "tensorcraft/core/type_traits.hpp"
#include "tensorcraft/memory/aligned_vector.hpp"

namespace tensorcraft {
namespace kernels {

// Generic elementwise kernel with functor
template<typename T, typename Func, int VecSize = 4>
__global__ void elementwise_kernel(const T* __restrict__ input,
                                    T* __restrict__ output,
                                    size_t n,
                                    Func func) {
    using VecT = AlignedVector<T, VecSize>;
    
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize;
    size_t stride = blockDim.x * gridDim.x * VecSize;
    
    for (size_t i = idx; i < n; i += stride) {
        if (i + VecSize <= n) {
            VecT in_vec = *reinterpret_cast<const VecT*>(&input[i]);
            VecT out_vec;
            
            #pragma unroll
            for (int k = 0; k < VecSize; ++k) {
                out_vec[k] = func(in_vec[k]);
            }
            
            *reinterpret_cast<VecT*>(&output[i]) = out_vec;
        } else {
            // Handle tail elements
            for (size_t j = i; j < n; ++j) {
                output[j] = func(input[j]);
            }
        }
    }
}

// Activation function functors
struct ReLU {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        return x > T(0) ? x : T(0);
    }
};

struct SiLU {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        return x / (T(1) + exp(-x));
    }
};

struct GeLU {
    template<typename T>
    __device__ __forceinline__ T operator()(T x) const {
        constexpr T c = T(0.7978845608028654);  // sqrt(2/pi)
        constexpr T k = T(0.044715);
        T inner = c * (x + k * x * x * x);
        return T(0.5) * x * (T(1) + tanh(inner));
    }
};

template<typename T>
struct LeakyReLU {
    T alpha;
    
    __device__ __forceinline__ T operator()(T x) const {
        return x > T(0) ? x : alpha * x;
    }
};

// Launcher functions
template<typename T, typename Func>
void launch_elementwise(const T* input, T* output, size_t n, 
                        Func func, cudaStream_t stream = nullptr) {
    constexpr int block_size = 256;
    constexpr int vec_size = 4;
    int grid_size = (n + block_size * vec_size - 1) / (block_size * vec_size);
    grid_size = std::min(grid_size, 65535);
    
    elementwise_kernel<T, Func, vec_size><<<grid_size, block_size, 0, stream>>>(
        input, output, n, func);
}

} // namespace kernels
} // namespace tensorcraft
```

#### 3.2 Reduction Kernels (Softmax)

```cpp
// include/tensorcraft/kernels/softmax.hpp
#pragma once
#include "tensorcraft/core/cuda_check.hpp"

namespace tensorcraft {
namespace kernels {

// Online softmax using warp shuffle
template<typename T, int BLOCK_SIZE = 256>
__global__ void softmax_kernel(const T* __restrict__ input,
                                T* __restrict__ output,
                                int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const T* row_input = input + row * cols;
    T* row_output = output + row * cols;
    
    // Phase 1: Find max (online)
    T thread_max = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        thread_max = max(thread_max, row_input[i]);
    }
    
    // Warp reduce max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    // Block reduce max using shared memory
    __shared__ T shared_max[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane == 0) shared_max[warp_id] = thread_max;
    __syncthreads();
    
    if (warp_id == 0) {
        thread_max = (lane < (BLOCK_SIZE / 32)) ? shared_max[lane] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
        }
    }
    
    __shared__ T row_max;
    if (threadIdx.x == 0) row_max = thread_max;
    __syncthreads();
    
    // Phase 2: Compute exp and sum
    T thread_sum = T(0);
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        T val = exp(row_input[i] - row_max);
        row_output[i] = val;  // Store intermediate
        thread_sum += val;
    }
    
    // Warp reduce sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    __shared__ T shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = thread_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        thread_sum = (lane < (BLOCK_SIZE / 32)) ? shared_sum[lane] : T(0);
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
    }
    
    __shared__ T row_sum;
    if (threadIdx.x == 0) row_sum = thread_sum;
    __syncthreads();
    
    // Phase 3: Normalize
    T inv_sum = T(1) / row_sum;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        row_output[i] *= inv_sum;
    }
}

template<typename T>
void launch_softmax(const T* input, T* output, int rows, int cols,
                    cudaStream_t stream = nullptr) {
    softmax_kernel<T, 256><<<rows, 256, 0, stream>>>(input, output, rows, cols);
    TC_CUDA_CHECK_LAST();
}

} // namespace kernels
} // namespace tensorcraft
```


#### 3.3 GEMM Kernels

```cpp
// include/tensorcraft/kernels/gemm.hpp
#pragma once
#include "tensorcraft/core/features.hpp"
#include "tensorcraft/core/cuda_check.hpp"

namespace tensorcraft {
namespace kernels {

// ============================================================================
// GEMM v1: Naive Implementation
// ============================================================================
template<typename T>
__global__ void gemm_naive(const T* __restrict__ A,
                           const T* __restrict__ B,
                           T* __restrict__ C,
                           int M, int N, int K,
                           T alpha, T beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = T(0);
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ============================================================================
// GEMM v2: Shared Memory Tiling
// ============================================================================
template<typename T, int TILE_SIZE = 32>
__global__ void gemm_tiled(const T* __restrict__ A,
                           const T* __restrict__ B,
                           T* __restrict__ C,
                           int M, int N, int K,
                           T alpha, T beta) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    T sum = T(0);
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;
        
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : T(0);
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : T(0);
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ============================================================================
// GEMM v3: Double Buffering
// ============================================================================
template<typename T, int TILE_SIZE = 32>
__global__ void gemm_double_buffer(const T* __restrict__ A,
                                    const T* __restrict__ B,
                                    T* __restrict__ C,
                                    int M, int N, int K,
                                    T alpha, T beta) {
    __shared__ T As[2][TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[2][TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    T sum = T(0);
    
    // Prefetch first tile
    int a_col = tx;
    int b_row = ty;
    As[0][ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : T(0);
    Bs[0][ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : T(0);
    __syncthreads();
    
    for (int t = 0; t < num_tiles; ++t) {
        int curr = t % 2;
        int next = (t + 1) % 2;
        
        // Prefetch next tile while computing current
        if (t + 1 < num_tiles) {
            a_col = (t + 1) * TILE_SIZE + tx;
            b_row = (t + 1) * TILE_SIZE + ty;
            As[next][ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : T(0);
            Bs[next][ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : T(0);
        }
        
        // Compute with current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[curr][ty][k] * Bs[curr][k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// ============================================================================
// GEMM v4: Tensor Core (WMMA) - CUDA 11.0+
// ============================================================================
#ifdef TC_HAS_WMMA
#include <mma.h>

template<int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
__global__ void gemm_wmma(const half* __restrict__ A,
                          const half* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
    using namespace nvcuda::wmma;
    
    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    fill_fragment(c_frag, 0.0f);
    
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32 * WMMA_N;
    
    // Accumulate over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        if (warp_row < M && k < K) {
            load_matrix_sync(a_frag, A + warp_row * K + k, K);
        }
        if (k < K && warp_col < N) {
            load_matrix_sync(b_frag, B + k * N + warp_col, N);
        }
        
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    if (warp_row < M && warp_col < N) {
        store_matrix_sync(C + warp_row * N + warp_col, c_frag, N, mem_row_major);
    }
}
#endif

// ============================================================================
// GEMM Launcher with Version Selection
// ============================================================================
enum class GemmVersion {
    Naive,
    Tiled,
    DoubleBuffer,
    TensorCore,
    WGMMA_TMA  // CUDA 12.0+ only
};

template<typename T>
void launch_gemm(const T* A, const T* B, T* C,
                 int M, int N, int K,
                 T alpha, T beta,
                 GemmVersion version = GemmVersion::Tiled,
                 cudaStream_t stream = nullptr) {
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    
    switch (version) {
        case GemmVersion::Naive:
            gemm_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
            break;
        case GemmVersion::Tiled:
            gemm_tiled<T, 32><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
            break;
        case GemmVersion::DoubleBuffer:
            gemm_double_buffer<T, 32><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
            break;
#ifdef TC_HAS_WMMA
        case GemmVersion::TensorCore:
            // Requires half precision input
            static_assert(std::is_same_v<T, half>, "TensorCore requires half precision");
            break;
#endif
        default:
            gemm_tiled<T, 32><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }
    TC_CUDA_CHECK_LAST();
}

} // namespace kernels
} // namespace tensorcraft
```

#### 3.4 Normalization Kernels

```cpp
// include/tensorcraft/kernels/normalization.hpp
#pragma once
#include "tensorcraft/memory/aligned_vector.hpp"

namespace tensorcraft {
namespace kernels {

// ============================================================================
// RMSNorm Kernel
// ============================================================================
template<typename T, int BLOCK_SIZE = 256>
__global__ void rmsnorm_kernel(const T* __restrict__ input,
                                const T* __restrict__ weight,
                                T* __restrict__ output,
                                int hidden_size,
                                float eps) {
    int row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;
    
    // Compute sum of squares
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = static_cast<float>(row_input[i]);
        thread_sum += val * val;
    }
    
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Block reduce
    __shared__ float shared_sum[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane == 0) shared_sum[warp_id] = thread_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        thread_sum = (lane < (BLOCK_SIZE / 32)) ? shared_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
    }
    
    __shared__ float rms_inv;
    if (threadIdx.x == 0) {
        rms_inv = rsqrtf(thread_sum / hidden_size + eps);
    }
    __syncthreads();
    
    // Normalize and scale
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = static_cast<float>(row_input[i]);
        row_output[i] = static_cast<T>(val * rms_inv * static_cast<float>(weight[i]));
    }
}

// ============================================================================
// LayerNorm Kernel
// ============================================================================
template<typename T, int BLOCK_SIZE = 256>
__global__ void layernorm_kernel(const T* __restrict__ input,
                                  const T* __restrict__ gamma,
                                  const T* __restrict__ beta,
                                  T* __restrict__ output,
                                  int hidden_size,
                                  float eps) {
    int row = blockIdx.x;
    const T* row_input = input + row * hidden_size;
    T* row_output = output + row * hidden_size;
    
    // Compute mean
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        thread_sum += static_cast<float>(row_input[i]);
    }
    
    // Warp + block reduce for mean
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    if (lane == 0) shared[warp_id] = thread_sum;
    __syncthreads();
    
    if (warp_id == 0) {
        thread_sum = (lane < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
    }
    
    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = thread_sum / hidden_size;
    }
    __syncthreads();
    
    // Compute variance
    float thread_var = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float diff = static_cast<float>(row_input[i]) - mean;
        thread_var += diff * diff;
    }
    
    // Reduce variance
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
    }
    
    if (lane == 0) shared[warp_id] = thread_var;
    __syncthreads();
    
    if (warp_id == 0) {
        thread_var = (lane < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
        }
    }
    
    __shared__ float inv_std;
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(thread_var / hidden_size + eps);
    }
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
        float val = (static_cast<float>(row_input[i]) - mean) * inv_std;
        row_output[i] = static_cast<T>(val * static_cast<float>(gamma[i]) 
                                       + static_cast<float>(beta[i]));
    }
}

// Launcher functions
template<typename T>
void launch_rmsnorm(const T* input, const T* weight, T* output,
                    int batch_size, int hidden_size, float eps = 1e-6f,
                    cudaStream_t stream = nullptr) {
    rmsnorm_kernel<T, 256><<<batch_size, 256, 0, stream>>>(
        input, weight, output, hidden_size, eps);
}

template<typename T>
void launch_layernorm(const T* input, const T* gamma, const T* beta, T* output,
                      int batch_size, int hidden_size, float eps = 1e-5f,
                      cudaStream_t stream = nullptr) {
    layernorm_kernel<T, 256><<<batch_size, 256, 0, stream>>>(
        input, gamma, beta, output, hidden_size, eps);
}

} // namespace kernels
} // namespace tensorcraft
```


#### 3.5 Attention Kernels (FlashAttention Style)

```cpp
// include/tensorcraft/kernels/attention.hpp
#pragma once
#include "tensorcraft/core/features.hpp"

namespace tensorcraft {
namespace kernels {

// ============================================================================
// FlashAttention-style Kernel (Simplified)
// ============================================================================
template<typename T, int BLOCK_M = 64, int BLOCK_N = 64, int HEAD_DIM = 64>
__global__ void flash_attention_kernel(
    const T* __restrict__ Q,  // [batch, heads, seq_len, head_dim]
    const T* __restrict__ K,  // [batch, heads, seq_len, head_dim]
    const T* __restrict__ V,  // [batch, heads, seq_len, head_dim]
    T* __restrict__ O,        // [batch, heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    float scale) {
    
    // Shared memory for Q, K, V tiles
    __shared__ T Qs[BLOCK_M][HEAD_DIM];
    __shared__ T Ks[BLOCK_N][HEAD_DIM];
    __shared__ T Vs[BLOCK_N][HEAD_DIM];
    __shared__ float scores[BLOCK_M][BLOCK_N];
    
    int batch_head = blockIdx.z;
    int batch_idx = batch_head / num_heads;
    int head_idx = batch_head % num_heads;
    int m_block = blockIdx.x;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Offset pointers
    int offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const T* q_ptr = Q + offset;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset;
    
    // Initialize output accumulators and running max/sum
    float o_acc[HEAD_DIM / 32] = {0};  // Per-thread output accumulator
    float m_prev = -INFINITY;  // Running max
    float l_prev = 0.0f;       // Running sum of exp
    
    int m_start = m_block * BLOCK_M;
    
    // Load Q tile (persistent across K/V blocks)
    for (int d = tx; d < HEAD_DIM; d += blockDim.x) {
        int m_idx = m_start + ty;
        Qs[ty][d] = (m_idx < seq_len) ? q_ptr[m_idx * HEAD_DIM + d] : T(0);
    }
    __syncthreads();
    
    // Iterate over K/V blocks
    for (int n_block = 0; n_block < (seq_len + BLOCK_N - 1) / BLOCK_N; ++n_block) {
        int n_start = n_block * BLOCK_N;
        
        // Load K, V tiles
        for (int d = tx; d < HEAD_DIM; d += blockDim.x) {
            int n_idx = n_start + ty;
            Ks[ty][d] = (n_idx < seq_len) ? k_ptr[n_idx * HEAD_DIM + d] : T(0);
            Vs[ty][d] = (n_idx < seq_len) ? v_ptr[n_idx * HEAD_DIM + d] : T(0);
        }
        __syncthreads();
        
        // Compute QK^T for this block
        float qk = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            qk += static_cast<float>(Qs[ty][d]) * static_cast<float>(Ks[tx][d]);
        }
        qk *= scale;
        scores[ty][tx] = qk;
        __syncthreads();
        
        // Online softmax update
        float m_curr = m_prev;
        for (int n = 0; n < BLOCK_N && (n_start + n) < seq_len; ++n) {
            m_curr = fmaxf(m_curr, scores[ty][n]);
        }
        
        float l_curr = l_prev * expf(m_prev - m_curr);
        for (int n = 0; n < BLOCK_N && (n_start + n) < seq_len; ++n) {
            l_curr += expf(scores[ty][n] - m_curr);
        }
        
        // Update output accumulator
        float scale_prev = l_prev * expf(m_prev - m_curr) / l_curr;
        float scale_curr = 1.0f / l_curr;
        
        for (int d = 0; d < HEAD_DIM / 32; ++d) {
            o_acc[d] *= scale_prev;
            for (int n = 0; n < BLOCK_N && (n_start + n) < seq_len; ++n) {
                float p = expf(scores[ty][n] - m_curr) * scale_curr;
                o_acc[d] += p * static_cast<float>(Vs[n][d * 32 + tx]);
            }
        }
        
        m_prev = m_curr;
        l_prev = l_curr;
        __syncthreads();
    }
    
    // Write output
    int m_idx = m_start + ty;
    if (m_idx < seq_len) {
        for (int d = 0; d < HEAD_DIM / 32; ++d) {
            o_ptr[m_idx * HEAD_DIM + d * 32 + tx] = static_cast<T>(o_acc[d]);
        }
    }
}

// ============================================================================
// RoPE (Rotary Position Embedding) Kernel
// ============================================================================
template<typename T>
__global__ void rope_kernel(T* __restrict__ x,  // [batch, seq_len, num_heads, head_dim]
                            const float* __restrict__ cos_cache,  // [max_seq, head_dim/2]
                            const float* __restrict__ sin_cache,  // [max_seq, head_dim/2]
                            int batch_size,
                            int seq_len,
                            int num_heads,
                            int head_dim,
                            int start_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * num_heads * (head_dim / 2);
    
    if (idx >= total) return;
    
    // Decode indices
    int half_head_dim = head_dim / 2;
    int d = idx % half_head_dim;
    int remaining = idx / half_head_dim;
    int h = remaining % num_heads;
    remaining /= num_heads;
    int s = remaining % seq_len;
    int b = remaining / seq_len;
    
    // Get position
    int pos = start_pos + s;
    
    // Load cos/sin for this position and dimension
    float cos_val = cos_cache[pos * half_head_dim + d];
    float sin_val = sin_cache[pos * half_head_dim + d];
    
    // Compute offset in x
    int base_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
    
    // Load pair of values
    float x0 = static_cast<float>(x[base_offset + d]);
    float x1 = static_cast<float>(x[base_offset + d + half_head_dim]);
    
    // Apply rotation
    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x0 * sin_val + x1 * cos_val;
    
    // Store
    x[base_offset + d] = static_cast<T>(y0);
    x[base_offset + d + half_head_dim] = static_cast<T>(y1);
}

// Launcher functions
template<typename T>
void launch_flash_attention(const T* Q, const T* K, const T* V, T* O,
                            int batch_size, int num_heads, int seq_len, int head_dim,
                            float scale, cudaStream_t stream = nullptr) {
    dim3 block(32, 64 / 32);  // 32 threads per warp, 2 warps
    dim3 grid((seq_len + 63) / 64, 1, batch_size * num_heads);
    
    flash_attention_kernel<T, 64, 64, 64><<<grid, block, 0, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, scale);
}

template<typename T>
void launch_rope(T* x, const float* cos_cache, const float* sin_cache,
                 int batch_size, int seq_len, int num_heads, int head_dim,
                 int start_pos = 0, cudaStream_t stream = nullptr) {
    int total = batch_size * seq_len * num_heads * (head_dim / 2);
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    rope_kernel<<<grid_size, block_size, 0, stream>>>(
        x, cos_cache, sin_cache, batch_size, seq_len, num_heads, head_dim, start_pos);
}

} // namespace kernels
} // namespace tensorcraft
```

#### 3.6 Convolution Kernels

```cpp
// include/tensorcraft/kernels/conv2d.hpp
#pragma once

namespace tensorcraft {
namespace kernels {

// ============================================================================
// Conv2D Naive Implementation
// ============================================================================
template<typename T>
__global__ void conv2d_naive(const T* __restrict__ input,   // [N, C, H, W]
                              const T* __restrict__ weight,  // [K, C, R, S]
                              T* __restrict__ output,        // [N, K, OH, OW]
                              int N, int C, int H, int W,
                              int K, int R, int S,
                              int OH, int OW,
                              int stride, int padding) {
    int n = blockIdx.z;
    int k = blockIdx.y;
    int oh = blockIdx.x * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;
    
    if (oh >= OH || ow >= OW) return;
    
    T sum = T(0);
    
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int ih = oh * stride - padding + r;
                int iw = ow * stride - padding + s;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    T in_val = input[((n * C + c) * H + ih) * W + iw];
                    T w_val = weight[((k * C + c) * R + r) * S + s];
                    sum += in_val * w_val;
                }
            }
        }
    }
    
    output[((n * K + k) * OH + oh) * OW + ow] = sum;
}

// ============================================================================
// Im2Col + GEMM based Convolution
// ============================================================================
template<typename T>
__global__ void im2col_kernel(const T* __restrict__ input,  // [N, C, H, W]
                               T* __restrict__ col,          // [N, C*R*S, OH*OW]
                               int N, int C, int H, int W,
                               int R, int S,
                               int OH, int OW,
                               int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * R * S * OH * OW;
    
    if (idx >= total) return;
    
    // Decode index
    int ow = idx % OW;
    int remaining = idx / OW;
    int oh = remaining % OH;
    remaining /= OH;
    int s = remaining % S;
    remaining /= S;
    int r = remaining % R;
    remaining /= R;
    int c = remaining % C;
    int n = remaining / C;
    
    int ih = oh * stride - padding + r;
    int iw = ow * stride - padding + s;
    
    T val = T(0);
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        val = input[((n * C + c) * H + ih) * W + iw];
    }
    
    // col layout: [N, C*R*S, OH*OW]
    int col_idx = (n * (C * R * S) + (c * R * S + r * S + s)) * (OH * OW) + (oh * OW + ow);
    col[col_idx] = val;
}

// Launcher
template<typename T>
void launch_conv2d_naive(const T* input, const T* weight, T* output,
                         int N, int C, int H, int W,
                         int K, int R, int S,
                         int stride, int padding,
                         cudaStream_t stream = nullptr) {
    int OH = (H + 2 * padding - R) / stride + 1;
    int OW = (W + 2 * padding - S) / stride + 1;
    
    dim3 block(32, 8);
    dim3 grid((OH + 7) / 8, K, N);
    
    conv2d_naive<<<grid, block, 0, stream>>>(
        input, weight, output, N, C, H, W, K, R, S, OH, OW, stride, padding);
}

} // namespace kernels
} // namespace tensorcraft
```

## Data Models

### Tensor Data Layout

```cpp
// Supported tensor layouts
enum class TensorLayout {
    NCHW,    // Batch, Channel, Height, Width (default for conv)
    NHWC,    // Batch, Height, Width, Channel (optimized for Tensor Core)
    NC,      // Batch, Features (for linear layers)
    NLC,     // Batch, Sequence, Features (for transformers)
    BNHD     // Batch, Heads, Sequence, HeadDim (for attention)
};

// Data type enumeration
enum class DataType {
    FP32,
    FP16,
    BF16,
    FP8_E4M3,  // CUDA 12.0+
    FP8_E5M2,  // CUDA 12.0+
    INT8
};
```

### Kernel Configuration

```cpp
// Kernel launch configuration
struct KernelConfig {
    int block_size_x = 256;
    int block_size_y = 1;
    int block_size_z = 1;
    int shared_memory_bytes = 0;
    cudaStream_t stream = nullptr;
    
    dim3 block() const { return dim3(block_size_x, block_size_y, block_size_z); }
};

// GEMM specific configuration
struct GemmConfig {
    int tile_m = 128;
    int tile_n = 128;
    int tile_k = 32;
    int warp_m = 64;
    int warp_n = 64;
    int stages = 3;  // Pipeline stages for double/triple buffering
    bool use_tensor_core = true;
};
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Tensor RAII Memory Management

*For any* Tensor object created with a given shape, GPU memory SHALL be allocated on construction and freed on destruction, with no memory leaks.

**Validates: Requirements 2.3**

### Property 2: Aligned Vector Alignment

*For any* AlignedVector<T, N> type, the alignment SHALL equal sizeof(T) * N bytes.

**Validates: Requirements 2.5**

### Property 3: VectorAdd Correctness

*For any* two input vectors A and B of the same size, VectorAdd(A, B) SHALL produce a vector C where C[i] = A[i] + B[i] for all valid indices i.

**Validates: Requirements 3.1**

### Property 4: Activation Function Mathematical Correctness

*For any* input tensor X and activation function F ∈ {ReLU, SiLU, GeLU, LeakyReLU, ELU, Swish}, the kernel output SHALL match the mathematical definition within floating-point tolerance:
- ReLU(x) = max(0, x)
- SiLU(x) = x * sigmoid(x)
- GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

**Validates: Requirements 3.2, 3.3**

### Property 5: Optimization Level Numerical Equivalence

*For any* kernel with multiple optimization levels (naive, tiled, tensor core, etc.), all versions SHALL produce numerically equivalent outputs within tolerance (rtol=1e-3, atol=1e-5 for FP32; rtol=1e-2, atol=1e-3 for FP16).

**Validates: Requirements 3.5, 10.2**

### Property 6: Softmax Row Sum Invariant

*For any* input matrix X, the Softmax output S SHALL satisfy:
1. sum(S[i, :]) = 1.0 for all rows i (within tolerance)
2. S[i, j] >= 0 for all elements
3. S[i, j] = exp(X[i,j] - max(X[i,:])) / sum(exp(X[i,:] - max(X[i,:])))

**Validates: Requirements 4.1**

### Property 7: LayerNorm Statistical Properties

*For any* input tensor X with hidden dimension H, LayerNorm output Y SHALL satisfy:
1. mean(Y[i, :]) ≈ 0 (within tolerance)
2. var(Y[i, :]) ≈ 1 (within tolerance)
3. Y = gamma * (X - mean) / sqrt(var + eps) + beta

**Validates: Requirements 4.2**

### Property 8: RMSNorm Correctness

*For any* input tensor X with hidden dimension H, RMSNorm output Y SHALL satisfy:
Y[i, :] = X[i, :] / RMS(X[i, :]) * weight, where RMS(x) = sqrt(mean(x²) + eps)

**Validates: Requirements 4.2**

### Property 9: BatchNorm Correctness

*For any* input batch X with shape [N, C, H, W], BatchNorm output Y SHALL normalize across the batch dimension for each channel.

**Validates: Requirements 4.3**

### Property 10: GEMM Mathematical Correctness

*For any* matrices A[M×K], B[K×N] and scalars alpha, beta, GEMM SHALL compute C = alpha * A @ B + beta * C correctly within floating-point tolerance.

**Validates: Requirements 5.1**

### Property 11: Matrix Transpose Round-Trip

*For any* matrix A, transpose(transpose(A)) SHALL equal A exactly.

**Validates: Requirements 5.5**

### Property 12: FlashAttention Equivalence

*For any* Q, K, V tensors with valid attention dimensions, FlashAttention output SHALL match standard attention computation: softmax(Q @ K^T / sqrt(d)) @ V within tolerance.

**Validates: Requirements 6.1**

### Property 13: RoPE Transformation Correctness

*For any* input tensor X and position indices, RoPE SHALL apply the correct rotational transformation using precomputed cos/sin caches.

**Validates: Requirements 6.3**

### Property 14: Operator Fusion Equivalence

*For any* input tensor X, fused operations (e.g., Bias+GeLU) SHALL produce the same result as applying operations sequentially.

**Validates: Requirements 7.1**

### Property 15: Convolution Algorithm Equivalence

*For any* input tensor and filter, all convolution implementations (naive, im2col+GEMM, Winograd) SHALL produce numerically equivalent outputs within tolerance.

**Validates: Requirements 11.2, 11.3**

### Property 16: Sparse-Dense Operation Equivalence

*For any* sparse matrix in CSR/CSC format and corresponding dense matrix, SpMV and SpMM operations SHALL produce results equivalent to dense matrix operations.

**Validates: Requirements 12.3, 12.4**

### Property 17: Sparse Format Round-Trip

*For any* dense matrix A, converting to sparse format and back to dense SHALL produce an equivalent matrix.

**Validates: Requirements 12.6**

## Error Handling

### CUDA Error Handling Strategy

```cpp
// All CUDA API calls wrapped with TC_CUDA_CHECK macro
// Throws std::runtime_error with file, line, and error description

// Example error handling pattern
try {
    Tensor<float> t({1024, 1024});
    launch_gemm(...);
    TC_CUDA_CHECK(cudaDeviceSynchronize());
} catch (const std::runtime_error& e) {
    std::cerr << "CUDA Error: " << e.what() << std::endl;
    // Handle error or rethrow
}
```

### Numerical Error Handling

```cpp
// Tolerance configuration for numerical comparisons
struct ToleranceConfig {
    float rtol_fp32 = 1e-5f;
    float atol_fp32 = 1e-8f;
    float rtol_fp16 = 1e-3f;
    float atol_fp16 = 1e-3f;
    float rtol_bf16 = 1e-2f;
    float atol_bf16 = 1e-2f;
};

// Comparison function
template<typename T>
bool allclose(const T* a, const T* b, size_t n, 
              float rtol, float atol) {
    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
        float threshold = atol + rtol * std::abs(static_cast<float>(b[i]));
        if (diff > threshold) return false;
    }
    return true;
}
```

### Input Validation

```cpp
// Dimension validation for GEMM
void validate_gemm_dims(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) {
        throw std::invalid_argument("GEMM dimensions must be positive");
    }
    if (M > 65535 || N > 65535) {
        throw std::invalid_argument("GEMM dimensions exceed maximum grid size");
    }
}

// Alignment validation for vectorized access
template<typename T, int VecSize>
void validate_alignment(const T* ptr) {
    if (reinterpret_cast<uintptr_t>(ptr) % (sizeof(T) * VecSize) != 0) {
        throw std::invalid_argument("Pointer not aligned for vectorized access");
    }
}
```

## Testing Strategy

### Dual Testing Approach

本项目采用双重测试策略：

1. **单元测试 (Unit Tests)**: 使用 GoogleTest 验证特定示例和边界情况
2. **属性测试 (Property-Based Tests)**: 使用随机生成的输入验证通用属性

### Property-Based Testing Configuration

```cpp
// 使用 rapidcheck 或自定义随机测试框架
// 每个属性测试至少运行 100 次迭代

// 示例：GEMM 正确性属性测试
// Feature: tensorcraft-hpc, Property 10: GEMM Mathematical Correctness
TEST(GemmPropertyTest, MathematicalCorrectness) {
    for (int iter = 0; iter < 100; ++iter) {
        // 随机生成维度
        int M = rand() % 512 + 1;
        int N = rand() % 512 + 1;
        int K = rand() % 512 + 1;
        
        // 随机生成输入
        auto A = generate_random_matrix<float>(M, K);
        auto B = generate_random_matrix<float>(K, N);
        auto C = generate_random_matrix<float>(M, N);
        
        // 计算参考结果
        auto C_ref = reference_gemm(A, B, C, 1.0f, 0.0f);
        
        // 测试所有 GEMM 版本
        for (auto version : {GemmVersion::Naive, GemmVersion::Tiled, 
                             GemmVersion::DoubleBuffer}) {
            auto C_test = C;
            launch_gemm(A.data(), B.data(), C_test.data(), M, N, K, 
                       1.0f, 0.0f, version);
            
            EXPECT_TRUE(allclose(C_test.data(), C_ref.data(), M * N, 
                                1e-3f, 1e-5f));
        }
    }
}
```

### Test Categories

1. **Correctness Tests**: 验证数值正确性
   - 与参考实现（PyTorch/NumPy）对比
   - 属性测试验证数学不变量

2. **Edge Case Tests**: 边界情况
   - 空输入、单元素输入
   - 非对齐内存
   - 极大/极小值

3. **Performance Tests**: 性能回归测试
   - Google Benchmark 基准测试
   - 与 cuBLAS/cuDNN 对比

### Python Verification

```python
# tests/python/test_kernels.py
import torch
import tensorcraft_ops as tc
import numpy as np

def test_softmax_correctness():
    """Property 6: Softmax Row Sum Invariant"""
    for _ in range(100):
        rows = np.random.randint(1, 256)
        cols = np.random.randint(1, 1024)
        
        x = torch.randn(rows, cols, device='cuda')
        
        # TensorCraft implementation
        y_tc = tc.softmax(x)
        
        # PyTorch reference
        y_ref = torch.softmax(x, dim=-1)
        
        # Verify row sums
        assert torch.allclose(y_tc.sum(dim=-1), torch.ones(rows, device='cuda'), 
                             rtol=1e-5, atol=1e-5)
        
        # Verify against reference
        assert torch.allclose(y_tc, y_ref, rtol=1e-3, atol=1e-5)

def test_gemm_equivalence():
    """Property 5 & 10: GEMM Optimization Level Equivalence"""
    for _ in range(100):
        M, N, K = [np.random.randint(32, 512) for _ in range(3)]
        
        A = torch.randn(M, K, device='cuda')
        B = torch.randn(K, N, device='cuda')
        
        # Test all versions produce same result
        results = []
        for version in ['naive', 'tiled', 'double_buffer']:
            C = tc.gemm(A, B, version=version)
            results.append(C)
        
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], rtol=1e-3, atol=1e-5)
```

### Benchmark Framework

```cpp
// benchmarks/gemm_benchmark.cpp
#include <benchmark/benchmark.h>
#include "tensorcraft/kernels/gemm.hpp"

static void BM_GEMM_Naive(benchmark::State& state) {
    int M = state.range(0);
    int N = state.range(0);
    int K = state.range(0);
    
    Tensor<float> A({M, K}), B({K, N}), C({M, N});
    
    for (auto _ : state) {
        launch_gemm(A.data(), B.data(), C.data(), M, N, K, 
                   1.0f, 0.0f, GemmVersion::Naive);
        cudaDeviceSynchronize();
    }
    
    // Report GFLOPS
    double gflops = 2.0 * M * N * K / 1e9;
    state.counters["GFLOPS"] = benchmark::Counter(
        gflops, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_GEMM_Naive)->RangeMultiplier(2)->Range(256, 4096);
BENCHMARK(BM_GEMM_Tiled)->RangeMultiplier(2)->Range(256, 4096);
BENCHMARK(BM_GEMM_DoubleBuffer)->RangeMultiplier(2)->Range(256, 4096);
```

## Build System Design

### CMakeLists.txt Structure

```cmake
cmake_minimum_required(VERSION 3.20)
project(TensorCraft-HPC LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect C++20/23 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" COMPILER_SUPPORTS_CXX20)
check_cxx_compiler_flag("-std=c++23" COMPILER_SUPPORTS_CXX23)

if(COMPILER_SUPPORTS_CXX23)
    set(CMAKE_CXX_STANDARD 23)
    add_compile_definitions(TC_CPP23=1)
elseif(COMPILER_SUPPORTS_CXX20)
    set(CMAKE_CXX_STANDARD 20)
    add_compile_definitions(TC_CPP20=1)
endif()

# CUDA Configuration
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

# Feature detection based on CUDA version
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
    add_compile_definitions(TC_HAS_TMA=1 TC_HAS_WGMMA=1 TC_HAS_FP8=1)
endif()
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
    add_compile_definitions(TC_HAS_WMMA=1)
endif()

# Dependencies via FetchContent
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)

FetchContent_MakeAvailable(googletest benchmark pybind11)

# Library target
add_library(tensorcraft INTERFACE)
target_include_directories(tensorcraft INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tensorcraft INTERFACE CUDA::cudart)

# Tests
enable_testing()
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)

# Python bindings
add_subdirectory(src/python_ops)
```

### CMakePresets.json

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "default",
            "hidden": true,
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build/${presetName}",
            "cacheVariables": {
                "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
            }
        },
        {
            "name": "debug",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "CMAKE_CUDA_FLAGS": "-G -lineinfo"
            }
        },
        {
            "name": "release",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_CUDA_FLAGS": "-O3 --use_fast_math"
            }
        },
        {
            "name": "profile",
            "inherits": "default",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "RelWithDebInfo",
                "CMAKE_CUDA_FLAGS": "-O3 -lineinfo"
            }
        }
    ],
    "buildPresets": [
        {"name": "debug", "configurePreset": "debug"},
        {"name": "release", "configurePreset": "release"},
        {"name": "profile", "configurePreset": "profile"}
    ],
    "testPresets": [
        {
            "name": "default",
            "configurePreset": "debug",
            "output": {"outputOnFailure": true}
        }
    ]
}
```
