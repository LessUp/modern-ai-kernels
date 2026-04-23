# API Specification: TensorCraft-HPC

> **Version**: 2.0.0
> **Last Updated**: 2026-04-17

---

## Overview

This document defines the public API specification for TensorCraft-HPC. All implementations must conform to these interface definitions.

---

## Module Overview

| Module | Header Path | Description |
|--------|-------------|-------------|
| Core | `tensorcraft/core/` | Error handling, feature detection, type traits |
| Memory | `tensorcraft/memory/` | Tensor wrapper, memory pool, aligned vectors |
| Kernels | `tensorcraft/kernels/` | Compute kernels (GEMM, attention, etc.) |
| Python | `tensorcraft_ops` | Python bindings |

---

## 1. Core Module API

### 1.1 Error Handling (`cuda_check.hpp`)

#### CudaException

```cpp
namespace tensorcraft;

class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& file, int line, cudaError_t error);

    cudaError_t error() const noexcept;
    const std::string& file() const noexcept;
    int line() const noexcept;
};
```

**Behavior:**

- SHALL throw on any CUDA API error
- SHALL include file, line, and error description
- SHALL be catchable as `std::runtime_error`

#### Error Checking Macros

```cpp
TC_CUDA_CHECK(call)       // Check CUDA API call
TC_CUDA_CHECK_LAST()      // Check last kernel launch error
TC_CUDA_SYNC_CHECK()      // Synchronize and check all errors
```

### 1.2 Feature Detection (`features.hpp`)

#### C++ Version Macros

| Macro | Condition |
|-------|-----------|
| `TC_CPP17` | C++17 or higher |
| `TC_CPP20` | C++20 or higher |
| `TC_CPP23` | C++23 or higher |

#### CUDA Version Macros

| Macro | Condition |
|-------|-----------|
| `TC_CUDA_VERSION` | Numeric CUDA version (e.g., 12080) |
| `TC_CUDA_10` | CUDA 10.x or higher |
| `TC_CUDA_11` | CUDA 11.x or higher |
| `TC_CUDA_12` | CUDA 12.x or higher |
| `TC_CUDA_13` | CUDA 13.x or higher |

#### Feature Macros

| Macro | Feature |
|-------|---------|
| `TC_HAS_WMMA` | Warp Matrix Multiply-Accumulate |
| `TC_HAS_BF16` | BFloat16 support |
| `TC_HAS_FP8` | FP8 support (CUDA 12.0+) |
| `TC_HAS_TMA` | Tensor Memory Accelerator |
| `TC_HAS_WGMMA` | Warp Group MMA |

#### Runtime Functions

```cpp
namespace tensorcraft;

int get_cuda_runtime_version();
int get_cuda_driver_version();
std::pair<int, int> get_compute_capability(int device = 0);
bool has_tensor_cores(int device = 0);
bool has_tma(int device = 0);
```

### 1.3 Type Traits (`type_traits.hpp`)

```cpp
namespace tensorcraft;

template<typename T> inline constexpr bool is_half_v;
template<typename T> inline constexpr bool is_fp8_v;
template<typename T> inline constexpr bool is_floating_v;
template<typename T> inline constexpr bool is_numeric_v;

// C++20 Concepts (when available)
template<typename T> concept Numeric = is_numeric_v<T>;
template<typename T> concept FloatingPoint = is_floating_v<T>;
template<typename T> concept HalfPrecision = is_half_v<T>;

// Type conversion
template<typename T> TC_HOST_DEVICE_INLINE float to_float(T val);
template<typename T> TC_HOST_DEVICE_INLINE T from_float(float val);

// DataType enumeration
enum class DataType {
    FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, INT8, INT32, INT64
};

template<typename T> constexpr DataType get_dtype();
constexpr size_t dtype_size(DataType dtype);
```

### 1.4 Warp Utilities (`warp_utils.hpp`)

```cpp
namespace tensorcraft;

// Warp reduction
template<typename T> TC_DEVICE_INLINE T warp_reduce_max(T val);
template<typename T> TC_DEVICE_INLINE T warp_reduce_sum(T val);
template<typename T> TC_DEVICE_INLINE T warp_reduce_min(T val);
template<typename T> TC_DEVICE_INLINE T warp_broadcast(T val, int src_lane = 0);

// Block reduction
template<typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_sum(T val, T* shared);

template<typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_max(T val, T* shared);
```

---

## 2. Memory Module API

### 2.1 Tensor Wrapper (`tensor.hpp`)

```cpp
namespace tensorcraft;

template<typename T>
class Tensor {
public:
    // Constructors
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    ~Tensor();

    // Assignment
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(Tensor&&) noexcept;

    // Accessors
    T* data() noexcept;
    const T* data() const noexcept;
    size_t size() const noexcept;
    const std::vector<size_t>& shape() const noexcept;

    // Operations
    void fill(T value);
    void copy_from(const T* host_data);
    void copy_to(T* host_data) const;

    // Factory
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
};

// Type aliases
using FloatTensor = Tensor<float>;
using HalfTensor = Tensor<half>;
```

**Behavior:**

- SHALL allocate GPU memory on construction
- SHALL free GPU memory on destruction (RAII)
- SHALL NOT allow copying (move-only)
- SHALL throw `CudaException` on allocation failure

### 2.2 Memory Pool (`memory_pool.hpp`)

```cpp
namespace tensorcraft;

class MemoryPool {
public:
    explicit MemoryPool(size_t initial_size = 1 << 20);
    ~MemoryPool();

    void* allocate(size_t size);
    void deallocate(void* ptr);
    void clear();

    size_t capacity() const noexcept;
    size_t used() const noexcept;
};
```

**Behavior:**

- SHALL be thread-safe
- SHALL track allocated blocks
- SHALL NOT free in-use memory on `clear()`

### 2.3 Aligned Vector (`aligned_vector.hpp`)

```cpp
namespace tensorcraft;

template<typename T, size_t Alignment>
class AlignedVector {
public:
    explicit AlignedVector(size_t size);
    ~AlignedVector();

    T* data() noexcept;
    const T* data() const noexcept;
    size_t size() const noexcept;

    T& operator[](size_t i);
    const T& operator[](size_t i) const;
};
```

---

## 3. Kernels Module API

### 3.1 Elementwise Operations (`elementwise.hpp`)

#### Activation Functors

```cpp
namespace tensorcraft::kernels;

struct ReLU { template<typename T> TC_DEVICE_INLINE T operator()(T x) const; };
struct SiLU { template<typename T> TC_DEVICE_INLINE T operator()(T x) const; };
struct GeLU { template<typename T> TC_DEVICE_INLINE T operator()(T x) const; };
struct Sigmoid { template<typename T> TC_DEVICE_INLINE T operator()(T x) const; };
struct Tanh { template<typename T> TC_DEVICE_INLINE T operator()(T x) const; };

template<typename T = float>
struct LeakyReLU {
    T alpha;
    TC_DEVICE_INLINE T operator()(T x) const;
};

template<typename T = float>
struct Swish {
    T beta;
    TC_DEVICE_INLINE T operator()(T x) const;
};
```

#### Launcher Functions

```cpp
template<typename T, typename Func>
void launch_elementwise(const T* input, T* output, size_t n, Func func,
                        cudaStream_t stream = nullptr);

template<typename T> void relu(const T* in, T* out, size_t n, cudaStream_t s = nullptr);
template<typename T> void silu(const T* in, T* out, size_t n, cudaStream_t s = nullptr);
template<typename T> void gelu(const T* in, T* out, size_t n, cudaStream_t s = nullptr);
template<typename T> void sigmoid(const T* in, T* out, size_t n, cudaStream_t s = nullptr);
```

### 3.2 Softmax (`softmax.hpp`)

```cpp
namespace tensorcraft::kernels;

template<typename T>
void softmax(const T* input, T* output, size_t batch_size, size_t dim,
             cudaStream_t stream = nullptr);
```

**Correctness Properties:**

- Output values SHALL be >= 0
- Row sums SHALL equal 1.0 (within tolerance)

### 3.3 Normalization (`normalization.hpp`)

```cpp
namespace tensorcraft::kernels;

template<typename T>
void layernorm(const T* input, const T* gamma, const T* beta, T* output,
               size_t batch_size, size_t hidden_size, float eps = 1e-5f,
               cudaStream_t stream = nullptr);

template<typename T>
void rmsnorm(const T* input, const T* weight, T* output,
             size_t batch_size, size_t hidden_size, float eps = 1e-6f,
             cudaStream_t stream = nullptr);

template<typename T>
void launch_batchnorm(const T* input, const T* gamma, const T* beta,
                      const float* running_mean, const float* running_var, T* output,
                      int N, int C, int H, int W, float eps = 1e-5f,
                      bool fuse_relu = false, cudaStream_t stream = nullptr);
```

### 3.4 GEMM (`gemm.hpp`)

```cpp
namespace tensorcraft::kernels;

enum class GemmVersion {
    Naive,          // Basic implementation
    Tiled,          // Shared memory tiling
    DoubleBuffer,   // Double buffering
    TensorCore,     // WMMA tensor core
    Auto            // Automatic selection
};

template<typename T>
void gemm(const T* A, const T* B, T* C, size_t M, size_t N, size_t K,
          T alpha = T(1), T beta = T(0), cudaStream_t stream = nullptr);

template<typename T>
void launch_gemm(const T* A, const T* B, T* C, int M, int N, int K,
                 T alpha, T beta, GemmVersion version,
                 cudaStream_t stream = nullptr);

#ifdef TC_HAS_WMMA
void launch_gemm_wmma(const half* A, const half* B, float* C,
                      int M, int N, int K, float alpha = 1.0f, float beta = 0.0f,
                      cudaStream_t stream = nullptr);
#endif
```

**Correctness Properties:**

- For matrices A[M×K], B[K×N], computes C = alpha *A @ B + beta* C
- All versions SHALL produce numerically equivalent outputs (within tolerance)

### 3.5 Attention (`attention.hpp`)

```cpp
namespace tensorcraft::kernels;

template<typename T>
void flash_attention(const T* Q, const T* K, const T* V, T* O,
                     size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
                     cudaStream_t stream = nullptr);

template<typename T>
void launch_rope(T* x, const float* cos_cache, const float* sin_cache,
                 int batch_size, int seq_len, int num_heads, int head_dim,
                 int start_pos = 0, cudaStream_t stream = nullptr);

void precompute_rope_cache(float* cos_cache, float* sin_cache,
                           int max_seq_len, int head_dim, float base = 10000.0f,
                           cudaStream_t stream = nullptr);

template<typename T>
void launch_moe_router(const T* gate_logits, int* expert_indices, float* expert_weights,
                       int batch_size, int num_experts, int top_k,
                       cudaStream_t stream = nullptr);
```

### 3.6 Convolution (`conv2d.hpp`)

```cpp
namespace tensorcraft::kernels;

template<typename T>
void conv2d(const T* input, const T* weight, const T* bias, T* output,
            size_t N, size_t C, size_t H, size_t W, size_t K, size_t R, size_t S,
            int stride = 1, int padding = 0, cudaStream_t stream = nullptr);

template<typename T>
void conv2d_depthwise(const T* input, const T* weight, const T* bias, T* output,
                      size_t N, size_t C, size_t H, size_t W, size_t R, size_t S,
                      int stride = 1, int padding = 0, cudaStream_t stream = nullptr);
```

### 3.7 Sparse Operations (`sparse.hpp`)

```cpp
namespace tensorcraft::kernels;

template<typename T>
struct CSRMatrix {
    T* values;
    int* col_indices;
    int* row_ptrs;
    int rows, cols, nnz;
};

template<typename T>
void spmv(const CSRMatrix<T>& A, const T* x, T* y, cudaStream_t stream = nullptr);

template<typename T>
void spmm(const CSRMatrix<T>& A, const T* B, T* C, int N, cudaStream_t stream = nullptr);
```

### 3.8 Fusion & Quantization (`fusion.hpp`)

```cpp
namespace tensorcraft::kernels;

// Fused operations
template<typename T>
void gemm_bias_relu(const T* A, const T* B, const T* bias, T* C,
                    size_t M, size_t N, size_t K, cudaStream_t stream = nullptr);

template<typename T>
void gemm_bias_gelu(const T* A, const T* B, const T* bias, T* C,
                    size_t M, size_t N, size_t K, cudaStream_t stream = nullptr);

// INT8 Quantization
template<typename T>
void quantize_int8(const T* input, int8_t* output, float scale, int zero_point, size_t n,
                   cudaStream_t stream = nullptr);

template<typename T>
void dequantize_int8(const int8_t* input, T* output, float scale, int zero_point, size_t n,
                     cudaStream_t stream = nullptr);

// FP8 Quantization (CUDA 12.0+)
#ifdef TC_HAS_FP8
template<typename T>
void quantize_fp8_e4m3(const T* input, __nv_fp8_e4m3* output, float scale, size_t n,
                       cudaStream_t stream = nullptr);
#endif
```

---

## 4. Python API (`tensorcraft_ops`)

### Module Functions

```python
import tensorcraft_ops as tc

# Version
tc.__version__  # str

# GEMM
def tc.gemm(A: np.ndarray, B: np.ndarray,
            alpha: float = 1.0, beta: float = 0.0) -> np.ndarray

# Activation
def tc.relu(x: np.ndarray) -> np.ndarray
def tc.silu(x: np.ndarray) -> np.ndarray
def tc.gelu(x: np.ndarray) -> np.ndarray
def tc.sigmoid(x: np.ndarray) -> np.ndarray

# Normalization
def tc.layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                 eps: float = 1e-5) -> np.ndarray
def tc.rmsnorm(x: np.ndarray, weight: np.ndarray,
               eps: float = 1e-6) -> np.ndarray
def tc.softmax(x: np.ndarray, dim: int = -1) -> np.ndarray

# Attention
def tc.flash_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                       scale: Optional[float] = None) -> np.ndarray

# Convolution
def tc.conv2d(input: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
              stride: int = 1, padding: int = 0) -> np.ndarray
```

**Interface Requirements:**

- All functions SHALL accept NumPy arrays with `float32` or `float16` dtype
- All functions SHALL return NumPy arrays
- Memory management SHALL be automatic (no explicit free required)

---

## Error Handling

All API functions SHALL:

1. Throw `CudaException` on CUDA errors
2. Throw `std::invalid_argument` on invalid arguments
3. Never return error codes (use exceptions)

---

## Version Compatibility

| API Version | CUDA Version | C++ Standard |
|-------------|--------------|--------------|
| 2.0.0 | 11.0 - 13.1 | C++17 |
| 1.x | 11.0 - 12.x | C++17 |

---

## See Also

- [RFC 0001: Core Architecture](../rfc/0001-core-architecture.md)
- [Product Spec: TensorCraft-HPC](../product/tensorcraft-hpc.md)
