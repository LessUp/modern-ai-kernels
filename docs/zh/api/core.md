# Core Module API

The Core module provides fundamental utilities for CUDA error handling, feature detection, type traits, and warp-level operations.

## Headers

| Header | Description |
|--------|-------------|
| `cuda_check.hpp` | CUDA error checking macros and exceptions |
| `features.hpp` | Compile-time feature detection |
| `type_traits.hpp` | Type traits and concepts |
| `warp_utils.hpp` | Warp-level reduction primitives |

---

## cuda_check.hpp

CUDA error checking utilities with file and line tracking.

### CudaException

```cpp
namespace tensorcraft;

class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& file, int line, cudaError_t error);
    
    cudaError_t error() const noexcept;  // CUDA error code
    const std::string& file() const noexcept;  // Source file
    int line() const noexcept;  // Line number
};
```

### Error Checking Macros

```cpp
// Check CUDA API call
TC_CUDA_CHECK(call)
// Example:
TC_CUDA_CHECK(cudaMalloc(&ptr, size));

// Check last CUDA error (for kernel launches)
TC_CUDA_CHECK_LAST()
// Example:
my_kernel<<<grid, block>>>(...);
TC_CUDA_CHECK_LAST();

// Synchronize and check all errors
TC_CUDA_SYNC_CHECK()
```

### Example

```cpp
#include "tensorcraft/core/cuda_check.hpp"

void allocate_memory(float** ptr, size_t n) {
    TC_CUDA_CHECK(cudaMalloc(ptr, n * sizeof(float)));
}

void run_kernel(float* data, size_t n) {
    my_kernel<<<grid, block>>>(data, n);
    TC_CUDA_CHECK_LAST();  // Check kernel launch error
    TC_CUDA_SYNC_CHECK();  // Synchronize and check
}
```

---

## features.hpp

Compile-time detection of C++ and CUDA features.

### C++ Version Macros

```cpp
TC_CPP17  // C++17 available
TC_CPP20  // C++20 available
TC_CPP23  // C++23 available
```

### CUDA Version Macros

```cpp
TC_CUDA_VERSION   // CUDA version (e.g., 12080 for 12.8)
TC_CUDA_10        // CUDA 10.x+
TC_CUDA_11        // CUDA 11.x+
TC_CUDA_12        // CUDA 12.x+
TC_CUDA_13        // CUDA 13.x+
```

### Feature Macros

```cpp
TC_HAS_WMMA       // Warp Matrix Multiply-Accumulate (Volta+)
TC_HAS_BF16       // BFloat16 support
TC_HAS_FP8        // FP8 support (CUDA 12.0+)
TC_HAS_TMA        // Tensor Memory Accelerator (Hopper+)
TC_HAS_WGMMA      // Warp Group MMA (Hopper+)
```

### Architecture Macros (Device Code)

```cpp
TC_ARCH_VOLTA     // SM 70+
TC_ARCH_TURING    // SM 75+
TC_ARCH_AMPERE    // SM 80+
TC_ARCH_ADA       // SM 89
TC_ARCH_HOPPER    // SM 90+
TC_ARCH_BLACKWELL // SM 100+
TC_HAS_TENSOR_CORE  // Tensor Core available
```

### Runtime Functions

```cpp
namespace tensorcraft;

// Get CUDA runtime version
int get_cuda_runtime_version();

// Get CUDA driver version
int get_cuda_driver_version();

// Get compute capability
std::pair<int, int> get_compute_capability(int device = 0);

// Check if Tensor Cores available
bool has_tensor_cores(int device = 0);

// Check if TMA available
bool has_tma(int device = 0);
```

---

## type_traits.hpp

Type traits for numeric types and CUDA half-precision types.

### Type Detection

```cpp
namespace tensorcraft;

// Check if T is half-precision (FP16 or BF16)
template<typename T>
inline constexpr bool is_half_v;

// Check if T is FP8
template<typename T>
inline constexpr bool is_fp8_v;

// Check if T is any floating-point type
template<typename T>
inline constexpr bool is_floating_v;

// Check if T is numeric (arithmetic or half)
template<typename T>
inline constexpr bool is_numeric_v;
```

### C++20 Concepts (when available)

```cpp
template<typename T>
concept Numeric = is_numeric_v<T>;

template<typename T>
concept FloatingPoint = is_floating_v<T>;

template<typename T>
concept HalfPrecision = is_half_v<T>;

template<typename T>
concept StandardFloat = std::is_floating_point_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;
```

### Type Conversion

```cpp
namespace tensorcraft;

// Convert to float for computation
template<typename T>
TC_HOST_DEVICE_INLINE float to_float(T val);

// Convert from float to target type
template<typename T>
TC_HOST_DEVICE_INLINE T from_float(float val);
```

### DataType Enumeration

```cpp
enum class DataType {
    FP32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    INT8,
    INT32,
    INT64
};

// Get DataType from C++ type
template<typename T>
constexpr DataType get_dtype();

// Get size in bytes
constexpr size_t dtype_size(DataType dtype);
```

---

## warp_utils.hpp

Warp-level reduction and shuffle utilities.

### Warp Reduction

```cpp
namespace tensorcraft;

// Warp-level max reduction
template<typename T>
TC_DEVICE_INLINE T warp_reduce_max(T val);

// Warp-level sum reduction
template<typename T>
TC_DEVICE_INLINE T warp_reduce_sum(T val);

// Warp-level min reduction
template<typename T>
TC_DEVICE_INLINE T warp_reduce_min(T val);

// Broadcast value from lane 0
template<typename T>
TC_DEVICE_INLINE T warp_broadcast(T val, int src_lane = 0);
```

### Block Reduction

```cpp
// Block-level sum reduction using shared memory
template<typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_sum(T val, T* shared);

// Block-level max reduction
template<typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_max(T val, T* shared);
```

### Example

```cpp
#include "tensorcraft/core/warp_utils.hpp"

__global__ void softmax_kernel(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    
    // Load value
    float val = input[tid];
    
    // Warp reduce max
    float max_val = tensorcraft::warp_reduce_max(val);
    
    // Broadcast max from lane 0
    max_val = tensorcraft::warp_broadcast(max_val);
    
    // Compute exp and normalize
    float exp_val = expf(val - max_val);
    float sum = tensorcraft::warp_reduce_sum(exp_val);
    
    output[tid] = exp_val / tensorcraft::warp_broadcast(sum);
}
```
