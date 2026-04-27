---
title: 核心模块 API
lang: zh
---

# 核心模块 API

核心模块提供 CUDA 错误处理、特性检测、类型特征和 Warp 级操作的基础工具。

## 头文件

| 头文件 | 描述 |
|--------|------|
| `cuda_check.hpp` | CUDA 错误检查宏和异常类 |
| `features.hpp` | 编译时特性检测 |
| `type_traits.hpp` | 类型特征和概念 |
| `warp_utils.hpp` | Warp 级归约原语 |

---

## cuda_check.hpp

CUDA 错误检查工具，支持文件和行号追踪。

### CudaException 异常类

```cpp
namespace tensorcraft;

class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& file, int line, cudaError_t error);
    
    cudaError_t error() const noexcept;  // CUDA 错误码
    const std::string& file() const noexcept;  // 源文件
    int line() const noexcept;  // 行号
};
```

### 错误检查宏

```cpp
// 检查 CUDA API 调用
TC_CUDA_CHECK(call)
// 示例:
TC_CUDA_CHECK(cudaMalloc(&ptr, size));

// 检查最后一个 CUDA 错误（用于内核启动）
TC_CUDA_CHECK_LAST()
// 示例:
my_kernel<<<grid, block>>>(...);
TC_CUDA_CHECK_LAST();

// 同步并检查所有错误
TC_CUDA_SYNC_CHECK()
```

### 示例

```cpp
#include "tensorcraft/core/cuda_check.hpp"

void allocate_memory(float** ptr, size_t n) {
    TC_CUDA_CHECK(cudaMalloc(ptr, n * sizeof(float)));
}

void run_kernel(float* data, size_t n) {
    my_kernel<<<grid, block>>>(data, n);
    TC_CUDA_CHECK_LAST();  // 检查内核启动错误
    TC_CUDA_SYNC_CHECK();  // 同步并检查
}
```

---

## features.hpp

C++ 和 CUDA 特性的编译时检测。

### C++ 版本宏

```cpp
TC_CPP17  // C++17 可用
TC_CPP20  // C++20 可用
TC_CPP23  // C++23 可用
```

### CUDA 版本宏

```cpp
TC_CUDA_VERSION   // CUDA 版本（如 12080 表示 12.8）
TC_CUDA_10        // CUDA 10.x+
TC_CUDA_11        // CUDA 11.x+
TC_CUDA_12        // CUDA 12.x+
TC_CUDA_13        // CUDA 13.x+
```

### 特性宏

```cpp
TC_HAS_WMMA       // Warp 矩阵乘累加（Volta+）
TC_HAS_BF16       // BFloat16 支持
TC_HAS_FP8        // FP8 支持（CUDA 12.0+）
TC_HAS_TMA        // 张量内存加速器（Hopper+）
TC_HAS_WGMMA      // Warp 组矩阵乘累加（Hopper+）
```

### 架构宏（设备代码）

```cpp
TC_ARCH_VOLTA     // SM 70+
TC_ARCH_TURING    // SM 75+
TC_ARCH_AMPERE    // SM 80+
TC_ARCH_ADA       // SM 89
TC_ARCH_HOPPER    // SM 90+
TC_ARCH_BLACKWELL // SM 100+
TC_HAS_TENSOR_CORE  // Tensor Core 可用
```

### 运行时函数

```cpp
namespace tensorcraft;

// 获取 CUDA 运行时版本
int get_cuda_runtime_version();

// 获取 CUDA 驱动版本
int get_cuda_driver_version();

// 获取计算能力
std::pair<int, int> get_compute_capability(int device = 0);

// 检查是否有 Tensor Core
bool has_tensor_cores(int device = 0);

// 检查是否有 TMA
bool has_tma(int device = 0);
```

---

## type_traits.hpp

数值类型和 CUDA 半精度类型的类型特征。

### 类型检测

```cpp
namespace tensorcraft;

// 检查 T 是否为半精度（FP16 或 BF16）
template<typename T>
inline constexpr bool is_half_v;

// 检查 T 是否为 FP8
template<typename T>
inline constexpr bool is_fp8_v;

// 检查 T 是否为任何浮点类型
template<typename T>
inline constexpr bool is_floating_v;

// 检查 T 是否为数值类型（算术类型或半精度）
template<typename T>
inline constexpr bool is_numeric_v;
```

### C++20 概念（如可用）

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

### 类型转换

```cpp
namespace tensorcraft;

// 转换为 float 用于计算
template<typename T>
TC_HOST_DEVICE_INLINE float to_float(T val);

// 从 float 转换为目标类型
template<typename T>
TC_HOST_DEVICE_INLINE T from_float(float val);
```

### DataType 枚举

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

// 从 C++ 类型获取 DataType
template<typename T>
constexpr DataType get_dtype();

// 获取字节大小
constexpr size_t dtype_size(DataType dtype);
```

---

## warp_utils.hpp

Warp 级归约和 Shuffle 工具。

### Warp 归约

```cpp
namespace tensorcraft;

// Warp 级最大值归约
template<typename T>
TC_DEVICE_INLINE T warp_reduce_max(T val);

// Warp 级求和归约
template<typename T>
TC_DEVICE_INLINE T warp_reduce_sum(T val);

// Warp 级最小值归约
template<typename T>
TC_DEVICE_INLINE T warp_reduce_min(T val);

// 从 lane 0 广播值
template<typename T>
TC_DEVICE_INLINE T warp_broadcast(T val, int src_lane = 0);
```

### Block 归约

```cpp
// 使用共享内存的 Block 级求和归约
template<typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_sum(T val, T* shared);

// Block 级最大值归约
template<typename T, int BLOCK_SIZE>
TC_DEVICE_INLINE T block_reduce_max(T val, T* shared);
```

### 示例

```cpp
#include "tensorcraft/core/warp_utils.hpp"

__global__ void softmax_kernel(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    
    // 加载值
    float val = input[tid];
    
    // Warp 归约求最大值
    float max_val = tensorcraft::warp_reduce_max(val);
    
    // 从 lane 0 广播最大值
    max_val = tensorcraft::warp_broadcast(max_val);
    
    // 计算 exp 并归一化
    float exp_val = expf(val - max_val);
    float sum = tensorcraft::warp_reduce_sum(exp_val);
    
    output[tid] = exp_val / tensorcraft::warp_broadcast(sum);
}
```
