# TensorCraft-HPC

English | [简体中文](README.zh-CN.md)

<p align="center">
  <strong>Demystifying High-Performance AI Kernels with Modern C++ & CUDA</strong>
</p>

[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/modern-ai-kernels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17/20-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#contributing">Contributing</a>
</p>

---

TensorCraft-HPC 是一个现代化的、教学友好且工业级的高性能 AI 算子优化库。它展示了从朴素实现到极致优化的渐进式优化技术，涵盖了 LLM 和深度学习中最关键的算子。

## 🎯 项目愿景

本项目旨在创建一个现代化的算子优化知识库，主要体现在"新"和"深"：

- **新**：使用 C++17/20/23 标准，CMake 3.20+ 构建系统，支持 CUDA 11.0-13.1
- **深**：不只是写一个矩阵乘法，而是深入到 Tensor Core、FlashAttention、量化等前沿技术

## ✨ Features

### 核心算子库

| 类别 | 算子 | 优化级别 |
|------|------|----------|
| **Elementwise** | ReLU, SiLU, GeLU, Sigmoid, Tanh, Softplus | 向量化加载 |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm | Warp Shuffle |
| **GEMM** | 矩阵乘法 | Naive → Tiled → Double Buffer → Tensor Core |
| **Attention** | FlashAttention, RoPE, PagedAttention, MoE | Online Softmax |
| **Convolution** | Conv2D, Im2Col, Depthwise, Pointwise | 多算法支持 |
| **Sparse** | CSR/CSC SpMV, SpMM | 向量化 SpMV |
| **Fusion** | Bias+GeLU, Bias+ReLU | Epilogue 模式 |
| **Quantization** | INT8, FP8 (CUDA 12.0+) | 量化/反量化 |

### 技术特性

- 🚀 **现代 C++**: C++17 基础，C++20/23 可选特性（Concepts, constexpr if）
- 🎮 **多架构支持**: Volta (SM 7.0+), Ampere (SM 8.0+), Hopper (SM 9.0+)
- 📦 **Header-Only**: 核心库为纯头文件，易于集成
- 🐍 **Python 绑定**: 通过 pybind11 提供 Python 接口
- 🧪 **完整测试**: GoogleTest 单元测试 + 属性测试
- 📊 **性能基准**: Google Benchmark 性能测试

## 🚀 Quick Start

### 环境要求

- CMake 3.20+
- CUDA Toolkit 11.0+ (推荐 12.0+ 以获得 FP8 支持)
- C++17 兼容编译器 (GCC 9+, Clang 10+, MSVC 2019+)
- (可选) Python 3.8+ 用于 Python 绑定

### 构建项目

```bash
# 克隆仓库
git clone https://github.com/your-username/tensorcraft-hpc.git
cd tensorcraft-hpc

# 使用 CMake Presets 配置（推荐）
cmake --preset release

# 或手动配置
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# 构建
cmake --build build/release -j$(nproc)

# 运行测试
ctest --test-dir build/release --output-on-failure

# 运行基准测试
./build/release/benchmarks/gemm_benchmark
```

### 在项目中使用

**方式 1: 作为子目录**

```cmake
# CMakeLists.txt
add_subdirectory(tensorcraft-hpc)
target_link_libraries(your_target PRIVATE tensorcraft)
```

**方式 2: 直接包含头文件**

```cmake
target_include_directories(your_target PRIVATE path/to/tensorcraft-hpc/include)
target_link_libraries(your_target PRIVATE CUDA::cudart)
```

## 📖 Documentation

### 使用示例

#### Elementwise 操作

```cpp
#include "tensorcraft/kernels/elementwise.hpp"

using namespace tensorcraft::kernels;

// 激活函数
relu(d_input, d_output, n);
gelu(d_input, d_output, n);
silu(d_input, d_output, n);

// 向量运算
vector_add(d_a, d_b, d_c, n);
vector_mul(d_a, d_b, d_c, n);

// 自定义激活函数
launch_elementwise(d_input, d_output, n, LeakyReLU<float>{0.01f});
```

#### GEMM 矩阵乘法

```cpp
#include "tensorcraft/kernels/gemm.hpp"

using namespace tensorcraft::kernels;

// 默认使用 Tiled 版本
gemm(d_A, d_B, d_C, M, N, K);

// 选择优化级别
launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::Naive);
launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::Tiled);
launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::DoubleBuffer);

// Tensor Core (需要 half 精度)
#ifdef TC_HAS_WMMA
launch_gemm_wmma(d_A_half, d_B_half, d_C_float, M, N, K);
#endif

// 矩阵转置
transpose(d_input, d_output, rows, cols);
```

#### 归一化层

```cpp
#include "tensorcraft/kernels/normalization.hpp"

using namespace tensorcraft::kernels;

// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
layernorm(d_input, d_gamma, d_beta, d_output, batch_size, hidden_size);

// RMSNorm: y = x / RMS(x) * weight (LLaMA, Mistral 等模型使用)
rmsnorm(d_input, d_weight, d_output, batch_size, hidden_size);

// BatchNorm (推理模式)
launch_batchnorm(d_input, d_gamma, d_beta, d_mean, d_var, d_output,
                 N, C, H, W, eps, /*fuse_relu=*/false);
```

#### Attention 机制

```cpp
#include "tensorcraft/kernels/attention.hpp"

using namespace tensorcraft::kernels;

// FlashAttention 风格的注意力计算
float scale = 1.0f / sqrtf(head_dim);
launch_flash_attention(d_Q, d_K, d_V, d_O,
                       batch_size, num_heads, seq_len, head_dim, scale);

// RoPE 位置编码
precompute_rope_cache(d_cos, d_sin, max_seq_len, head_dim);
launch_rope(d_x, d_cos, d_sin, batch_size, seq_len, num_heads, head_dim, start_pos);

// MoE 路由
launch_moe_router(d_gate_logits, d_expert_indices, d_expert_weights,
                  batch_size, num_experts, top_k);
```

#### 卷积操作

```cpp
#include "tensorcraft/kernels/conv2d.hpp"

using namespace tensorcraft::kernels;

// 标准卷积
conv2d(d_input, d_weight, d_bias, d_output,
       N, C, H, W, K, R, S, stride, padding);

// Depthwise 卷积 (MobileNet 等)
conv2d_depthwise(d_input, d_weight, d_bias, d_output,
                 N, C, H, W, R, S, stride, padding);

// Im2Col 变换 (用于 Im2Col + GEMM 卷积)
launch_im2col(d_input, d_col, N, C, H, W, R, S, stride, stride, pad, pad);
```

#### 稀疏矩阵

```cpp
#include "tensorcraft/kernels/sparse.hpp"

using namespace tensorcraft::kernels;

// CSR 格式的 SpMV: y = A * x
launch_spmv_csr(d_values, d_col_indices, d_row_ptrs, d_x, d_y, rows);

// CSR 格式的 SpMM: C = A * B
launch_spmm_csr(d_A_values, d_A_col_indices, d_A_row_ptrs,
                d_B, d_C, M, K, N);
```

#### 算子融合与量化

```cpp
#include "tensorcraft/kernels/fusion.hpp"

using namespace tensorcraft::kernels;

// GEMM + Bias + GeLU 融合
gemm_bias_gelu(d_A, d_B, d_bias, d_C, M, N, K);

// GEMM + Bias + ReLU 融合
gemm_bias_relu(d_A, d_B, d_bias, d_C, M, N, K);

// INT8 量化
quantize_int8(d_input, d_output_int8, scale, zero_point, n);
dequantize_int8(d_input_int8, d_output, scale, zero_point, n);
```

### Python 接口

```python
import tensorcraft_ops as tc
import numpy as np

# 激活函数
input_data = np.random.randn(1024, 512).astype(np.float32)
output = tc.relu(input_data)
output = tc.gelu(input_data)
output = tc.silu(input_data)

# Softmax
output = tc.softmax(input_data)

# 归一化
gamma = np.ones(512, dtype=np.float32)
beta = np.zeros(512, dtype=np.float32)
output = tc.layernorm(input_data, gamma, beta)

weight = np.ones(512, dtype=np.float32)
output = tc.rmsnorm(input_data, weight)

# GEMM
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 128).astype(np.float32)
C = tc.gemm(A, B, version='tiled')  # 'naive', 'tiled', 'double_buffer'
```

## 📊 Benchmarks

### GEMM 性能对比

在 NVIDIA RTX 3090 上的测试结果：

| 矩阵大小 | Naive | Tiled | Double Buffer | cuBLAS |
|----------|-------|-------|---------------|--------|
| 512x512  | 15 GFLOPS | 180 GFLOPS | 220 GFLOPS | 280 GFLOPS |
| 1024x1024 | 18 GFLOPS | 350 GFLOPS | 450 GFLOPS | 520 GFLOPS |
| 2048x2048 | 20 GFLOPS | 480 GFLOPS | 620 GFLOPS | 750 GFLOPS |

### 运行基准测试

```bash
# GEMM 基准测试
./build/release/benchmarks/gemm_benchmark

# Attention 基准测试
./build/release/benchmarks/attention_benchmark

# 卷积基准测试
./build/release/benchmarks/conv_benchmark
```

## 📁 项目结构

```
TensorCraft-HPC/
├── include/tensorcraft/
│   ├── core/                    # 核心工具
│   │   ├── cuda_check.hpp       # CUDA 错误检查
│   │   ├── features.hpp         # 特性检测
│   │   └── type_traits.hpp      # 类型特征
│   ├── memory/                  # 内存管理
│   │   ├── aligned_vector.hpp   # 对齐向量
│   │   ├── tensor.hpp           # Tensor 封装
│   │   └── memory_pool.hpp      # 内存池
│   └── kernels/                 # 算子实现
│       ├── elementwise.hpp      # Elementwise 算子
│       ├── softmax.hpp          # Softmax
│       ├── normalization.hpp    # 归一化层
│       ├── gemm.hpp             # GEMM
│       ├── attention.hpp        # Attention
│       ├── conv2d.hpp           # 卷积
│       ├── sparse.hpp           # 稀疏矩阵
│       └── fusion.hpp           # 融合与量化
├── src/python_ops/              # Python 绑定
├── tests/                       # 单元测试
├── benchmarks/                  # 性能基准
├── docs/                        # 文档
│   ├── modern_cpp_cuda.md       # Modern C++ 指南
│   ├── optimization_guide.md    # 优化指南
│   ├── api_reference.md         # API 参考
│   └── architecture.md          # 架构设计
├── CMakeLists.txt
├── CMakePresets.json
└── README.md
```

## 🔧 配置选项

### CMake 选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `TC_BUILD_TESTS` | ON | 构建测试 |
| `TC_BUILD_BENCHMARKS` | ON | 构建基准测试 |
| `TC_BUILD_PYTHON` | ON | 构建 Python 绑定 |
| `TC_ENABLE_FP16` | ON | 启用 FP16 支持 |
| `TC_ENABLE_BF16` | ON | 启用 BF16 支持 |

### CMake Presets

```bash
cmake --preset debug      # 调试构建，启用 CUDA 调试
cmake --preset release    # 发布构建，最大优化
cmake --preset profile    # 性能分析构建
```

## 🤝 Contributing

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

### 开发路线图

- [x] Phase 1: 基础设施和核心算子
- [x] Phase 2: GEMM 优化和 Attention
- [x] Phase 3: 卷积和稀疏矩阵
- [x] Phase 4: 融合和量化
- [ ] Phase 5: CUDA 12+ 高级特性 (TMA, WGMMA)
- [ ] Phase 6: 更多 LLM 算子 (KV Cache, Speculative Decoding)

## 📚 详细文档

- [API 参考](docs/api_reference.md) - 完整的 API 文档
- [架构设计](docs/architecture.md) - 系统架构和设计决策
- [Modern C++ 指南](docs/modern_cpp_cuda.md) - 现代 C++ 在 CUDA 中的应用
- [优化指南](docs/optimization_guide.md) - Kernel 优化技术详解

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 🙏 Acknowledgments

- NVIDIA CUTLASS - GEMM 优化模式的灵感来源
- FlashAttention - Attention 优化技术
- PyTorch/TensorFlow - API 设计参考
- CUDA 社区 - 持续的学习资源

---

<p align="center">
  Made with ❤️ for the HPC and AI community
</p>
