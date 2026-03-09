# TensorCraft-HPC

[English](README.md) | 简体中文

<p align="center">
  <strong>现代化高性能 AI 算子优化库 — Modern C++17/CUDA</strong>
</p>

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/modern-ai-kernels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20%2F23-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

---

教学友好且工业级的高性能 AI 算子优化库。展示从朴素实现到极致优化的渐进式优化技术，涵盖 LLM 和深度学习中最关键的算子。

## 项目愿景

本项目旨在创建一个现代化的算子优化知识库，主要体现在"新"和"深"：

- **新**：使用 C++17/20/23 标准，CMake 3.20+ 构建系统，支持 CUDA 11.0 ~ 13.1
- **深**：不只是写一个矩阵乘法，而是深入到 Tensor Core、FlashAttention、量化等前沿技术

## 核心算子库

| 类别 | 算子 | 优化技术 |
|------|------|----------|
| **Elementwise** | ReLU, SiLU, GeLU, Sigmoid, Tanh, Softplus | 向量化加载 (128-bit)、Functor 模式 |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm | Warp Shuffle 归约、Welford 算法 |
| **GEMM** | 矩阵乘法 (4 级优化) | Shared Memory → Double Buffer → WMMA Tensor Core |
| **Attention** | FlashAttention, RoPE, PagedAttention, MoE | Online Softmax、Tiled I/O、KV Cache |
| **Convolution** | Conv2D, Im2Col, Depthwise, Pointwise | Im2Col + GEMM、Shared Memory |
| **Sparse** | CSR/CSC SpMV, SpMM, 格式转换 | 向量化 SpMV、Warp 协作 |
| **Fusion** | Bias+GeLU, Bias+ReLU | Epilogue Functor 模式 |
| **Quantization** | INT8, FP8 (E4M3) | CAS-based Atomic Min/Max |

## 技术特性

- **现代 C++** — C++17 基础，C++20 Concepts / C++23 可选特性
- **多架构支持** — Volta (SM 70) → Turing (75) → Ampere (80/86) → Ada (89) → Hopper (90)
- **Header-Only** — 核心库为纯头文件，`#include` 即可使用
- **Python 绑定** — 通过 pybind11 提供 NumPy 互操作接口
- **渐进式优化** — 每个算子提供 Naive → 极致优化的多个版本
- **完整测试** — GoogleTest 单元测试 + Google Benchmark 性能基准

## 快速开始

### 环境要求

- CMake 3.20+
- CUDA Toolkit 11.0+ (推荐 12.0+ 以获得 FP8 支持)
- C++17 兼容编译器 (GCC 9+, Clang 10+, MSVC 2019+)
- (可选) Python 3.8+ 用于 Python 绑定

### 构建项目

```bash
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 使用 CMake Presets 配置（推荐）
cmake --preset release
cmake --build build/release -j$(nproc)

# 运行测试
ctest --test-dir build/release --output-on-failure

# Python 绑定
pip install -e .
```

### CMake Presets

```bash
cmake --preset debug      # 调试构建，启用 CUDA 调试
cmake --preset release    # 发布构建，最大优化
cmake --preset profile    # 性能分析构建
```

### 集成方式

```cmake
# 方式 1: 作为子目录
add_subdirectory(modern-ai-kernels)
target_link_libraries(your_target PRIVATE tensorcraft)

# 方式 2: 直接包含头文件
target_include_directories(your_target PRIVATE path/to/include)
target_link_libraries(your_target PRIVATE CUDA::cudart)
```

## 使用示例

```cpp
#include "tensorcraft/kernels/elementwise.hpp"
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/kernels/attention.hpp"
#include "tensorcraft/kernels/normalization.hpp"

using namespace tensorcraft::kernels;

// 激活函数
relu(d_input, d_output, n);
gelu(d_input, d_output, n);
silu(d_input, d_output, n);

// GEMM — 选择优化级别
launch_gemm(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f, GemmVersion::TensorCore);

// FlashAttention
float scale = 1.0f / sqrtf(head_dim);
launch_flash_attention(d_Q, d_K, d_V, d_O,
                       batch_size, num_heads, seq_len, head_dim, scale);

// LayerNorm / RMSNorm
layernorm(d_input, d_gamma, d_beta, d_output, batch_size, hidden_size);
rmsnorm(d_input, d_weight, d_output, batch_size, hidden_size);
```

## 项目结构

```
modern-ai-kernels/
├── include/tensorcraft/
│   ├── core/                    # 核心工具 (错误检查、特性检测、类型系统)
│   ├── memory/                  # 内存管理 (Tensor RAII、内存池、向量化加载)
│   └── kernels/                 # 算子实现 (8 大类 30+ 算子)
├── src/python_ops/              # Python 绑定 (pybind11)
├── tests/                       # GoogleTest 单元测试
├── benchmarks/                  # Google Benchmark 性能基准
├── docs/                        # 文档 (API、架构、优化指南)
├── examples/                    # 使用示例
├── CMakeLists.txt               # 构建配置
└── CMakePresets.json             # CMake 预设
```

## GPU 架构支持

| 架构 | SM | 代表 GPU | CUDA 最低版本 |
|------|------|----------|---------------|
| Volta | 70 | V100 | 11.0 |
| Turing | 75 | RTX 2080, T4 | 11.0 |
| Ampere | 80, 86 | A100, RTX 3090 | 11.0 |
| Ada Lovelace | 89 | RTX 4090, L40 | 11.8 |
| Hopper | 90 | H100 | 12.0 |

## 文档

- [API 参考](docs/api_reference.md) — 完整的 C++ & Python API 文档
- [架构设计](docs/architecture.md) — 模块架构和设计决策
- [Modern C++ 指南](docs/modern_cpp_cuda.md) — 现代 C++ 在 CUDA 中的应用
- [优化指南](docs/optimization_guide.md) — Kernel 优化技术详解
- [安装指南](docs/INSTALL.md) — 多平台环境搭建
- [问题排查](docs/TROUBLESHOOTING.md) — 常见问题

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。
