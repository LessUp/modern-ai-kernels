---
layout: default
title: TensorCraft-HPC — 现代化高性能 AI 算子库
---

# TensorCraft-HPC

**Demystifying High-Performance AI Kernels with Modern C++ & CUDA**

现代化的、教学友好且工业级的高性能 AI 算子优化库。展示从朴素实现到极致优化的渐进式优化技术，涵盖 LLM 和深度学习中最关键的算子。

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20%2F23-00599C?logo=c%2B%2B&logoColor=white)

---

## 架构总览

```
tensorcraft/
├── core/               ← 基础设施 (错误检查、特性检测、类型系统)
├── memory/             ← 内存管理 (Tensor RAII、内存池、向量化加载)
└── kernels/            ← 算子实现 (8 大类 30+ 算子)
    ├── elementwise     ← ReLU, GeLU, SiLU, Sigmoid, Tanh …
    ├── normalization   ← LayerNorm, RMSNorm, BatchNorm
    ├── gemm            ← Naive → Tiled → Double Buffer → Tensor Core
    ├── attention       ← FlashAttention, RoPE, PagedAttention, MoE
    ├── conv2d          ← Conv2D, Im2Col, Depthwise, Pointwise
    ├── sparse          ← CSR/CSC SpMV, SpMM
    ├── fusion          ← Bias+GeLU, Bias+ReLU (Epilogue 模式)
    └── quantization    ← INT8, FP8 (CUDA 12.0+)
```

## 算子矩阵

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

## 核心特性

| 特性 | 详情 |
|------|------|
| **现代 C++** | C++17 基础，C++20 Concepts / C++23 可选 |
| **多架构** | Volta (SM 70) → Turing (75) → Ampere (80/86) → Ada (89) → Hopper (90) |
| **Header-Only** | 纯头文件设计，`#include` 即可使用 |
| **Python 绑定** | pybind11 提供 NumPy 互操作接口 |
| **渐进式优化** | 每个算子提供 Naive → 极致优化的多个版本 |
| **完整测试** | GoogleTest 单元测试 + Google Benchmark 性能基准 |
| **CUDA 兼容** | CUDA 11.0 ~ 13.1，FP8 需 CUDA 12.0+ |

## 快速开始

```bash
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# CMake Presets 构建 (推荐)
cmake --preset release
cmake --build build/release -j$(nproc)

# 运行测试
ctest --test-dir build/release --output-on-failure

# Python 绑定
pip install -e .
```

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/kernels/attention.hpp"

using namespace tensorcraft::kernels;

// GEMM — 选择优化级别
launch_gemm(A, B, C, M, N, K, 1.0f, 0.0f, GemmVersion::TensorCore);

// FlashAttention
launch_flash_attention(Q, K, V, O, batch, heads, seq_len, head_dim, scale);
```

## GPU 架构支持

| 架构 | SM | 代表 GPU | CUDA 最低版本 |
|------|------|----------|---------------|
| Volta | 70 | V100 | 11.0 |
| Turing | 75 | RTX 2080, T4 | 11.0 |
| Ampere | 80, 86 | A100, RTX 3090 | 11.0 |
| Ada Lovelace | 89 | RTX 4090, L40 | 11.8 |
| Hopper | 90 | H100 | 12.0 |

## 文档导航

| 文档 | 说明 |
|------|------|
| [README](README.md) | 项目概述、完整使用示例 |
| [安装指南](docs/INSTALL.md) | 多平台环境搭建 (Linux / Windows / macOS) |
| [API 参考](docs/api_reference.md) | 完整的 C++ & Python API 文档 |
| [架构设计](docs/architecture.md) | 模块架构、设计模式、扩展指南 |
| [优化指南](docs/optimization_guide.md) | GEMM / Softmax / Attention 优化技术详解 |
| [Modern C++ in CUDA](docs/modern_cpp_cuda.md) | C++17/20/23 在 CUDA 中的最佳实践 |
| [问题排查](docs/TROUBLESHOOTING.md) | 常见构建 & 运行时问题 |
| [CHANGELOG](CHANGELOG.md) | 版本变更记录 |
| [CONTRIBUTING](CONTRIBUTING.md) | 贡献指南 |

## 项目链接

- [GitHub 仓库](https://github.com/LessUp/modern-ai-kernels)
- [问题反馈](https://github.com/LessUp/modern-ai-kernels/issues)
- [README (English)](README.md) ｜ [README (中文)](README.zh-CN.md)
