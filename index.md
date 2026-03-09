---
layout: default
title: TensorCraft-HPC
---

# TensorCraft-HPC

Modern C++17/CUDA AI Kernel Library — 现代化 AI 算子库，覆盖 Elementwise、GEMM、FlashAttention、Conv2D、SpMV、FP8 量化。

## 算子矩阵

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

## 技术特性

- **现代 C++** — C++17 基础，C++20/23 可选特性（Concepts, constexpr if）
- **多架构支持** — Volta (SM 7.0+), Ampere (SM 8.0+), Hopper (SM 9.0+)
- **Header-Only** — 核心库为纯头文件，易于集成
- **Python 绑定** — 通过 pybind11 提供 Python 接口

## 文档

- [README](README.md) — 项目概述
- [安装指南](docs/INSTALL.md) — 环境搭建
- [API 参考](docs/api_reference.md) — 接口文档
- [架构设计](docs/architecture.md) — 系统架构
- [优化指南](docs/optimization_guide.md) — 性能优化技巧
- [现代 C++/CUDA](docs/modern_cpp_cuda.md) — 现代 C++ 在 CUDA 中的实践
- [问题排查](docs/TROUBLESHOOTING.md) — 常见问题

## 快速开始

```bash
# 使用 CMake Presets 构建
cmake --preset release
cmake --build build/release -j$(nproc)

# 运行测试
cd build/release && ctest --output-on-failure

# Python 绑定
pip install -e .
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | CUDA C++17, Python |
| 构建 | CMake 3.20+ |
| GPU | SM 70+ (Volta → Hopper) |
| 绑定 | pybind11 |
| 测试 | Google Test, Google Benchmark |

## 链接

- [GitHub 仓库](https://github.com/LessUp/modern-ai-kernels)
- [README](README.md)
