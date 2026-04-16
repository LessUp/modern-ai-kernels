# Design Document: TensorCraft-HPC

## Overview

TensorCraft-HPC 是一个模块化的高性能 AI 算子优化库，采用分层架构设计，支持从 CUDA 11.0 到 CUDA 13.1 的多版本兼容。项目核心设计原则：

1. **渐进式优化**: 每个算子提供从 naive 到极致优化的多个版本
2. **编译期特性检测**: 通过宏和模板在编译期选择最优实现路径
3. **零成本抽象**: 使用现代 C++ 模板技术实现高性能泛型编程
4. **教学友好**: 代码结构清晰，注释详尽，便于学习理解

## Documentation Structure

```
docs/
├── README.md                 # 文档导航入口
├── getting-started/          # 快速入门
│   ├── README.md
│   ├── installation.md       # 安装指南
│   └── troubleshooting.md    # 问题排查
├── guides/                   # 指南
│   ├── README.md
│   ├── architecture.md       # 架构设计
│   ├── optimization.md       # 优化指南
│   └── modern-cpp-cuda.md    # 现代 C++ CUDA
├── api/                      # API 参考
│   ├── README.md
│   ├── core.md               # Core 模块 API
│   ├── memory.md             # Memory 模块 API
│   ├── kernels.md            # Kernels 模块 API
│   └── python.md             # Python API
├── examples/                 # 示例
│   ├── README.md
│   ├── basic-gemm.md         # GEMM 示例
│   ├── attention.md          # Attention 示例
│   ├── normalization.md      # 归一化示例
│   └── python-usage.md       # Python 使用示例
└── reference/                # 参考文档
    ├── README.md
    ├── contributing.md       # 贡献指南
    ├── changelog.md          # 版本变更
    ├── code-of-conduct.md    # 行为准则
    └── security.md           # 安全策略
```

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

- `cuda_check.hpp`: CUDA 错误检查宏和异常处理
- `features.hpp`: 编译时特性检测 (C++17/20/23, CUDA 11/12/13)
- `type_traits.hpp`: 类型特征和 Concepts
- `warp_utils.hpp`: Warp 级别规约原语

### 2. Memory Management (include/tensorcraft/memory/)

- `aligned_vector.hpp`: 对齐向量类型，支持向量化内存访问
- `tensor.hpp`: RAII 风格的 GPU Tensor 封装
- `memory_pool.hpp`: 线程安全的 GPU 内存池

### 3. Kernels (include/tensorcraft/kernels/)

- `elementwise.hpp`: 逐元素操作和激活函数
- `softmax.hpp`: 数值稳定的 Softmax
- `normalization.hpp`: LayerNorm, RMSNorm, BatchNorm
- `gemm.hpp`: 矩阵乘法 (Naive → Tiled → Double Buffer → Tensor Core)
- `attention.hpp`: FlashAttention, RoPE, MoE Router
- `conv2d.hpp`: 2D 卷积操作
- `sparse.hpp`: 稀疏矩阵操作 (CSR/CSC)
- `fusion.hpp`: 融合算子和量化

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 2.0.0 | 2026-03-09 | MemoryPool bug fix, atomicMin/Max fix, warp_utils extraction, FlashAttention rewrite |
| 1.1.0 | 2026-01-08 | Build system fixes for CUDA-optional environments |
| 1.0.1 | 2025-02-13 | Project infrastructure improvements |
| 1.0.0 | 2024-01-01 | Initial release |

## Build System

### CMake Presets

| Preset | Purpose |
|--------|---------|
| `dev` | Recommended CUDA development preset |
| `python-dev` | Lighter CUDA build for Python bindings |
| `release` | Full release build with benchmarks |
| `debug` | Debug-oriented CUDA build |
| `cpu-smoke` | CPU-only configure/install validation |

### Dependencies

- CUDA Toolkit 12.8 (targeted)
- CMake 3.20+
- C++17 compiler
- pybind11 (for Python bindings)

## Correctness Properties

### Property 1: Tensor RAII Memory Management

For any Tensor object created with a given shape, GPU memory SHALL be allocated on construction and freed on destruction, with no memory leaks.

### Property 2: GEMM Mathematical Correctness

For any matrices A[M×K], B[K×N] and scalars alpha, beta, GEMM SHALL compute C = alpha * A @ B + beta * C correctly within floating-point tolerance.

### Property 3: Softmax Row Sum Invariant

For any input matrix X, the Softmax output S SHALL satisfy:
1. sum(S[i, :]) = 1.0 for all rows i (within tolerance)
2. S[i, j] >= 0 for all elements

### Property 4: Optimization Level Numerical Equivalence

For any kernel with multiple optimization levels, all versions SHALL produce numerically equivalent outputs within tolerance.

## Testing Strategy

1. **Unit Tests**: GoogleTest framework for specific examples and edge cases
2. **Property-Based Tests**: Random input validation for general properties
3. **Python Verification**: Comparison with PyTorch reference implementations
4. **Benchmarks**: Google Benchmark for performance regression detection

## Error Handling

All CUDA API calls wrapped with `TC_CUDA_CHECK` macro that throws `CudaException` with file, line, and error description.
