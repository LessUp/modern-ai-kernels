# TensorCraft-HPC

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20-00599C?logo=c%2B%2B&logoColor=white)

[English](README.md) | 简体中文

现代化的、教学友好且工业级的高性能 AI 算子优化库。展示从朴素实现到极致优化的渐进式优化技术。

## 核心算子库

| 类别 | 算子 | 优化级别 |
|------|------|----------|
| **Elementwise** | ReLU, SiLU, GeLU, Sigmoid, Tanh | 向量化加载 |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm | Warp Shuffle |
| **GEMM** | 矩阵乘法 | Naive → Tiled → Double Buffer → Tensor Core |
| **Attention** | FlashAttention, RoPE, PagedAttention, MoE | Online Softmax |
| **Convolution** | Conv2D, Im2Col, Depthwise, Pointwise | 多算法支持 |
| **Sparse** | CSR/CSC SpMV, SpMM | 向量化 SpMV |
| **Fusion** | Bias+GeLU, Bias+ReLU | Epilogue 模式 |
| **Quantization** | INT8, FP8 (CUDA 12.0+) | 量化/反量化 |

## 技术特性

- 现代 C++ (C++17 基础, C++20/23 可选)
- 多架构支持: Volta (SM 7.0+), Ampere (SM 8.0+), Hopper (SM 9.0+)
- Header-Only 核心库
- Python 绑定 (pybind11)
- 完整测试 (GoogleTest + 属性测试)
- 性能基准 (Google Benchmark)

## 快速开始

```bash
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

cmake --preset release
cmake --build build/release -j$(nproc)
ctest --test-dir build/release --output-on-failure
```

### 集成方式

```cmake
# 作为子目录
add_subdirectory(modern-ai-kernels)
target_link_libraries(your_target PRIVATE tensorcraft)

# 或直接包含头文件
target_include_directories(your_target PRIVATE path/to/include)
```

## 使用示例

```cpp
#include "tensorcraft/kernels/elementwise.hpp"
#include "tensorcraft/kernels/gemm.hpp"

using namespace tensorcraft::kernels;

// 激活函数
relu(d_input, d_output, n);
gelu(d_input, d_output, n);

// GEMM
gemm(d_A, d_B, d_C, M, N, K, GemmVersion::RegisterBlocked);
```

## 环境要求

- CMake 3.20+, CUDA 11.0+ (推荐 12.0+), C++17 编译器

## 许可证

MIT License
