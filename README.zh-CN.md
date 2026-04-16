# TensorCraft-HPC

<div align="center">

**现代 C++ / CUDA AI 高性能计算内核库**

Modern C++ / CUDA AI Kernel Library for High-Performance Computing

[English](README.md) | [简体中文](README.zh-CN.md) | [文档](docs/) | [API 参考](docs/zh/api/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20%2F23-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

</div>

---

## 项目简介

**TensorCraft-HPC** 是一个现代化的 C++/CUDA 库，专为学习、验证和实现高性能 AI 计算内核而设计。它提供了一系列全面的优化实现，涵盖深度学习和 AI 工作负载中使用的基础算子。

### 核心特性

- **GEMM 内核**: 从朴素实现到 Tensor Core (WMMA) 实现
  - 朴素、平铺、双缓冲和张量核心版本
  - 性能比较和优化研究
  
- **注意力机制**: 内存高效的注意力计算
  - FlashAttention 风格的融合注意力
  - RoPE (旋转位置编码)
  - MoE (专家混合) 路由器
  
- **归一化**: 标准归一化层
  - LayerNorm、RMSNorm、BatchNorm
  - 线程束优化实现
  
- **卷积**: 2D 卷积操作
  - 朴素、Im2Col 和深度可分离卷积
  
- **稀疏操作**: CSR/CSC 格式支持
  - 稀疏矩阵-向量 (SpMV) 和矩阵-矩阵 (SpMM) 乘法
  
- **量化**: INT8 和 FP8 (CUDA 12.0+) 支持
  - 与量化融合的算子
  
- **Python 绑定**: 通过 pybind11 提供 NumPy 兼容接口

---

## 快速开始

### 环境要求

- **CUDA Toolkit**: 12.8
- **CMake**: 3.20+
- **C++ 编译器**: 支持 C++17
- **NVIDIA GPU**: 推荐用于运行测试

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 使用开发预设构建
cmake --preset dev
cmake --build --preset dev --parallel 2

# 运行测试
ctest --preset dev --output-on-failure

# 安装 Python 绑定
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

### 快速示例

**C++:**
```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

// 使用 RAII Tensor 封装
tensorcraft::FloatTensor A({256, 512});
tensorcraft::FloatTensor B({512, 128});
tensorcraft::FloatTensor C({256, 128});

// GEMM 操作
tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 256, 128, 512);
```

**Python:**
```python
import tensorcraft_ops as tc
import numpy as np

# 矩阵乘法
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 128).astype(np.float32)
C = tc.gemm(A, B)

# 激活与归一化
x = np.random.randn(32, 256).astype(np.float32)
y = tc.gelu(tc.layernorm(x, gamma, beta))
```

---

## 文档

### 双语文档 | Bilingual Documentation

我们提供 **英文** 和 **简体中文** 的完整文档：

- **English**: [docs/en/](docs/en/README.md)
- **中文**: [docs/zh/](docs/zh/README.md)

### 文档结构

| 章节 | 描述 | 链接 |
|------|------|------|
| 入门指南 | 安装和故障排除 | [en](docs/en/getting-started/) / [zh](docs/zh/getting-started/) |
| 开发指南 | 架构设计和优化 | [en](docs/en/guides/) / [zh](docs/zh/guides/) |
| API 参考 | 完整的 API 文档 | [en](docs/en/api/) / [zh](docs/zh/api/) |
| 示例教程 | 代码示例和教程 | [en](docs/en/examples/) / [zh](docs/zh/examples/) |
| 变更日志 | 版本历史 | [CHANGELOG.md](CHANGELOG.md) |

### 在线文档

📚 **https://lessup.github.io/modern-ai-kernels/**

---

## GPU 架构支持

| 架构 | SM | Tensor Core | TMA | WGMMA |
|------|-----|-------------|-----|-------|
| Volta | 70 | ✅ | ❌ | ❌ |
| Turing | 75 | ✅ | ❌ | ❌ |
| Ampere | 80 | ✅ | ❌ | ❌ |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ |
| Hopper | 90 | ✅ | ✅ | ✅ |

---

## 项目结构

```
modern-ai-kernels/
├── specs/                  # 规范文档（单一事实来源）
│   ├── product/           # 产品功能定义 (PRD)
│   ├── rfc/               # 技术设计文档
│   ├── api/               # API 规范定义
│   ├── db/                # 数据库 Schema 设计
│   └── testing/           # 测试用例规范与实现计划
├── include/tensorcraft/    # 头文件内核库
│   ├── core/              # CUDA 错误处理、类型特征
│   ├── memory/            # Tensor、内存池
│   └── kernels/           # 所有计算内核
├── src/python_ops/        # Python 绑定
├── tests/                 # 单元测试
├── benchmarks/            # 性能基准
├── docs/                  # 文档 (en/, zh/)
├── changelog/             # 开发变更日志
└── examples/              # 示例代码
```

---

## 构建预设

| 预设 | 用途 |
|------|------|
| `dev` | 推荐的 CUDA 开发预设 |
| `python-dev` | 专注于 Python 绑定的轻量构建 |
| `release` | 带基准测试的完整发布构建 |
| `debug` | 面向调试的 CUDA 构建 |
| `cpu-smoke` | 仅 CPU 的配置/安装验证 |

---

## 参与贡献

我们欢迎各种形式的贡献！详情请参阅我们的 [贡献指南](docs/zh/reference/contributing.md)。

- 🐛 [提交 Issue](https://github.com/LessUp/modern-ai-kernels/issues)
- 💡 [请求功能](https://github.com/LessUp/modern-ai-kernels/issues)
- 🔀 [提交 Pull Request](docs/zh/reference/contributing.md)

---

## 许可协议

本项目采用 [MIT 许可证](LICENSE)。

---

## 致谢

- 受 CUTLASS、FlashAttention 和其他优秀 CUDA 库的启发
- 使用现代 C++17/20 特性和 CUDA 12.8 构建

---

<div align="center">

**为 AI HPC 社区精心打造 ❤️**

</div>
