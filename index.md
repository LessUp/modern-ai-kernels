---
layout: default
title: TensorCraft-HPC — Documentation Hub
description: Modern C++ / CUDA AI Kernel Library for GEMM, Attention, Convolution, Normalization, Sparse Operators, and Quantization
---

# TensorCraft-HPC

<div align="center">

**Modern C++ / CUDA AI Kernel Library for High-Performance Computing**

[![GitHub Pages](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml)
[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20%2F23-00599C?logo=c%2B%2B&logoColor=white)

</div>

---

## Language Selection | 语言选择

<div align="center">

| Language | Documentation | GitHub |
|:--------:|:-------------:|:------:|
| **English** | [View Docs →](docs/en/) | [README](README.md) |
| **简体中文** | [查看文档 →](docs/zh/) | [中文 README](README.zh-CN.md) |

</div>

---

## Features | 特性

- **GEMM Kernels**: Naive → Tiled → Double Buffer → Tensor Core (WMMA)
- **Attention**: FlashAttention-style, RoPE, MoE Router
- **Normalization**: LayerNorm, RMSNorm, BatchNorm, Softmax
- **Convolution**: Naive, Im2Col, Depthwise Separable
- **Sparse**: CSR/CSC formats, SpMV, SpMM
- **Quantization**: INT8 and FP8 (CUDA 12.0+) support
- **Python Bindings**: NumPy-compatible interface

---

## Quick Start | 快速开始

```bash
# Clone the repository | 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Build | 构建
cmake --preset dev
cmake --build --preset dev --parallel 2

# Run tests | 运行测试
ctest --preset dev --output-on-failure

# Install Python bindings | 安装 Python 绑定
python -m pip install -e .
```

---

## Documentation Structure | 文档结构

```
docs/
├── en/           # English Documentation | 英文文档
│   ├── getting-started/
│   ├── guides/
│   ├── api/
│   ├── examples/
│   └── reference/
├── zh/           # 简体中文文档 | Simplified Chinese
│   ├── getting-started/
│   ├── guides/
│   ├── api/
│   ├── examples/
│   └── reference/
└── README.md     # Language selection | 语言选择
```

---

## GPU Architecture Support | GPU 架构支持

| Architecture | SM | Tensor Core | TMA | WGMMA |
|--------------|-----|-------------|-----|-------|
| Volta | 70 | ✅ | ❌ | ❌ |
| Turing | 75 | ✅ | ❌ | ❌ |
| Ampere | 80 | ✅ | ❌ | ❌ |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ |
| Hopper | 90 | ✅ | ✅ | ✅ |

---

## Changelog | 变更日志

See [CHANGELOG.md](CHANGELOG.md) for version history.

查看 [CHANGELOG.md](CHANGELOG.md) 了解版本历史。

---

## External Links | 外部链接

- **GitHub Repository**: https://github.com/LessUp/modern-ai-kernels
- **Issue Tracker**: https://github.com/LessUp/modern-ai-kernels/issues
- **License**: [MIT](LICENSE)

---

<div align="center">

**Made with ❤️ for the AI HPC community | 为 AI HPC 社区精心打造**

</div>
