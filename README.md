# TensorCraft-HPC

<div align="center">

**Modern C++/CUDA AI Kernel Library for High-Performance Computing**

Modern C++ / CUDA AI 高性能计算内核库

[English](README.md) | [简体中文](README.zh-CN.md) | [Documentation](docs/) | [API Reference](docs/en/api/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20%2F23-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

</div>

---

## Overview

**TensorCraft-HPC** is a modern C++/CUDA library designed for learning, validating, and implementing high-performance AI computing kernels. It provides a comprehensive collection of optimized implementations covering fundamental operations used in deep learning and AI workloads.

### Key Features

- **GEMM Kernels**: From naive to Tensor Core (WMMA) implementations
  - Naive, Tiled, Double-Buffer, and Tensor Core versions
  - Performance comparison and optimization study
  
- **Attention Mechanisms**: Memory-efficient attention computation
  - FlashAttention-style fused attention
  - RoPE (Rotary Positional Embeddings)
  - MoE (Mixture of Experts) Router
  
- **Normalization**: Standard normalization layers
  - LayerNorm, RMSNorm, BatchNorm
  - Warp-optimized implementations
  
- **Convolution**: 2D convolution operations
  - Naive, Im2Col, and Depthwise Separable
  
- **Sparse Operations**: CSR/CSC format support
  - Sparse Matrix-Vector (SpMV) and Matrix-Matrix (SpMM) multiplication
  
- **Quantization**: INT8 and FP8 (CUDA 12.0+) support
  - Fused operations with quantization
  
- **Python Bindings**: NumPy-compatible interface via pybind11

---

## Quick Start

### Prerequisites

- **CUDA Toolkit**: 12.8
- **CMake**: 3.20+
- **C++ Compiler**: C++17-capable
- **NVIDIA GPU**: Recommended for running tests

### Installation

```bash
# Clone the repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Build with development preset
cmake --preset dev
cmake --build --preset dev --parallel 2

# Run tests
ctest --preset dev --output-on-failure

# Install Python bindings
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

### Quick Example

**C++:**
```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

// Using RAII Tensor wrapper
tensorcraft::FloatTensor A({256, 512});
tensorcraft::FloatTensor B({512, 128});
tensorcraft::FloatTensor C({256, 128});

// GEMM operation
tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 256, 128, 512);
```

**Python:**
```python
import tensorcraft_ops as tc
import numpy as np

# Matrix multiplication
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 128).astype(np.float32)
C = tc.gemm(A, B)

# Activation & Normalization  
x = np.random.randn(32, 256).astype(np.float32)
y = tc.gelu(tc.layernorm(x, gamma, beta))
```

---

## Documentation

### Bilingual Documentation | 双语文档

We provide comprehensive documentation in both **English** and **简体中文**:

- **English**: [docs/en/](docs/en/README.md)
- **中文**: [docs/zh/](docs/zh/README.md)

### Documentation Structure

| Section | Description | Link |
|---------|-------------|------|
| Getting Started | Installation and troubleshooting | [en](docs/en/getting-started/) / [zh](docs/zh/getting-started/) |
| Guides | Architecture and optimization | [en](docs/en/guides/) / [zh](docs/zh/guides/) |
| API Reference | Complete API documentation | [en](docs/en/api/) / [zh](docs/zh/api/) |
| Examples | Code examples and tutorials | [en](docs/en/examples/) / [zh](docs/zh/examples/) |
| Changelog | Version history | [CHANGELOG.md](CHANGELOG.md) |

### Online Documentation

📚 **https://lessup.github.io/modern-ai-kernels/**

---

## GPU Architecture Support

| Architecture | SM | Tensor Core | TMA | WGMMA |
|--------------|-----|-------------|-----|-------|
| Volta | 70 | ✅ | ❌ | ❌ |
| Turing | 75 | ✅ | ❌ | ❌ |
| Ampere | 80 | ✅ | ❌ | ❌ |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ |
| Hopper | 90 | ✅ | ✅ | ✅ |

---

## Project Structure

```
modern-ai-kernels/
├── include/tensorcraft/    # Header-only kernel library
│   ├── core/              # CUDA error handling, type traits
│   ├── memory/            # Tensor, memory pool
│   └── kernels/           # All compute kernels
├── src/python_ops/        # Python bindings
├── tests/                 # Unit tests
├── benchmarks/            # Performance benchmarks
├── docs/                  # Documentation (en/, zh/)
├── changelog/             # Development changelog
└── examples/              # Example code
```

---

## Build Presets

| Preset | Purpose |
|--------|---------|
| `dev` | Recommended CUDA development preset |
| `python-dev` | Lighter build focused on Python bindings |
| `release` | Full release build with benchmarks |
| `debug` | Debug-oriented CUDA build |
| `cpu-smoke` | CPU-only configure/install validation |

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/en/reference/contributing.md) for details.

- 🐛 [Report Issues](https://github.com/LessUp/modern-ai-kernels/issues)
- 💡 [Request Features](https://github.com/LessUp/modern-ai-kernels/issues)
- 🔀 [Submit Pull Requests](docs/en/reference/contributing.md)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- Inspired by CUTLASS, FlashAttention, and other excellent CUDA libraries
- Built with modern C++17/20 features and CUDA 12.8

---

<div align="center">

**Made with ❤️ for the AI HPC community**

</div>
