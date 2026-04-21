---
layout: home
title: TensorCraft-HPC
nav_order: 1
permalink: /
---

# TensorCraft-HPC

{: .highlight }
**Demystifying High-Performance AI Kernels with Modern C++ & CUDA**

**Modern C++/CUDA AI High-Performance Computing Kernel Library**

[![GitHub stars](https://img.shields.io/github/stars/LessUp/modern-ai-kernels?style=social)](https://github.com/LessUp/modern-ai-kernels/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/LessUp/modern-ai-kernels/blob/master/LICENSE)

---

## 🚀 Quick Links

<div class="code-example" markdown="block">

| For Beginners | For Developers | For Researchers |
|:--------------|:---------------|:----------------|
| [Installation Guide](en/getting-started/installation.md) | [API Reference](en/api/) | [Optimization Guide](en/guides/optimization.md) |
| [Examples](en/examples/) | [Architecture](en/guides/architecture.md) | [Performance Benchmarks](en/guides/optimization.md#performance) |

</div>

---

## Overview

TensorCraft-HPC is a **comprehensive, header-only GPU kernel library** implementing core deep learning operations with **progressive optimization levels**—from naive implementations to Tensor Core-optimized kernels.

### Core Features

| Category | Operations | Performance |
|----------|------------|-------------|
| **GEMM** | Naive → Tiled → Double Buffer → Tensor Core (WMMA) | 85-95% of cuBLAS |
| **Attention** | FlashAttention, RoPE, MoE Router | 80-90% of cuDNN |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm, Softmax | 90-95% of cuDNN |
| **Convolution** | Naive, Im2Col, Depthwise Separable | 75-85% of cuDNN |
| **Sparse** | CSR/CSC, SpMV, SpMM | Optimized for sparsity |
| **Quantization** | INT8, FP8 (CUDA 12.0+) | Reduced precision acceleration |

---

## Quick Start

### Prerequisites

- **CMake** ≥ 3.20
- **CUDA** 12.8 (11.x - 13.1 compatible)
- **C++ Compiler** with C++17 support
- **Python** 3.8+ (for bindings)

### Build & Install

```bash
# Clone repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Configure and build
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)

# Run tests
ctest --preset dev --output-on-failure

# Install Python bindings (optional)
pip install -e .
```

### Python Usage

```python
import tensorcraft_ops as tc

# Create tensors
a = tc.tensor([[1.0, 2.0], [3.0, 4.0]])
b = tc.tensor([[5.0, 6.0], [7.0, 8.0]])

# Matrix multiplication
c = tc.matmul(a, b)
print(c.numpy())
```

---

## GPU Architecture Support

| Architecture | SM | Tensor Core | TMA | WGMMA |
|--------------|-----|-------------|-----|-------|
| Volta | 70 | ✅ | ❌ | ❌ |
| Turing | 75 | ✅ | ❌ | ❌ |
| Ampere | 80 | ✅ | ❌ | ❌ |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ |
| **Hopper** ⭐ | 90 | ✅ | ✅ | ✅ |
| Blackwell | 100 | ✅ | ✅ | ✅ |

---

## Documentation

{: .nav-list }
- 📖 **[Getting Started](en/getting-started/)** - Installation and setup guides
- 🔧 **[User Guides](en/guides/)** - Architecture and optimization deep-dives
- 📚 **[API Reference](en/api/)** - Complete API documentation
- 💻 **[Examples](en/examples/)** - Usage examples
- 🌐 **[中文文档](zh/)** - Complete Chinese documentation

---

## Project Structure

```
├── include/tensorcraft/     # Header-only library
│   ├── core/               # Core utilities
│   ├── kernels/            # GPU kernel implementations
│   └── memory/             # Memory management
├── src/python_ops/         # Python bindings (pybind11)
├── tests/                  # Unit tests (GoogleTest)
├── benchmarks/             # Performance benchmarks
├── examples/               # Example code
└── specs/                  # Specification documents (SDD)
```

---

## Key Design Principles

{: .important }
**Header-Only Design**: Just include and use. No separate compilation needed.

{: .note }
**Progressive Optimization**: Learn kernel optimization step-by-step: Naive → Tiled → Double Buffer → Tensor Core

{: .warning }
**Modern C++ & CUDA**: Leverages C++17/20/23 features with CUDA 12.8 for maximum performance

---

## Performance

TensorCraft-HPC delivers production-grade performance across all kernel types:

| Kernel | Performance vs cuBLAS/cuDNN |
|--------|----------------------------|
| GEMM (Tensor Core) | 85-95% |
| FlashAttention | 80-90% |
| LayerNorm | 90-95% |
| Conv2D (Im2Col) | 75-85% |

> Performance numbers vary by GPU architecture and problem size.

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/LessUp/modern-ai-kernels/blob/master/CONTRIBUTING.md) for details.

### Development Workflow

This project follows **Spec-Driven Development (SDD)**. All implementations must be traceable to specification documents under `/specs/`.

1. Review specs in `/specs/`
2. Update specs before code changes
3. Implement according to spec
4. Write tests based on spec acceptance criteria

---

## License

This project is licensed under the [MIT License](https://github.com/LessUp/modern-ai-kernels/blob/master/LICENSE).

---

## Community

- 🐛 [Report Issues](https://github.com/LessUp/modern-ai-kernels/issues)
- 💬 [Discussions](https://github.com/LessUp/modern-ai-kernels/discussions)
- 📖 [Documentation](https://lessup.github.io/modern-ai-kernels/)

{: .text-center }
**Made with ❤️ for the AI HPC community**
