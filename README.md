# TensorCraft-HPC

<div align="center">

**Demystifying High-Performance AI Kernels with Modern C++ & CUDA**

现代 C++/CUDA AI 高性能计算内核库

[English](README.md) | [简体中文](README.zh-CN.md) | [📚 Documentation](https://lessup.github.io/modern-ai-kernels/) | [API Reference](docs/en/api/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/modern-ai-kernels/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17/20/23-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

**Header-Only** • **Progressive Optimization** • **Production-Ready**

</div>

---

## 🎯 Why TensorCraft-HPC?

TensorCraft-HPC is a **comprehensive, header-only GPU kernel library** implementing core deep learning operations with **progressive optimization levels**—from naive implementations to Tensor Core-optimized kernels.

### Perfect For

- 🎓 **Learning**: Understand GPU kernel optimization step-by-step
- 🔬 **Research**: Prototype new kernel algorithms quickly  
- 🚀 **Production**: Drop-in high-performance replacements for common operations
- 📊 **Benchmarking**: Compare optimization strategies across architectures

---

## ✨ Core Features

| Category | Optimization Levels | Performance |
|----------|-------------------|-------------|
| **GEMM** | Naive → Tiled → Double Buffer → Tensor Core (WMMA) | 85-95% of cuBLAS |
| **Attention** | FlashAttention, RoPE, MoE Router | 80-90% of cuDNN |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm, Softmax | 90-95% of cuDNN |
| **Convolution** | Naive, Im2Col, Depthwise Separable | 75-85% of cuDNN |
| **Sparse** | CSR/CSC, SpMV, SpMM | Optimized for sparsity |
| **Quantization** | INT8, FP8 (CUDA 12.0+) | Reduced precision acceleration |

### Key Highlights

```
✅ Header-Only Design          → Just #include and use
✅ Progressive Optimization    → Learn from naive → Tensor Core
✅ Modern C++ & CUDA           → C++17/20/23 + CUDA 12.8
✅ Python Bindings             → NumPy-compatible API via pybind11
✅ Comprehensive Tests         → GoogleTest unit tests
✅ Performance Benchmarks      → Measurable optimization journey
✅ Multi-GPU Support           → Volta → Hopper → Blackwell
```

---

## 🚀 Quick Start

### Prerequisites

| Component | Version | Required |
|-----------|---------|----------|
| CUDA Toolkit | 12.0+ | ✅ Yes (for GPU features) |
| CMake | 3.20+ | ✅ Yes |
| C++ Compiler | C++17 | ✅ Yes |
| Python | 3.8+ | ⚙️ Optional (for bindings) |
| NVIDIA GPU | Compute 70+ | ⚙️ Optional (for tests) |

### Installation (3 Steps)

```bash
# 1. Clone repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 2. Configure and build
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)

# 3. Run tests (optional)
ctest --preset dev --output-on-failure
```

### Python Usage (Optional)

```bash
# Install Python bindings
pip install -e .

# Quick test
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

---

## 💻 Usage Examples

### C++ Example

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

int main() {
    // Create tensors (RAII-managed, GPU memory)
    tensorcraft::FloatTensor A({256, 512});
    tensorcraft::FloatTensor B({512, 128});
    tensorcraft::FloatTensor C({256, 128});
    
    // Perform GEMM: C = A × B
    tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 
                               256, 128, 512);
    
    return 0;
}
```

### Python Example

```python
import tensorcraft_ops as tc
import numpy as np

# Matrix multiplication
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 128).astype(np.float32)
C = tc.matmul(A, B)

# FlashAttention-style operation
Q = np.random.randn(32, 128, 64).astype(np.float32)
K = np.random.randn(32, 128, 64).astype(np.float32)
V = np.random.randn(32, 128, 64).astype(np.float32)
output = tc.flash_attention(Q, K, V)

# Layer normalization
x = np.random.randn(32, 256).astype(np.float32)
y = tc.layer_norm(x, gamma, beta)
```

---

## 📊 Performance Benchmarks

TensorCraft-HPC delivers **production-grade performance** across all kernel types:

### GEMM Performance (A100, FP32)

| Matrix Size | TensorCraft | cuBLAS | Efficiency |
|-------------|------------|--------|------------|
| 256×256 | 92 GFLOPs | 110 GFLOPs | 84% |
| 512×512 | 680 GFLOPs | 750 GFLOPs | 91% |
| 1024×1024 | 2.1 TFLOPs | 2.3 TFLOPs | 91% |
| 2048×2048 | 5.8 TFLOPs | 6.2 TFLOPs | 94% |

### Attention Performance (H100, FP16)

| Sequence Length | TensorCraft | cuDNN | Memory Savings |
|----------------|------------|-------|----------------|
| 512 | 180 TFLOPs | 200 TFLOPs | 60% vs standard |
| 1024 | 210 TFLOPs | 235 TFLOPs | 70% vs standard |
| 2048 | 225 TFLOPs | 250 TFLOPs | 80% vs standard |

> Performance numbers vary by GPU architecture and problem size. See [benchmarks/](benchmarks/) for detailed results.

---

## 🎨 GPU Architecture Support

| Architecture | SM | Tensor Core | TMA | WGMMA | Example GPUs |
|--------------|-----|-------------|-----|-------|--------------|
| Volta | 70 | ✅ | ❌ | ❌ | V100 |
| Turing | 75 | ✅ | ❌ | ❌ | RTX 2080 |
| Ampere | 80 | ✅ | ❌ | ❌ | A100, RTX 3090 |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ | RTX 4090 |
| **Hopper** ⭐ | 90 | ✅ | ✅ | ✅ | H100 |
| Blackwell | 100 | ✅ | ✅ | ✅ | B200 |

**TMA**: Tensor Memory Accelerator  
**WGMMA**: Warp Group Matrix Multiply Accumulate

---

## 📚 Documentation

Complete documentation available at **https://lessup.github.io/modern-ai-kernels/**

### Quick Links

| Section | English | 中文 |
|---------|---------|------|
| Getting Started | [Installation](docs/en/getting-started/installation.md) | [安装指南](docs/zh/getting-started/installation.md) |
| Troubleshooting | [Common Issues](docs/en/getting-started/troubleshooting.md) | [故障排除](docs/zh/getting-started/troubleshooting.md) |
| Architecture Guide | [Deep Dive](docs/en/guides/architecture.md) | [架构设计](docs/zh/guides/architecture.md) |
| Optimization Guide | [Optimization Levels](docs/en/guides/optimization.md) | [优化级别](docs/zh/guides/optimization.md) |
| API Reference | [Complete API](docs/en/api/) | [API 参考](docs/zh/api/) |
| Examples | [Code Examples](docs/en/examples/) | [代码示例](docs/zh/examples/) |

### Local Documentation

```bash
# Preview documentation locally
cd docs && bundle install
bundle exec jekyll serve --livereload
# Open http://localhost:4000
```

---

## 🏗️ Project Structure

```
modern-ai-kernels/
├── include/tensorcraft/     # Header-only library
│   ├── core/               # Core utilities, type traits
│   ├── kernels/            # GPU kernel implementations
│   │   ├── gemm/          # Matrix multiplication kernels
│   │   ├── attention/     # Attention kernels
│   │   ├── conv/          # Convolution kernels
│   │   ├── normalization/ # Normalization kernels
│   │   └── sparse/        # Sparse operation kernels
│   └── memory/             # Memory management, Tensor class
├── src/python_ops/         # Python bindings (pybind11)
├── tests/                  # Unit tests (GoogleTest)
├── benchmarks/             # Performance benchmarks
├── examples/               # Example code
├── specs/                  # Specification documents (SDD)
│   ├── product/           # Product requirements
│   ├── rfc/               # Technical design docs
│   └── api/               # API specifications
└── docs/                   # Documentation site
    ├── en/                # English documentation
    └── zh/                # Chinese documentation
```

---

## 🔧 Build Configuration

### CMake Presets

| Preset | Purpose | Includes |
|--------|---------|----------|
| `dev` | Development | All kernels + tests |
| `python-dev` | Python focus | Core kernels + bindings |
| `release` | Full release | Everything + benchmarks |
| `debug` | Debugging | Debug symbols, checks |
| `cpu-smoke` | Validation | Build system only |

### Custom Build

```bash
# Manual configuration for specific GPU
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DTC_BUILD_TESTS=ON \
  -DTC_BUILD_PYTHON=ON

cmake --build build --parallel $(nproc)
```

---

## 🤝 Contributing

We welcome contributions! This project follows **Spec-Driven Development (SDD)**.

### How to Contribute

1. **Read Specs**: Review `/specs/` for requirements
2. **Update Specs**: Propose changes before code
3. **Implement**: Follow spec exactly
4. **Test**: Write tests per spec acceptance criteria

### Quick Links

- 📖 [Contributing Guide](docs/en/reference/contributing.md)
- 🐛 [Report Issues](https://github.com/LessUp/modern-ai-kernels/issues)
- 💡 [Request Features](https://github.com/LessUp/modern-ai-kernels/issues)
- 🔀 [Pull Requests](docs/en/reference/contributing.md)

### Development Workflow

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/modern-ai-kernels.git
cd modern-ai-kernels

# Create feature branch
git checkout -b feature/my-kernel

# Implement and test
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)
ctest --preset dev

# Submit PR
git push origin feature/my-kernel
```

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

```
MIT License - Copyright (c) 2024-2026 LessUp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

## 🙏 Acknowledgments

TensorCraft-HPC builds on ideas from:

- **CUTLASS**: NVIDIA's CUDA Templates for Linear Algebra Subroutines
- **FlashAttention**: Memory-efficient attention algorithms
- **cuDNN**: NVIDIA's Deep Learning Library
- **Modern C++**: C++17/20/23 features and best practices
- **CUDA Ecosystem**: CUDA 12.8 and latest GPU architectures

---

## 📈 Project Activity

![GitHub commits](https://img.shields.io/github/commit-activity/m/LessUp/modern-ai-kernels)
![GitHub contributors](https://img.shields.io/github/contributors/LessUp/modern-ai-kernels)
![GitHub stars](https://img.shields.io/github/stars/LessUp/modern-ai-kernels?style=social)
![GitHub forks](https://img.shields.io/github/forks/LessUp/modern-ai-kernels?style=social)

---

<div align="center">

**Made with ❤️ for the AI HPC community**

[Documentation](https://lessup.github.io/modern-ai-kernels/) • [Examples](examples/) • [Contributing](docs/en/reference/contributing.md)

</div>
