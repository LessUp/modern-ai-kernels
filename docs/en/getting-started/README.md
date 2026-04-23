---
title: Getting Started
nav_order: 1
has_children: true
---

# Getting Started

This section will help you set up, build, and validate TensorCraft-HPC on your system.

## In This Section

{: .toc }

- [**Installation Guide**](installation.md) - Complete setup instructions
- [**Troubleshooting**](troubleshooting.md) - Common issues and solutions

## Quick Start

Choose the build preset that matches your needs:

### Development (Recommended)

For CUDA development on a GPU-equipped machine:

```bash
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)
ctest --preset dev --output-on-failure
```

### Python Bindings Only

If you mainly need the Python interface:

```bash
cmake --preset python-dev
cmake --build --preset python-dev --parallel $(nproc)
python3 -m pip install -e .
python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

### Full Release Build

For benchmarks and complete validation:

```bash
cmake --preset release
cmake --build --preset release --parallel $(nproc)
ctest --test-dir build/release --output-on-failure
```

### CPU-Only Validation

To test build infrastructure without CUDA:

```bash
cmake --preset cpu-smoke
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
```

## Next Steps

Once you've successfully built the project:

1. **Explore Examples** → [Examples Section](../examples/)
2. **Understand Architecture** → [Architecture Guide](../guides/architecture.md)
3. **Reference API** → [API Documentation](../api/)
4. **Learn Optimization** → [Optimization Guides](../guides/optimization.md)

## Prerequisites Overview

| Component | Version | Required |
|-----------|---------|----------|
| CUDA Toolkit | 12.0+ | Yes (for GPU features) |
| CMake | 3.20+ | Yes |
| C++ Compiler | C++17 | Yes |
| Python | 3.8+ | No (for bindings) |
| NVIDIA GPU | Compute 70+ | No (for tests/benchmarks) |

For detailed prerequisites and troubleshooting, see the [Installation Guide](installation.md).
