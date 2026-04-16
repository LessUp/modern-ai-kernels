# Getting Started

This chapter helps you quickly get started with TensorCraft-HPC.

## Content Overview

| Document | Description |
|----------|-------------|
| [Installation Guide](installation.md) | System requirements, installation steps, and build configuration |
| [Troubleshooting](troubleshooting.md) | Common problem diagnosis and solutions |

## Quick Start

If you already have a CUDA development environment, you can use the following commands directly:

```bash
# Clone the repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Build (recommended development preset)
cmake --preset dev
cmake --build --preset dev --parallel 2

# Run tests
ctest --preset dev --output-on-failure

# Install Python bindings
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

## System Requirements

- **CUDA Toolkit**: 12.8
- **CMake**: 3.20+
- **C++ Compiler**: C++17-capable host compiler
- **NVIDIA GPU**: Recommended for tests and Python bindings

For detailed requirements, please refer to the [Installation Guide](installation.md).
