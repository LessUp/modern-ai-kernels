# Getting Started

This section helps you set up TensorCraft-HPC and get running quickly.

## Quick Links

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | Detailed build requirements and instructions |
| [Troubleshooting](troubleshooting.md) | Solutions to common problems |

## 5-Minute Quick Start

### Prerequisites

- **CUDA Toolkit**: 12.8
- **CMake**: 3.20+
- **C++ Compiler**: C++17 capable
- **NVIDIA GPU**: For tests and Python bindings

### Minimal Build

```bash
# Clone the repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Configure and build (recommended preset)
cmake --preset dev
cmake --build --preset dev --parallel 2

# Run tests
ctest --preset dev --output-on-failure

# Install Python bindings
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

### Build Presets

| Preset | Purpose |
|--------|---------|
| `dev` | Recommended for CUDA development |
| `python-dev` | Lighter build focused on Python bindings |
| `release` | Full release build with benchmarks |
| `cpu-smoke` | CPU-only configure/install smoke test |

### Next Steps

- Read the [Architecture Guide](../guides/architecture.md) to understand the module structure
- Check out [Examples](../examples/basic-gemm.md) for usage patterns
- Browse the [API Reference](../api/core.md) for detailed documentation
