# Examples

Welcome to the TensorCraft-HPC examples! This section provides hands-on tutorials and practical code samples to help you get started with high-performance AI kernel development.

## Quick Links

| Example | Description | Difficulty |
|---------|-------------|------------|
| [GEMM Tutorial](/en/examples/gemm-tutorial) | Build GEMM from scratch with progressive optimization | 🟢 Beginner |
| [FlashAttention](/en/examples/flash-attention) | Memory-efficient attention implementation | 🟡 Intermediate |
| [Python Bindings](/en/examples/python-bindings) | Use TensorCraft from Python | 🟢 Beginner |

## Prerequisites

Before running the examples, make sure you have:

1. **CUDA Toolkit 11.0+** installed
2. **CMake 3.18+** for building
3. **C++17 compatible compiler** (GCC 9+, Clang 10+, MSVC 19.28+)
4. **Python 3.8+** (optional, for Python bindings)

## Running the Examples

### C++ Examples

```bash
# Clone and build
git clone https://github.com/AICL-Lab/modern-ai-kernels.git
cd modern-ai-kernels

# Build with CUDA support
cmake --preset dev
cmake --build --preset dev

# Run an example
./build/dev/examples/gemm_example
```

### Python Examples

```bash
# Install Python package
pip install -e .

# Run Python example
python examples/python/gemm_demo.py
```

## Learning Path

We recommend following this order for the best learning experience:

1. **Start with GEMM Tutorial** — Learn the fundamentals of CUDA kernel optimization
2. **Explore FlashAttention** — Understand memory-efficient computing patterns
3. **Try Python Bindings** — Integrate TensorCraft into your Python workflow

## Need Help?

- Check the [API Reference](/en/api/gemm) for detailed documentation
- Browse [Learning Resources](/en/references/resources) for more tutorials
- Open an issue on [GitHub](https://github.com/AICL-Lab/modern-ai-kernels/issues) if you encounter problems