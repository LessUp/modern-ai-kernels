# Frequently Asked Questions

## General

### What is TensorCraft-HPC?

TensorCraft-HPC is a header-only C++/CUDA library for learning high-performance AI kernel implementation. It provides progressive optimization paths from naive to production-grade performance, with clear annotations at every step.

### Who is this project for?

- **GPU Kernel Developers** seeking to understand optimization techniques
- **ML Infrastructure Engineers** evaluating kernel implementations
- **Researchers** studying high-performance computing patterns
- **Students** learning CUDA programming

### How does this differ from CUTLASS or cuBLAS?

| Aspect | TensorCraft-HPC | CUTLASS | cuBLAS |
|--------|----------------|---------|--------|
| Purpose | Learning | Production | Production |
| Code Style | Readable | Template-heavy | Closed-source |
| Performance | 92% cuBLAS | ~100% | 100% (baseline) |
| Build | Header-only | CMake + builds | Pre-installed |

TensorCraft-HPC prioritizes **educational clarity** over maximum performance. CUTLASS is the reference for production deployment.

---

## Technical

### What GPU architectures are supported?

| Architecture | Compute Capability | Status |
|-------------|-------------------|--------|
| Volta | SM70 (7.0) | Supported |
| Turing | SM75 (7.5) | Supported |
| Ampere | SM80/SM86 | Supported |
| Hopper | SM90 (9.0) | Supported |
| Blackwell | SM100 (10.0) | Supported |

### What CUDA versions are supported?

CUDA 11.0 through 13.1. CUDA 12.0+ is recommended for FP8 and Transformer Engine features.

### Why header-only?

1. **Zero build friction**: Just `#include` and go
2. **Easy integration**: Copy headers into any project
3. **Transparency**: All code is visible and auditable
4. **No ABI issues**: Everything compiles with your flags

### Can I use this in production?

TensorCraft-HPC is primarily designed for learning. While the kernels achieve good performance (up to 95% of NVIDIA libraries for some operations), for production deployments we recommend:

- **GEMM**: Use cuBLAS or CUTLASS
- **Attention**: Use FlashAttention official
- **Normalization**: Use cuDNN

### What precision types are supported?

| Type | Size | Architecture |
|------|------|-------------|
| FP32 | 32-bit | All |
| FP16 (half) | 16-bit | SM70+ |
| BF16 | 16-bit | SM80+ |
| TF32 | 19-bit | SM80+ |
| FP8 (E4M3/E5M2) | 8-bit | SM90+ |
| INT8 | 8-bit | SM75+ |

---

## Building & Integration

### How do I add TensorCraft-HPC to my CMake project?

```cmake
# Simple: just add the include directory
target_include_directories(your_target PRIVATE
    path/to/modern-ai-kernels/include
)

# Link against CUDA
find_package(CUDA REQUIRED)
target_link_libraries(your_target CUDA::cudart)
```

### How do I use Python bindings?

```bash
# Install from source
cd modern-ai-kernels
pip install -e .
```

```python
import tensorcraft_ops as tc
import numpy as np

# Use NumPy-compatible API
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = tc.gemm(A, B)
```

### How do I run tests?

```bash
# Build with development preset
cmake --preset dev
cmake --build --preset dev

# Run all tests
ctest --preset dev --output-on-failure

# Run specific test
ctest --preset dev -R gemm
```

### How do I run benchmarks?

```bash
# Build benchmarks
cmake --preset release
cmake --build --preset release --parallel 2

# Run GEMM benchmark
./build/release/benchmarks/gemm_benchmark

# Run with specific filter
./build/release/benchmarks/gemm_benchmark --benchmark_filter="FP16"
```

---

## Troubleshooting

### Build fails with "CUDA not found"

Ensure CUDA toolkit is installed and `CUDA_PATH` is set:

```bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### "Unsupported GPU architecture" error

Check your GPU's compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

TensorCraft-HPC requires SM70 (Volta) or later.

### Numerical results differ from cuBLAS

Small numerical differences are expected due to:

1. **Floating-point order**: Different reduction orders produce different rounding
2. **Mixed precision**: Tensor Core uses mixed-precision accumulation
3. **Fused operations**: Fused kernels may use different intermediate precision

For numerical validation, use relative error with an appropriate tolerance:

```cpp
float rel_error = std::abs(your_result - reference) / std::abs(reference);
EXPECT_LT(rel_error, 1e-3);  // 0.1% tolerance
```

### Performance is lower than expected

Check these common issues:

1. **Clock speeds**: GPU may be power-limited. Check with `nvidia-smi -q -d CLOCK`
2. **Memory bandwidth**: Ensure data fits in GPU memory
3. **Warp divergence**: Check for branch divergence in your kernel
4. **Shared memory bank conflicts**: Use padding to avoid conflicts

---

## Contributing

### How do I contribute a new kernel?

1. Create a specification in `openspec/changes/`
2. Implement the kernel header in `include/tensorcraft/kernels/`
3. Add GoogleTest tests in `tests/kernels/`
4. Add benchmarks in `benchmarks/`
5. Submit a pull request

See the [Methodology](/en/whitepaper/methodology) section for detailed guidelines.

### How do I report a bug?

Please file an issue at [GitHub Issues](https://github.com/AICL-Lab/modern-ai-kernels/issues) with:

- GPU model and driver version
- CUDA version
- Minimal reproduction steps
- Expected vs. actual behavior
