# API Reference

This section provides detailed API documentation for all TensorCraft-HPC modules.

## Module Overview

```
tensorcraft/
├── core/           # Core utilities
│   ├── cuda_check.hpp
│   ├── features.hpp
│   ├── type_traits.hpp
│   └── warp_utils.hpp
├── memory/         # Memory management
│   ├── aligned_vector.hpp
│   ├── tensor.hpp
│   └── memory_pool.hpp
└── kernels/        # Compute kernels
    ├── elementwise.hpp
    ├── softmax.hpp
    ├── normalization.hpp
    ├── gemm.hpp
    ├── attention.hpp
    ├── conv2d.hpp
    ├── sparse.hpp
    └── fusion.hpp
```

## API Documentation Index

| Module | Description |
|--------|-------------|
| [Core](core.md) | Error handling, feature detection, type traits, warp utilities |
| [Memory](memory.md) | Tensor class, memory pool, vectorized access |
| [Kernels](kernels.md) | All compute kernels (GEMM, attention, etc.) |
| [Python](python.md) | Python bindings (tensorcraft_ops) |

## Common Patterns

### Error Handling

All CUDA operations use exception-based error handling:

```cpp
#include "tensorcraft/core/cuda_check.hpp"

try {
    tensorcraft::kernels::gemm(A, B, C, M, N, K);
} catch (const tensorcraft::CudaException& e) {
    std::cerr << "CUDA error at " << e.file() << ":" << e.line() 
              << " - " << e.what() << std::endl;
}
```

### Stream Support

All kernel launchers support optional CUDA streams:

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// With stream
tensorcraft::kernels::gemm(A, B, C, M, N, K, 1.0f, 0.0f, 
                           tensorcraft::kernels::GemmVersion::Tiled, stream);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

### Type Support

The library supports multiple data types:

| Type | Header | Description |
|------|--------|-------------|
| `float` | Built-in | 32-bit floating point |
| `__half` | `<cuda_fp16.h>` | 16-bit floating point |
| `__nv_bfloat16` | `<cuda_bf16.h>` | BF16 (CUDA 11.0+) |
| `int8_t` | Built-in | 8-bit integer |
| `__nv_fp8_e4m3` | `<cuda_fp8.h>` | FP8 E4M3 (CUDA 12.0+) |
