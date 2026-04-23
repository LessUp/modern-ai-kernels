---
title: Examples Overview
lang: zh
---

# Examples Overview

This section provides practical examples for using TensorCraft-HPC kernels.

## Available Examples

| Example | Description |
|---------|-------------|
| [Basic GEMM](basic-gemm.md) | GEMM optimization journey from naive to Tensor Core |
| [Attention](attention.md) | FlashAttention and RoPE usage |
| [Normalization](normalization.md) | LayerNorm and RMSNorm examples |
| [Python Usage](python-usage.md) | Complete Python binding examples |

## Prerequisites

Before running the examples, ensure you have:

1. Built the project with CUDA support:

   ```bash
   cmake --preset dev
   cmake --build --preset dev --parallel 2
   ```

2. For Python examples, install the bindings:

   ```bash
   python3 -m pip install -e .
   ```

## Quick Reference

### C++ Usage

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

using namespace tensorcraft;

// Using Tensor class
FloatTensor A({128, 256});
FloatTensor B({256, 512});
FloatTensor C({128, 512});

// Initialize data...
A.copy_from_host(host_data);

// Run GEMM
kernels::gemm(A.data(), B.data(), C.data(), 128, 512, 256);
```

### Python Usage

```python
import tensorcraft_ops as tc
import numpy as np

# Create arrays
A = np.random.randn(128, 256).astype(np.float32)
B = np.random.randn(256, 512).astype(np.float32)

# Run GEMM
C = tc.gemm(A, B)
```

## Building Examples

The examples in the `examples/` directory can be built as part of the main project:

```bash
cmake --preset dev
cmake --build --preset dev
./build/dev/examples/basic_gemm
```

## Performance Benchmarking

For performance benchmarking, see the `benchmarks/` directory:

```bash
cmake --preset release
cmake --build --preset release
./build/release/benchmarks/gemm_benchmark
```
