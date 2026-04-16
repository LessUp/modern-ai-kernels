# RFC 0001: TensorCraft-HPC Core Architecture

> **RFC Number**: 0001
> **Title**: Core Architecture
> **Status**: ✅ Accepted — Implemented
> **Type**: Architecture
> **Created**: 2024-01-01
> **Last Updated**: 2026-04-17

---

## Summary

This RFC defines the core architecture of TensorCraft-HPC, a modular high-performance AI operator optimization library supporting CUDA 11.0 to CUDA 13.1.

---

## Motivation

Modern AI workloads require highly optimized kernel implementations. However, existing libraries often lack:

1. Clear teaching-friendly code structure
2. Progressive optimization examples (naive → optimized)
3. Multi-version CUDA compatibility
4. Modern C++ best practices

TensorCraft-HPC aims to fill this gap by providing a comprehensive, well-documented kernel library.

---

## Design Principles

1. **Progressive Optimization**: Each operator provides multiple versions from naive to optimized
2. **Compile-Time Feature Detection**: Macros and templates select optimal implementation paths at compile time
3. **Zero-Cost Abstraction**: Modern C++ template techniques for high-performance generic programming
4. **Teaching-Friendly**: Clear code structure with detailed comments for learning

---

## Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Python Bindings (pybind11)                 │
├─────────────────────────────────────────────────────────────────┤
│                         Kernel Launchers                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │Elementwise│ │Reduction │ │  GEMM    │ │Attention │ │  Conv  │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                        Core Utilities                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │  Tensor  │ │  Memory  │ │   Math   │ │  Config  │            │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                     CUDA Runtime / Driver API                    │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
modern-ai-kernels/
├── specs/                     # Specification documents
│   ├── product/              # Product requirements (PRDs)
│   ├── rfc/                  # Technical design (RFCs)
│   ├── api/                  # API specifications
│   ├── db/                   # Data structure specifications
│   └── testing/              # Test specifications
├── include/tensorcraft/      # Header-only kernel library
│   ├── core/                 # CUDA error handling, type traits
│   │   ├── cuda_check.hpp
│   │   ├── features.hpp
│   │   ├── type_traits.hpp
│   │   └── warp_utils.hpp
│   ├── memory/               # Tensor, memory pool
│   │   ├── aligned_vector.hpp
│   │   ├── tensor.hpp
│   │   └── memory_pool.hpp
│   └── kernels/              # All compute kernels
│       ├── elementwise.hpp
│       ├── softmax.hpp
│       ├── normalization.hpp
│       ├── gemm.hpp
│       ├── attention.hpp
│       ├── conv2d.hpp
│       ├── sparse.hpp
│       └── fusion.hpp
├── src/python_ops/           # Python bindings
├── tests/                    # Unit tests
├── benchmarks/               # Performance benchmarks
├── docs/                     # Documentation (en/, zh/)
├── examples/                 # Example code
└── changelog/                # Development changelog
```

---

## Components

### 1. Core Utilities (`include/tensorcraft/core/`)

| File | Purpose |
|------|---------|
| `cuda_check.hpp` | CUDA error checking macros and exception handling |
| `features.hpp` | Compile-time feature detection (C++17/20/23, CUDA 11/12/13) |
| `type_traits.hpp` | Type traits and Concepts |
| `warp_utils.hpp` | Warp-level reduction primitives |

### 2. Memory Management (`include/tensorcraft/memory/`)

| File | Purpose |
|------|---------|
| `aligned_vector.hpp` | Aligned vector types for vectorized memory access |
| `tensor.hpp` | RAII-style GPU Tensor wrapper |
| `memory_pool.hpp` | Thread-safe GPU memory pool |

### 3. Kernels (`include/tensorcraft/kernels/`)

| File | Purpose |
|------|---------|
| `elementwise.hpp` | Element-wise operations and activation functions |
| `softmax.hpp` | Numerically stable Softmax |
| `normalization.hpp` | LayerNorm, RMSNorm, BatchNorm |
| `gemm.hpp` | Matrix multiplication (Naive → Tiled → Double Buffer → Tensor Core) |
| `attention.hpp` | FlashAttention, RoPE, MoE Router |
| `conv2d.hpp` | 2D convolution operations |
| `sparse.hpp` | Sparse matrix operations (CSR/CSC) |
| `fusion.hpp` | Fused operators and quantization |

---

## Build System

### CMake Presets

| Preset | Purpose |
|--------|---------|
| `dev` | Recommended CUDA development preset |
| `python-dev` | Lighter CUDA build for Python bindings |
| `release` | Full release build with benchmarks |
| `debug` | Debug-oriented CUDA build |
| `cpu-smoke` | CPU-only configure/install validation |

### Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| CUDA Toolkit | 12.8 (targeted) | GPU compute |
| CMake | 3.20+ | Build system |
| C++ Compiler | C++17 | Code compilation |
| pybind11 | 2.x | Python bindings |
| GoogleTest | 1.x | Unit testing |

---

## Correctness Properties

### Property 1: Tensor RAII Memory Management

For any Tensor object created with a given shape, GPU memory SHALL be allocated on construction and freed on destruction, with no memory leaks.

### Property 2: GEMM Mathematical Correctness

For any matrices A[M×K], B[K×N] and scalars alpha, beta, GEMM SHALL compute C = alpha * A @ B + beta * C correctly within floating-point tolerance.

### Property 3: Softmax Row Sum Invariant

For any input matrix X, the Softmax output S SHALL satisfy:

1. `sum(S[i, :]) = 1.0` for all rows i (within tolerance)
2. `S[i, j] >= 0` for all elements

### Property 4: Optimization Level Numerical Equivalence

For any kernel with multiple optimization levels, all versions SHALL produce numerically equivalent outputs within tolerance.

---

## Testing Strategy

| Test Type | Framework | Purpose |
|-----------|-----------|---------|
| Unit Tests | GoogleTest | Specific examples and edge cases |
| Property-Based Tests | Custom | Random input validation for general properties |
| Python Verification | PyTorch | Comparison with reference implementations |
| Benchmarks | Google Benchmark | Performance regression detection |

---

## Error Handling

All CUDA API calls wrapped with `TC_CUDA_CHECK` macro that throws `CudaException` with:

- Source file name
- Line number
- Error description

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 2.0.0 | 2026-03-09 | MemoryPool bug fix, atomicMin/Max fix, warp_utils extraction, FlashAttention rewrite |
| 1.1.0 | 2026-01-08 | Build system fixes for CUDA-optional environments |
| 1.0.1 | 2025-02-13 | Project infrastructure improvements |
| 1.0.0 | 2024-01-01 | Initial release |

---

## References

- [API Specification](../api/cxx-api.md)
- [Data Structure Specification](../db/data-structures.md)
- [Product Specification: TensorCraft-HPC](../product/tensorcraft-hpc.md)
