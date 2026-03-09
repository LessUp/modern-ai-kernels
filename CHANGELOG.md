# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-09

### Fixed
- **MemoryPool lifecycle bug**: `clear()` was erasing tracking for in-use blocks; `deallocate()` left stale entries
- **atomicMin/atomicMax for negative floats**: `compute_quant_params_kernel` gave incorrect results for negative values; replaced with CAS-based atomic float min/max

### Added
- `core/warp_utils.hpp`: Shared warp-level reduction primitives (`warp_reduce_max/sum/min`, `warp_broadcast`, `block_reduce_sum/max`)
- `detail::fill_kernel`: GPU-side fill kernel for `Tensor::fill` on non-byte types

### Changed
- **FlashAttention kernel rewrite**: Moved output accumulator from per-thread registers to shared memory, reducing register pressure from 256 bytes/thread; cooperative tile loading; reduced default block sizes
- `normalization.hpp` no longer depends on `softmax.hpp` (uses `warp_utils.hpp` directly)
- `Tensor::fill` now uses a GPU kernel instead of host-memory roundtrip for non-byte types

## [Unreleased]

## [1.0.0] - 2024-01-01

### Added
- Initial release with core kernel implementations
- **GEMM Kernels**
  - Naive GEMM implementation
  - Tiled GEMM with shared memory optimization
  - Double-buffered GEMM for latency hiding
  - Tensor Core GEMM using WMMA API
- **Attention Kernels**
  - FlashAttention-style fused attention kernel
  - Memory-efficient attention computation
- **Normalization Kernels**
  - LayerNorm implementation
  - RMSNorm implementation
  - BatchNorm implementation
- **Convolution Kernels**
  - Naive 2D convolution
  - Im2Col-based convolution
  - Depthwise separable convolution
- **Sparse Operations**
  - CSR and CSC sparse matrix formats
  - Sparse Matrix-Vector multiplication (SpMV)
  - Sparse Matrix-Matrix multiplication (SpMM)
- **Elementwise Operations**
  - Fused elementwise kernel support
  - Common activation functions (ReLU, GELU, SiLU)
- **Memory Management**
  - Memory pool for efficient GPU memory allocation
  - Aligned vector for CPU-side data
  - Tensor abstraction with automatic memory management
- **Python Bindings**
  - pybind11-based Python interface
  - NumPy array interoperability
- **Documentation**
  - API reference documentation
  - Architecture overview
  - Optimization guide
  - Modern C++/CUDA best practices guide
- **Benchmarks**
  - GEMM performance benchmarks
  - Attention kernel benchmarks
  - Convolution benchmarks
- **Testing**
  - Unit tests for all kernel implementations
  - Correctness validation against reference implementations

### Dependencies
- CUDA Toolkit 11.0+ (12.x recommended)
- CMake 3.18+
- C++17 compatible compiler
- pybind11 (for Python bindings)

[Unreleased]: https://github.com/LessUp/modern-ai-kernels/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/LessUp/modern-ai-kernels/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/LessUp/modern-ai-kernels/releases/tag/v1.0.0
