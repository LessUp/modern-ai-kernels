# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project polish: code quality tools, CI/CD, documentation

### Changed
- Nothing yet

### Fixed
- Nothing yet

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

[Unreleased]: https://github.com/username/tensorcraft-hpc/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/username/tensorcraft-hpc/releases/tag/v1.0.0
