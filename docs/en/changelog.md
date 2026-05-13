# Changelog

All notable changes to TensorCraft-HPC are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] - 2024-12-XX

### Added
- OpenSpec-driven specification workflow
- VitePress documentation site with NVIDIA-style theme
- Mermaid architecture diagrams
- Academic references section
- Chinese documentation (中文文档)
- LLM-ready documentation generation

### Changed
- Migrated from Jekyll to VitePress
- New NVIDIA-inspired dark theme
- Reorganized documentation structure

---

## [1.0.0] - 2024-06-XX

### Added
- Header-only C++/CUDA kernel library
- GEMM kernels with progressive optimization
  - Naive implementation
  - Tiled for shared memory reuse
  - Double buffer for compute/transfer overlap
  - Tensor Core (WMMA) support
- FlashAttention implementation
- Normalization kernels (LayerNorm, RMSNorm, BatchNorm)
- Softmax with numerical stability
- 2D convolution (Im2Col, Winograd)
- Sparse operations (CSR, CSC formats)
- Elementwise activations (ReLU, GeLU, SiLU, Sigmoid)
- Python bindings via pybind11
- CMake build system with presets
- GoogleTest unit tests
- Google Benchmark integration

### Performance
- GEMM: 85-95% cuBLAS parity
- FlashAttention: 80-90% cuDNN parity
- Normalization: 90-95% cuDNN parity

---

## Architecture Support History

| Architecture | SM | Added |
|-------------|-----|-------|
| Volta | 70 | v1.0.0 |
| Turing | 75 | v1.0.0 |
| Ampere | 80/86 | v1.0.0 |
| Ada Lovelace | 89 | v1.0.0 |
| Hopper | 90 | v1.0.0 |
| Blackwell | 100 | v2.0.0 |

---

## CUDA Version History

| Version | CUDA Requirement |
|---------|------------------|
| v1.0.0 | CUDA 11.0+ |
| v2.0.0 | CUDA 11.0+ (12.0+ recommended for FP8) |

---

[2.0.0]: https://github.com/LessUp/modern-ai-kernels/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/LessUp/modern-ai-kernels/releases/tag/v1.0.0