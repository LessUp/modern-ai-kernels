# Changelog | 变更日志

All notable changes to this project will be documented in this file.

本项目的所有显著变更都将记录在此文件中。

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased] | [未发布]

### Added | 新增

- Complete bilingual (EN/ZH) documentation structure | 完整的中英文双语文档结构
- Root-level CHANGELOG.md with bilingual support | 根目录双语 CHANGELOG.md
- Professional documentation index and navigation | 专业的文档索引和导航

---

## [3.0.0] - 2025-04-16 | v3.0.0 - 2025年4月16日

### Changed | 变更

- **Documentation Reconstruction** | **文档重构**
  - Reorganized docs/ into bilingual structure (en/, zh/) | 将 docs/ 重组为双语结构（en/、zh/）
  - Professional documentation landing page | 专业的文档首页
  - Comprehensive bilingual navigation | 全面的双语导航

---

## [2.0.0] - 2026-03-09 | v2.0.0 - 2026年3月9日

### Fixed | 修复

- **MemoryPool lifecycle bug (Critical)** | **MemoryPool 生命周期错误（严重）**
  - `clear()` was erasing tracking for in-use blocks | `clear()` 删除了正在使用块的跟踪
  - `deallocate()` left stale entries | `deallocate()` 留下过期条目
  - Added `freed_sizes_` map for proper pool state management | 添加 `freed_sizes_` 映射以正确管理池状态

- **atomicMin/atomicMax for negative floats (Critical)** | **负浮点数的 atomicMin/atomicMax（严重）**
  - `compute_quant_params_kernel` gave incorrect results for negative values | `compute_quant_params_kernel` 对负值给出错误结果
  - Replaced with CAS-based atomic float min/max | 替换为基于 CAS 的原子浮点 min/max

### Added | 新增

- `core/warp_utils.hpp`: Shared warp-level reduction primitives | 共享线程束级归约原语
  - `warp_reduce_max/sum/min` | `warp_reduce_max/sum/min`
  - `warp_broadcast` | `warp_broadcast`
  - `block_reduce_sum/max` | `block_reduce_sum/max`
- `detail::fill_kernel`: GPU-side fill kernel for `Tensor::fill` | 用于 `Tensor::fill` 的 GPU 端填充内核

### Changed | 变更

- **FlashAttention kernel rewrite** | **FlashAttention 内核重写**
  - Moved output accumulator from per-thread registers to shared memory | 将输出累加器从每线程寄存器移动到共享内存
  - Reduced register pressure from 256 bytes/thread | 将寄存器压力从 256 字节/线程降低
  - Cooperative tile loading | 协作瓦片加载
  - Reduced default block sizes | 减少默认块大小
- `normalization.hpp` no longer depends on `softmax.hpp` | `normalization.hpp` 不再依赖 `softmax.hpp`
- `Tensor::fill` now uses a GPU kernel instead of host-memory roundtrip | `Tensor::fill` 现在使用 GPU 内核而非主机内存往返

---

## [1.1.0] - 2026-01-08 | v1.1.0 - 2026年1月8日

### Fixed | 修复

- Python bindings CMake configuration | Python 绑定的 CMake 配置
  - Fixed `src/python_ops/CMakeLists.txt` referencing non-existent source files | 修复 `src/python_ops/CMakeLists.txt` 引用不存在的源文件
  - `tensor_bindings.cpp` / `kernel_bindings.cpp` | `tensor_bindings.cpp` / `kernel_bindings.cpp`
- CUDA-optional builds | 可选 CUDA 构建
  - CMake now gracefully handles environments without CUDA Toolkit | CMake 现在优雅处理没有 CUDA Toolkit 的环境
  - Auto-disabling tests, benchmarks, and Python bindings | 自动禁用测试、基准和 Python 绑定

---

## [1.0.1] - 2025-02-13 | v1.0.1 - 2025年2月13日

### Added | 新增

- Project infrastructure files | 项目基础设施文件
  - `.gitignore` for CUDA/Python/IDE rules | `.gitignore` 用于 CUDA/Python/IDE 规则
  - `.editorconfig` for unified code formatting | `.editorconfig` 用于统一代码格式
- Standardized badges in README | README 中的标准化徽章
  - License, CUDA, C++17/20, CMake, Python | 许可证、CUDA、C++17/20、CMake、Python
  
### Changed | 变更

- Changelog files restructured into changelog/ directory | 变更日志文件重组到 changelog/ 目录

---

## [1.0.0] - 2024-01-01 | v1.0.0 - 2024年1月1日

### Added | 新增

- **GEMM Kernels** | **GEMM 内核**
  - Naive GEMM implementation | 朴素 GEMM 实现
  - Tiled GEMM with shared memory optimization | 带共享内存优化的平铺 GEMM
  - Double-buffered GEMM for latency hiding | 用于延迟隐藏的双缓冲 GEMM
  - Tensor Core GEMM using WMMA API (CUDA 11.0+) | 使用 WMMA API 的张量核心 GEMM

- **Attention Kernels** | **注意力内核**
  - FlashAttention-style fused attention kernel | FlashAttention 风格的融合注意力内核
  - Memory-efficient attention computation | 内存高效注意力计算
  - RoPE (Rotary Positional Embeddings) kernel | RoPE（旋转位置嵌入）内核
  - Simplified PagedAttention kernel | 简化版 PagedAttention 内核
  - MoE (Mixture of Experts) router kernel | MoE（专家混合）路由器内核

- **Normalization Kernels** | **归一化内核**
  - LayerNorm implementation | LayerNorm 实现
  - RMSNorm implementation | RMSNorm 实现
  - BatchNorm implementation | BatchNorm 实现
  - Softmax with online algorithm | 使用在线算法的 Softmax

- **Convolution Kernels** | **卷积内核**
  - Naive 2D convolution | 朴素二维卷积
  - Im2Col-based convolution | 基于 Im2Col 的卷积
  - Depthwise separable convolution | 深度可分离卷积

- **Sparse Operations** | **稀疏操作**
  - CSR and CSC sparse matrix formats | CSR 和 CSC 稀疏矩阵格式
  - Sparse Matrix-Vector multiplication (SpMV) | 稀疏矩阵-向量乘法 (SpMV)
  - Sparse Matrix-Matrix multiplication (SpMM) | 稀疏矩阵-矩阵乘法 (SpMM)

- **Elementwise Operations** | **逐元素操作**
  - Fused elementwise kernel support | 融合逐元素内核支持
  - Common activation functions | 常用激活函数 (ReLU, GELU, SiLU, LeakyReLU, ELU, Swish)

- **Operator Fusion & Quantization** | **算子融合与量化**
  - Fused Bias+GeLU epilogue | 融合的 Bias+GeLU 后记
  - INT8 quantization support | INT8 量化支持
  - FP8 quantization support (CUDA 12.0+) | FP8 量化支持

- **Memory Management** | **内存管理**
  - Memory pool for efficient GPU memory allocation | 用于高效 GPU 内存分配的内存池
  - Aligned vector for CPU-side data | 用于 CPU 端数据的对齐向量
  - Tensor abstraction with automatic memory management | 自动内存管理的张量抽象

- **Python Bindings** | **Python 绑定**
  - pybind11-based Python interface | 基于 pybind11 的 Python 接口
  - NumPy array interoperability | NumPy 数组互操作性

- **Testing** | **测试**
  - Unit tests for all kernel implementations | 所有内核实现的单元测试
  - Correctness validation against reference implementations | 与参考实现的正确性验证

- **Benchmarks** | **基准测试**
  - GEMM performance benchmarks | GEMM 性能基准
  - Attention kernel benchmarks | 注意力内核基准
  - Convolution benchmarks | 卷积基准

### Dependencies | 依赖

- CUDA Toolkit 11.0+ (12.x recommended) | CUDA Toolkit 11.0+（推荐 12.x）
- CMake 3.20+ | CMake 3.20+
- C++17 compatible compiler | 兼容 C++17 的编译器
- pybind11 (for Python bindings) | pybind11（用于 Python 绑定）

---

## Version History Summary | 版本历史汇总

| Version | Date | Description | 描述 |
|---------|------|-------------|------|
| 3.0.0 | 2025-04-16 | Bilingual documentation, CHANGELOG professionalization | 双语文档，CHANGELOG 专业化 |
| 2.0.0 | 2026-03-09 | Critical bug fixes, architecture improvements | 关键错误修复，架构改进 |
| 1.1.0 | 2026-01-08 | Build system fixes for CUDA-optional environments | 可选 CUDA 环境的构建系统修复 |
| 1.0.1 | 2025-02-13 | Project infrastructure improvements | 项目基础设施改进 |
| 1.0.0 | 2024-01-01 | Initial release | 初始发布 |

---

## Links | 链接

- [Unreleased]: https://github.com/LessUp/modern-ai-kernels/compare/v3.0.0...HEAD
- [3.0.0]: https://github.com/LessUp/modern-ai-kernels/compare/v2.0.0...v3.0.0
- [2.0.0]: https://github.com/LessUp/modern-ai-kernels/compare/v1.1.0...v2.0.0
- [1.1.0]: https://github.com/LessUp/modern-ai-kernels/compare/v1.0.1...v1.1.0
- [1.0.1]: https://github.com/LessUp/modern-ai-kernels/compare/v1.0.0...v1.0.1
- [1.0.0]: https://github.com/LessUp/modern-ai-kernels/releases/tag/v1.0.0
