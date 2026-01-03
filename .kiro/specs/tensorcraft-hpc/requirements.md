# Requirements Document

## Introduction

TensorCraft-HPC 是一个现代化的高性能计算（HPC）与 AI 算子优化知识库项目。项目使用现代 C++（C++17 及以上）标准、CMake 3.20+ 构建系统，支持 CUDA 11.x 到 13.1 多版本，针对从 Ampere 到 Blackwell 的多代 GPU 架构进行优化。旨在创建一个教学友好但工业级的算子优化代码仓库。

项目口号：Demystifying High-Performance AI Kernels with Modern C++ & CUDA

### 项目目标

开发一个包含常见算子优化案例的 GitHub 项目，旨在帮助开发者在 AI 和高性能计算领域更好地实现和优化算法。每个优化案例包含：
- 介绍：简要介绍算子和优化目标
- 优化方法：深入讲解优化思路，包括算法分析和理论支持
- 实现代码：具体的 C++/CUDA 实现，代码注释详尽
- 性能对比：优化前后的性能对比数据（时间、内存、带宽利用率等）
- 测试和验证：验证优化效果的测试代码

### 技术栈

- 现代 C++（C++17/20/23）
- 现代 CMake（3.20+）
- 现代 CUDA（11.x - 13.1）
- AI 模型优化（卷积、全连接层、注意力机制等）
- 高性能计算优化技术

## Glossary

- **TensorCraft_System**: 整个 TensorCraft-HPC 项目系统
- **Build_System**: CMake 3.28+ 构建系统，负责项目编译和依赖管理
- **Kernel**: CUDA GPU 计算核函数
- **TMA**: Tensor Memory Accelerator，Hopper/Blackwell 架构的异步张量搬运加速器
- **WGMMA**: Warp Group Matrix Multiply-Accumulate，Hopper 架构的矩阵乘法指令
- **Tensor_Core**: NVIDIA GPU 上专用于矩阵运算的硬件单元
- **Python_Binding**: 通过 pybind11/nanobind 生成的 Python 接口
- **Profiler**: Nsight Compute 性能分析工具
- **Roofline_Model**: 用于分析算子性能瓶颈的可视化模型

- **Sparse_Matrix**: 稀疏矩阵，使用压缩格式（CSR、CSC）存储的矩阵
- **Winograd**: Winograd 算法，用于加速卷积计算的数学变换方法
- **cuDNN**: NVIDIA 深度神经网络库，提供优化的卷积等操作
- **cuSPARSE**: NVIDIA 稀疏矩阵库
- **TensorRT**: NVIDIA 深度学习推理优化器
- **Pinned_Memory**: 固定内存/锁页内存，用于加速 CPU-GPU 数据传输
- **Memory_Pool**: 内存池，预分配内存以减少分配开销

## Requirements

### Requirement 1: 项目构建系统

**User Story:** As a developer, I want a modern CMake-based build system, so that I can easily compile the project with different configurations and manage dependencies automatically.

#### Acceptance Criteria

1. THE Build_System SHALL use CMake 3.20+ with CMakePresets.json for configuration management
2. WHEN a developer runs cmake --preset=<preset_name>, THE Build_System SHALL configure the project with predefined settings for Debug, Release, and Profile builds
3. THE Build_System SHALL use FetchContent to automatically download and configure GoogleTest, Google Benchmark, and pybind11 dependencies
4. THE Build_System SHALL support CUDA versions from 11.0 to 13.1, with feature detection for architecture-specific optimizations
5. WHEN CUDA version is below 11.0, THE Build_System SHALL report a clear error message indicating the minimum required CUDA version
6. THE Build_System SHALL support GCC 9+, Clang 10+, or MSVC 2019+ compilers with C++17 standard as minimum requirement
7. WHEN C++20 or C++23 features are available, THE Build_System SHALL enable additional optimizations and modern syntax
8. WHEN building Python bindings, THE Build_System SHALL generate a shared library (.so/.pyd) that can be imported directly in Python

### Requirement 2: 核心工具库

**User Story:** As a kernel developer, I want a set of core utilities and abstractions, so that I can focus on algorithm implementation without boilerplate code.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide a CUDA error checking macro that reports file, line, and error description
2. THE TensorCraft_System SHALL provide a Tensor wrapper class supporting FP32, FP16, BF16, and FP8 data types
3. WHEN a Tensor is created, THE TensorCraft_System SHALL allocate GPU memory and track ownership using RAII
4. THE TensorCraft_System SHALL provide type traits for constraining numeric template parameters (using C++20 Concepts when available, SFINAE for C++17)
5. THE TensorCraft_System SHALL provide aligned vector types (AlignedVector<T, N>) for vectorized memory access
6. WHEN converting between precision types, THE TensorCraft_System SHALL provide explicit conversion functions with proper rounding
7. THE TensorCraft_System SHALL use compile-time feature detection macros to enable architecture-specific code paths

### Requirement 3: 基础算子 - Elementwise 操作与激活函数

**User Story:** As a learner, I want to understand basic GPU programming patterns through elementwise operations and activation functions, so that I can build a foundation for more complex kernels.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide a VectorAdd kernel demonstrating coalesced memory access patterns
2. THE TensorCraft_System SHALL provide standard activation function kernels (ReLU, SiLU, GeLU) using modern C++ templates
3. THE TensorCraft_System SHALL provide custom activation function kernels (Leaky ReLU, ELU, Swish) with configurable parameters
4. WHEN a user provides a lambda function, THE TensorCraft_System SHALL generate an elementwise kernel at compile time using templates
5. THE TensorCraft_System SHALL provide both naive and optimized (vectorized load, SIMD) versions of each elementwise kernel
6. THE TensorCraft_System SHALL demonstrate CUDA Streams usage for asynchronous activation function execution
7. THE TensorCraft_System SHALL provide memory alignment techniques for custom activation functions
8. WHEN profiled, THE optimized elementwise kernels SHALL achieve at least 80% of theoretical memory bandwidth

### Requirement 4: 规约算子与归一化层

**User Story:** As a kernel developer, I want optimized reduction and normalization operations, so that I can implement efficient normalization layers for neural networks.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide a Softmax kernel using online algorithm to minimize global memory access
2. THE TensorCraft_System SHALL provide LayerNorm and RMSNorm kernels with warp-level reduction
3. THE TensorCraft_System SHALL provide BatchNorm kernel with optimized forward and backward pass
4. WHEN the hidden dimension exceeds warp size, THE reduction kernels SHALL use shared memory for cross-warp reduction
5. THE TensorCraft_System SHALL provide vectorized load (LDS.128) versions of normalization kernels
6. THE TensorCraft_System SHALL demonstrate BatchNorm optimization by merging computations and reducing memory access
7. THE TensorCraft_System SHALL provide fused BatchNorm+Activation kernels to reduce memory round-trips
8. WHEN profiled, THE optimized reduction kernels SHALL show reduced global memory transactions compared to naive versions

### Requirement 5: GEMM 矩阵乘法

**User Story:** As a performance engineer, I want to understand GEMM optimization progression, so that I can apply these techniques to other compute-bound kernels.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide GEMM implementations in progressive optimization levels:
   - v1: Naive implementation (all CUDA versions)
   - v2: Shared memory tiling (all CUDA versions)
   - v3: Double buffering for latency hiding (all CUDA versions)
   - v4: Tensor Core using WMMA API (CUDA 11.0+, Volta+)
   - v5: WGMMA + TMA for Hopper/Blackwell architecture (CUDA 12.0+, Hopper+)
2. WHEN using Tensor Cores, THE GEMM kernel SHALL support FP16 and BF16 input types; FP8 support requires CUDA 12.0+
3. THE TensorCraft_System SHALL provide clear documentation explaining each optimization step
4. WHEN profiled on supported hardware, THE optimized GEMM kernels SHALL achieve at least 70% of peak TFLOPS for large matrices (M,N,K >= 4096)
5. THE TensorCraft_System SHALL provide a matrix transpose kernel demonstrating shared memory bank conflict elimination
6. WHEN advanced features (TMA, WGMMA) are not available, THE Build_System SHALL gracefully fall back to compatible implementations

### Requirement 6: LLM 关键算子

**User Story:** As an AI infrastructure engineer, I want optimized LLM-specific operators, so that I can understand and implement efficient inference systems.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide a FlashAttention-style kernel with Q/K/V tiling and online softmax
2. WHEN using Hopper architecture (CUDA 12.0+), THE FlashAttention kernel SHALL utilize TMA for asynchronous Q/K/V loading
3. THE TensorCraft_System SHALL provide a RoPE (Rotational Positional Embeddings) kernel with efficient complex rotation
4. THE TensorCraft_System SHALL provide a simplified PagedAttention kernel demonstrating non-contiguous KV cache access
5. WHEN the sequence length exceeds shared memory capacity, THE attention kernels SHALL use proper tiling strategies
6. THE TensorCraft_System SHALL provide MoE (Mixture of Experts) router kernel for expert selection
7. WHEN advanced architecture features are unavailable, THE LLM kernels SHALL provide fallback implementations using standard CUDA APIs

### Requirement 7: 算子融合与量化

**User Story:** As an optimization engineer, I want to understand operator fusion and quantization techniques, so that I can reduce memory bandwidth bottlenecks.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide a fused Bias+GeLU epilogue that integrates with GEMM output
2. THE TensorCraft_System SHALL demonstrate C++ template functors for flexible epilogue design
3. WHEN CUDA 12.0+ is available, THE TensorCraft_System SHALL provide FP8 GEMM kernel using __nv_fp8_e4m3 type
4. WHEN performing quantization, THE TensorCraft_System SHALL provide data layout transformation utilities
5. THE TensorCraft_System SHALL document the performance impact of fusion on memory bandwidth utilization
6. THE TensorCraft_System SHALL provide INT8 quantization support for CUDA 11.0+ environments

### Requirement 8: Python 绑定与验证

**User Story:** As a researcher, I want Python bindings for all kernels, so that I can integrate them with PyTorch and verify numerical correctness.

#### Acceptance Criteria

1. THE Python_Binding SHALL expose all kernel functions with numpy-compatible interfaces
2. WHEN a kernel is called from Python, THE Python_Binding SHALL handle GPU memory allocation and data transfer automatically
3. THE TensorCraft_System SHALL provide verification scripts comparing results with torch.nn.functional equivalents
4. WHEN comparing results, THE verification scripts SHALL use torch.allclose with appropriate tolerances for each precision
5. THE TensorCraft_System SHALL provide benchmark scripts that generate performance comparison plots

### Requirement 9: 性能分析与文档

**User Story:** As a learner, I want comprehensive profiling documentation, so that I can understand how to analyze and optimize GPU kernels.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL include Nsight Compute profile reports (.ncu-rep) for key kernels
2. THE documentation SHALL include Roofline Model analysis showing memory-bound vs compute-bound characteristics
3. WHEN showing optimization progression, THE documentation SHALL include before/after memory bandwidth utilization metrics
4. THE TensorCraft_System SHALL provide a "Modern C++ for CUDA" guide demonstrating C++17/20/23 features in kernel code
5. THE documentation SHALL explain bank conflict, memory coalescing, and occupancy concepts with visual diagrams
6. THE documentation SHALL clearly indicate which features require specific CUDA versions or GPU architectures

### Requirement 10: 测试与持续集成

**User Story:** As a contributor, I want comprehensive testing infrastructure, so that I can verify correctness and prevent regressions.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide GTest unit tests for all kernel implementations
2. WHEN a kernel has multiple optimization levels, THE tests SHALL verify numerical equivalence between versions
3. THE TensorCraft_System SHALL provide Google Benchmark tests for performance regression detection
4. IF a test fails due to numerical precision, THEN THE TensorCraft_System SHALL report the maximum absolute and relative errors
5. THE TensorCraft_System SHALL support CI/CD pipeline configuration for automated testing

### Requirement 11: 卷积层优化

**User Story:** As a deep learning engineer, I want optimized convolution implementations, so that I can understand and apply convolution optimization techniques in neural networks.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide a naive Conv2D implementation as baseline
2. THE TensorCraft_System SHALL provide im2col + GEMM based convolution implementation
3. THE TensorCraft_System SHALL provide Winograd algorithm based convolution for small filter sizes (3x3, 5x5)
4. THE TensorCraft_System SHALL provide depthwise separable convolution implementation
5. THE TensorCraft_System SHALL demonstrate cuDNN integration for comparison benchmarks
6. WHEN profiled, THE optimized convolution kernels SHALL show significant speedup over naive implementation
7. THE documentation SHALL explain the trade-offs between different convolution algorithms (memory vs compute)

### Requirement 12: 稀疏矩阵优化

**User Story:** As a performance engineer, I want sparse matrix operations, so that I can optimize models with sparse weight matrices.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL provide CSR (Compressed Sparse Row) format implementation
2. THE TensorCraft_System SHALL provide CSC (Compressed Sparse Column) format implementation
3. THE TensorCraft_System SHALL provide sparse matrix-vector multiplication (SpMV) kernel
4. THE TensorCraft_System SHALL provide sparse matrix-matrix multiplication (SpMM) kernel
5. THE TensorCraft_System SHALL demonstrate cuSPARSE integration for comparison benchmarks
6. THE TensorCraft_System SHALL provide format conversion utilities between dense and sparse representations
7. WHEN the sparsity ratio exceeds 90%, THE sparse kernels SHALL outperform dense equivalents

### Requirement 13: 内存优化技术

**User Story:** As a CUDA developer, I want to understand memory optimization techniques, so that I can reduce memory bottlenecks in my kernels.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL demonstrate shared memory usage patterns for reducing global memory access
2. THE TensorCraft_System SHALL demonstrate pinned memory (page-locked memory) for faster CPU-GPU transfers
3. THE TensorCraft_System SHALL provide memory pool implementation for reducing allocation overhead
4. THE TensorCraft_System SHALL demonstrate CUDA Streams and Events for asynchronous execution
5. THE TensorCraft_System SHALL provide examples of memory coalescing optimization
6. THE TensorCraft_System SHALL demonstrate bank conflict avoidance in shared memory
7. THE documentation SHALL include memory access pattern analysis with Nsight Compute screenshots

### Requirement 14: 深度学习推理优化

**User Story:** As an ML engineer, I want to understand inference optimization techniques, so that I can deploy efficient models in production.

#### Acceptance Criteria

1. THE TensorCraft_System SHALL demonstrate TensorRT integration for model optimization
2. THE TensorCraft_System SHALL provide examples of operator fusion for inference
3. THE TensorCraft_System SHALL demonstrate CUDA Graphs for reducing kernel launch overhead
4. THE TensorCraft_System SHALL provide mixed-precision inference examples (FP16/INT8)
5. THE documentation SHALL explain the inference optimization pipeline from trained model to deployment
6. WHEN using TensorRT, THE optimized model SHALL show measurable latency reduction compared to native PyTorch
