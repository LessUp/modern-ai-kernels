# Requirements Document

## Introduction

TensorCraft-HPC 是一个现代化的高性能计算（HPC）与 AI 算子优化知识库项目。项目使用现代 C++（C++17 及以上）标准、CMake 3.20+ 构建系统，支持 CUDA 11.x 到 13.1 多版本，针对从 Ampere 到 Blackwell 的多代 GPU 架构进行优化。旨在创建一个教学友好但工业级的算子优化代码仓库。

项目口号：Demystifying High-Performance AI Kernels with Modern C++ & CUDA

### 项目目标

开发一个包含常见算子优化案例的 GitHub 项目，旨在帮助开发者在 AI 和高性能计算领域更好地实现和优化算法。每个优化案例包含：
- 介绍：简要介绍算子和优化目标
- 优化方法：深入讲解优化思路，包括算法分析和理论支持
- 实现代码：具体的 C++/CUDA 实现，代码注释详尽
- 性能对比：优化前后的性能对比数据
- 测试和验证：验证优化效果的测试代码

### 技术栈

- 现代 C++（C++17/20/23）
- 现代 CMake（3.20+）
- 现代 CUDA（12.8 targeted, 11.x - 13.1 compatible）
- AI 模型优化（卷积、全连接层、注意力机制等）
- 高性能计算优化技术

## Glossary

- **TensorCraft_System**: 整个 TensorCraft-HPC 项目系统
- **Build_System**: CMake 3.28+ 构建系统，负责项目编译和依赖管理
- **Kernel**: CUDA GPU 计算核函数
- **TMA**: Tensor Memory Accelerator，Hopper/Blackwell 架构的异步张量搬运加速器
- **WGMMA**: Warp Group Matrix Multiply-Accumulate，Hopper 架构的矩阵乘法指令
- **Tensor_Core**: NVIDIA GPU 上专用于矩阵运算的硬件单元
- **Python_Binding**: 通过 pybind11 生成的 Python 接口

## Requirements

### Requirement 1: 项目构建系统 ✅

**User Story:** As a developer, I want a modern CMake-based build system, so that I can easily compile the project with different configurations.

**Acceptance Criteria:**

1. THE Build_System SHALL use CMake 3.20+ with CMakePresets.json for configuration management
2. THE Build_System SHALL provide presets: dev, python-dev, release, debug, cpu-smoke
3. THE Build_System SHALL support CUDA 12.8 as the primary target
4. THE Build_System SHALL gracefully handle environments without CUDA by auto-disabling GPU features

### Requirement 2: 核心工具库 ✅

**User Story:** As a kernel developer, I want a set of core utilities and abstractions.

**Acceptance Criteria:**

1. THE TensorCraft_System SHALL provide CUDA error checking macros (TC_CUDA_CHECK)
2. THE TensorCraft_System SHALL provide compile-time feature detection (features.hpp)
3. THE TensorCraft_System SHALL provide type traits for numeric types
4. THE TensorCraft_System SHALL provide warp-level reduction utilities

### Requirement 3: GEMM 矩阵乘法 ✅

**User Story:** As a performance engineer, I want to understand GEMM optimization progression.

**Acceptance Criteria:**

1. THE TensorCraft_System SHALL provide GEMM implementations in progressive optimization levels:
   - v1: Naive implementation
   - v2: Shared memory tiling
   - v3: Double buffering
   - v4: Tensor Core using WMMA API
2. THE optimized GEMM kernels SHALL achieve good performance on supported hardware

### Requirement 4: LLM 关键算子 ✅

**User Story:** As an AI infrastructure engineer, I want optimized LLM-specific operators.

**Acceptance Criteria:**

1. THE TensorCraft_System SHALL provide FlashAttention-style kernel
2. THE TensorCraft_System SHALL provide RoPE kernel
3. THE TensorCraft_System SHALL provide MoE router kernel

### Requirement 5: 归一化算子 ✅

**User Story:** As a kernel developer, I want optimized normalization operations.

**Acceptance Criteria:**

1. THE TensorCraft_System SHALL provide LayerNorm kernel
2. THE TensorCraft_System SHALL provide RMSNorm kernel
3. THE TensorCraft_System SHALL provide BatchNorm kernel
4. THE TensorCraft_System SHALL provide Softmax with online algorithm

### Requirement 6: Python 绑定 ✅

**User Story:** As a researcher, I want Python bindings for all kernels.

**Acceptance Criteria:**

1. THE Python_Binding SHALL expose kernel functions with NumPy-compatible interfaces
2. THE module SHALL be named `tensorcraft_ops`
3. THE Python_Binding SHALL handle GPU memory automatically

### Requirement 7: 测试与持续集成 ✅

**User Story:** As a contributor, I want comprehensive testing infrastructure.

**Acceptance Criteria:**

1. THE TensorCraft_System SHALL provide GTest unit tests
2. CI SHALL validate format check and CPU-only configure/install
3. CUDA tests SHALL be validated on GPU machines locally

### Requirement 8: 文档 ✅

**User Story:** As a learner, I want comprehensive documentation.

**Acceptance Criteria:**

1. THE documentation SHALL be organized in docs/ directory
2. THE documentation SHALL include: getting-started/, guides/, api/, examples/, reference/
3. THE documentation SHALL be deployed via GitHub Pages
4. THE documentation SHALL include README, Installation, Architecture, API Reference

### Requirement 9: 社区治理 ✅

**User Story:** As a contributor, I want clear community guidelines.

**Acceptance Criteria:**

1. THE project SHALL provide docs/reference/code-of-conduct.md
2. THE project SHALL provide docs/reference/security.md
3. THE project SHALL provide docs/reference/contributing.md
4. THE project SHALL provide docs/reference/changelog.md
5. THE project SHALL provide GitHub Issue and PR templates

## Implementation Status

All requirements have been implemented. See `.kiro/specs/tensorcraft-hpc/tasks.md` for detailed task completion.
