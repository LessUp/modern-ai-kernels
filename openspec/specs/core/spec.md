# Core Specifications

> **Domain**: Core Product Requirements
> **Version**: 2.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2026-04-23

---

## Overview

This specification defines the core product requirements for TensorCraft-HPC, a modern high-performance computing (HPC) and AI operator optimization library. The project uses modern C++ (C++17+), CMake 3.20+ build system, supports CUDA 11.0 to 13.1, and is optimized for GPU architectures from Volta (SM70) to Blackwell (SM100).

**Project Slogan**: *Demystifying High-Performance AI Kernels with Modern C++ & CUDA*

---

## ADDED Requirements

### Requirement: Project Build System (REQ-001)

**User Story:** As a developer, I want a modern CMake-based build system, so that I can easily compile the project with different configurations.

#### Scenario: CMake Configuration
- **WHEN** a developer configures the project
- **THEN** the Build_System SHALL use CMake 3.20+ with CMakePresets.json for configuration management

#### Scenario: Build Presets
- **WHEN** a developer lists available presets
- **THEN** the Build_System SHALL provide presets: `dev`, `python-dev`, `release`, `debug`, `cpu-smoke`

#### Scenario: CUDA Version Support
- **WHEN** building with CUDA
- **THEN** the Build_System SHALL support CUDA 12.8 as the primary target version

#### Scenario: CUDA-Optional Environment
- **WHEN** CUDA is not available in the build environment
- **THEN** the Build_System SHALL gracefully handle by auto-disabling GPU features

---

### Requirement: Core Utility Library (REQ-002)

**User Story:** As a kernel developer, I want a set of core utilities and abstractions, so that I can focus on kernel logic rather than boilerplate.

#### Scenario: CUDA Error Checking
- **WHEN** a CUDA API call fails
- **THEN** the TensorCraft_System SHALL provide `TC_CUDA_CHECK` macro that throws `CudaException` with file, line, and error description

#### Scenario: Feature Detection
- **WHEN** compiling the project
- **THEN** the TensorCraft_System SHALL provide compile-time feature detection macros via `features.hpp` (TC_CUDA_12, TC_HAS_WMMA, etc.)

#### Scenario: Type Traits
- **WHEN** working with numeric types
- **THEN** the TensorCraft_System SHALL provide type traits (`is_half_v`, `is_fp8_v`, `is_numeric_v`) and C++20 concepts (`Numeric`, `FloatingPoint`)

#### Scenario: Warp Utilities
- **WHEN** implementing reduction operations
- **THEN** the TensorCraft_System SHALL provide warp-level reduction utilities (`warp_reduce_max`, `warp_reduce_sum`, `block_reduce_sum`)

---

### Requirement: GEMM Matrix Multiplication (REQ-003)

**User Story:** As a performance engineer, I want to understand GEMM optimization progression, so that I can learn optimization techniques.

#### Scenario: Progressive Optimization Levels
- **WHEN** using GEMM operations
- **THEN** the TensorCraft_System SHALL provide implementations in progressive optimization levels:
  - v1 (Naive): Basic implementation
  - v2 (Tiled): Shared memory tiling
  - v3 (Double Buffer): Double buffering optimization
  - v4 (Tensor Core WMMA): Tensor Core acceleration

#### Scenario: Performance Target
- **WHEN** benchmarking optimized GEMM kernels
- **THEN** the kernels SHALL achieve good performance on supported hardware relative to theoretical peak

#### Scenario: Numerical Equivalence
- **WHEN** comparing outputs from different GEMM versions
- **THEN** all versions SHALL produce numerically equivalent outputs within floating-point tolerance

---

### Requirement: LLM Key Operators (REQ-004)

**User Story:** As an AI infrastructure engineer, I want optimized LLM-specific operators, so that I can efficiently run transformer models.

#### Scenario: FlashAttention Kernel
- **WHEN** computing self-attention in transformers
- **THEN** the TensorCraft_System SHALL provide FlashAttention-style kernel that is memory-efficient

#### Scenario: RoPE Kernel
- **WHEN** applying positional embeddings
- **THEN** the TensorCraft_System SHALL provide RoPE (Rotary Positional Embeddings) kernel with precomputed cos/sin cache

#### Scenario: MoE Router Kernel
- **WHEN** routing tokens to mixture-of-experts
- **THEN** the TensorCraft_System SHALL provide MoE router kernel that computes top-k expert selection with softmax weights

---

### Requirement: Normalization Operators (REQ-005)

**User Story:** As a kernel developer, I want optimized normalization operations, so that I can efficiently normalize tensors in neural networks.

#### Scenario: LayerNorm Kernel
- **WHEN** normalizing across a layer dimension
- **THEN** the TensorCraft_System SHALL provide LayerNorm kernel with gamma and beta parameters

#### Scenario: RMSNorm Kernel
- **WHEN** normalizing with root mean square
- **THEN** the TensorCraft_System SHALL provide RMSNorm kernel (LLM-preferred normalization)

#### Scenario: BatchNorm Kernel
- **WHEN** normalizing across a batch dimension
- **THEN** the TensorCraft_System SHALL provide BatchNorm kernel with optional ReLU fusion

#### Scenario: Softmax Kernel
- **WHEN** computing softmax probabilities
- **THEN** the TensorCraft_System SHALL provide numerically stable Softmax with online algorithm

---

### Requirement: Python Bindings (REQ-006)

**User Story:** As a researcher, I want Python bindings for all kernels, so that I can use them from Python code.

#### Scenario: NumPy-Compatible Interface
- **WHEN** calling kernels from Python
- **THEN** the Python_Binding SHALL expose functions with NumPy-compatible interfaces (accept and return `np.ndarray`)

#### Scenario: Module Name
- **WHEN** importing the module
- **THEN** the module SHALL be named `tensorcraft_ops`

#### Scenario: Memory Management
- **WHEN** working with GPU data from Python
- **THEN** the Python_Binding SHALL handle GPU memory automatically (no explicit allocation/deallocation required)

---

### Requirement: Testing and Continuous Integration (REQ-007)

**User Story:** As a contributor, I want comprehensive testing infrastructure, so that I can verify correctness and catch regressions.

#### Scenario: Unit Tests
- **WHEN** running the test suite
- **THEN** the TensorCraft_System SHALL provide GTest-based unit tests covering all kernel functionality

#### Scenario: CI Validation
- **WHEN** the CI pipeline runs
- **THEN** CI SHALL validate code formatting (clang-format) and CPU-only configure/install

#### Scenario: Local CUDA Tests
- **WHEN** running tests on GPU machines
- **THEN** CUDA tests SHALL be validated locally (GitHub runners lack GPUs)

---

### Requirement: Documentation (REQ-008)

**User Story:** As a learner, I want comprehensive documentation, so that I can understand and use the library effectively.

#### Scenario: Documentation Directory
- **WHEN** exploring documentation
- **THEN** the documentation SHALL be organized in `docs/` directory

#### Scenario: Documentation Structure
- **WHEN** navigating the docs
- **THEN** the documentation SHALL include sections: `getting-started/`, `guides/`, `api/`, `examples/`, `reference/`

#### Scenario: Documentation Deployment
- **WHEN** documentation is published
- **THEN** the documentation SHALL be deployed via GitHub Pages

#### Scenario: Bilingual Support
- **WHEN** reading documentation
- **THEN** the documentation SHALL be available in both English and Chinese (Simplified)

---

### Requirement: Community Governance (REQ-009)

**User Story:** As a contributor, I want clear community guidelines, so that I know how to participate in the project.

#### Scenario: Code of Conduct
- **WHEN** joining the community
- **THEN** the project SHALL provide `CODE_OF_CONDUCT.md` defining community standards

#### Scenario: Security Policy
- **WHEN** reporting security vulnerabilities
- **THEN** the project SHALL provide `SECURITY.md` with security policy

#### Scenario: Contributing Guide
- **WHEN** making contributions
- **THEN** the project SHALL provide `CONTRIBUTING.md` with contribution guidelines

#### Scenario: Changelog
- **WHEN** tracking project changes
- **THEN** the project SHALL provide `CHANGELOG.md` following Keep a Changelog format

#### Scenario: Issue and PR Templates
- **WHEN** creating issues or pull requests
- **THEN** the project SHALL provide GitHub Issue and PR templates for consistency

---

## Correctness Properties

### Property 1: Tensor RAII Memory Management
For any Tensor object created with a given shape, GPU memory SHALL be allocated on construction and freed on destruction, with no memory leaks.

### Property 2: GEMM Mathematical Correctness
For any matrices A[M×K], B[K×N] and scalars alpha, beta, GEMM SHALL compute C = alpha * A @ B + beta * C correctly within floating-point tolerance.

### Property 3: Softmax Row Sum Invariant
For any input matrix X, the Softmax output S SHALL satisfy:
- `sum(S[i, :]) = 1.0` for all rows i (within tolerance)
- `S[i, j] >= 0` for all elements

### Property 4: Optimization Level Numerical Equivalence
For any kernel with multiple optimization levels, all versions SHALL produce numerically equivalent outputs within tolerance.

---

## See Also

- [Polish Specifications](../polish/spec.md) — REQ-010 to REQ-017
- [API Specifications](../api/spec.md) — C++ API contracts
- [Data Structures](../data-structures/spec.md) — Memory layouts and types
- [Architecture](../architecture/spec.md) — Design decisions
