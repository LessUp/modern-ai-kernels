# Product Specification: TensorCraft-HPC

> **Document Type**: Product Requirements Document (PRD)
> **Version**: 2.0.0
> **Status**: ✅ Accepted — Implemented
> **Last Updated**: 2026-04-17

---

## Introduction

TensorCraft-HPC is a modern high-performance computing (HPC) and AI operator optimization knowledge base project. The project uses modern C++ (C++17 and above), CMake 3.20+ build system, supports CUDA 11.x to 13.1 multi-version, and is optimized for multiple generations of GPU architectures from Ampere to Blackwell. The goal is to create a teaching-friendly but industrial-grade operator optimization code repository.

### Project Slogan

**Demystifying High-Performance AI Kernels with Modern C++ & CUDA**

### Project Goals

Develop a GitHub project containing common operator optimization cases to help developers better implement and optimize algorithms in the AI and HPC fields. Each optimization case includes:

1. **Introduction**: Brief introduction to the operator and optimization objectives
2. **Optimization Methods**: In-depth explanation of optimization approaches, including algorithm analysis and theoretical support
3. **Implementation Code**: Specific C++/CUDA implementations with detailed comments
4. **Performance Comparison**: Performance comparison data before and after optimization
5. **Testing and Validation**: Test code validating optimization effectiveness

---

## Glossary

| Term | Definition |
|------|------------|
| TensorCraft_System | The entire TensorCraft-HPC project system |
| Build_System | CMake 3.20+ build system for project compilation and dependency management |
| Kernel | CUDA GPU compute kernel function |
| TMA | Tensor Memory Accelerator, Hopper/Blackwell architecture async tensor memory transfer accelerator |
| WGMMA | Warp Group Matrix Multiply-Accumulate, Hopper architecture matrix multiply instructions |
| Tensor_Core | NVIDIA GPU dedicated hardware unit for matrix operations |
| Python_Binding | Python interface generated via pybind11 |

---

## Requirements

### REQ-001: Project Build System

**User Story:** As a developer, I want a modern CMake-based build system, so that I can easily compile the project with different configurations.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-001-AC1 | The Build_System SHALL use CMake 3.20+ with CMakePresets.json for configuration management |
| REQ-001-AC2 | The Build_System SHALL provide presets: `dev`, `python-dev`, `release`, `debug`, `cpu-smoke` |
| REQ-001-AC3 | The Build_System SHALL support CUDA 12.8 as the primary target |
| REQ-001-AC4 | The Build_System SHALL gracefully handle environments without CUDA by auto-disabling GPU features |

**Status:** ✅ Implemented

---

### REQ-002: Core Utility Library

**User Story:** As a kernel developer, I want a set of core utilities and abstractions.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-002-AC1 | The TensorCraft_System SHALL provide CUDA error checking macros (`TC_CUDA_CHECK`) |
| REQ-002-AC2 | The TensorCraft_System SHALL provide compile-time feature detection (`features.hpp`) |
| REQ-002-AC3 | The TensorCraft_System SHALL provide type traits for numeric types |
| REQ-002-AC4 | The TensorCraft_System SHALL provide warp-level reduction utilities |

**Status:** ✅ Implemented

---

### REQ-003: GEMM Matrix Multiplication

**User Story:** As a performance engineer, I want to understand GEMM optimization progression.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-003-AC1 | The TensorCraft_System SHALL provide GEMM implementations in progressive optimization levels: v1 (Naive), v2 (Tiled), v3 (Double Buffer), v4 (Tensor Core WMMA) |
| REQ-003-AC2 | The optimized GEMM kernels SHALL achieve good performance on supported hardware |
| REQ-003-AC3 | All GEMM versions SHALL produce numerically equivalent outputs within tolerance |

**Status:** ✅ Implemented

---

### REQ-004: LLM Key Operators

**User Story:** As an AI infrastructure engineer, I want optimized LLM-specific operators.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-004-AC1 | The TensorCraft_System SHALL provide FlashAttention-style kernel |
| REQ-004-AC2 | The TensorCraft_System SHALL provide RoPE (Rotary Positional Embeddings) kernel |
| REQ-004-AC3 | The TensorCraft_System SHALL provide MoE (Mixture of Experts) router kernel |

**Status:** ✅ Implemented

---

### REQ-005: Normalization Operators

**User Story:** As a kernel developer, I want optimized normalization operations.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-005-AC1 | The TensorCraft_System SHALL provide LayerNorm kernel |
| REQ-005-AC2 | The TensorCraft_System SHALL provide RMSNorm kernel |
| REQ-005-AC3 | The TensorCraft_System SHALL provide BatchNorm kernel |
| REQ-005-AC4 | The TensorCraft_System SHALL provide Softmax with online algorithm |

**Status:** ✅ Implemented

---

### REQ-006: Python Bindings

**User Story:** As a researcher, I want Python bindings for all kernels.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-006-AC1 | The Python_Binding SHALL expose kernel functions with NumPy-compatible interfaces |
| REQ-006-AC2 | The module SHALL be named `tensorcraft_ops` |
| REQ-006-AC3 | The Python_Binding SHALL handle GPU memory automatically |

**Status:** ✅ Implemented

---

### REQ-007: Testing and Continuous Integration

**User Story:** As a contributor, I want comprehensive testing infrastructure.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-007-AC1 | The TensorCraft_System SHALL provide GTest unit tests |
| REQ-007-AC2 | CI SHALL validate format check and CPU-only configure/install |
| REQ-007-AC3 | CUDA tests SHALL be validated on GPU machines locally |

**Status:** ✅ Implemented

---

### REQ-008: Documentation

**User Story:** As a learner, I want comprehensive documentation.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-008-AC1 | The documentation SHALL be organized in `docs/` directory |
| REQ-008-AC2 | The documentation SHALL include: `getting-started/`, `guides/`, `api/`, `examples/`, `reference/` |
| REQ-008-AC3 | The documentation SHALL be deployed via GitHub Pages |
| REQ-008-AC4 | The documentation SHALL be bilingual (English and Chinese) |

**Status:** ✅ Implemented

---

### REQ-009: Community Governance

**User Story:** As a contributor, I want clear community guidelines.

**Acceptance Criteria:**

| ID | Criterion |
|----|-----------|
| REQ-009-AC1 | The project SHALL provide `CODE_OF_CONDUCT.md` |
| REQ-009-AC2 | The project SHALL provide `SECURITY.md` |
| REQ-009-AC3 | The project SHALL provide `CONTRIBUTING.md` |
| REQ-009-AC4 | The project SHALL provide `CHANGELOG.md` |
| REQ-009-AC5 | The project SHALL provide GitHub Issue and PR templates |

**Status:** ✅ Implemented

---

## Implementation Status

| Requirement | Status | Implementation Plan |
|-------------|--------|---------------------|
| REQ-001 | ✅ Implemented | [tensorcraft-hpc-impl.md](../testing/tensorcraft-hpc-impl.md) |
| REQ-002 | ✅ Implemented | [tensorcraft-hpc-impl.md](../testing/tensorcraft-hpc-impl.md) |
| REQ-003 | ✅ Implemented | [tensorcraft-hpc-impl.md](../testing/tensorcraft-hpc-impl.md) |
| REQ-004 | ✅ Implemented | [tensorcraft-hpc-impl.md](../testing/tensorcraft-hpc-impl.md) |
| REQ-005 | ✅ Implemented | [tensorcraft-hpc-impl.md](../testing/tensorcraft-hpc-impl.md) |
| REQ-006 | ✅ Implemented | [tensorcraft-hpc-impl.md](../testing/tensorcraft-hpc-impl.md) |
| REQ-007 | ✅ Implemented | [tensorcraft-hpc-impl.md](../testing/tensorcraft-hpc-impl.md) |
| REQ-008 | ✅ Implemented | [tensorcraft-hpc-impl.md](../testing/tensorcraft-hpc-impl.md) |
| REQ-009 | ✅ Implemented | [project-polish.md](project-polish.md) |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-04-17 | Converted to new SDD format |
| 1.1.0 | 2026-03-09 | Critical bug fixes |
| 1.0.0 | 2024-01-01 | Initial release |

---

## See Also

- [RFC 0001: Core Architecture](../rfc/0001-core-architecture.md)
- [API Specification](../api/cxx-api.md)
- [Data Structure Specification](../db/data-structures.md)
- [Implementation Plan](../testing/tensorcraft-hpc-impl.md)
