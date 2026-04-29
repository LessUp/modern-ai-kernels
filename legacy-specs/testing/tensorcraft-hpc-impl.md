# Implementation Plan: TensorCraft-HPC

> **Document Type**: Implementation Plan
> **Related Spec**: [Product Specification: TensorCraft-HPC](../product/tensorcraft-hpc.md)
> **Version**: 2.0.0
> **Last Updated**: 2026-04-17

---

## Overview

This implementation plan decomposes TensorCraft-HPC into executable coding tasks, organized in progressive development order. Each task references specific requirements to ensure complete feature coverage.

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Infrastructure | 4 | ✅ Complete |
| Phase 2: Core Utilities | 4 | ✅ Complete |
| Phase 3: Basic Operators | 8 | ✅ Complete |
| Phase 4: GEMM | 5 | ✅ Complete |
| Phase 5: LLM Operators | 3 | ✅ Complete |
| Phase 6: Convolution | 3 | ✅ Complete |
| Phase 7: Sparse | 4 | ✅ Complete |
| Phase 8: Fusion | 3 | ✅ Complete |
| Phase 9: Python | 3 | ✅ Complete |
| Phase 10: Documentation | 7 | ✅ Complete |
| Phase 11: CI/CD | 3 | ✅ Complete |
| Phase 12: Community | 5 | ✅ Complete |

---

## Phase 1: Project Infrastructure Setup

**Related Requirements:** REQ-001

### Tasks

- [x] **T001**: Create project directory structure and CMakeLists.txt
- [x] **T002**: Create CMakePresets.json configuration
- [x] **T003**: Configure FetchContent dependency management
- [x] **T004**: Write build system unit tests

**Deliverables:**

```
modern-ai-kernels/
├── CMakeLists.txt
├── CMakePresets.json
└── tests/
    └── cmake_test.cpp
```

---

## Phase 2: Core Utility Library Implementation

**Related Requirements:** REQ-002

### Tasks

- [x] **T005**: Implement CUDA error checking macros (`cuda_check.hpp`)
- [x] **T006**: Implement feature detection header (`features.hpp`)
- [x] **T007**: Implement type traits and Concepts (`type_traits.hpp`)
- [x] **T008**: Implement warp-level reduction primitives (`warp_utils.hpp`)

**Deliverables:**

```
include/tensorcraft/core/
├── cuda_check.hpp
├── features.hpp
├── type_traits.hpp
└── warp_utils.hpp
```

---

## Phase 3: Basic Operators Implementation

**Related Requirements:** REQ-005

### Tasks

- [x] **T009**: Implement generic Elementwise Kernel framework
- [x] **T010**: Implement VectorAdd Kernel
- [x] **T011**: Implement standard activation functions (ReLU, SiLU, GeLU)
- [x] **T012**: Implement custom activation functions (LeakyReLU, ELU, Swish)
- [x] **T013**: Implement Softmax Kernel (online algorithm)
- [x] **T014**: Implement LayerNorm Kernel
- [x] **T015**: Implement RMSNorm Kernel
- [x] **T016**: Implement BatchNorm Kernel

**Deliverables:**

```
include/tensorcraft/kernels/
├── elementwise.hpp
├── softmax.hpp
└── normalization.hpp

tests/
├── test_elementwise.cpp
├── test_softmax.cpp
└── test_normalization.cpp
```

---

## Phase 4: GEMM Matrix Multiplication Implementation

**Related Requirements:** REQ-003

### Tasks

- [x] **T017**: Implement GEMM v1 (Naive)
- [x] **T018**: Implement GEMM v2 (Shared Memory Tiling)
- [x] **T019**: Implement GEMM v3 (Double Buffering)
- [x] **T020**: Implement GEMM v4 (Tensor Core WMMA)
- [x] **T021**: Implement matrix transpose Kernel

**Deliverables:**

```
include/tensorcraft/kernels/
└── gemm.hpp

tests/
└── test_gemm.cpp

benchmarks/
└── bench_gemm.cpp
```

---

## Phase 5: LLM Key Operators Implementation

**Related Requirements:** REQ-004

### Tasks

- [x] **T022**: Implement FlashAttention Kernel
- [x] **T023**: Implement RoPE Kernel
- [x] **T024**: Implement MoE Router Kernel

**Deliverables:**

```
include/tensorcraft/kernels/
└── attention.hpp

tests/
└── test_attention.cpp
```

---

## Phase 6: Convolution Layer Implementation

**Related Requirements:** REQ-003 (extended)

### Tasks

- [x] **T025**: Implement Conv2D Naive Kernel
- [x] **T026**: Implement Im2Col Kernel
- [x] **T027**: Implement Depthwise Separable convolution

**Deliverables:**

```
include/tensorcraft/kernels/
└── conv2d.hpp

tests/
└── test_conv2d.cpp
```

---

## Phase 7: Sparse Matrix Implementation

**Related Requirements:** REQ-003 (extended)

### Tasks

- [x] **T028**: Implement CSR format
- [x] **T029**: Implement CSC format
- [x] **T030**: Implement SpMV Kernel
- [x] **T031**: Implement SpMM Kernel

**Deliverables:**

```
include/tensorcraft/kernels/
└── sparse.hpp

tests/
└── test_sparse.cpp
```

---

## Phase 8: Operator Fusion and Quantization

**Related Requirements:** REQ-003, REQ-004 (extended)

### Tasks

- [x] **T032**: Implement Bias+GeLU fusion Epilogue
- [x] **T033**: Implement INT8 quantization support
- [x] **T034**: Implement FP8 quantization support (CUDA 12.0+)

**Deliverables:**

```
include/tensorcraft/kernels/
└── fusion.hpp

tests/
└── test_fusion.cpp
```

---

## Phase 9: Python Bindings Implementation

**Related Requirements:** REQ-006

### Tasks

- [x] **T035**: Configure pybind11 build
- [x] **T036**: Implement core operator bindings
- [x] **T037**: Implement LLM operator bindings

**Deliverables:**

```
src/python_ops/
├── CMakeLists.txt
├── module.cpp
├── tensor_bindings.cpp
└── kernel_bindings.cpp

pyproject.toml
```

---

## Phase 10: Documentation

**Related Requirements:** REQ-008

### Tasks

- [x] **T038**: Write README.md (English)
- [x] **T039**: Write README.zh-CN.md (Chinese)
- [x] **T040**: Write Modern C++ for CUDA guide
- [x] **T041**: Write operator optimization tutorials
- [x] **T042**: Write API reference documentation
- [x] **T043**: Write example documentation
- [x] **T044**: Write contributing guide

**Deliverables:**

```
docs/
├── en/
│   ├── README.md
│   ├── getting-started/
│   ├── guides/
│   ├── api/
│   ├── examples/
│   └── reference/
└── zh/
    └── (same structure)
```

---

## Phase 11: CI/CD Configuration

**Related Requirements:** REQ-007

### Tasks

- [x] **T045**: Create ci.yml workflow
- [x] **T046**: Create release.yml workflow
- [x] **T047**: Create pages.yml workflow

**Deliverables:**

```
.github/workflows/
├── ci.yml
├── release.yml
└── pages.yml
```

---

## Phase 12: GitHub Community Configuration

**Related Requirements:** REQ-009

### Tasks

- [x] **T048**: Create Issue templates (bug_report.yml, feature_request.yml)
- [x] **T049**: Create PR template
- [x] **T050**: Create CODEOWNERS
- [x] **T051**: Create Code of Conduct
- [x] **T052**: Create Security Policy

**Deliverables:**

```
.github/
├── ISSUE_TEMPLATE/
│   ├── bug_report.yml
│   └── feature_request.yml
├── PULL_REQUEST_TEMPLATE.md
└── CODEOWNERS

CODE_OF_CONDUCT.md
SECURITY.md
CONTRIBUTING.md
CHANGELOG.md
```

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 2.0.0 | 2026-03-09 | Critical bug fixes, architecture improvements |
| 1.1.0 | 2026-01-08 | Build system fixes |
| 1.0.1 | 2025-02-13 | Infrastructure improvements |
| 1.0.0 | 2024-01-01 | Initial release |

---

## Notes

- All tasks completed
- Each task references specific requirements for traceability
- Property-based tests validate general correctness properties
- Unit tests verify specific examples and edge cases
