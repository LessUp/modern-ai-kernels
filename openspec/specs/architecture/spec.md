# Architecture Specifications

> **Domain**: Design Decisions
> **Version**: 2.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2026-04-23

---

## Overview

This document captures the key architectural decisions for TensorCraft-HPC, a modular high-performance AI operator optimization library supporting CUDA 11.0 to CUDA 13.1.

---

## Design Principles

1. **Progressive Optimization**: Each operator provides multiple versions from naive to optimized
2. **Compile-Time Feature Detection**: Macros and templates select optimal implementation paths at compile time
3. **Zero-Cost Abstraction**: Modern C++ template techniques for high-performance generic programming
4. **Teaching-Friendly**: Clear code structure with detailed comments for learning

---

## ADDED Requirements

### Requirement: Layered Architecture (ARCH-001)

**User Story:** As an architect, I want a clear layered design, so that the codebase is maintainable and extensible.

#### Scenario: Layer Separation
- **WHEN** organizing the codebase
- **THEN** the architecture SHALL follow these layers:

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

---

### Requirement: Directory Structure (ARCH-002)

**User Story:** As a developer, I want a clear directory layout, so that I can find files easily.

#### Scenario: Project Layout
- **WHEN** navigating the project
- **THEN** the following directory structure SHALL be used:

```
modern-ai-kernels/
├── openspec/                 # OpenSpec workflow
├── specs/                    # Legacy SDD specs (archive)
├── include/tensorcraft/      # Header-only kernel library
│   ├── core/                 # CUDA error handling, type traits
│   ├── memory/               # Tensor, memory pool
│   └── kernels/              # All compute kernels
├── src/python_ops/           # Python bindings
├── tests/                    # Unit tests
├── benchmarks/               # Performance benchmarks
├── docs/                     # Documentation (en/, zh/)
├── examples/                 # Example code
└── changelog/                # Development changelog
```

---

### Requirement: Component Organization (ARCH-003)

**User Story:** As a kernel developer, I want clear component boundaries, so that I can develop features independently.

#### Scenario: Core Utilities
- **WHEN** developing core functionality
- **THEN** the following components SHALL be in `include/tensorcraft/core/`:

| File | Purpose |
|------|---------|
| `cuda_check.hpp` | CUDA error checking macros and exception handling |
| `features.hpp` | Compile-time feature detection |
| `type_traits.hpp` | Type traits and Concepts |
| `warp_utils.hpp` | Warp-level reduction primitives |

#### Scenario: Memory Management
- **WHEN** managing GPU memory
- **THEN** the following components SHALL be in `include/tensorcraft/memory/`:

| File | Purpose |
|------|---------|
| `aligned_vector.hpp` | Aligned vector types for vectorized memory access |
| `tensor.hpp` | RAII-style GPU Tensor wrapper |
| `memory_pool.hpp` | Thread-safe GPU memory pool |

- **AND** `Tensor::fill()` SHALL delegate to the shared memory-operations module so fill behavior
  lives behind one seam

#### Scenario: Kernels
- **WHEN** implementing compute kernels
- **THEN** the following components SHALL be in `include/tensorcraft/kernels/`:

| File | Purpose |
|------|---------|
| `elementwise.hpp` | Element-wise operations and activation functions |
| `softmax.hpp` | Numerically stable Softmax |
| `normalization.hpp` | LayerNorm, RMSNorm, BatchNorm |
| `gemm.hpp` | Matrix multiplication (progressive optimization) |
| `attention.hpp` | FlashAttention, RoPE, MoE Router |
| `conv2d.hpp` | 2D convolution operations |
| `sparse.hpp` | Sparse matrix operations (CSR/CSC) |
| `fusion.hpp` | Fused operators and quantization |

- **AND** `CSRMatrixView<T>` SHALL be the primary seam for sparse launchers, with `CSRMatrix<T>`
  acting as the owning adapter for that seam

---

### Requirement: Build System Architecture (ARCH-004)

**User Story:** As a build engineer, I want a configurable build system, so that I can build for different targets.

#### Scenario: CMake Presets
- **WHEN** building the project
- **THEN** the following presets SHALL be available:

| Preset | Purpose |
|--------|---------|
| `dev` | Recommended CUDA development preset |
| `python-dev` | Lighter CUDA build for Python bindings |
| `release` | Full release build with benchmarks |
| `debug` | Debug-oriented CUDA build |
| `cpu-smoke` | CPU-only configure/install validation |

#### Scenario: Dependencies
- **WHEN** building the project
- **THEN** the following dependencies SHALL be managed:

| Dependency | Version | Purpose |
|------------|---------|---------|
| CUDA Toolkit | 12.8 (targeted) | GPU compute |
| CMake | 3.20+ | Build system |
| C++ Compiler | C++17+ | Code compilation |
| pybind11 | 2.x | Python bindings |
| GoogleTest | 1.x | Unit testing |

---

### Requirement: Testing Strategy (ARCH-005)

**User Story:** As a QA engineer, I want comprehensive testing, so that I can verify correctness.

#### Scenario: Test Types
- **WHEN** testing the codebase
- **THEN** the following test types SHALL be used:

| Test Type | Framework | Purpose |
|-----------|-----------|---------|
| Unit Tests | GoogleTest | Specific examples and edge cases |
| Property-Based Tests | Custom | Random input validation for general properties |
| Python Verification | PyTorch | Comparison with reference implementations |
| Benchmarks | Google Benchmark | Performance regression detection |

---

### Requirement: Error Handling Strategy (ARCH-006)

**User Story:** As a developer, I want consistent error handling, so that I can diagnose issues easily.

#### Scenario: CUDA Error Propagation
- **WHEN** a CUDA error occurs
- **THEN** the error SHALL be wrapped in `CudaException` with:
  - Source file name
  - Line number
  - Error description

#### Scenario: Exception Hierarchy
- **WHEN** catching errors
- **THEN** `CudaException` SHALL be catchable as `std::runtime_error`

---

### Requirement: GPU Architecture Support (ARCH-007)

**User Story:** As a deployment engineer, I want broad GPU support, so that the library works on many systems.

#### Scenario: Supported Architectures
- **WHEN** deploying the library
- **THEN** the following GPU architectures SHALL be supported:

| Architecture | SM | Tensor Core | TMA | WGMMA |
|--------------|-----|-------------|-----|-------|
| Volta | 70 | ✅ | ❌ | ❌ |
| Turing | 75 | ✅ | ❌ | ❌ |
| Ampere | 80 | ✅ | ❌ | ❌ |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ |
| Hopper | 90 | ✅ | ✅ | ✅ |
| Blackwell | 100 | ✅ | ✅ | ✅ |

---

## Architectural Decisions

### Decision: Allocator Seam for Memory Management (ARCH-008)

**Context:** Tensor classes need flexible memory allocation strategies for different use cases (production pooling, testing isolation, custom hardware).

**Decision:** Introduce an `Allocator` abstraction layer between `Tensor` and `MemoryPool`.

```cpp
template <typename T, typename Allocator = PoolAllocator>
class Tensor { ... };
```

**Consequences:**
- ✅ Enables testing with `DirectAllocator` (isolated memory)
- ✅ Supports custom allocators for special hardware (UVM, stream-ordered)
- ✅ Maintains backward compatibility with default `PoolAllocator`

**Related Files:** `include/tensorcraft/memory/allocator.hpp`, `tensor.hpp`

---

### Decision: CSRMatrix Direct MemoryPool Usage (ARCH-009)

**Context:** `CSRMatrix` manages three arrays (`values`, `col_indices`, `row_ptrs`) of different types.

**Decision:** `CSRMatrix` uses `MemoryPool` directly without the `Allocator` template parameter.

**Rationale:**
1. **Complexity Trade-off:** Adding `Allocator` template would complicate the interface for minimal practical benefit
2. **Sparse Matrix Use Case:** Typically used in batch processing where memory pooling provides clear performance benefit
3. **Consistency with Dense Operations:** Most sparse operations follow similar allocation patterns

**Consequences:**
- ⚠️ Cannot use `DirectAllocator` for testing isolation (minor limitation)
- ✅ Simpler interface for users
- ✅ Consistent performance characteristics
- ✅ `CSRMatrixView<T>` is the non-owning launch seam for sparse kernels
- ✅ Raw pointer bundles do not remain as a parallel public interface

**Future Consideration:** If testing isolation becomes necessary, can add `CSRMatrix<T, Allocator>` in a future version.

**Related Files:** `include/tensorcraft/kernels/sparse.hpp`

---

### Decision: Operation Registry as Design Reserve (ARCH-010)

**Context:** `op_registry.hpp` provides macros (`TC_REGISTER_UNARY_OP`, `TC_REGISTER_BINARY_OP`) for declarative operation registration.

**Decision:** Keep the registry infrastructure as a **design reserve** for future use.

**Current State:**
- Macros are defined but not yet used in production code
- `ops::` namespace provides factory functions for operation entries
- Python bindings are currently hand-written

**Rationale:**
1. **Low Maintenance Cost:** ~220 lines of code with minimal dependencies
2. **Future Value:** Can auto-generate Python bindings when operation count grows
3. **Similar to PyTorch:** Registry pattern proven in production frameworks

**When to Activate:**
- Operation count exceeds ~20 (current: ~10)
- Need for automatic Python binding generation
- Need for runtime operation discovery

**Related Files:** `include/tensorcraft/kernels/op_registry.hpp`

---

### Decision: Unified Test Infrastructure (ARCH-011)

**Context:** Test files previously used manual `cudaMalloc`/`cudaFree` patterns.

**Decision:** Provide `test_utils.hpp` with `DeviceBuffer<T, Allocator>` and `CudaTest` base class.

**Usage:**
```cpp
#include "test_utils.hpp"
using tensorcraft::test::DeviceBuffer;
using tensorcraft::test::CudaTest;

class MyTest : public CudaTest { ... };
```

**Consequences:**
- ✅ Unified test memory management
- ✅ `DirectAllocator` provides test isolation from production `MemoryPool`
- ✅ Common CUDA device setup via `CudaTest` base class

**Migration Status:** `test_attention.cpp` migrated; other tests can follow incrementally.

**Related Files:** `tests/test_utils.hpp`, `tests/test_attention.cpp`

---

## See Also

- [Core Specifications](../core/spec.md) — Product requirements
- [API Specifications](../api/spec.md) — API contracts
- [Data Structures](../data-structures/spec.md) — Memory layouts
