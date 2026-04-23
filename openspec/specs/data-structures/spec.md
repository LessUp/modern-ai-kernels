# Data Structure Specifications

> **Domain**: Memory Layouts and Types
> **Version**: 2.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2026-04-23

---

## Overview

This specification defines the data structures and memory layouts used in TensorCraft-HPC. All implementations must conform to these specifications for correctness and performance.

---

## ADDED Requirements

### Requirement: Tensor Memory Layout (DATA-001)

**User Story:** As a kernel developer, I want a defined tensor memory layout, so that I can write correct memory access patterns.

#### Scenario: Row-Major Layout
- **WHEN** storing tensor data
- **THEN** tensors SHALL use row-major (C-style) order

```
Tensor<T> Memory Layout:
┌─────────────────────────────────────────────────────────────┐
│                    GPU Global Memory                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Element[0]  Element[1]  ...  Element[N-1]              ││
│  │  (sizeof(T)) (sizeof(T))       (sizeof(T))              ││
│  └─────────────────────────────────────────────────────────┘│
│  Shape: [D0, D1, D2, ..., Dk]                               │
│  Total Elements: D0 * D1 * D2 * ... * Dk                    │
│  Strides: [D1*D2*...*Dk, D2*...*Dk, ..., Dk, 1] (row-major) │
└─────────────────────────────────────────────────────────────┘
```

#### Scenario: Shape Representation
- **WHEN** representing tensor shape
- **THEN** shape SHALL be stored as `std::vector<size_t>` with dimensions in row-major order

---

### Requirement: Supported Data Types (DATA-002)

**User Story:** As a developer, I want a defined set of supported data types, so that I know what precision options are available.

#### Scenario: Floating-Point Types
- **WHEN** using floating-point data
- **THEN** the following types SHALL be supported:

| Type | C++ Type | Size (bytes) | Range |
|------|----------|--------------|-------|
| FP32 | `float` | 4 | IEEE 754 single |
| FP16 | `half` | 2 | IEEE 754 half |
| BF16 | `__nv_bfloat16` | 2 | Brain float |
| FP8_E4M3 | `__nv_fp8_e4m3` | 1 | E4M3 format |
| FP8_E5M2 | `__nv_fp8_e5m2` | 1 | E5M2 format |

#### Scenario: Integer Types
- **WHEN** using integer data
- **THEN** the following types SHALL be supported:

| Type | C++ Type | Size (bytes) | Range |
|------|----------|--------------|-------|
| INT8 | `int8_t` | 1 | [-128, 127] |
| INT32 | `int32_t` | 4 | [-2^31, 2^31-1] |
| INT64 | `int64_t` | 8 | [-2^63, 2^63-1] |

---

### Requirement: Memory Pool Structure (DATA-003)

**User Story:** As a performance engineer, I want a memory pool, so that I can reduce allocation overhead.

#### Scenario: Block Tracking
- **WHEN** managing memory blocks
- **THEN** the MemoryPool SHALL track allocated blocks with:

```
MemoryPool Block:
┌────────────────────────────────────────────────────────────┐
│  Block Header (Host)                                        │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ void* ptr    │ size_t size  │ bool in_use  │            │
│  └──────────────┴──────────────┴──────────────┘            │
│  Tracking Maps:                                             │
│  - std::unordered_map<void*, size_t> allocated_sizes_      │
│  - std::unordered_map<void*, size_t> freed_sizes_          │
└────────────────────────────────────────────────────────────┘
```

#### Scenario: Thread Safety
- **WHEN** using the memory pool from multiple threads
- **THEN** the MemoryPool SHALL be thread-safe via mutex

---

### Requirement: Matrix Formats (DATA-004)

**User Story:** As a kernel developer, I want standard matrix formats, so that I can support both dense and sparse operations.

#### Scenario: Dense Matrix (Row-Major)
- **WHEN** storing dense matrices
- **THEN** matrices SHALL use row-major layout:

```
Dense Matrix MxN:
Row 0: [A(0,0), A(0,1), ..., A(0,N-1)]
Row 1: [A(1,0), A(1,1), ..., A(1,N-1)]
...
Row M-1: [A(M-1,0), A(M-1,1), ..., A(M-1,N-1)]

Memory: [A(0,0), ..., A(0,N-1), A(1,0), ..., A(M-1,N-1)]
Stride: N elements per row
```

#### Scenario: CSR Sparse Matrix
- **WHEN** storing sparse matrices in CSR format
- **THEN** the structure SHALL be:

```
CSR Format (Compressed Sparse Row):
values[]:     [v0, v1, ..., v(nnz-1)]     // Non-zero values
col_indices[]: [c0, c1, ..., c(nnz-1)]    // Column indices
row_ptrs[]:   [0, r1, r2, ..., rM, nnz]  // Row pointers (M+1 elements)
```

---

### Requirement: GEMM Tiling Structure (DATA-005)

**User Story:** As a kernel developer, I want defined tiling parameters, so that I can optimize for shared memory.

#### Scenario: Block Tiling
- **WHEN** implementing tiled GEMM
- **THEN** the following tile dimensions SHALL be used:

```
Block Tile Dimensions:
- BLOCK_M: 128 (rows of C per block)
- BLOCK_N: 128 (cols of C per block)
- BLOCK_K: 32  (K dimension per iteration)

Shared Memory Usage:
- A_tile: 128 × 32 × sizeof(T) = 16KB
- B_tile: 32 × 128 × sizeof(T) = 16KB
- Total: 32KB per block
```

#### Scenario: Thread Tiling
- **WHEN** implementing thread-level tiling
- **THEN** the following dimensions SHALL be used:

```
Thread Tile Dimensions:
- THREAD_M: 8 (rows of C per thread)
- THREAD_N: 8 (cols of C per thread)
- Threads per block: 256
```

---

### Requirement: FlashAttention Tile Structure (DATA-006)

**User Story:** As a kernel developer, I want defined attention tiling, so that I can fit in shared memory.

#### Scenario: Attention Tiling
- **WHEN** implementing FlashAttention
- **THEN** the following tile dimensions SHALL be used:

```
Tile Dimensions:
- BLOCK_M: 64 (query rows per block)
- BLOCK_N: 64 (key/value rows per block)
- HEAD_DIM: 64 (head dimension)

Shared Memory Usage:
- Q_tile: 64 × 64 × sizeof(T) = 8KB
- K_tile: 64 × 64 × sizeof(T) = 8KB
- V_tile: 64 × 64 × sizeof(T) = 8KB
- O_accum: 64 × 64 × 4 = 16KB
- S_tile: 64 × 64 × 4 = 16KB
- Total: ~56KB
```

---

### Requirement: Convolution Data Layout (DATA-007)

**User Story:** As a kernel developer, I want defined convolution layouts, so that I can implement correct convolutions.

#### Scenario: NCHW Layout
- **WHEN** storing input/output tensors
- **THEN** tensors SHALL use NCHW layout:

```
Dimension order: [Batch, Channels, Height, Width]
Memory offset: n * C*H*W + c * H*W + h * W + w
Strides: [C*H*W, H*W, W, 1]
```

#### Scenario: OIHW Layout (Weights)
- **WHEN** storing convolution weights
- **THEN** weights SHALL use OIHW layout:

```
Dimension order: [OutChannels, InChannels, H, W]
For groups > 1: [OutChannels, InChannels/groups, H, W]
```

---

### Requirement: Memory Alignment (DATA-008)

**User Story:** As a performance engineer, I want defined memory alignment, so that I can optimize memory access.

#### Scenario: Data Type Alignment
- **WHEN** allocating memory for data types
- **THEN** the following alignments SHALL be used:

| Data Type | Alignment (bytes) |
|-----------|-------------------|
| float | 4 |
| half | 2 |
| int8_t | 1 |
| __nv_bfloat16 | 2 |
| __nv_fp8_e4m3 | 1 |

#### Scenario: Memory Access Alignment
- **WHEN** optimizing memory access
- **THEN** the following alignments SHALL be used:

| Memory Type | Alignment (bytes) | Reason |
|-------------|-------------------|--------|
| Shared memory | 128 | Bank conflicts |
| Global memory | 256 | Coalescing |

---

## See Also

- [Core Specifications](../core/spec.md) — Product requirements
- [API Specifications](../api/spec.md) — API contracts
- [Architecture](../architecture/spec.md) — Design decisions
