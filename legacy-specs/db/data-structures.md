# Data Structure Specification: TensorCraft-HPC

> **Version**: 2.0.0
> **Last Updated**: 2026-04-17

---

## Overview

This document defines the data structures and memory layouts used in TensorCraft-HPC. All implementations must conform to these specifications.

---

## 1. Tensor Data Structure

### 1.1 Tensor Layout

```
Tensor<T> Memory Layout:
┌─────────────────────────────────────────────────────────────┐
│                    GPU Global Memory                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Element[0]  Element[1]  ...  Element[N-1]              ││
│  │  (sizeof(T)) (sizeof(T))       (sizeof(T))              ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  Shape: [D0, D1, D2, ..., Dk]                               │
│  Total Elements: D0 * D1 * D2 * ... * Dk                    │
│  Strides: [D1*D2*...*Dk, D2*...*Dk, ..., Dk, 1] (row-major) │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Tensor Shape Representation

```cpp
// Shape stored as std::vector<size_t>
// Row-major order (C-style)
// Example: shape = {batch, height, width, channels}
//          strides = {height*width*channels, width*channels, channels, 1}
```

### 1.3 Supported Data Types

| Type | C++ Type | Size (bytes) | Range |
|------|----------|--------------|-------|
| FP32 | `float` | 4 | IEEE 754 single |
| FP16 | `half` | 2 | IEEE 754 half |
| BF16 | `__nv_bfloat16` | 2 | Brain float |
| FP8_E4M3 | `__nv_fp8_e4m3` | 1 | E4M3 format |
| FP8_E5M2 | `__nv_fp8_e5m2` | 1 | E5M2 format |
| INT8 | `int8_t` | 1 | [-128, 127] |
| INT32 | `int32_t` | 4 | [-2^31, 2^31-1] |
| INT64 | `int64_t` | 8 | [-2^63, 2^63-1] |

---

## 2. Memory Pool Structure

### 2.1 Block Structure

```
MemoryPool Block:
┌────────────────────────────────────────────────────────────┐
│  Block Header (Host)                                        │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │ void* ptr    │ size_t size  │ bool in_use  │            │
│  │ (8 bytes)    │ (8 bytes)    │ (1 byte)     │            │
│  └──────────────┴──────────────┴──────────────┘            │
│                                                             │
│  Tracking Maps (Host):                                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ std::unordered_map<void*, size_t> allocated_sizes_   │  │
│  │ std::unordered_map<void*, size_t> freed_sizes_       │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 2.2 Allocation Strategy

1. **First-fit with coalescing**: Find first free block that fits
2. **Block splitting**: Split large blocks if needed
3. **Block merging**: Merge adjacent free blocks on deallocation

### 2.3 Thread Safety

- Uses mutex for all operations
- Safe for concurrent allocate/deallocate

---

## 3. Matrix Formats

### 3.1 Dense Matrix (Row-Major)

```
Dense Matrix MxN:
┌─────────────────────────────────────────────────────┐
│  Row 0: [A(0,0), A(0,1), ..., A(0,N-1)]            │
│  Row 1: [A(1,0), A(1,1), ..., A(1,N-1)]            │
│  ...                                                │
│  Row M-1: [A(M-1,0), A(M-1,1), ..., A(M-1,N-1)]    │
└─────────────────────────────────────────────────────┘

Memory: [A(0,0), A(0,1), ..., A(0,N-1), A(1,0), ..., A(M-1,N-1)]
Stride: N elements per row
```

### 3.2 CSR Sparse Matrix

```
CSR Format (Compressed Sparse Row):

values[]:     [v0, v1, v2, ..., v(nnz-1)]     // Non-zero values
col_indices[]: [c0, c1, c2, ..., c(nnz-1)]    // Column indices
row_ptrs[]:   [0, r1, r2, ..., rM, nnz]      // Row pointers (M+1 elements)

Example:
Matrix:       [1, 0, 2]
              [0, 3, 0]
              [4, 5, 6]

values:      [1, 2, 3, 4, 5, 6]
col_indices: [0, 2, 1, 0, 1, 2]
row_ptrs:    [0, 2, 3, 6]
```

### 3.3 CSC Sparse Matrix

```
CSC Format (Compressed Sparse Column):

values[]:     [v0, v1, v2, ..., v(nnz-1)]     // Non-zero values
row_indices[]: [r0, r1, r2, ..., r(nnz-1)]    // Row indices
col_ptrs[]:   [0, c1, c2, ..., cN, nnz]      // Column pointers (N+1 elements)
```

---

## 4. GEMM Tiling Structure

### 4.1 Block Tiling

```
GEMM Tiling (Shared Memory):

Block Tile Dimensions:
- BLOCK_M: 128 (rows of C per block)
- BLOCK_N: 128 (cols of C per block)
- BLOCK_K: 32  (K dimension per iteration)

Shared Memory Usage:
- A_tile: BLOCK_M × BLOCK_K × sizeof(T) = 128 × 32 × 4 = 16KB
- B_tile: BLOCK_K × BLOCK_N × sizeof(T) = 32 × 128 × 4 = 16KB
- Total: 32KB per block
```

### 4.2 Thread Tiling

```
Thread Tile Dimensions:
- THREAD_M: 8 (rows of C per thread)
- THREAD_N: 8 (cols of C per thread)
- Threads per block: (BLOCK_M/THREAD_M) × (BLOCK_N/THREAD_N) = 16 × 16 = 256

Register Usage per Thread:
- C_accumulator: THREAD_M × THREAD_N × sizeof(float) = 8 × 8 × 4 = 256 bytes
- A_fragment: THREAD_M × sizeof(T) = 8 × 4 = 32 bytes
- B_fragment: THREAD_N × sizeof(T) = 8 × 4 = 32 bytes
```

---

## 5. Attention Data Structures

### 5.1 FlashAttention Tile Structure

```
FlashAttention Tiling:

Tile Dimensions:
- BLOCK_M: 64 (query rows per block)
- BLOCK_N: 64 (key/value rows per block)
- HEAD_DIM: 64 (head dimension, fixed)

Shared Memory Usage:
- Q_tile: BLOCK_M × HEAD_DIM × sizeof(T) = 64 × 64 × 2 = 8KB
- K_tile: BLOCK_N × HEAD_DIM × sizeof(T) = 64 × 64 × 2 = 8KB
- V_tile: BLOCK_N × HEAD_DIM × sizeof(T) = 64 × 64 × 2 = 8KB
- O_accum: BLOCK_M × HEAD_DIM × sizeof(float) = 64 × 64 × 4 = 16KB
- S_tile (attention scores): BLOCK_M × BLOCK_N × sizeof(float) = 64 × 64 × 4 = 16KB

Total Shared Memory: ~56KB
```

### 5.2 RoPE Cache Structure

```
RoPE Cache:
┌──────────────────────────────────────────────────────┐
│  cos_cache[max_seq_len][head_dim/2]                  │
│  sin_cache[max_seq_len][head_dim/2]                  │
│                                                       │
│  Precomputed for positions 0 to max_seq_len-1        │
│  cos(pos, dim) = cos(pos / 10000^(2*dim/head_dim))   │
│  sin(pos, dim) = sin(pos / 10000^(2*dim/head_dim))   │
└──────────────────────────────────────────────────────┘
```

---

## 6. Quantization Parameters

### 6.1 INT8 Quantization

```cpp
struct QuantParams {
    float scale;        // Scaling factor
    int zero_point;     // Zero point offset
};

// Quantization formula:
// q = round(x / scale) + zero_point
// q = clamp(q, -128, 127)

// Dequantization formula:
// x = (q - zero_point) * scale
```

### 6.2 FP8 Quantization

```cpp
// FP8 E4M3 format:
// - 1 sign bit
// - 4 exponent bits
// - 3 mantissa bits
// - Range: [0, 448], no infinities

// FP8 E5M2 format:
// - 1 sign bit
// - 5 exponent bits
// - 2 mantissa bits
// - Range: same as FP16, lower precision
```

---

## 7. Convolution Data Layout

### 7.1 NCHW Layout (Input/Output)

```
NCHW Layout:
┌─────────────────────────────────────────────────────┐
│  Dimension order: [Batch, Channels, Height, Width] │
│                                                      │
│  Memory offset for (n, c, h, w):                    │
│    offset = n * C*H*W + c * H*W + h * W + w         │
│                                                      │
│  Strides: [C*H*W, H*W, W, 1]                        │
└─────────────────────────────────────────────────────┘
```

### 7.2 Kernel Layout (OIHW)

```
OIHW Layout (Convolution Weights):
┌─────────────────────────────────────────────────────┐
│  Dimension order: [OutChannels, InChannels, H, W]  │
│                                                      │
│  For groups > 1:                                     │
│    Shape: [OutChannels, InChannels/groups, H, W]   │
└─────────────────────────────────────────────────────┘
```

### 7.3 Im2Col Buffer

```
Im2Col Buffer Layout:
┌─────────────────────────────────────────────────────┐
│  Input: [N, C, H, W]                                │
│  Kernel: [K, C, R, S]                               │
│                                                      │
│  Im2Col Output: [N*H_out*W_out, C*R*S]             │
│                                                      │
│  Where:                                              │
│    H_out = (H + 2*pad - R) / stride + 1            │
│    W_out = (W + 2*pad - S) / stride + 1            │
└─────────────────────────────────────────────────────┘
```

---

## 8. MoE Router Data Structures

### 8.1 Expert Routing Output

```
MoE Router Output:
┌─────────────────────────────────────────────────────┐
│  expert_indices[batch_size][top_k]                  │
│    - Expert ID for each selected expert             │
│    - Range: [0, num_experts)                        │
│                                                      │
│  expert_weights[batch_size][top_k]                  │
│    - Softmax weight for each selected expert        │
│    - Range: [0, 1], sum(top_k weights) ≈ 1         │
└─────────────────────────────────────────────────────┘
```

---

## Memory Alignment Requirements

| Data Type | Alignment (bytes) |
|-----------|-------------------|
| float | 4 |
| half | 2 |
| int8_t | 1 |
| __nv_bfloat16 | 2 |
| __nv_fp8_e4m3 | 1 |
| Shared memory | 128 (bank conflicts) |
| Global memory | 256 (coalescing) |

---

## See Also

- [RFC 0001: Core Architecture](../rfc/0001-core-architecture.md)
- [API Specification](../api/cxx-api.md)
