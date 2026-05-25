# Architecture Overview

This document describes the high-level architecture of TensorCraft-HPC.

## Design Philosophy {#philosophy}

TensorCraft-HPC follows three core principles:

1. **Readability First** — Code is meant to be read. Each kernel shows the optimization progression.
2. **Header-Only** — Zero build complexity for C++ users. Just include and go.
3. **OpenSpec-Driven** — Active work starts in `openspec/changes/`, while accepted baselines live in `openspec/specs/`.

---

## System Architecture {#system}

```mermaid
flowchart TB
    subgraph UserAPI["User API Layer"]
        direction TB
        CPP["C++ Headers<br/>(Header-Only)"]
        PY["Python Bindings<br/>(tensorcraft_ops)"]
    end

    subgraph KernelLayer["Kernel Layer"]
        direction TB
        GEMM["GEMM Kernels<br/>(Naive → Tensor Core)"]
        ATTN["Attention Kernels<br/>(FlashAttention)"]
        NORM["Normalization<br/>(LayerNorm, RMSNorm)"]
        CONV["Convolution<br/>(Im2Col, Winograd)"]
        SPARSE["Sparse Ops<br/>(CSR, CSC)"]
        QUANT["Quantization<br/>(INT8, FP8)"]
    end

    subgraph MemoryLayer["Memory Layer"]
        direction TB
        TENSOR["FloatTensor<br/>(RAII GPU Memory)"]
        POOL["MemoryPool<br/>(Optional Pooling)"]
        ALIGNED["AlignedVector<br/>(Cache-Friendly)"]
    end

    subgraph CoreLayer["Core Utilities"]
        direction TB
        CUDA_CHECK["cuda_check.hpp<br/>(Error Handling)"]
        FEATURES["features.hpp<br/>(Compile-Time Detection)"]
        TYPE_TRAITS["type_traits.hpp<br/>(Type Utilities)"]
        WARP_UTILS["warp_utils.hpp<br/>(Warp Primitives)"]
    end

    subgraph Hardware["Hardware Abstraction"]
        SM70["SM70<br/>(Volta)"]
        SM75["SM75<br/>(Turing)"]
        SM80["SM80<br/>(Ampere)"]
        SM90["SM90<br/>(Hopper)"]
        SM100["SM100<br/>(Blackwell)"]
    end

    CPP --> KernelLayer
    PY --> KernelLayer
    KernelLayer --> MemoryLayer
    MemoryLayer --> CoreLayer
    CoreLayer --> Hardware
```

---

## Directory Structure {#directories}

```
modern-ai-kernels/
├── include/tensorcraft/       # Header-only library
│   ├── core/                  # Utilities (error handling, type traits)
│   │   ├── cuda_check.hpp     # CUDA error checking macros
│   │   ├── features.hpp       # Compile-time GPU feature detection
│   │   ├── type_traits.hpp    # Type manipulation utilities
│   │   └── warp_utils.hpp     # Warp-level primitives
│   ├── memory/                # Memory management
│   │   ├── tensor.hpp         # RAII GPU tensor wrapper
│   │   ├── memory_pool.hpp    # Optional memory pooling
│   │   └── aligned_vector.hpp # Cache-aligned vectors
│   └── kernels/               # All compute kernels
│       ├── gemm.hpp           # Matrix multiplication
│       ├── attention.hpp      # Attention mechanisms
│       ├── normalization.hpp  # LayerNorm, RMSNorm, etc.
│       ├── softmax.hpp        # Softmax variants
│       ├── conv2d.hpp         # 2D convolution
│       ├── sparse.hpp         # Sparse operations
│       ├── elementwise.hpp    # ReLU, GeLU, etc.
│       ├── memory_ops.hpp     # Copy, transpose
│       └── fusion.hpp         # Fused operators and quantization helpers
├── src/python_ops/            # Python bindings (pybind11)
├── tests/                     # Unit tests (GoogleTest)
├── benchmarks/                # Performance benchmarks
├── examples/                  # Usage examples
├── docs/                      # VitePress documentation
└── openspec/                  # Specification workflow
    ├── specs/                 # Accepted specifications
    ├── changes/               # Active change proposals
    └── archive/               # Completed changes
```

---

## GEMM Optimization Path {#gemm-path}

The GEMM kernel demonstrates the progressive optimization approach:

```mermaid
flowchart LR
    A["Naive<br/>(O(N³) Global Memory)"]
    B["Tiled<br/>(Shared Memory)"]
    C["Double Buffer<br/>(Overlap Copy/Compute)"]
    D["Tensor Core<br/>(WMMA)"]
    E["cuBLAS Parity<br/>(85-95%)"]

    A -->|"Tile for<br/>reuse"| B
    B -->|"Double buffer<br/>for overlap"| C
    C -->|"Use Tensor Cores<br/>(WMMA)"| D
    D -->|"Fine-tune<br/>parameters"| E

```

### Performance Characteristics

| Stage | Memory Traffic | Compute Efficiency | Relative Speed |
|-------|----------------|-------------------|----------------|
| Naive | O(N³) global | ~1% | 1x |
| Tiled | O(N²) global | ~10% | 10x |
| Double Buffer | O(N²) global | ~30% | 30x |
| Tensor Core | O(N²) global | ~80% | 80x |

---

## FlashAttention Implementation {#flash-attention}

```mermaid
sequenceDiagram
    participant Host as Host
    participant SRAM as Shared Memory
    participant Reg as Registers
    participant DRAM as Global Memory

    Note over Host,DRAM: FlashAttention Tiling

    Host->>DRAM: Allocate Q, K, V, O
    loop For each tile
        DRAM->>SRAM: Load Q_tile, K_tile, V_tile
        SRAM->>Reg: Load to registers
        Reg->>Reg: Compute QK^T (partial)
        Reg->>Reg: Update softmax state
        Reg->>Reg: Accumulate output
        Reg->>SRAM: Store partial O
    end
    SRAM->>DRAM: Write final O

    Note over Host,DRAM: Memory: O(N²) instead of O(N²d)
```

### Key Innovations

1. **Tiling** — Process attention in tiles that fit in SRAM
2. **Online Softmax** — Update softmax statistics incrementally
3. **Recomputation** — Recompute attention weights instead of storing

---

## Memory Management {#memory}

### RAII Pattern

```cpp
// Automatic memory management
{
    tensorcraft::FloatTensor A({4096, 4096});
    // Use A...
} // Automatically freed when scope exits
```

### Memory Pool (Optional)

```mermaid
flowchart LR
    REQ["Kernel Request"] --> POOL{"Pool Available?"}
    POOL -->|"Yes"| ALLOC["Return from Pool"]
    POOL -->|"No"| NEW["cudaMalloc"]
    NEW --> POOL
    ALLOC --> KERNEL["Kernel Execution"]
    KERNEL --> RET["Return to Pool"]
    RET --> POOL
```

---

## Compile-Time Feature Detection {#features}

The `features.hpp` header provides compile-time GPU capability detection:

```cpp
// Automatically detected at compile time
#if TENSORCRAFT_HAS_WMMA
    // Use Tensor Cores (SM70+)
#endif

#if TENSORCRAFT_HAS_FP8
    // Use FP8 types (SM90+)
#endif

#if TENSORCRAFT_HAS_TMA
    // Use Tensor Memory Accelerator (SM90+)
#endif
```

---

## OpenSpec Workflow {#openspec}

```mermaid
flowchart TB
    IDEA["New Idea"] --> PROPOSAL["Create Proposal<br/>openspec/changes/"]
    PROPOSAL --> REVIEW["Review & Discuss"]
    REVIEW -->|"Start work"| IMPL["Implement from<br/>openspec/changes/"]
    REVIEW -->|"Reject"| ARCHIVE["Archive with<br/>rationale"]
    IMPL --> VERIFY["Verify against change"]
    VERIFY -->|"Accept"| SPEC["Promote baseline to<br/>openspec/specs/"]
    SPEC --> DONE["Complete"]
```

### Specification Structure

Each accepted baseline in `openspec/specs/` contains:
- **Requirements** — What the component must do
- **Contracts** — API guarantees and invariants
- **Acceptance Criteria** — How to verify compliance

---

## Testing Strategy {#testing}

| Level | Tool | Purpose |
|-------|------|---------|
| Unit | GoogleTest | Per-kernel correctness |
| Integration | pytest | Python bindings |
| Benchmark | Google Benchmark | Performance regression |
| Validation | Custom | Numerical accuracy |

### Running Tests

```bash
# All tests
ctest --preset dev --output-on-failure

# Specific kernel
ctest --preset dev -R gemm

# Benchmarks
cmake --preset release
cmake --build --preset release --parallel 2
./build/release/benchmarks/gemm_benchmark
```
