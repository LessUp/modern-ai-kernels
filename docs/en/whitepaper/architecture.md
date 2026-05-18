# Architecture

This document describes the system architecture, module design, and extension points of TensorCraft-HPC.

---

## Design Philosophy

TensorCraft-HPC follows three core principles:

1. **Readability first** — Code is written to be read; each kernel demonstrates the optimization progression.
2. **Header-only** — Zero build complexity for C++ users; include and go.
3. **OpenSpec-driven** — The specifications in `openspec/specs/` are the authoritative source for implementation.

---

## System Architecture

```mermaid
flowchart TB
    subgraph UserAPI["User API Layer"]
        direction TB
        CPP["C++ Headers<br/>(Header-Only)"]
        PY["Python Bindings<br/>(tensorcraft_ops)"]
    end

    subgraph KernelLayer["Kernel Layer"]
        direction TB
        GEMM["GEMM Kernels<br/>(Naive to Tensor Core)"]
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
    end

    subgraph CoreLayer["Core Utilities"]
        direction TB
        CUDA_CHECK["cuda_check.hpp<br/>(Error Handling)"]
        FEATURES["features.hpp<br/>(Compile-Time Detection)"]
    end

    subgraph Hardware["Hardware Abstraction"]
        SM70["SM70<br/>(Volta)"]
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

## Directory Structure

```
modern-ai-kernels/
├── include/tensorcraft/       # Header-only library
│   ├── core/                  # Utilities
│   │   ├── cuda_check.hpp     # CUDA error checking
│   │   ├── features.hpp       # Compile-time GPU detection
│   │   └── type_traits.hpp    # Type utilities
│   ├── memory/                # Memory management
│   │   ├── tensor.hpp         # RAII GPU tensor
│   │   └── memory_pool.hpp    # Optional pooling
│   └── kernels/               # All compute kernels
│       ├── gemm.hpp           # Matrix multiplication
│       ├── attention.hpp      # Attention mechanisms
│       ├── normalization.hpp  # LayerNorm, RMSNorm
│       ├── softmax.hpp        # Softmax variants
│       ├── conv2d.hpp         # 2D convolution
│       ├── sparse.hpp         # Sparse operations
│       └── fusion.hpp         # Fused operators and quantization helpers
├── src/python_ops/            # Python bindings (pybind11)
├── tests/                     # Unit tests (GoogleTest)
├── benchmarks/                # Performance benchmarks
├── docs/                      # VitePress documentation
└── openspec/                  # Specification workflow
    ├── specs/                 # Accepted specifications
    ├── changes/               # Active change proposals
    └── archive/               # Completed changes
```

---

## GEMM Optimization Path

The GEMM kernel demonstrates the progressive optimization approach:

```mermaid
flowchart LR
    A["Naive<br/>(O(N^3) Global Memory)"]
    B["Tiled<br/>(Shared Memory)"]
    C["Double Buffer<br/>(Overlap Copy/Compute)"]
    D["Tensor Core<br/>(WMMA)"]
    E["cuBLAS Parity<br/>(92%)"]

    A -->|"Tile for<br/>reuse"| B
    B -->|"Double buffer<br/>for overlap"| C
    C -->|"Use Tensor Cores<br/>(WMMA)"| D
    D -->|"Fine-tune<br/>parameters"| E
```

### Performance Characteristics

| Stage | Memory Traffic | Compute Efficiency | Relative Speed |
|-------|----------------|--------------------|----------------|
| Naive | O(N³) global | ~1% | 1x |
| Tiled | O(N²) global | ~10% | 10x |
| Double Buffer | O(N²) global | ~30% | 30x |
| Tensor Core | O(N²) global | ~80% | 80x |

---

## FlashAttention Implementation

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

    Note over Host,DRAM: Memory: O(N) instead of O(N^2)
```

### Key Innovations

1. **Tiled computation** — Process attention blocks that fit in SRAM.
2. **Online softmax** — Incrementally update softmax statistics.
3. **Recomputation** — Recompute attention weights rather than storing them.

---

## Memory Management

### RAII Pattern

```cpp
// Automatic memory management
{
    tensorcraft::FloatTensor A({4096, 4096});
    // use A...
} // Released automatically when scope exits
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

## Compile-Time Feature Detection

`features.hpp` provides compile-time GPU capability detection:

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

## OpenSpec Workflow

```mermaid
flowchart TB
    IDEA["New Idea"] --> PROPOSAL["Create Proposal<br/>openspec/changes/"]
    PROPOSAL --> REVIEW["Review and Discuss"]
    REVIEW -->|"Accept"| SPEC["Move to<br/>openspec/specs/"]
    REVIEW -->|"Reject"| ARCHIVE["Archive with<br/>rationale"]
    SPEC --> IMPL["Implement"]
    IMPL --> VERIFY["Verify against Spec"]
    VERIFY --> DONE["Complete"]
```

### Spec Structure

Each specification in `openspec/specs/` contains:

- **Requirements** — What the component must do.
- **Contracts** — API guarantees and invariants.
- **Acceptance Criteria** — How to verify compliance.

---

## Extension Points

### Adding a New Kernel

1. Create a spec proposal in `openspec/changes/`.
2. After review, move to `openspec/specs/`.
3. Implement the header in `include/tensorcraft/kernels/`.
4. Add GoogleTest unit tests.
5. Add performance benchmarks.
6. Update documentation.

### Adding Python Bindings

```cpp
// src/python_ops/bindings.cpp
m.def("my_kernel", &tensorcraft::kernels::my_kernel,
    "A new kernel",
    py::arg("input"),
    py::arg("output"));
```
