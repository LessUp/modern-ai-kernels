# Technical Whitepaper

::: abstract
**Abstract**

TensorCraft-HPC is a header-only C++/CUDA library designed for learning high-performance AI kernel implementation. This whitepaper presents the architectural decisions, optimization strategies, and performance analysis that guide the project. Our goal is to demystify GPU kernel development by providing clear, progressive optimization paths from naive implementations to production-grade performance.

**Key Results**

- 92% cuBLAS performance on FP16 GEMM with Tensor Core
- 85% cuDNN performance on FlashAttention
- Support for NVIDIA SM70-SM100 architectures
- Zero build complexity via header-only design
:::

---

## Executive Summary

Modern AI systems depend critically on high-performance GPU kernels for operations like matrix multiplication, attention, and normalization. However, the path from understanding the math to achieving production-grade performance is often obscured by complexity.

TensorCraft-HPC addresses this gap by:

1. **Explicit Progression**: Each kernel evolves through well-defined optimization stages
2. **Educational Clarity**: Code is optimized for readability, not just performance
3. **OpenSpec Governance**: Specifications drive implementation, ensuring correctness

---

## Project Philosophy

### Why This Project Exists

The CUDA ecosystem has excellent production libraries (cuBLAS, cuDNN, CUTLASS), but they are optimized for deployment, not learning. When a developer asks "How do I write an efficient GEMM kernel?", the answer often points to thousands of lines of template metaprogramming.

TensorCraft-HPC provides an alternative: kernels that start simple and evolve, with each optimization step justified and explained.

### Design Principles

| Principle | Implication |
|-----------|-------------|
| **Readability First** | Code comments explain *why*, not just *what* |
| **Progressive Complexity** | Each stage is a complete, working kernel |
| **Specification-Driven** | OpenSpec files define contracts before implementation |
| **Zero Build Friction** | Header-only for C++, optional pip for Python |

---

## Core Contributions

### 1. Progressive Optimization Framework

Every kernel follows a documented optimization path:

```
Naive → Tiled → Double Buffer → Tensor Core → Production Parity
```

Each stage:
- Is a complete, testable implementation
- Has clear performance characteristics
- Demonstrates specific optimization techniques

### 2. Multi-Architecture Support

Compile-time feature detection enables:

```cpp
#if TENSORCRAFT_HAS_WMMA
    // Tensor Core path (SM70+)
#elif TENSORCRAFT_HAS_FP8
    // FP8 path (SM90+)
#else
    // Fallback path
#endif
```

### 3. OpenSpec Workflow

Specifications in `openspec/specs/` define:

- **Requirements**: What the component must do
- **Contracts**: API guarantees and invariants
- **Acceptance Criteria**: How to verify compliance

---

## What this project is not

TensorCraft-HPC is **not** trying to replace cuBLAS, cuDNN, CUTLASS, or Triton as a full production
kernel stack. It is also not trying to maximize feature count at the expense of coherence.

Instead, the project optimizes for a rarer combination:

- code that is still readable after optimization begins
- benchmark claims that stay attached to methodology and caveats
- architecture that can be explained in an interview or design review
- documentation that helps readers learn why the implementation looks the way it does

## How to evaluate this repository

| Lens | What to look for |
|------|------------------|
| **Architecture** | Clear boundaries between kernel layer, memory abstractions, feature detection, and public surfaces |
| **Implementation quality** | Progressive optimization steps instead of opaque "final form" kernels |
| **Evidence discipline** | Benchmark numbers paired with methodology, references, and honest limits |
| **Project coherence** | README, GitHub Pages, OpenSpec, and workflows telling the same story |

---

## Evolution notes

TensorCraft-HPC is intentionally biased toward explainability over maximal breadth. The project evolves by making the optimization path more legible, the benchmark claims more defensible, and the architectural boundaries easier to discuss in design review. That means the site should preserve evidence, caveats, and implementation trade-offs even when the project grows.

## Related open-source projects

The project sits in conversation with several important surfaces:

- **CUTLASS** for highly optimized, template-heavy CUDA kernel construction
- **Triton** for compiler-led kernel authoring and research ergonomics
- **FlashAttention** for attention-specific algorithmic and systems co-design
- **cuBLAS / cuDNN** for production baselines and performance reference points

These projects matter here not as competitors in a marketing sense, but as reference frames for understanding why TensorCraft-HPC emphasizes pedagogy, architecture visibility, and benchmark honesty.

---

## Target Audience

This whitepaper is intended for:

- **GPU Kernel Developers** seeking to understand optimization techniques
- **ML Infrastructure Engineers** evaluating kernel implementations
- **Researchers** studying high-performance computing patterns
- **Students** learning CUDA programming

---

## Document Structure

| Section | Content |
|---------|---------|
| [Architecture](/en/whitepaper/architecture) | System design, layering, and extension points |
| [Performance](/en/whitepaper/performance) | Benchmarking methodology and analysis |
| [Methodology](/en/whitepaper/methodology) | OpenSpec workflow and contribution guidelines |
| [Papers & Citations](/en/references/papers) | Academic and ecosystem references behind the design |

---

## Quick Start

::: code-group
```bash [Clone]
git clone https://github.com/AICL-Lab/modern-ai-kernels.git
cd modern-ai-kernels
```

```cpp [C++]
#include "tensorcraft/kernels/gemm.hpp"

tensorcraft::FloatTensor A({4096, 4096});
tensorcraft::FloatTensor B({4096, 4096});
tensorcraft::FloatTensor C({4096, 4096});

tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 4096, 4096, 4096);
```

```python [Python]
import tensorcraft_ops as tc
import numpy as np

A = np.random.randn(4096, 4096).astype(np.float16)
B = np.random.randn(4096, 4096).astype(np.float16)
C = tc.gemm(A, B)  # GPU-accelerated
```
:::

---

## Citation

If you reference TensorCraft-HPC in academic work:

```bibtex
@software{tensorcraft-hpc,
  title = {TensorCraft-HPC: Demystifying High-Performance AI Kernels
           with Modern C++ and CUDA},
  author = {TensorCraft-HPC Contributors},
  year = {2024},
  url = {https://github.com/AICL-Lab/modern-ai-kernels}
}
```
