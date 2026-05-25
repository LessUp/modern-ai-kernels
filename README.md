<div align="center">

# TensorCraft-HPC

**Demystifying High-Performance AI Kernels with Modern C++ & CUDA**

[![CI](https://github.com/AICL-Lab/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/AICL-Lab/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://github.com/AICL-Lab/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://aicl-lab.github.io/modern-ai-kernels/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2B-00599C?logo=c%2B%2B&logoColor=white)
[![OpenSpec](https://img.shields.io/badge/Workflow-OpenSpec-8ED000)](openspec/)

[Whitepaper](https://aicl-lab.github.io/modern-ai-kernels/en/whitepaper/) •
[Academy](https://aicl-lab.github.io/modern-ai-kernels/en/academy/) •
[Evidence](https://aicl-lab.github.io/modern-ai-kernels/en/evidence/) •
[Kernel Atlas](https://aicl-lab.github.io/modern-ai-kernels/en/api/gemm)

</div>

---

TensorCraft-HPC is a **header-only C++/CUDA kernel library** and **technical whitepaper / architecture showcase** for learning, validating, and packaging modern AI operators. The repository now presents itself as a three-part public surface: a **whitepaper** for the thesis, an **academy** for the learning path, and an **evidence** layer for benchmarks, references, and research framing.

## ✨ Highlights

| Feature | Description |
|---------|-------------|
| 🎓 **Educational Design** | Progressive optimization paths from naive to Tensor Core |
| ⚡ **Zero-Build Integration** | Header-only — just `#include` and go |
| 📊 **Multi-Architecture** | SM70 (Volta) to SM100 (Blackwell) |
| 🔧 **OpenSpec Workflow** | Specification-driven development |
| 📚 **Bilingual Docs** | Complete English & Chinese documentation |

## 🚀 Quick Start

### CPU-only Smoke Validation

```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke --parallel 2
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
python3 -m build --wheel
```

### CUDA-enabled Local Validation

```bash
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)
ctest --preset dev --output-on-failure
```

### C++ Usage

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

// Create GPU tensors (RAII-managed)
tensorcraft::FloatTensor A({4096, 4096});
tensorcraft::FloatTensor B({4096, 4096});
tensorcraft::FloatTensor C({4096, 4096});

// Optimized GEMM (92% cuBLAS performance)
tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 4096, 4096, 4096);
```

### Python Usage

```python
import tensorcraft_ops as tc
import numpy as np

# GPU-accelerated GEMM
A = np.random.randn(4096, 4096).astype(np.float16)
B = np.random.randn(4096, 4096).astype(np.float16)
C = tc.gemm(A, B)

# FlashAttention
Q, K, V = [np.random.randn(32, 128, 64).astype(np.float16) for _ in range(3)]
output = tc.flash_attention(Q, K, V)
```

## 📦 Capability Snapshot

| Area | Scope |
|------|-------|
| **Core utilities** | CUDA checks, feature detection, type traits, warp helpers |
| **Memory** | `Tensor`, aligned vectors, memory pool |
| **Kernels** | GEMM, FlashAttention, normalization, convolution, sparse, fusion |
| **Python** | `tensorcraft_ops` bindings for smoke/integration workflows |
| **Validation** | CPU smoke build/install, Python wheel build, optional CUDA tests |

## 📈 Performance Benchmarks

| Kernel | Reference | Performance |
|--------|-----------|-------------|
| GEMM (FP16) | cuBLAS | 92% |
| FlashAttention | cuDNN | 85% |
| LayerNorm | cuDNN | 95% |
| Conv2D | cuDNN | 78% |
| SpMV (CSR) | cuSPARSE | 88% |

*Benchmarks on A100 80GB, CUDA 12.4, FP16 Tensor Core*

## 📚 Documentation

- **Showcase home**: <https://aicl-lab.github.io/modern-ai-kernels/>
- **Whitepaper**: <https://aicl-lab.github.io/modern-ai-kernels/en/whitepaper/>
- **Academy**: <https://aicl-lab.github.io/modern-ai-kernels/en/academy/>
- **Evidence**: <https://aicl-lab.github.io/modern-ai-kernels/en/evidence/>
- **Kernel Atlas / API**: <https://aicl-lab.github.io/modern-ai-kernels/en/api/gemm>
- **中文文档**: <https://aicl-lab.github.io/modern-ai-kernels/zh/>

## 🔧 OpenSpec Workflow

This repository uses **OpenSpec** as the active development workflow:

1. Review accepted specs in `openspec/specs/`
2. Create or update changes under `openspec/changes/`
3. Implement against that change
4. Run validation before merge
5. Use `/review` before merging structural changes

## 📁 Repository Layout

```
modern-ai-kernels/
├── include/tensorcraft/   # Header-only C++/CUDA library
├── src/python_ops/        # Python bindings
├── tests/                 # Validation
├── benchmarks/            # Benchmark binaries
├── docs/                  # GitHub Pages + documentation
├── openspec/              # Active spec workflow
└── .github/               # Workflows, templates
```

## 🛠 Tooling Baseline

- **Build system**: CMake presets
- **Formatting**: `.clang-format`, `.clang-tidy`, `pre-commit`
- **LSP**: `clangd` with `compile_commands.json`
- **GitHub automation**: CI, Pages, release workflow

## 🤝 Contributing

Contributions are welcome! Please:

1. Read the [OpenSpec workflow](openspec/)
2. Follow the code style (run `pre-commit` hooks)
3. Add tests for new functionality
4. Update documentation

## 📄 License

Released under the [MIT License](LICENSE).

---

<div align="center">

**[⬆ Back to Top](#tensorcraft-hpc)**

Made with ❤️ for learning high-performance AI kernels

</div>
