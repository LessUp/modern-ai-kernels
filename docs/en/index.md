---
layout: home

hero:
  name: TensorCraft-HPC
  text: Demystifying High-Performance AI Kernels
  tagline: A header-only C++/CUDA library for learning modern AI operators — progressive optimization paths, readable code, OpenSpec-driven development.
  actions:
    - theme: brand
      text: Get Started
      link: /en/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/LessUp/modern-ai-kernels
    - theme: alt
      text: Papers & Citations
      link: /en/references/papers
---

<script setup>
import ArchitectureImg from '/images/diagrams/architecture.svg'
import GEMMPathImg from '/images/diagrams/gemm-optimization-path.svg'
import BenchmarksImg from '/images/diagrams/performance-benchmarks.svg'
</script>

<div class="home-hero-badges">
  <span class="badge cuda">CUDA 11.0+</span>
  <span class="badge arch">SM70-SM100</span>
  <span class="badge header">Header-Only</span>
  <span class="badge openspec">OpenSpec</span>
</div>

## Why TensorCraft-HPC?

<div class="feature-grid">

<div class="feature-card">
  <h3>🎓 Educational Design</h3>
  <p>Each kernel evolves from <strong>naive to optimized</strong>, making the learning process explicit and accessible. No magic, just clear code.</p>
</div>

<div class="feature-card">
  <h3>🚀 Progressive Optimization</h3>
  <p>GEMM implementation demonstrates 4 optimization stages: Naive → Tiled → Double Buffer → Tensor Core, achieving <strong>92% cuBLAS</strong>.</p>
</div>

<div class="feature-card">
  <h3>⚡ Zero-Build Integration</h3>
  <p>Header-only architecture — just <code>#include "tensorcraft/"</code> in your project. Optional Python bindings via <code>pip install</code>.</p>
</div>

<div class="feature-card">
  <h3>📊 Multi-Architecture Support</h3>
  <p>Compile-time feature detection for <strong>Volta (SM70)</strong> through <strong>Blackwell (SM100)</strong>, with Tensor Core, FP8, and BF16 support.</p>
</div>

</div>

## Architecture

<div class="diagram-container">
  <img :src="ArchitectureImg" alt="TensorCraft-HPC Architecture" />
</div>

## GEMM Optimization Path

<div class="diagram-container">
  <img :src="GEMMPathImg" alt="GEMM Optimization Path" />
</div>

## Performance Benchmarks

<div class="diagram-container">
  <img :src="BenchmarksImg" alt="Performance Benchmarks" />
</div>

<div class="benchmark-note">
  <span class="note-icon">📊</span>
  <span>Benchmarks measured on A100 80GB, CUDA 12.4, FP16 Tensor Core enabled. Relative performance vs NVIDIA libraries.</span>
</div>

## Quick Start

::: code-group
```bash [Install]
# Clone the repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Header-only: just include the headers
# For CMake projects:
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke
```

```cpp [C++ Usage]
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

// Create GPU tensors (RAII-managed)
tensorcraft::FloatTensor A({4096, 4096});
tensorcraft::FloatTensor B({4096, 4096});
tensorcraft::FloatTensor C({4096, 4096});

// Optimized GEMM
tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 4096, 4096, 4096);
```

```python [Python Usage]
import tensorcraft_ops as tc
import numpy as np

# Use NumPy-compatible API
A = np.random.randn(4096, 4096).astype(np.float32)
B = np.random.randn(4096, 4096).astype(np.float32)
C = tc.gemm(A, B)  # GPU-accelerated

# FlashAttention
Q, K, V = [np.random.randn(32, 128, 64).astype(np.float32) for _ in range(3)]
output = tc.flash_attention(Q, K, V)
```
:::

## Project Status

| Aspect | Status |
|--------|--------|
| Repository Mode | Stabilization / Closeout |
| Core Kernels | Complete (GEMM, Attention, Norm, Conv) |
| Documentation | OpenSpec-driven, bilingual |
| CUDA Support | 11.0 - 13.1 |
| Architecture Support | SM70 - SM100 (Volta → Blackwell) |

## Citation

If you use TensorCraft-HPC in your research or learning materials, please cite:

```bibtex
@software{tensorcraft-hpc,
  title = {TensorCraft-HPC: Demystifying High-Performance AI Kernels},
  author = {LessUp},
  year = {2024},
  url = {https://github.com/LessUp/modern-ai-kernels}
}
```

<style>
.home-hero-badges {
  display: flex;
  justify-content: center;
  gap: 12px;
  margin-top: 16px;
  flex-wrap: wrap;
}

.badge {
  font-size: 12px;
  padding: 4px 12px;
  border-radius: 6px;
  font-weight: 500;
  font-family: var(--vp-font-family-mono);
}

.badge.cuda {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border: 1px solid var(--vp-c-brand-1);
}

.badge.arch {
  background: rgba(0, 212, 255, 0.1);
  color: #00D4FF;
  border: 1px solid #00D4FF;
}

.badge.header {
  background: rgba(255, 197, 23, 0.1);
  color: #FFB800;
  border: 1px solid #FFB800;
}

.badge.openspec {
  background: rgba(142, 208, 0, 0.1);
  color: #8ED000;
  border: 1px solid #8ED000;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin: 24px 0;
}

.feature-card {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 20px;
  transition: all 0.2s ease;
}

.feature-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
}

.feature-card h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.feature-card p {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}

.feature-card code {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
}

.diagram-container {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 16px;
  margin: 24px 0;
  text-align: center;
}

.diagram-container img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
}

.benchmark-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 16px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  font-size: 12px;
  color: var(--vp-c-text-3);
  margin-top: 16px;
}

.note-icon {
  font-size: 16px;
}

@media (max-width: 768px) {
  .feature-grid {
    grid-template-columns: 1fr;
  }

  .home-hero-badges {
    gap: 8px;
  }

  .badge {
    font-size: 11px;
    padding: 3px 8px;
  }
}
</style>