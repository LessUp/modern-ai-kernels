---
layout: home

hero:
  name: TensorCraft-HPC
  text: 解密高性能 AI 内核
  tagline: 一个仅头文件的 C++/CUDA 库，用于学习现代 AI 算子 — 渐进式优化路径、可读代码、OpenSpec 驱动开发。
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/getting-started
    - theme: alt
      text: GitHub 仓库
      link: https://github.com/LessUp/modern-ai-kernels
    - theme: alt
      text: 论文引用
      link: /zh/references/papers
---

<script setup>
import ArchitectureImg from '/images/diagrams/architecture.svg'
import GEMMPathImg from '/images/diagrams/gemm-optimization-path.svg'
import BenchmarksImg from '/images/diagrams/performance-benchmarks.svg'
</script>

<div class="home-hero-badges">
  <span class="badge cuda">CUDA 11.0+</span>
  <span class="badge arch">SM70-SM100</span>
  <span class="badge header">仅头文件</span>
  <span class="badge openspec">OpenSpec</span>
</div>

## 为什么选择 TensorCraft-HPC？

<div class="feature-grid">

<div class="feature-card">
  <h3>🎓 教育性设计</h3>
  <p>每个内核从<strong>朴素实现到优化版本</strong>渐进演进，让学习过程清晰可见。没有魔法，只有清晰的代码。</p>
</div>

<div class="feature-card">
  <h3>🚀 渐进式优化</h3>
  <p>GEMM 实现展示 4 个优化阶段：朴素 → 分块 → 双缓冲 → Tensor Core，达到 <strong>92% cuBLAS</strong> 性能。</p>
</div>

<div class="feature-card">
  <h3>⚡ 零构建集成</h3>
  <p>仅头文件架构 — 只需 <code>#include "tensorcraft/"</code> 即可使用。可选 Python 绑定通过 <code>pip install</code> 安装。</p>
</div>

<div class="feature-card">
  <h3>📊 多架构支持</h3>
  <p>编译时特性检测，支持 <strong>Volta (SM70)</strong> 到 <strong>Blackwell (SM100)</strong>，包含 Tensor Core、FP8 和 BF16 支持。</p>
</div>

</div>

## 架构设计

<div class="diagram-container">
  <img :src="ArchitectureImg" alt="TensorCraft-HPC 架构" />
</div>

## GEMM 优化路径

<div class="diagram-container">
  <img :src="GEMMPathImg" alt="GEMM 优化路径" />
</div>

## 性能基准

<div class="diagram-container">
  <img :src="BenchmarksImg" alt="性能基准" />
</div>

<div class="benchmark-note">
  <span class="note-icon">📊</span>
  <span>基准测试在 A100 80GB、CUDA 12.4、FP16 Tensor Core 环境下测量。相对性能对比 NVIDIA 库。</span>
</div>

## 快速开始

::: code-group
```bash [安装]
# 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 仅头文件：只需包含头文件
# 对于 CMake 项目：
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke
```

```cpp [C++ 使用]
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

// 创建 GPU 张量 (RAII 管理)
tensorcraft::FloatTensor A({4096, 4096});
tensorcraft::FloatTensor B({4096, 4096});
tensorcraft::FloatTensor C({4096, 4096});

// 优化的 GEMM
tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 4096, 4096, 4096);
```

```python [Python 使用]
import tensorcraft_ops as tc
import numpy as np

# 使用 NumPy 兼容 API
A = np.random.randn(4096, 4096).astype(np.float32)
B = np.random.randn(4096, 4096).astype(np.float32)
C = tc.gemm(A, B)  # GPU 加速

# FlashAttention
Q, K, V = [np.random.randn(32, 128, 64).astype(np.float32) for _ in range(3)]
output = tc.flash_attention(Q, K, V)
```
:::

## 项目状态

| 方面 | 状态 |
|------|------|
| 仓库模式 | 稳定化 / 收尾 |
| 核心内核 | 完成 (GEMM, Attention, Norm, Conv) |
| 文档 | OpenSpec 驱动，双语 |
| CUDA 支持 | 11.0 - 13.1 |
| 架构支持 | SM70 - SM100 (Volta → Blackwell) |

## 引用

如果您在研究或学习材料中使用 TensorCraft-HPC，请引用：

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