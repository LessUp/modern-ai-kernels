---
layout: home

hero:
  name: TensorCraft-HPC
  text: 解密高性能 AI 内核
  tagline: 一个可读的 C++/CUDA 内核库，用于学习、验证和打包。保持快速路径，保持文档诚实，保持工作流足够小以维护。
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

features:
  - icon: 🧮
    title: GEMM 优化
    details: 从朴素实现到分块、双缓冲、Tensor Core (WMMA) 的渐进式优化。达到 cuBLAS 性能的 85-95%。
  - icon: 👁️
    title: FlashAttention
    details: 内存高效的注意力实现，支持 RoPE 位置编码和 MoE 路由。达到 cuDNN 性能的 80-90%。
  - icon: 📊
    title: 归一化
    details: LayerNorm、RMSNorm、BatchNorm 和 Softmax，采用优化的融合策略。达到 cuDNN 性能的 90-95%。
  - icon: 🔄
    title: 卷积
    details: 2D/3D 卷积，支持 Im2Col、Winograd 和深度可分离优化。
  - icon: 📉
    title: 稀疏操作
    details: CSR/CSC 格式支持，SpMV、SpMM 和结构化稀疏模式。
  - icon: ⚡
    title: 量化
    details: INT8 和 FP8 (CUDA 12.0+) 量化操作，精度损失最小。
---

<style>
.VPHero .name {
  background: linear-gradient(135deg, #ffffff 0%, #76B900 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
</style>

## 摘要 {#abstract}

<div class="abstract">
<div class="abstract-title">摘要</div>
<div class="abstract-content">

TensorCraft-HPC 是一个仅头文件的 C++/CUDA 内核库，专为学习、验证和打包现代 AI 算子而设计。与优先考虑原始性能的生产库不同，TensorCraft-HPC 强调**可读性**和**渐进式优化路径**——每个内核从朴素实现演进到优化版本，使学习过程明确且易于理解。

该仓库遵循 **OpenSpec 驱动的开发工作流**，其中 `openspec/specs/` 中的规范作为权威的真实来源。这种方法确保文档与实现保持同步，并为贡献者提供清晰的契约。

</div>
</div>

## 主要贡献 {#contributions}

<ul class="contributions">
<li><strong>教育性内核实现</strong> — 从朴素到 Tensor Core 的渐进式优化路径，包括 GEMM、FlashAttention 风格的内存高效注意力和融合归一化内核。</li>
<li><strong>仅头文件架构</strong> — C++ 项目零构建集成，通过 pybind11 提供可选的 Python 绑定用于实验。</li>
<li><strong>多架构支持</strong> — CUDA 内核目标为 SM70 (Volta) 到 SM100 (Blackwell)，具有编译时特性检测。</li>
<li><strong>OpenSpec 工作流</strong> — 规范优先开发，验收标准在 `openspec/specs/`，变更提案在 `openspec/changes/`。</li>
<li><strong>双语文档</strong> — 完整的中英文文档，配有 Mermaid 架构图。</li>
</ul>

## 架构概览 {#architecture}

```mermaid
flowchart TB
    subgraph UserAPI["用户 API 层"]
        CPP["C++ 头文件<br/>(仅头文件)"]
        PY["Python 绑定<br/>(tensorcraft_ops)"]
    end

    subgraph Kernels["内核层"]
        GEMM["GEMM<br/>(朴素 → Tensor Core)"]
        ATTN["Attention<br/>(FlashAttention)"]
        NORM["归一化<br/>(融合)"]
        CONV["卷积<br/>(Im2Col/Winograd)"]
    end

    subgraph Memory["内存层"]
        TENSOR["FloatTensor<br/>(RAII)"]
        POOL["MemoryPool<br/>(可选)"]
    end

    subgraph Hardware["硬件抽象"]
        SM70["SM70 (Volta)"]
        SM80["SM80 (Ampere)"]
        SM90["SM90 (Hopper)"]
        SM100["SM100 (Blackwell)"]
    end

    CPP --> Kernels
    PY --> Kernels
    Kernels --> Memory
    Memory --> Hardware
```

## 快速开始 {#quick-start}

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

## 项目状态 {#status}

| 方面 | 状态 |
|------|------|
| 仓库模式 | 稳定化 / 收尾 |
| 核心内核 | 完成 (GEMM, Attention, Norm, Conv) |
| 文档 | OpenSpec 驱动，双语 |
| CUDA 支持 | 11.0 - 13.1 |
| 架构支持 | SM70 - SM100 (Volta → Blackwell) |

## 引用 {#citation}

如果您在研究或学习材料中使用 TensorCraft-HPC，请引用：

```bibtex
@software{tensorcraft-hpc,
  title = {TensorCraft-HPC: Demystifying High-Performance AI Kernels},
  author = {LessUp},
  year = {2024},
  url = {https://github.com/LessUp/modern-ai-kernels}
}
```

## 参考资料 {#references}

参见 [论文引用](/zh/references/papers) 获取本仓库引用的学术论文和开源项目完整列表。