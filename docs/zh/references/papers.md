# 论文引用

本页面列出了影响 TensorCraft-HPC 设计和实现的学术论文和开源项目。我们鼓励用户阅读原始论文以获得更深入的理解。

## 如何使用这些引用

请把这个页面当作“带注释的参考层”，而不是单纯的书目列表。每条引用通常承担三种角色之一：

1. 解释仓库采用的某种 kernel 策略
2. 定义性能比较时所依赖的生产级 baseline
3. 说明 TensorCraft-HPC 为了可解释性而刻意保持更简单的地方

如果你在评估这个项目，推荐的方式是先读原始论文或库文档，再回看 TensorCraft-HPC 对应的白皮书、证据页或 atlas 页面。重点不是宣称全面对标生产系统，而是展示这些思想如何被转译成一个可学习、可评估的工程表面。

## GEMM 优化 {#gemm}

### 基础论文

<div class="citation">
<span class="citation-author">CUTLASS Team (NVIDIA)</span> —
<span class="citation-title">CUTLASS: CUDA Templates for Linear Algebra Subroutines</span><br/>
<a class="citation-link" href="https://github.com/NVIDIA/cutlass">https://github.com/NVIDIA/cutlass</a>
</div>

Tensor Core 编程模式的主要参考。TensorCraft-HPC 的 GEMM 实现遵循 CUTLASS 的分块和流水线策略。

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">cuBLAS 文档</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/cuda/cublas/">https://docs.nvidia.com/cuda/cublas/</a>
</div>

性能比较的基准。所有 GEMM 基准测试报告相对于 cuBLAS 的性能。

### Tensor Core 编程

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">Tensor Core 编程指南</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-cores">CUDA C++ 编程指南</a>
</div>

理解 WMMA (Warp Matrix Multiply-Accumulate) 操作的必读材料。

---

## Attention 机制 {#attention}

### FlashAttention

<div class="citation">
<span class="citation-author">Tri Dao, Daniel Y. Fu, et al.</span> —
<span class="citation-title">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</span><br/>
NeurIPS 2022<br/>
<a class="citation-link" href="https://arxiv.org/abs/2205.14135">arXiv:2205.14135</a> |
<a class="citation-link" href="https://github.com/Dao-AILab/flash-attention">GitHub</a>
</div>

内存高效注意力机制的基础论文。TensorCraft-HPC 实现了论文中描述的分块策略。

<div class="citation">
<span class="citation-author">Tri Dao</span> —
<span class="citation-title">FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning</span><br/>
ICLR 2024<br/>
<a class="citation-link" href="https://arxiv.org/abs/2307.08691">arXiv:2307.08691</a>
</div>

注意力计算的改进并行策略。

### RoPE (旋转位置编码)

<div class="citation">
<span class="citation-author">苏剑林等</span> —
<span class="citation-title">RoFormer: Enhanced Transformer with Rotary Position Embedding</span><br/>
<a class="citation-link" href="https://arxiv.org/abs/2104.09864">arXiv:2104.09864</a>
</div>

---

## 归一化 {#normalization}

<div class="citation">
<span class="citation-author">Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton</span> —
<span class="citation-title">Layer Normalization</span><br/>
<a class="citation-link" href="https://arxiv.org/abs/1607.06450">arXiv:1607.06450</a>
</div>

<div class="citation">
<span class="citation-author">Biao Zhang, Rico Sennrich</span> —
<span class="citation-title">Root Mean Square Layer Normalization</span><br/>
NeurIPS 2019<br/>
<a class="citation-link" href="https://arxiv.org/abs/1911.12247">arXiv:1911.12247</a>
</div>

RMSNorm 是 LLaMA 和许多现代大语言模型使用的归一化层。

---

## 量化 {#quantization}

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">FP8 Formats for Deep Learning</span><br/>
<a class="citation-link" href="https://arxiv.org/abs/2209.05433">arXiv:2209.05433</a>
</div>

定义 Hopper 架构中使用的 E4M3 和 E5M2 FP8 格式的论文。

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">FP8 Training with NVIDIA Hopper</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html">Transformer Engine 文档</a>
</div>

---

## 稀疏操作 {#sparse}

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">cuSPARSE 文档</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/cuda/cusparse/">https://docs.nvidia.com/cuda/cusparse/</a>
</div>

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">2:4 结构化稀疏</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#structured-sparse-matrix-storage">CUDA 编程指南</a>
</div>

Ampere 架构支持 2:4 结构化稀疏，可提供 2 倍吞吐量提升。

---

## 相关项目 {#projects}

| 项目 | 描述 | 许可证 |
|------|------|--------|
| [CUTLASS](https://github.com/NVIDIA/cutlass) | CUDA 线性代数模板 | BSD-3 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | 内存高效注意力 | BSD-3 |
| [xFormers](https://github.com/facebookresearch/xformers) | Facebook 注意力内核 | BSD-3 |
| [Triton](https://github.com/openai/triton) | OpenAI GPU 编程语言 | MIT |
| [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/) | NVIDIA 深度学习库 | 私有 |

---

## 对比框架

这些引用不应该被看作平铺的“推荐清单”，而应该被看作一套对比框架：

- **CUTLASS 与 cuBLAS** 给出 GEMM 相关结论的生产级性能参考面
- **FlashAttention** 给出 attention kernel 的算法与系统协同设计参考
- **Triton 与 xFormers** 展示了相邻生态对 GPU kernel 的组织方式
- **NVIDIA 官方文档** 锚定硬件事实、API 约束与能力边界

这也是为什么白皮书、证据页与 atlas 页面应该联动阅读。

---

## 引用 TensorCraft-HPC {#citing}

如果您在研究或教学材料中使用 TensorCraft-HPC，请引用：

```bibtex
@software{tensorcraft-hpc,
  title = {TensorCraft-HPC: Demystifying High-Performance AI Kernels
           with Modern C++ and CUDA},
  author = {TensorCraft-HPC Contributors},
  year = {2024},
  url = {https://github.com/AICL-Lab/modern-ai-kernels},
  note = {Header-only C++/CUDA kernel library for learning}
}
```
