# Papers & Citations

This page lists the academic papers and open-source projects that inform the design and implementation of TensorCraft-HPC. We encourage users to read the original papers for deeper understanding.

## GEMM Optimization {#gemm}

### Foundational Papers

<div class="citation">
<span class="citation-author">CUTLASS Team (NVIDIA)</span> —
<span class="citation-title">CUTLASS: CUDA Templates for Linear Algebra Subroutines</span><br/>
<a class="citation-link" href="https://github.com/NVIDIA/cutlass">https://github.com/NVIDIA/cutlass</a>
</div>

The primary reference for Tensor Core programming patterns. TensorCraft-HPC's GEMM implementation follows CUTLASS's tiled and pipeline strategies.

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">cuBLAS Documentation</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/cuda/cublas/">https://docs.nvidia.com/cuda/cublas/</a>
</div>

The baseline for performance comparison. All GEMM benchmarks report relative performance to cuBLAS.

### Tensor Core Programming

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">Tensor Core Programming Guide</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-cores">CUDA C++ Programming Guide</a>
</div>

Essential reading for understanding WMMA (Warp Matrix Multiply-Accumulate) operations.

---

## Attention Mechanisms {#attention}

### FlashAttention

<div class="citation">
<span class="citation-author">Tri Dao, Daniel Y. Fu, et al.</span> —
<span class="citation-title">FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</span><br/>
NeurIPS 2022<br/>
<a class="citation-link" href="https://arxiv.org/abs/2205.14135">arXiv:2205.14135</a> |
<a class="citation-link" href="https://github.com/Dao-AILab/flash-attention">GitHub</a>
</div>

The foundational paper on memory-efficient attention. TensorCraft-HPC implements the tiling strategy described in this paper.

<div class="citation">
<span class="citation-author">Tri Dao</span> —
<span class="citation-title">FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning</span><br/>
ICLR 2024<br/>
<a class="citation-link" href="https://arxiv.org/abs/2307.08691">arXiv:2307.08691</a>
</div>

Improved parallelism strategies for attention computation.

### RoPE (Rotary Position Embedding)

<div class="citation">
<span class="citation-author">Jianlin Su, et al.</span> —
<span class="citation-title">RoFormer: Enhanced Transformer with Rotary Position Embedding</span><br/>
<a class="citation-link" href="https://arxiv.org/abs/2104.09864">arXiv:2104.09864</a>
</div>

---

## Normalization {#normalization}

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

RMSNorm is the normalization layer used in LLaMA and many modern LLMs.

---

## Quantization {#quantization}

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">FP8 Formats for Deep Learning</span><br/>
<a class="citation-link" href="https://arxiv.org/abs/2209.05433">arXiv:2209.05433</a>
</div>

The paper defining the E4M3 and E5M2 FP8 formats used in Hopper architecture.

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">FP8 Training with NVIDIA Hopper</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html">Transformer Engine Documentation</a>
</div>

---

## Sparse Operations {#sparse}

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">cuSPARSE Documentation</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/cuda/cusparse/">https://docs.nvidia.com/cuda/cusparse/</a>
</div>

<div class="citation">
<span class="citation-author">NVIDIA</span> —
<span class="citation-title">2:4 Structured Sparsity</span><br/>
<a class="citation-link" href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#structured-sparse-matrix-storage">CUDA Programming Guide</a>
</div>

Ampere architecture supports 2:4 structured sparsity for 2x throughput improvement.

---

## Related Projects {#projects}

| Project | Description | License |
|---------|-------------|---------|
| [CUTLASS](https://github.com/NVIDIA/cutlass) | CUDA Templates for Linear Algebra | BSD-3 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Memory-efficient attention | BSD-3 |
| [xFormers](https://github.com/facebookresearch/xformers) | Facebook's attention kernels | BSD-3 |
| [Triton](https://github.com/openai/triton) | OpenAI's GPU programming language | MIT |
| [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/) | NVIDIA Deep Learning library | Proprietary |

---

## Citing TensorCraft-HPC {#citing}

If you use TensorCraft-HPC in your research or teaching materials, please cite:

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
