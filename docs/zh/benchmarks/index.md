# 性能基准

本节展示 TensorCraft-HPC 内核与 NVIDIA 优化库（cuBLAS、cuDNN、cuSPARSE）的性能对比。

## 概述

所有基准测试在以下环境测量：

| 参数 | 值 |
|------|-----|
| GPU | NVIDIA A100 80GB |
| CUDA 版本 | 12.4 |
| 数据类型 | FP16 (Tensor Core) |
| 测量方式 | 100 次运行平均值 |

## 性能总览

| 内核 | 参考库 | 相对性能 |
|------|--------|----------|
| GEMM (FP16) | cuBLAS | 92% |
| FlashAttention | cuDNN | 85% |
| LayerNorm | cuDNN | 95% |
| Conv2D | cuDNN | 78% |
| SpMV (CSR) | cuSPARSE | 88% |

## 详细基准

- [GEMM 性能](/zh/benchmarks/gemm) — 不同矩阵大小的 GEMM 基准详情
- [Attention 性能](/zh/benchmarks/attention) — FlashAttention 基准分析

## 基准测试理念

TensorCraft-HPC 优先考虑**可读性和教育价值**而非原始性能。我们的基准测试用于：

1. **验证正确性** — 确保优化版本产生准确结果
2. **展示进展** — 显示从朴素到优化的改进
3. **指导优化** — 识别瓶颈和优化机会

::: tip 性能 vs 可读性
虽然我们追求有竞争力的性能，但有时我们会选择更清晰的代码而非边际速度提升。目标是学习，而非超越 cuBLAS。
:::

## 运行自己的基准测试

```bash
# 构建基准测试
cmake --preset dev
cmake --build --preset dev

# 运行 GEMM 基准测试
./build/dev/benchmarks/gemm_benchmark

# 运行所有基准测试
ctest --preset dev -L benchmark
```

## 基准测试配置

可通过环境变量配置基准测试：

```bash
# 设置 GEMM 基准测试的矩阵大小
export TENSORCRAFT_BENCH_SIZE=4096

# 设置预热运行次数
export TENSORCRAFT_BENCH_WARMUP=10

# 设置测量运行次数
export TENSORCRAFT_BENCH_RUNS=100
```