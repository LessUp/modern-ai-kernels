# Benchmarks

This section presents performance benchmarks for TensorCraft-HPC kernels compared against NVIDIA's optimized libraries (cuBLAS, cuDNN, cuSPARSE).

## Overview

All benchmarks are measured on:

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100 80GB |
| CUDA Version | 12.4 |
| Data Type | FP16 (Tensor Core) |
| Measurements | Average of 100 runs |

## Performance Summary

| Kernel | Reference | Relative Performance |
|--------|-----------|---------------------|
| GEMM (FP16) | cuBLAS | 92% |
| FlashAttention | cuDNN | 85% |
| LayerNorm | cuDNN | 95% |
| Conv2D | cuDNN | 78% |
| SpMV (CSR) | cuSPARSE | 88% |

## Detailed Benchmarks

- [GEMM Performance](/en/benchmarks/gemm) — Detailed GEMM benchmark across matrix sizes
- [Attention Performance](/en/benchmarks/attention) — FlashAttention benchmark analysis

## Benchmarking Philosophy

TensorCraft-HPC prioritizes **readability and educational value** over raw performance. Our benchmarks serve to:

1. **Validate correctness** — Ensure optimized versions produce accurate results
2. **Demonstrate progress** — Show improvement from naive to optimized
3. **Guide optimization** — Identify bottlenecks and optimization opportunities

::: tip Performance vs Readability
While we strive for competitive performance, we sometimes choose clearer code over marginal speed improvements. The goal is learning, not beating cuBLAS.
:::

## Running Your Own Benchmarks

```bash
# Build benchmarks
cmake --preset dev
cmake --build --preset dev

# Run GEMM benchmark
./build/dev/benchmarks/gemm_benchmark

# Run all benchmarks
ctest --preset dev -L benchmark
```

## Benchmark Configuration

Benchmarks can be configured via environment variables:

```bash
# Set matrix size for GEMM benchmark
export TENSORCRAFT_BENCH_SIZE=4096

# Set number of warmup runs
export TENSORCRAFT_BENCH_WARMUP=10

# Set number of measured runs
export TENSORCRAFT_BENCH_RUNS=100
```