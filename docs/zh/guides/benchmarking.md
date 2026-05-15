# 性能测试指南

如何测量和分析内核性能。

## 运行基准测试 {#running}

```bash
cmake --preset release
cmake --build --preset release --parallel 2

./build/release/benchmarks/gemm_benchmark
./build/release/benchmarks/attention_benchmark
./build/release/benchmarks/conv_benchmark
```

## 分析工具 {#profiling}

### Nsight Compute

```bash
ncu --set full -o profile_report ./build/release/benchmarks/gemm_benchmark
```

### Nsight Systems

```bash
nsys profile -o timeline ./build/release/benchmarks/gemm_benchmark
```
