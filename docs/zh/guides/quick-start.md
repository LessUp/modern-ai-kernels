# 快速入门指南

5分钟快速上手 TensorCraft-HPC。

## 1. 验证环境 {#verify}

```bash
nvidia-smi
nvcc --version
```

## 2. 构建库 {#build}

```bash
git clone https://github.com/AICL-Lab/modern-ai-kernels.git
cd modern-ai-kernels

cmake --preset cpu-smoke
cmake --build --preset cpu-smoke

cmake --preset dev
cmake --build --preset dev --parallel 4
```

## 3. 运行测试 {#test}

```bash
ctest --preset dev --output-on-failure
```

## 下一步 {#next}

- [架构概览](/zh/architecture)
- [API 参考](/zh/api/gemm)