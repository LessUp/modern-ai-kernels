---
title: 故障排除指南
lang: zh
---

# 故障排除指南

本指南涵盖构建或使用 TensorCraft-HPC 时的常见问题。

## 构建问题

### 未找到 CUDA

**典型症状**

```text
CMake Error: Could not find CUDA
```

**本仓库的处理方式**

如果 CUDA 不可用，配置仍可成功，但 CMake 会强制：

- `TC_BUILD_TESTS=OFF`
- `TC_BUILD_BENCHMARKS=OFF`
- `TC_BUILD_PYTHON=OFF`

**解决方案**

```bash
nvcc --version
cmake --preset cpu-smoke
```

如果您确实需要 CUDA，请显式指定 toolkit 路径：

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda
```

## 不支持的 GPU 架构

**典型症状**

```text
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```

**解决方案**

从您的机器支持的单个架构开始：

```bash
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=75
```

如果您已知您的 GPU 架构，请使用该确切值。

## 构建过重或内存不足

**典型症状**

```text
nvcc fatal: Unsupported gpu architecture ...
nvcc error: ran out of memory during compilation
```

**解决方案**

1. 优先使用较轻量的预设：

   ```bash
   cmake --preset dev
   cmake --preset python-dev
   ```

2. 减少并行度：

   ```bash
   cmake --build --preset dev --parallel 2
   ```

3. 限制为单一架构：

   ```bash
   cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=75
   ```

## 测试或 Python 绑定意外禁用

这通常意味着 CMake 在没有可用 CUDA 的情况下进行了配置。

检查配置摘要，查找：

```text
CUDA Enabled:   ON
Build Tests:    ON
Build Python:   ON
```

如果 CUDA 关闭，测试和 Python 绑定将按设计被禁用。

## 可编辑安装成功但导入失败

使用仓库根目录并验证导入名称：

```bash
python3 -m pip install -e .
python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

同时确保两个命令使用相同的 Python 解释器。

## Python 环境显示 `Ignoring invalid distribution ~...`

这些警告来自 Python 环境中已存在的损坏包元数据，而非 TensorCraft 本身。

TensorCraft 仍可成功构建，但如果 pip 输出变得嘈杂，您可能需要清理该环境。

## `ModuleNotFoundError: No module named 'tensorcraft_ops'`

**检查项**

1. 从仓库根目录安装：

   ```bash
   python3 -m pip install -e .
   ```

2. 使用相同解释器验证导入：

   ```bash
   python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
   ```

3. 如需要，检查 pip 安装包的位置：

   ```bash
   python3 -m pip show -f tensorcraft-ops
   ```

## CUDA 版本兼容性

| 功能 | 所需 CUDA |
|------|-----------|
| 基本内核和核心构建 | 12.8 |
| BF16 相关路径 | 12.8 |
| FP8 相关路径 | 12.8 |
| Hopper 特定功能 | 12.8 |

本仓库现假定本地 CUDA `12.8` 工具链，不再保留 CUDA 10.x 兼容路径。

## GPU 运行时路径 vs CI

仓库文档记录的本地验证路径是 CUDA `dev` 预设加上 `ctest` 和 Python 导入检查。

当前 GitHub Actions 主要覆盖 CPU 配置/安装冒烟测试和打包冒烟测试。它们**不**替代在 GPU 机器上运行真正的 CUDA 路径。

## 推荐的验证命令

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
python3 -m pip install -e .
python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

## 仍然卡住？

报告问题时，请包含：

- `nvcc --version`
- `cmake --version`
- 您使用的预设或确切的 CMake 命令
- 完整的配置/构建错误输出
- 您的 `CMAKE_CUDA_ARCHITECTURES` 值（如已覆盖）
