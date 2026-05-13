# Python 绑定

::: info 即将推出
本指南正在开发中。请稍后回来查看完整的 Python 绑定文档。
:::

## 概述

TensorCraft-HPC 通过 pybind11 提供 Python 绑定，允许您从 Python 代码使用优化内核。

## 安装

```bash
pip install tensorcraft-ops
```

## 基本使用

```python
import tensorcraft_ops as tc
import numpy as np

# 创建矩阵
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

# GPU 加速 GEMM
C = tc.gemm(A, B)
```