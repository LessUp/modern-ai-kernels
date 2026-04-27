---
title: 示例概览
lang: zh
---

# 示例概览

本节提供使用 TensorCraft-HPC 内核的实际示例。

## 可用示例

| 示例 | 描述 |
|------|------|
| [基础 GEMM](basic-gemm.md) | GEMM 优化之旅：从朴素实现到 Tensor Core |
| [注意力机制](attention.md) | FlashAttention 和 RoPE 使用 |
| [归一化](normalization.md) | LayerNorm 和 RMSNorm 示例 |
| [Python 使用](python-usage.md) | 完整的 Python 绑定示例 |

## 前提条件

运行示例前，请确保：

1. 已启用 CUDA 支持构建项目：

   ```bash
   cmake --preset dev
   cmake --build --preset dev --parallel 2
   ```

2. 对于 Python 示例，安装绑定：

   ```bash
   python3 -m pip install -e .
   ```

## 快速参考

### C++ 使用

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

using namespace tensorcraft;

// 使用 Tensor 类
FloatTensor A({128, 256});
FloatTensor B({256, 512});
FloatTensor C({128, 512});

// 初始化数据...
A.copy_from_host(host_data);

// 运行 GEMM
kernels::gemm(A.data(), B.data(), C.data(), 128, 512, 256);
```

### Python 使用

```python
import tensorcraft_ops as tc
import numpy as np

# 创建数组
A = np.random.randn(128, 256).astype(np.float32)
B = np.random.randn(256, 512).astype(np.float32)

# 运行 GEMM
C = tc.gemm(A, B)
```

## 构建示例

`examples/` 目录中的示例可作为主项目的一部分构建：

```bash
cmake --preset dev
cmake --build --preset dev
./build/dev/examples/basic_gemm
```

## 性能基准测试

性能基准测试请参见 `benchmarks/` 目录：

```bash
cmake --preset release
cmake --build --preset release
./build/release/benchmarks/gemm_benchmark
```
