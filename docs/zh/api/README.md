---
title: API 参考
lang: zh
---

# API 参考

本章节提供 TensorCraft-HPC 所有模块的详细 API 文档。

## 模块概览

```
tensorcraft/
├── core/           # 核心工具
│   ├── cuda_check.hpp
│   ├── features.hpp
│   ├── type_traits.hpp
│   └── warp_utils.hpp
├── memory/         # 内存管理
│   ├── aligned_vector.hpp
│   ├── tensor.hpp
│   └── memory_pool.hpp
└── kernels/        # 计算算子
    ├── elementwise.hpp
    ├── softmax.hpp
    ├── normalization.hpp
    ├── gemm.hpp
    ├── attention.hpp
    ├── conv2d.hpp
    ├── sparse.hpp
    └── fusion.hpp
```

## API 文档索引

| 模块 | 描述 |
|------|------|
| [核心模块](core.md) | 错误处理、特性检测、类型特征、线程束工具 |
| [内存模块](memory.md) | Tensor 类、内存池、向量化访问 |
| [算子模块](kernels.md) | 所有计算算子（GEMM、注意力等） |
| [Python API](python.md) | Python 绑定（tensorcraft_ops） |

## 常用模式

### 错误处理

所有 CUDA 操作使用基于异常的错误处理：

```cpp
#include "tensorcraft/core/cuda_check.hpp"

try {
    tensorcraft::kernels::gemm(A, B, C, M, N, K);
} catch (const tensorcraft::CudaException& e) {
    std::cerr << "CUDA error at " << e.file() << ":" << e.line() 
              << " - " << e.what() << std::endl;
}
```

### 流支持

所有算子启动器支持可选的 CUDA 流：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// 使用流
tensorcraft::kernels::gemm(A, B, C, M, N, K, 1.0f, 0.0f, 
                           tensorcraft::kernels::GemmVersion::Tiled, stream);

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

### 类型支持

库支持多种数据类型：

| 类型 | 头文件 | 描述 |
|------|--------|------|
| `float` | 内置 | 32 位浮点数 |
| `__half` | `<cuda_fp16.h>` | 16 位浮点数 |
| `__nv_bfloat16` | `<cuda_bf16.h>` | BF16 (CUDA 11.0+) |
| `int8_t` | 内置 | 8 位整数 |
| `__nv_fp8_e4m3` | `<cuda_fp8.h>` | FP8 E4M3 (CUDA 12.0+) |
