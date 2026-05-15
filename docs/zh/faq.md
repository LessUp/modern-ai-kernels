# 常见问题

## 通用问题

### 什么是 TensorCraft-HPC？

TensorCraft-HPC 是一个仅头文件的 C++/CUDA 库，用于学习高性能 AI 内核实现。它提供从朴素到生产级性能的渐进式优化路径，每一步都有清晰的注释。

### 这个项目面向谁？

- **GPU 内核开发者**：寻求理解优化技术
- **ML 基础设施工程师**：评估内核实现
- **研究人员**：研究高性能计算模式
- **学生**：学习 CUDA 编程

### 这与 CUTLASS 或 cuBLAS 有什么不同？

| 方面 | TensorCraft-HPC | CUTLASS | cuBLAS |
|------|----------------|---------|--------|
| 目的 | 学习 | 生产 | 生产 |
| 代码风格 | 可读 | 模板密集 | 闭源 |
| 性能 | 92% cuBLAS | ~100% | 100%（基准） |
| 构建 | 仅头文件 | CMake + 构建 | 预安装 |

TensorCraft-HPC 优先考虑**教育性清晰度**而非最大性能。CUTLASS 是生产部署的参考。

---

## 技术问题

### 支持哪些 GPU 架构？

| 架构 | 计算能力 | 状态 |
|------|----------|------|
| Volta | SM70 (7.0) | 已支持 |
| Turing | SM75 (7.5) | 已支持 |
| Ampere | SM80/SM86 | 已支持 |
| Hopper | SM90 (9.0) | 已支持 |
| Blackwell | SM100 (10.0) | 已支持 |

### 支持哪些 CUDA 版本？

CUDA 11.0 到 13.1。建议使用 CUDA 12.0+ 以获取 FP8 和 Transformer Engine 功能。

### 为什么是仅头文件？

1. **零构建摩擦**：只需 `#include` 即可使用
2. **轻松集成**：将头文件复制到任何项目
3. **透明性**：所有代码可见可审计
4. **无 ABI 问题**：一切使用你的编译标志编译

### 我可以在生产环境中使用吗？

TensorCraft-HPC 主要为学习而设计。虽然内核达到了良好的性能（某些操作达到 NVIDIA 库的 95%），但对于生产部署，我们建议：

- **GEMM**：使用 cuBLAS 或 CUTLASS
- **Attention**：使用官方 FlashAttention
- **归一化**：使用 cuDNN

### 支持哪些精度类型？

| 类型 | 大小 | 架构 |
|------|------|------|
| FP32 | 32-bit | 全部 |
| FP16 (half) | 16-bit | SM70+ |
| BF16 | 16-bit | SM80+ |
| TF32 | 19-bit | SM80+ |
| FP8 (E4M3/E5M2) | 8-bit | SM90+ |
| INT8 | 8-bit | SM75+ |

---

## 构建与集成

### 如何将 TensorCraft-HPC 添加到 CMake 项目？

```cmake
# 简单：只需添加包含目录
target_include_directories(your_target PRIVATE
    path/to/modern-ai-kernels/include
)

# 链接 CUDA
find_package(CUDA REQUIRED)
target_link_libraries(your_target CUDA::cudart)
```

### 如何使用 Python 绑定？

```bash
# 从源码安装
cd modern-ai-kernels
pip install -e .
```

```python
import tensorcraft_ops as tc
import numpy as np

# 使用 NumPy 兼容 API
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)
C = tc.gemm(A, B)
```

### 如何运行测试？

```bash
# 使用开发预设构建
cmake --preset dev
cmake --build --preset dev

# 运行所有测试
ctest --preset dev --output-on-failure

# 运行特定测试
ctest --preset dev -R gemm
```

### 如何运行基准测试？

```bash
# 构建基准测试
cmake --preset release
cmake --build --preset release --parallel 2

# 运行 GEMM 基准测试
./build/release/benchmarks/gemm_benchmark

# 使用特定过滤器
./build/release/benchmarks/gemm_benchmark --benchmark_filter="FP16"
```

---

## 故障排除

### 构建失败，提示"CUDA not found"

确保已安装 CUDA 工具包并设置了 `CUDA_PATH`：

```bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### "Unsupported GPU architecture" 错误

检查 GPU 的计算能力：

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

TensorCraft-HPC 需要 SM70 (Volta) 或更高版本。

### 数值结果与 cuBLAS 不同

小的数值差异是预期的，原因是：

1. **浮点运算顺序**：不同的归约顺序产生不同的舍入
2. **混合精度**：Tensor Core 使用混合精度累加
3. **融合操作**：融合内核可能使用不同的中间精度

对于数值验证，使用适当的容差进行相对误差检查：

```cpp
float rel_error = std::abs(your_result - reference) / std::abs(reference);
EXPECT_LT(rel_error, 1e-3);  // 0.1% 容差
```

### 性能低于预期

检查以下常见问题：

1. **时钟速度**：GPU 可能受到功率限制。使用 `nvidia-smi -q -d CLOCK` 检查
2. **内存带宽**：确保数据适合 GPU 内存
3. **Warp 分歧**：检查内核中的分支分歧
4. **共享内存存储体冲突**：使用填充避免冲突

---

## 贡献

### 如何贡献新内核？

1. 在 `openspec/changes/` 创建规范
2. 在 `include/tensorcraft/kernels/` 实现内核头文件
3. 在 `tests/kernels/` 添加 GoogleTest 测试
4. 在 `benchmarks/` 添加基准测试
5. 提交 Pull Request

详见[开发方法论](/zh/whitepaper/methodology)章节。

### 如何报告 Bug？

请在 [GitHub Issues](https://github.com/AICL-Lab/modern-ai-kernels/issues) 提交问题，包含：

- GPU 型号和驱动版本
- CUDA 版本
- 最小复现步骤
- 期望行为 vs 实际行为
