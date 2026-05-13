# 示例

欢迎使用 TensorCraft-HPC 示例！本节提供实践教程和代码示例，帮助您开始高性能 AI 内核开发。

## 快速链接

| 示例 | 描述 | 难度 |
|------|------|------|
| [GEMM 教程](/zh/examples/gemm-tutorial) | 从零开始构建 GEMM，渐进式优化 | 🟢 初级 |
| [FlashAttention](/zh/examples/flash-attention) | 内存高效注意力实现 | 🟡 中级 |
| [Python 绑定](/zh/examples/python-bindings) | 从 Python 使用 TensorCraft | 🟢 初级 |

## 前置要求

运行示例前，请确保已安装：

1. **CUDA Toolkit 11.0+**
2. **CMake 3.18+**
3. **C++17 兼容编译器** (GCC 9+, Clang 10+, MSVC 19.28+)
4. **Python 3.8+**（可选，用于 Python 绑定）

## 运行示例

### C++ 示例

```bash
# 克隆并构建
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 使用 CUDA 支持构建
cmake --preset dev
cmake --build --preset dev

# 运行示例
./build/dev/examples/gemm_example
```

### Python 示例

```bash
# 安装 Python 包
pip install -e .

# 运行 Python 示例
python examples/python/gemm_demo.py
```

## 学习路径

我们建议按以下顺序学习以获得最佳体验：

1. **从 GEMM 教程开始** — 学习 CUDA 内核优化的基础
2. **探索 FlashAttention** — 理解内存高效计算模式
3. **尝试 Python 绑定** — 将 TensorCraft 集成到 Python 工作流

## 需要帮助？

- 查看 [API 参考](/zh/api/gemm) 获取详细文档
- 浏览 [学习资源](/zh/references/resources) 获取更多教程
- 如遇问题，在 [GitHub](https://github.com/LessUp/modern-ai-kernels/issues) 提交 issue