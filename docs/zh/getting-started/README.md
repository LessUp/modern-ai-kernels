---
title: 入门指南
lang: zh
---

# 入门指南

本章节帮助您快速开始使用 TensorCraft-HPC。

## 内容概览

| 文档 | 描述 |
|------|------|
| [安装指南](installation.md) | 系统要求、安装步骤和构建配置 |
| [故障排除](troubleshooting.md) | 常见问题诊断和解决方案 |

## 快速开始

如果您已经具备 CUDA 开发环境，可以直接使用以下命令：

```bash
# 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 构建（推荐开发预设）
cmake --preset dev
cmake --build --preset dev --parallel 2

# 运行测试
ctest --preset dev --output-on-failure

# 安装 Python 绑定
python3 -m pip install -e .
python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

## 系统要求

- **CUDA Toolkit**: 12.8
- **CMake**: 3.20+
- **C++ 编译器**: 支持 C++17 的主机编译器
- **NVIDIA GPU**: 推荐用于测试和 Python 绑定

详细要求请参阅 [安装指南](installation.md)。
