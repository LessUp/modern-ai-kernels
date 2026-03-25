# TensorCraft-HPC

[English](README.md) | 简体中文 | [文档站](https://lessup.github.io/modern-ai-kernels/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/modern-ai-kernels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

TensorCraft-HPC 是一个现代 C++ / CUDA AI kernel library，用于学习、验证与扩展高性能 GEMM、Attention、卷积、稀疏算子和量化实现。

## 仓库概览

- `include/tensorcraft/` 下是以头文件为主的核心算子库
- `src/python_ops/` 提供 Python 绑定
- `tests/` 与 `benchmarks/` 负责正确性验证与性能评估
- GitHub Pages 文档站负责文档导读、阅读路径与项目更新说明

## 快速开始

推荐在具备 CUDA 的开发机上使用：

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

`python -m pip install -e .` 会构建并安装当前的 pybind11 扩展模块，导入名为 `tensorcraft_ops`。Python 绑定当前依赖 CUDA；当 CUDA 不可用时，CMake 会自动关闭 tests、benchmarks 和 Python bindings。

## 构建预设

- `dev`：日常 CUDA 开发推荐预设；单架构、开启测试和 Python 绑定
- `python-dev`：更轻量的 CUDA 预设，聚焦构建 `tensorcraft_ops`
- `release`：更完整的发布构建，包含 benchmark
- `cpu-smoke`：仅做 CPU 配置/安装冒烟验证；测试与 Python 绑定会关闭

## 文档

- 项目文档：`https://lessup.github.io/modern-ai-kernels/`
- 站点首页提供项目定位、推荐阅读路径与关键文档入口
- 参与协作请查看 `CONTRIBUTING.md`

## 许可

MIT License
