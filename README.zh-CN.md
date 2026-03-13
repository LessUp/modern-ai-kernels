# TensorCraft-HPC

[English](README.md) | 简体中文 | [文档站](https://lessup.github.io/modern-ai-kernels/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/modern-ai-kernels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17/20-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

TensorCraft-HPC 是一个现代 C++ / CUDA AI kernel library，用于学习、验证与扩展高性能 GEMM、Attention、卷积、稀疏算子和量化实现。

## 仓库概览

- `include/tensorcraft/` 下是以头文件为主的核心算子库
- `src/python_ops/` 提供 Python 绑定
- `tests/` 与 `benchmarks/` 负责正确性验证与性能评估
- GitHub Pages 文档站负责文档导读、阅读路径与项目更新说明

## 快速开始

```bash
cmake --preset release
cmake --build build/release -j$(nproc)
ctest --test-dir build/release --output-on-failure
pip install -e .
```

## 文档

- 项目文档：`https://lessup.github.io/modern-ai-kernels/`
- 站点首页提供项目定位、推荐阅读路径与关键文档入口
- 参与协作请查看 `CONTRIBUTING.md`

## 许可

MIT License
