---
layout: default
title: TensorCraft-HPC — 文档入口
description: 现代 C++ / CUDA AI 算子库的文档首页：项目定位、阅读路径与核心文档导航
---

# TensorCraft-HPC

[![GitHub Pages](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml)
[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2F20%2F23-00599C?logo=c%2B%2B&logoColor=white)

TensorCraft-HPC 面向“理解并实践现代 AI 算子如何从可读实现演进到高性能版本”的学习与工程场景，覆盖 Elementwise、Normalization、GEMM、Attention、Conv2D、Sparse、Fusion 与 Quantization 等核心模块。

## 项目定位

这是一个把现代 C++/CUDA 编程、AI kernel 设计模式与工程化验证流程放在一起的学习型仓库。`README` 现在只保留仓库级入口，这个页面负责告诉你项目适合谁、从哪里开始，以及文档之间如何组织。

## 适合谁

- 想系统浏览现代 C++ / CUDA AI kernel 设计与优化模式的开发者
- 想按主题阅读安装、架构、API、优化与问题排查文档的工程师
- 需要快速进入测试、贡献流程和版本演进记录的维护者

## 从哪里开始

1. 先看 [README](README.md)，完成最小构建、测试与 Python 安装。
2. 再看 [安装指南](docs/INSTALL.md) 和 [架构设计](docs/architecture.md)，建立环境与模块边界认知。
3. 想深入实现细节时，继续阅读 [优化指南](docs/optimization_guide.md)、[API 参考](docs/api_reference.md) 与 [Modern C++ in CUDA](docs/modern_cpp_cuda.md)。

## 推荐阅读路径

### 我只想先编译并跑测试

- [README](README.md)
- [安装指南](docs/INSTALL.md)
- [问题排查](docs/TROUBLESHOOTING.md)

### 我想先理解架构与模块划分

- [架构设计](docs/architecture.md)
- [API 参考](docs/api_reference.md)
- [Modern C++ in CUDA](docs/modern_cpp_cuda.md)

### 我准备做优化或继续维护

- [优化指南](docs/optimization_guide.md)
- [CONTRIBUTING](CONTRIBUTING.md)
- [CHANGELOG](CHANGELOG.md)
- [changelog/](changelog/)

## 核心文档

| 类别 | 页面 | 说明 |
|------|------|------|
| 概览 | [README](README.md) | 仓库定位、最小构建命令与文档链接 |
| 快速开始 | [安装指南](docs/INSTALL.md) | 环境准备、平台差异与安装步骤 |
| 架构设计 | [架构设计](docs/architecture.md) | 模块边界、目录结构与设计思路 |
| 使用指南 | [优化指南](docs/optimization_guide.md) | GEMM / Attention / Kernel 优化路线 |
| 参考 | [API 参考](docs/api_reference.md) | C++ / Python 接口说明 |
| 开发指南 | [CONTRIBUTING](CONTRIBUTING.md) | 贡献流程、测试要求与代码规范 |
| 归档 | [CHANGELOG](CHANGELOG.md) / [changelog/](changelog/) | 版本记录与 Pages / 工作流调整记录 |

## 相关入口

- GitHub 仓库：`https://github.com/LessUp/modern-ai-kernels`
- 在线文档：`https://lessup.github.io/modern-ai-kernels/`
- Issues：`https://github.com/LessUp/modern-ai-kernels/issues`
