# TensorCraft-HPC

<div align="center">

**用现代 C++ 与 CUDA 揭开高性能 AI 内核的神秘面纱**

[English](README.md) | [简体中文](README.zh-CN.md) | [在线文档](https://lessup.github.io/modern-ai-kernels/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/modern-ai-kernels/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2B-00599C?logo=c%2B%2B&logoColor=white)

</div>

TensorCraft-HPC 是一个**纯头文件 C++/CUDA AI 内核库**，用于学习、验证与封装现代 AI
算子。仓库强调实现可读性、渐进式优化路径，以及适合收尾维护的轻量工程化表面。

**当前项目状态：** 仓库正在朝稳定收尾状态收敛，重点是代码可信、文档清晰、自动化可靠，
而不是继续无边界扩张。

## 这个仓库解决什么问题

- 提供结构清晰的 C++/CUDA 内核实现
- 为关键算子保留渐进式优化路径，便于学习与比较
- 使用 OpenSpec 统一开发流程
- 提供 CPU-only smoke validation 与可选本地 CUDA 校验
- 提供中英双语文档与 GitHub Pages

## 能力概览

| 模块 | 覆盖范围 |
|------|----------|
| Core utilities | CUDA 检查、特性检测、类型特征、warp 工具 |
| Memory | `Tensor`、对齐向量、内存池 |
| Kernels | GEMM、Attention、Normalization、Convolution、Sparse、Fusion |
| Python | `tensorcraft_ops` 绑定，用于 smoke / 集成验证 |
| Validation | CPU smoke build/install、Python wheel build、可选本地 CUDA 测试 |

## 快速开始

### CPU-only smoke validation

```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke --parallel 2
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
python3 -m build --wheel
```

### 启用 CUDA 的本地验证

```bash
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)
ctest --preset dev --output-on-failure
```

## 文档入口

- **文档中心**: <https://lessup.github.io/modern-ai-kernels/>
- **入门指南**: `docs/zh/getting-started/`
- **架构与优化指南**: `docs/zh/guides/`
- **API 文档**: `docs/zh/api/`
- **英文文档**: `docs/en/`

## OpenSpec 开发流程

本仓库以 **OpenSpec** 作为当前唯一的主动开发流程。

1. 先阅读 `openspec/specs/` 中的已接受规范。
2. 如果变更涉及行为、结构、workflow 或关键文档表面，先在 `openspec/changes/`
   下创建或更新 change。
3. 按 change 实施，并同步更新文档与配置。
4. 合并前运行校验。
5. 结构性或 workflow 变更在合并前使用 `/review`。

`specs/` 目录继续保留，但仅作为历史归档，不再作为新的主动事实来源。

## 仓库结构

```text
modern-ai-kernels/
├── AGENTS.md                      # 仓库级 AI 工作规则
├── CLAUDE.md                      # Claude 专用说明
├── .github/copilot-instructions.md
├── openspec/                      # 主动规范工作流
├── specs/                         # 历史归档
├── include/tensorcraft/           # 纯头文件 C++/CUDA 库
├── src/python_ops/                # Python 绑定
├── tests/                         # 验证
├── benchmarks/                    # 基准程序
└── docs/                          # GitHub Pages 与文档
```

## 工具链基线

- **构建系统**: CMake presets
- **格式化 / hooks**: `.clang-format`、`.clang-tidy`、`pre-commit`
- **LSP**: `clangd` + `build/dev/compile_commands.json`
- **GitHub 自动化**: CI、Pages、release workflow、Copilot setup steps

## 许可证

本项目基于 [MIT License](LICENSE) 发布。
