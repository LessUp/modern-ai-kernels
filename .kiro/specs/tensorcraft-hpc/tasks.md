# Implementation Plan: TensorCraft-HPC

## Overview

本实现计划将 TensorCraft-HPC 项目分解为可执行的编码任务，按照渐进式开发顺序组织。每个任务都引用具体的需求，确保完整覆盖所有功能。

## Tasks

- [x] 1. 项目基础设施搭建
  - [x] 1.1 创建项目目录结构和 CMakeLists.txt
  - [x] 1.2 创建 CMakePresets.json 配置文件
  - [x] 1.3 配置 FetchContent 依赖管理
  - [x] 1.4 编写构建系统单元测试

- [x] 2. 核心工具库实现
  - [x] 2.1 实现 CUDA 错误检查宏
  - [x] 2.2 实现特性检测头文件
  - [x] 2.3 实现类型特征和 Concepts
  - [x] 2.4 实现 warp 级别规约原语 (warp_utils.hpp)

- [x] 3. 内存管理模块实现
  - [x] 3.1 实现对齐向量类型
  - [x] 3.2 实现 Tensor 封装类
  - [x] 3.3 实现内存池

- [x] 4. Checkpoint - 基础设施验证

- [x] 5. Elementwise 算子实现
  - [x] 5.1 实现通用 Elementwise Kernel 框架
  - [x] 5.2 实现 VectorAdd Kernel
  - [x] 5.3 实现标准激活函数 (ReLU, SiLU, GeLU)
  - [x] 5.4 实现自定义激活函数 (LeakyReLU, ELU, Swish)

- [x] 6. 规约与归一化算子实现
  - [x] 6.1 实现 Softmax Kernel
  - [x] 6.2 实现 LayerNorm Kernel
  - [x] 6.3 实现 RMSNorm Kernel
  - [x] 6.4 实现 BatchNorm Kernel

- [x] 7. Checkpoint - 基础算子验证

- [x] 8. GEMM 矩阵乘法实现
  - [x] 8.1 实现 GEMM v1 (Naive)
  - [x] 8.2 实现 GEMM v2 (Shared Memory Tiling)
  - [x] 8.3 实现 GEMM v3 (Double Buffering)
  - [x] 8.4 实现 GEMM v4 (Tensor Core WMMA)
  - [x] 8.5 实现矩阵转置 Kernel

- [x] 9. LLM 关键算子实现
  - [x] 9.1 实现 FlashAttention Kernel
  - [x] 9.2 实现 RoPE Kernel
  - [x] 9.3 实现 MoE Router Kernel

- [x] 10. Checkpoint - 核心算子验证

- [x] 11. 卷积层实现
  - [x] 11.1 实现 Conv2D Naive Kernel
  - [x] 11.2 实现 Im2Col Kernel
  - [x] 11.3 实现 Depthwise Separable 卷积

- [x] 12. 稀疏矩阵实现
  - [x] 12.1 实现 CSR 格式
  - [x] 12.2 实现 CSC 格式
  - [x] 12.3 实现 SpMV Kernel
  - [x] 12.4 实现 SpMM Kernel

- [x] 13. 算子融合与量化
  - [x] 13.1 实现 Bias+GeLU 融合 Epilogue
  - [x] 13.2 实现 INT8 量化支持
  - [x] 13.3 实现 FP8 量化支持 (CUDA 12.0+)

- [x] 14. Checkpoint - 高级算子验证

- [x] 15. Python 绑定实现
  - [x] 15.1 配置 pybind11 构建
  - [x] 15.2 实现核心算子绑定
  - [x] 15.3 实现 LLM 算子绑定

- [x] 16. 性能基准测试
  - [x] 16.1 实现 GEMM Benchmark
  - [x] 16.2 实现 Attention Benchmark
  - [x] 16.3 实现 Conv2D Benchmark

- [x] 17. 文档编写
  - [x] 17.1 编写 README.md
  - [x] 17.2 编写 Modern C++ for CUDA 指南
  - [x] 17.3 编写算子优化教程
  - [x] 17.4 编写 API 参考文档
  - [x] 17.5 编写示例文档
  - [x] 17.6 编写贡献指南
  - [x] 17.7 重构文档目录结构

- [x] 18. CI/CD 配置
  - [x] 18.1 创建 ci.yml workflow
  - [x] 18.2 创建 release.yml workflow
  - [x] 18.3 创建 pages.yml workflow

- [x] 19. GitHub 社区配置
  - [x] 19.1 创建 Issue 模板
  - [x] 19.2 创建 PR 模板
  - [x] 19.3 创建 CODEOWNERS
  - [x] 19.4 创建 Code of Conduct
  - [x] 19.5 创建 Security Policy

- [x] 20. Final Checkpoint - 完整项目验证

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 2.0.0 | 2026-03-09 | Critical bug fixes, architecture improvements |
| 1.1.0 | 2026-01-08 | Build system fixes |
| 1.0.1 | 2025-02-13 | Infrastructure improvements |
| 1.0.0 | 2024-01-01 | Initial release |

## Notes

- 所有任务已完成
- 每个任务引用具体需求以确保可追溯性
- 属性测试验证通用正确性属性
- 单元测试验证特定示例和边界情况
