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
  - [x] 2.4 编写类型特征属性测试

- [x] 3. 内存管理模块实现
  - [x] 3.1 实现对齐向量类型
  - [x] 3.2 实现 Tensor 封装类
  - [x] 3.3 编写 Tensor RAII 属性测试
  - [x] 3.4 实现内存池

- [x] 4. Checkpoint - 基础设施验证

- [x] 5. Elementwise 算子实现
  - [x] 5.1 实现通用 Elementwise Kernel 框架
  - [x] 5.2 实现 VectorAdd Kernel
  - [x] 5.3 编写 VectorAdd 属性测试
  - [x] 5.4 实现标准激活函数 (ReLU, SiLU, GeLU)
  - [x] 5.5 实现自定义激活函数 (LeakyReLU, ELU, Swish)
  - [x] 5.6 编写激活函数属性测试
  - [x] 5.7 编写优化等级等价性测试

- [x] 6. 规约与归一化算子实现
  - [x] 6.1 实现 Softmax Kernel
  - [x] 6.2 编写 Softmax 属性测试
  - [x] 6.3 实现 LayerNorm Kernel
  - [x] 6.4 编写 LayerNorm 属性测试
  - [x] 6.5 实现 RMSNorm Kernel
  - [x] 6.6 编写 RMSNorm 属性测试
  - [x] 6.7 实现 BatchNorm Kernel
  - [x] 6.8 编写 BatchNorm 属性测试

- [x] 7. Checkpoint - 基础算子验证

- [x] 8. GEMM 矩阵乘法实现
  - [x] 8.1 实现 GEMM v1 (Naive)
  - [x] 8.2 实现 GEMM v2 (Shared Memory Tiling)
  - [x] 8.3 实现 GEMM v3 (Double Buffering)
  - [x] 8.4 实现 GEMM v4 (Tensor Core WMMA)
  - [x] 8.5 编写 GEMM 正确性属性测试
  - [x] 8.6 实现矩阵转置 Kernel
  - [x] 8.7 编写矩阵转置 Round-Trip 测试
  - [x] 8.8 实现 GEMM 版本选择器

- [x] 9. LLM 关键算子实现
  - [x] 9.1 实现 FlashAttention Kernel
  - [x] 9.2 编写 FlashAttention 等价性测试
  - [x] 9.3 实现 RoPE Kernel
  - [x] 9.4 编写 RoPE 正确性测试
  - [x] 9.5 实现简化版 PagedAttention Kernel
  - [x] 9.6 实现 MoE Router Kernel

- [x] 10. Checkpoint - 核心算子验证

- [x] 11. 卷积层实现
  - [x] 11.1 实现 Conv2D Naive Kernel
  - [x] 11.2 实现 Im2Col Kernel
  - [x] 11.3 实现 Im2Col + GEMM 卷积
  - [x] 11.4 实现 Winograd 卷积 (3x3) - 部分实现
  - [x] 11.5 实现 Depthwise Separable 卷积
  - [x] 11.6 编写卷积算法等价性测试

- [x] 12. 稀疏矩阵实现
  - [x] 12.1 实现 CSR 格式
  - [x] 12.2 实现 CSC 格式
  - [x] 12.3 实现 SpMV Kernel
  - [x] 12.4 实现 SpMM Kernel
  - [x] 12.5 编写稀疏操作等价性测试
  - [x] 12.6 编写稀疏格式 Round-Trip 测试

- [x] 13. 算子融合与量化
  - [x] 13.1 实现 Bias+GeLU 融合 Epilogue
  - [x] 13.2 编写融合算子等价性测试
  - [x] 13.3 实现 INT8 量化支持
  - [x] 13.4 实现 FP8 量化支持 (CUDA 12.0+)

- [x] 14. Checkpoint - 高级算子验证

- [x] 15. Python 绑定实现
  - [x] 15.1 配置 pybind11 构建
  - [x] 15.2 实现核心算子绑定
  - [x] 15.3 实现 LLM 算子绑定
  - [x] 15.4 编写 Python 验证脚本

- [x] 16. 性能基准测试
  - [x] 16.1 实现 GEMM Benchmark
  - [x] 16.2 实现 Attention Benchmark
  - [x] 16.3 实现 Conv2D Benchmark
  - [x] 16.4 创建性能对比绘图脚本

- [x] 17. 文档编写
  - [x] 17.1 编写 README.md
  - [x] 17.2 编写 Modern C++ for CUDA 指南
  - [x] 17.3 编写算子优化教程

- [x] 18. Final Checkpoint - 完整项目验证

## Notes

- 所有任务已完成
- 每个任务引用具体需求以确保可追溯性
- 属性测试验证通用正确性属性
- 单元测试验证特定示例和边界情况
