# TensorCraft-HPC 文档中心

欢迎来到 **TensorCraft-HPC** 文档中心。这是一个现代化的 C++/CUDA AI 高性能计算内核库，用于学习、验证和实现 GEMM、注意力机制、卷积、归一化、稀疏算子和量化等核心算法。

## 文档结构

```
docs/
├── getting-started/    # 快速入门和安装指南
├── guides/             # 架构设计和优化指南
├── api/                # API 参考文档
├── examples/           # 代码示例和教程
└── reference/          # 贡献指南、变更日志、社区准则
```

## 快速导航

### 入门指南

| 文档 | 描述 |
|------|------|
| [安装指南](getting-started/installation.md) | 构建要求和安装步骤 |
| [故障排除](getting-started/troubleshooting.md) | 常见问题和解决方案 |

### 开发指南

| 文档 | 描述 |
|------|------|
| [架构设计](guides/architecture.md) | 模块设计和代码组织结构 |
| [优化指南](guides/optimization.md) | 内核优化技术详解 |
| [现代 C++/CUDA](guides/modern-cpp-cuda.md) | CUDA 开发中的现代 C++ 特性 |

### API 参考

| 文档 | 描述 |
|------|------|
| [核心模块](api/core.md) | 错误处理、特性检测、类型特征 |
| [内存模块](api/memory.md) | Tensor 封装、内存池、向量化访问 |
| [算子模块](api/kernels.md) | GEMM、注意力、归一化等算子 |
| [Python API](api/python.md) | Python 绑定接口参考 |

### 示例教程

| 文档 | 描述 |
|------|------|
| [GEMM 示例](examples/basic-gemm.md) | 从朴素实现到 Tensor Core 优化 |
| [注意力示例](examples/attention.md) | FlashAttention 使用指南 |
| [归一化示例](examples/normalization.md) | LayerNorm/RMSNorm 使用指南 |
| [Python 示例](examples/python-usage.md) | Python 绑定使用示例 |

### 参考资料

| 文档 | 描述 |
|------|------|
| [贡献指南](reference/contributing.md) | 如何为项目做出贡献 |
| [变更日志](reference/changelog.md) | 版本历史和变更记录 |
| [行为准则](reference/code-of-conduct.md) | 社区行为准则 |
| [安全策略](reference/security.md) | 安全策略和漏洞报告 |

## 推荐阅读路径

### 快速构建和运行测试

1. [项目根目录 README](../../README.zh-CN.md) - 项目概览
2. [安装指南](getting-started/installation.md) - 构建说明
3. [故障排除](getting-started/troubleshooting.md) - 遇到问题时的解决方案

### 深入理解架构

1. [架构设计](guides/architecture.md) - 设计原则和模块结构
2. [API 参考](api/core.md) - 探索各个模块
3. [现代 C++/CUDA](guides/modern-cpp-cuda.md) - 使用的 C++17/20 特性

### 学习优化技术

1. [优化指南](guides/optimization.md) - 内核优化之旅
2. [GEMM 示例](examples/basic-gemm.md) - 实践示例
3. [API 参考](api/kernels.md) - 算子接口

### 参与项目贡献

1. [贡献指南](reference/contributing.md) - 贡献指南
2. [行为准则](reference/code-of-conduct.md) - 社区准则
3. [变更日志](reference/changelog.md) - 最新变更

## 语言切换

- **English**: [Documentation in English](../en/README.md)
- **简体中文**: 当前页面

## 外部链接

- **GitHub 仓库**: https://github.com/LessUp/modern-ai-kernels
- **在线文档**: https://lessup.github.io/modern-ai-kernels/
- **问题跟踪**: https://github.com/LessUp/modern-ai-kernels/issues
