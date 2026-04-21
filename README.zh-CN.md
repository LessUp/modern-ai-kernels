# TensorCraft-HPC

<div align="center">

**用现代 C++ 与 CUDA 揭开高性能 AI 内核的神秘面纱**

现代 C++/CUDA AI 高性能计算内核库

[English](README.md) | [简体中文](README.zh-CN.md) | [📚 在线文档](https://lessup.github.io/modern-ai-kernels/) | [API 参考](docs/zh/api/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/modern-ai-kernels/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17/20/23-00599C?logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
[![GitHub stars](https://img.shields.io/github/stars/LessUp/modern-ai-kernels?style=social)](https://github.com/LessUp/modern-ai-kernels/stargazers)

**纯头文件** • **渐进式优化** • **生产就绪**

</div>

---

## 🎯 为什么选择 TensorCraft-HPC？

<div align="center">

| 🎓 学习 | 🔬 研究 | 🚀 生产 | 📊 基准测试 |
|:------:|:------:|:-------:|:----------:|
| 渐进式 GPU 优化 | 快速内核原型 | 即插即用替换 | 跨架构对比 |
| 朴素 → Tensor Core | 测试新算法 | 85-95% cuBLAS 性能 | Volta → Blackwell |

</div>

---

TensorCraft-HPC 是一个**综合性的纯头文件 GPU 内核库**，实现了核心深度学习操作，并提供**渐进式优化级别**——从朴素实现到 Tensor Core 优化的内核。

---

## ✨ 核心特性

| 类别 | 优化级别 | 性能 |
|------|---------|------|
| **GEMM** | 朴素 → 平铺 → 双缓冲 → Tensor Core (WMMA) | cuBLAS 的 85-95% |
| **注意力** | FlashAttention, RoPE, MoE 路由器 | cuDNN 的 80-90% |
| **归一化** | LayerNorm, RMSNorm, BatchNorm, Softmax | cuDNN 的 90-95% |
| **卷积** | 朴素, Im2Col, 深度可分离 | cuDNN 的 75-85% |
| **稀疏** | CSR/CSC, SpMV, SpMM | 针对稀疏性优化 |
| **量化** | INT8, FP8 (CUDA 12.0+) | 降低精度加速 |

### 核心亮点

```
✅ 纯头文件设计              → 直接 #include 即可使用
✅ 渐进式优化                → 从朴素到 Tensor Core 的学习之旅
✅ 现代 C++ 与 CUDA          → C++17/20/23 + CUDA 12.8
✅ Python 绑定               → 通过 pybind11 提供 NumPy 兼容 API
✅ 全面测试                  → GoogleTest 单元测试
✅ 性能基准                  → 可测量的优化之旅
✅ 多 GPU 支持               → Volta → Hopper → Blackwell
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 版本 | 必需 |
|------|------|------|
| CUDA Toolkit | 12.0+ | ✅ 是（GPU 功能） |
| CMake | 3.20+ | ✅ 是 |
| C++ 编译器 | C++17 | ✅ 是 |
| Python | 3.8+ | ⚙️ 可选（绑定） |
| NVIDIA GPU | Compute 70+ | ⚙️ 可选（测试） |

### 安装步骤（3 步）

```bash
# 1. 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 2. 配置并构建
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)

# 3. 运行测试（可选）
ctest --preset dev --output-on-failure
```

### Python 使用（可选）

```bash
# 安装 Python 绑定
pip install -e .

# 快速测试
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

---

## 💻 使用示例

### C++ 示例

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

int main() {
    // 创建张量（RAII 管理，GPU 内存）
    tensorcraft::FloatTensor A({256, 512});
    tensorcraft::FloatTensor B({512, 128});
    tensorcraft::FloatTensor C({256, 128});
    
    // 执行 GEMM: C = A × B
    tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 
                               256, 128, 512);
    
    return 0;
}
```

### Python 示例

```python
import tensorcraft_ops as tc
import numpy as np

# 矩阵乘法
A = np.random.randn(256, 512).astype(np.float32)
B = np.random.randn(512, 128).astype(np.float32)
C = tc.matmul(A, B)

# FlashAttention 风格操作
Q = np.random.randn(32, 128, 64).astype(np.float32)
K = np.random.randn(32, 128, 64).astype(np.float32)
V = np.random.randn(32, 128, 64).astype(np.float32)
output = tc.flash_attention(Q, K, V)

# 层归一化
x = np.random.randn(32, 256).astype(np.float32)
y = tc.layer_norm(x, gamma, beta)
```

---

## 📊 性能基准

TensorCraft-HPC 在所有内核类型上均提供**生产级性能**：

### GEMM 性能（A100, FP32）

| 矩阵大小 | TensorCraft | cuBLAS | 效率 |
|---------|------------|--------|------|
| 256×256 | 92 GFLOPs | 110 GFLOPs | 84% |
| 512×512 | 680 GFLOPs | 750 GFLOPs | 91% |
| 1024×1024 | 2.1 TFLOPs | 2.3 TFLOPs | 91% |
| 2048×2048 | 5.8 TFLOPs | 6.2 TFLOPs | 94% |

### Attention 性能（H100, FP16）

| 序列长度 | TensorCraft | cuDNN | 内存节省 |
|---------|------------|-------|---------|
| 512 | 180 TFLOPs | 200 TFLOPs | 比标准节省 60% |
| 1024 | 210 TFLOPs | 235 TFLOPs | 比标准节省 70% |
| 2048 | 225 TFLOPs | 250 TFLOPs | 比标准节省 80% |

> 性能数据因 GPU 架构和问题规模而异。详见 [benchmarks/](benchmarks/)。

---

## 🎨 GPU 架构支持

| 架构 | SM | Tensor Core | TMA | WGMMA | 示例 GPU |
|------|-----|-------------|-----|-------|---------|
| Volta | 70 | ✅ | ❌ | ❌ | V100 |
| Turing | 75 | ✅ | ❌ | ❌ | RTX 2080 |
| Ampere | 80 | ✅ | ❌ | ❌ | A100, RTX 3090 |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ | RTX 4090 |
| **Hopper** ⭐ | 90 | ✅ | ✅ | ✅ | H100 |
| Blackwell | 100 | ✅ | ✅ | ✅ | B200 |

**TMA**: Tensor Memory Accelerator（张量内存加速器）  
**WGMMA**: Warp Group Matrix Multiply Accumulate（Warp 组矩阵乘累加）

---

## 📚 文档

完整文档请访问 **https://lessup.github.io/modern-ai-kernels/**

### 快速链接

| 章节 | 英文 | 中文 |
|------|------|------|
| 入门指南 | [安装](docs/en/getting-started/installation.md) | [安装指南](docs/zh/getting-started/installation.md) |
| 故障排除 | [常见问题](docs/en/getting-started/troubleshooting.md) | [故障排除](docs/zh/getting-started/troubleshooting.md) |
| 架构指南 | [深入](docs/en/guides/architecture.md) | [架构设计](docs/zh/guides/architecture.md) |
| 优化指南 | [优化级别](docs/en/guides/optimization.md) | [优化级别](docs/zh/guides/optimization.md) |
| API 参考 | [完整 API](docs/en/api/) | [API 参考](docs/zh/api/) |
| 示例教程 | [代码示例](docs/en/examples/) | [代码示例](docs/zh/examples/) |

### 本地文档

```bash
# 本地预览文档
cd docs && bundle install
bundle exec jekyll serve --livereload
# 打开 http://localhost:4000
```

---

## 🏗️ 项目结构

```
modern-ai-kernels/
├── include/tensorcraft/     # 纯头文件库
│   ├── core/               # 核心工具、类型特性
│   ├── kernels/            # GPU 内核实现
│   │   ├── gemm/          # 矩阵乘法内核
│   │   ├── attention/     # 注意力内核
│   │   ├── conv/          # 卷积内核
│   │   ├── normalization/ # 归一化内核
│   │   └── sparse/        # 稀疏操作内核
│   └── memory/             # 内存管理、Tensor 类
├── src/python_ops/         # Python 绑定（pybind11）
├── tests/                  # 单元测试（GoogleTest）
├── benchmarks/             # 性能基准测试
├── examples/               # 示例代码
├── specs/                  # 规范文档（SDD）
│   ├── product/           # 产品需求
│   ├── rfc/               # 技术设计文档
│   └── api/               # API 规范
└── docs/                   # 文档站点
    ├── en/                # 英文文档
    └── zh/                # 中文文档
```

---

## 🔧 构建配置

### CMake 预设

| 预设 | 用途 | 包含 |
|------|------|------|
| `dev` | 开发 | 所有内核 + 测试 |
| `python-dev` | Python 专注 | 核心内核 + 绑定 |
| `release` | 完整发布 | 全部 + 基准测试 |
| `debug` | 调试 | 调试符号、检查 |
| `cpu-smoke` | 验证 | 仅构建系统 |

### 自定义构建

```bash
# 为特定 GPU 手动配置
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DTC_BUILD_TESTS=ON \
  -DTC_BUILD_PYTHON=ON

cmake --build build --parallel $(nproc)
```

---

## 🤝 参与贡献

我们欢迎各种形式的贡献！本项目遵循**规范驱动开发（SDD）**。

### 贡献流程

1. **阅读规范**: 查看 `/specs/` 了解需求
2. **更新规范**: 在代码之前提出变更
3. **实现**: 严格按照规范执行
4. **测试**: 根据规范验收标准编写测试

### 快速链接

- 📖 [贡献指南](docs/zh/reference/contributing.md)
- 🐛 [报告问题](https://github.com/LessUp/modern-ai-kernels/issues)
- 💡 [请求功能](https://github.com/LessUp/modern-ai-kernels/issues)
- 🔀 [提交 PR](docs/zh/reference/contributing.md)

### 开发工作流

```bash
# Fork 并克隆
git clone https://github.com/YOUR_USERNAME/modern-ai-kernels.git
cd modern-ai-kernels

# 创建功能分支
git checkout -b feature/my-kernel

# 实现并测试
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)
ctest --preset dev

# 提交 PR
git push origin feature/my-kernel
```

---

## 📝 许可协议

本项目采用 [MIT 许可证](LICENSE)。

```
MIT License - Copyright (c) 2024-2026 LessUp

特此免费授予任何获得本软件副本及相关文档文件
（统称"软件"）的人无限制地处理软件的权利，
包括但不限于使用、复制、修改、合并、发布、
分发、再许可和/或销售软件副本的权利。
```

---

## 🙏 致谢

TensorCraft-HPC 借鉴了以下项目的思想：

- **CUTLASS**: NVIDIA 的 CUDA 线性代数子程序模板
- **FlashAttention**: 内存高效的注意力算法
- **cuDNN**: NVIDIA 的深度学习库
- **Modern C++**: C++17/20/23 特性和最佳实践
- **CUDA 生态**: CUDA 12.8 和最新 GPU 架构

---

## 📈 项目活跃度

![GitHub commits](https://img.shields.io/github/commit-activity/m/LessUp/modern-ai-kernels)
![GitHub contributors](https://img.shields.io/github/contributors/LessUp/modern-ai-kernels)
![GitHub stars](https://img.shields.io/github/stars/LessUp/modern-ai-kernels?style=social)
![GitHub forks](https://img.shields.io/github/forks/LessUp/modern-ai-kernels?style=social)

---

<div align="center">

**为 AI HPC 社区精心打造 ❤️**

[文档](https://lessup.github.io/modern-ai-kernels/) • [示例](examples/) • [贡献](docs/zh/reference/contributing.md)

</div>
