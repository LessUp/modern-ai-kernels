---
layout: default
title: Home
nav_order: 1
permalink: /
---

<!-- ═══════════════════════════════════════════════════════════════ -->
<!-- TENSORCRAFT-HPC 究极着陆页 -->
<!-- ═══════════════════════════════════════════════════════════════ -->

<div class="hero-section">
  <h1 class="hero-title">TensorCraft-HPC</h1>
  <p class="hero-subtitle">
    现代 C++/CUDA AI 高性能计算内核库<br>
    <span style="opacity: 0.8; font-size: 0.9em;">Modern C++/CUDA AI Kernel Library for HPC</span>
  </p>
  
  <div class="hero-badges">
    <img src="https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg" alt="GitHub Pages">
    <img src="https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg" alt="CI">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
    <img src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white" alt="CUDA">
    <img src="https://img.shields.io/badge/C%2B%2B-17%2F20%2F23-00599C?logo=c%2B%2B&logoColor=white" alt="C++">
  </div>
  
  <div class="hero-cta">
    <a href="docs/en/" class="btn btn-primary">🚀 Get Started</a>
    <a href="https://github.com/LessUp/modern-ai-kernels" class="btn btn-secondary">⭐ GitHub</a>
  </div>
</div>

---

## 🌍 Choose Your Language | 选择您的语言

<div class="lang-selector">
  <a href="docs/en/" class="lang-card">
    <span class="lang-flag">🇺🇸</span>
    <span class="lang-info">
      <span class="lang-name">English</span>
      <span class="lang-desc">Complete documentation in English</span>
    </span>
  </a>
  <a href="docs/zh/" class="lang-card">
    <span class="lang-flag">🇨🇳</span>
    <span class="lang-info">
      <span class="lang-name">简体中文</span>
      <span class="lang-desc">完整中文文档</span>
    </span>
  </a>
</div>

---

## ✨ Features | 核心特性

<div class="features-grid">
  <div class="feature-card" style="animation-delay: 0.1s;">
    <div class="feature-icon icon-gemm">🔢</div>
    <h3>GEMM Kernels</h3>
    <p>Naive → Tiled → Double Buffer → Tensor Core (WMMA)<br>完整矩阵乘法优化之旅</p>
  </div>
  
  <div class="feature-card" style="animation-delay: 0.2s;">
    <div class="feature-icon icon-attention">🎯</div>
    <h3>Attention</h3>
    <p>FlashAttention-style, RoPE, MoE Router<br>内存高效的注意力计算</p>
  </div>
  
  <div class="feature-card" style="animation-delay: 0.3s;">
    <div class="feature-icon icon-norm">📊</div>
    <h3>Normalization</h3>
    <p>LayerNorm, RMSNorm, BatchNorm, Softmax<br>标准化层完整支持</p>
  </div>
  
  <div class="feature-card" style="animation-delay: 0.4s;">
    <div class="feature-icon icon-conv">🌀</div>
    <h3>Convolution</h3>
    <p>Naive, Im2Col, Depthwise Separable<br>2D卷积操作优化</p>
  </div>
  
  <div class="feature-card" style="animation-delay: 0.5s;">
    <div class="feature-icon icon-sparse">🌐</div>
    <h3>Sparse Operations</h3>
    <p>CSR/CSC formats, SpMV, SpMM<br>稀疏矩阵高效计算</p>
  </div>
  
  <div class="feature-card" style="animation-delay: 0.6s;">
    <div class="feature-icon icon-quant">⚡</div>
    <h3>Quantization</h3>
    <p>INT8 and FP8 (CUDA 12.0+)<br>量化推理加速</p>
  </div>
  
  <div class="feature-card" style="animation-delay: 0.7s;">
    <div class="feature-icon icon-python">🐍</div>
    <h3>Python Bindings</h3>
    <p>NumPy-compatible interface via pybind11<br>Python 友好的 API</p>
  </div>
</div>

---

## 🚀 Quick Start | 快速开始

```bash
# Clone the repository | 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Build | 构建
cmake --preset dev
cmake --build --preset dev --parallel 2

# Run tests | 运行测试
ctest --preset dev --output-on-failure

# Install Python bindings | 安装 Python 绑定
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

---

## 🎨 GPU Architecture Support | GPU 架构支持

<table class="gpu-table">
  <thead>
    <tr>
      <th>Architecture</th>
      <th>SM</th>
      <th>Tensor Core</th>
      <th>TMA</th>
      <th>WGMMA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Volta</strong></td>
      <td>70</td>
      <td><span class="badge badge-support">✓</span></td>
      <td><span class="badge badge-unsupported">✗</span></td>
      <td><span class="badge badge-unsupported">✗</span></td>
    </tr>
    <tr>
      <td><strong>Turing</strong></td>
      <td>75</td>
      <td><span class="badge badge-support">✓</span></td>
      <td><span class="badge badge-unsupported">✗</span></td>
      <td><span class="badge badge-unsupported">✗</span></td>
    </tr>
    <tr>
      <td><strong>Ampere</strong></td>
      <td>80</td>
      <td><span class="badge badge-support">✓</span></td>
      <td><span class="badge badge-unsupported">✗</span></td>
      <td><span class="badge badge-unsupported">✗</span></td>
    </tr>
    <tr>
      <td><strong>Ada Lovelace</strong></td>
      <td>89</td>
      <td><span class="badge badge-support">✓</span></td>
      <td><span class="badge badge-unsupported">✗</span></td>
      <td><span class="badge badge-unsupported">✗</span></td>
    </tr>
    <tr>
      <td><strong>Hopper</strong> ⭐</td>
      <td>90</td>
      <td><span class="badge badge-support">✓</span></td>
      <td><span class="badge badge-support">✓</span></td>
      <td><span class="badge badge-support">✓</span></td>
    </tr>
  </tbody>
</table>

---

## 📚 Documentation Structure | 文档结构

```
docs/
├── en/                          # English Documentation
│   ├── getting-started/         # Installation & troubleshooting
│   ├── guides/                  # Architecture & optimization
│   ├── api/                     # API reference
│   ├── examples/                # Code examples
│   └── reference/               # Contributing & policies
│
└── zh/                          # 简体中文文档
    ├── getting-started/         # 安装与故障排除
    ├── guides/                  # 架构设计与优化
    ├── api/                     # API 参考
    ├── examples/                # 代码示例
    └── reference/               # 贡献指南与规范
```

---

## 💡 Key Highlights | 核心亮点

<div class="tip">
<b>Header-Only Design</b> | 纯头文件设计
<br>
Just include and use. No separate compilation needed. 只需包含即可使用，无需单独编译。
</div>

<div class="note">
<b>Progressive Optimization</b> | 渐进式优化
<br>
Learn kernel optimization step by step: Naive → Tiled → Double Buffer → Tensor Core
<br>
逐步学习内核优化：朴素实现 → 平铺 → 双缓冲 → 张量核心
</div>

<div class="important">
<b>Modern C++ & CUDA</b> | 现代 C++ 与 CUDA
<br>
Leveraging C++17/20/23 features with CUDA 12.8 for maximum performance
<br>
利用 C++17/20/23 特性和 CUDA 12.8 实现最大性能
</div>

---

## 📈 Version History | 版本历史

| Version | Date | Description |
|:-------:|:----:|-------------|
| **v3.0.0** | 2025-04 | 🌍 Bilingual documentation, GitHub Pages ultra optimization |
| **v2.0.0** | 2026-03 | 🔧 Critical bug fixes, architecture improvements |
| **v1.1.0** | 2026-01 | 🔨 Build system fixes |
| **v1.0.1** | 2025-02 | 📋 Project infrastructure |
| **v1.0.0** | 2024-01 | 🎉 Initial release |

[View Full Changelog →](CHANGELOG.md){: .btn .btn-primary }

---

## 🤝 Contributing | 参与贡献

We welcome contributions! See our [Contributing Guide](docs/en/reference/contributing.md) to get started.

欢迎各种形式的贡献！查看我们的[贡献指南](docs/zh/reference/contributing.md)开始参与。

---

<p align="center">
  <b>Made with ❤️ for the AI HPC community</b><br>
  <span style="opacity: 0.7;">为 AI HPC 社区精心打造</span>
</p>

<p align="center">
  <a href="https://github.com/LessUp/modern-ai-kernels">GitHub</a> •
  <a href="docs/en/">English Docs</a> •
  <a href="docs/zh/">中文文档</a> •
  <a href="CHANGELOG.md">Changelog</a>
</p>
