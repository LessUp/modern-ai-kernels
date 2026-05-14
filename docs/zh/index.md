---
layout: home

hero:
  name: TensorCraft-HPC
  text: 高性能 AI 内核
  tagline: 现代 GPU 计算技术白皮书 — 从 Volta 到 Blackwell
  actions:
    - theme: brand
      text: 阅读白皮书
      link: /zh/whitepaper/
    - theme: alt
      text: 快速开始
      link: /zh/getting-started
    - theme: alt
      text: API 参考
      link: /zh/api/gemm

features:
  - icon: 🎓
    title: 教育性设计
    details: 从朴素实现到 Tensor Core 的渐进式优化路径，每一步都有清晰的注释。
  - icon: ⚡
    title: 92% cuBLAS 性能
    details: FP16 GEMM 在 A100 上达到行业标准性能，充分利用 Tensor Core。
  - icon: 🔧
    title: 仅头文件架构
    details: 零构建复杂度 — 只需包含头文件。可选 Python 绑定通过 pip install。
  - icon: 🖥️
    title: 多架构支持
    details: 编译时特性检测支持 SM70-SM100，覆盖 Volta 到 Blackwell。
---

<style>
/* Minimal hero styling for professional look */
.VPHero .name {
  color: var(--vp-c-brand-1);
  letter-spacing: -0.5px;
}

.VPHero .tagline {
  font-style: italic;
  color: var(--vp-c-text-3);
}

.VPFeature .title {
  font-weight: 600;
}
</style>