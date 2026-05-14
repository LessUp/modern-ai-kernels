---
layout: home

hero:
  name: TensorCraft-HPC
  text: High-Performance AI Kernels
  tagline: A Technical Whitepaper on Modern GPU Computing — From Volta to Blackwell
  actions:
    - theme: brand
      text: Read Whitepaper
      link: /en/whitepaper/
    - theme: alt
      text: Getting Started
      link: /en/getting-started
    - theme: alt
      text: API Reference
      link: /en/api/gemm

features:
  - icon: 🎓
    title: Educational Design
    details: Progressive optimization paths from naive to Tensor Core, with clear annotations at every step.
  - icon: ⚡
    title: 92% cuBLAS Performance
    details: FP16 GEMM achieves industry-standard performance on A100 with full Tensor Core utilization.
  - icon: 🔧
    title: Header-Only Architecture
    details: Zero build complexity — simply include headers. Optional Python bindings via pip install.
  - icon: 🖥️
    title: Multi-Architecture Support
    details: Compile-time feature detection for SM70-SM100, covering Volta through Blackwell.
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