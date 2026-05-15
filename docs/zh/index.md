---
layout: home

hero:
  name: TensorCraft-HPC
  text: 技术白皮书 / 架构展示站
  tagline: 这是一个用 C++ / CUDA 讲清楚现代 AI 内核如何设计、优化、验证与表达的项目展示面 —— 从 Volta 到 Blackwell。
  actions:
    - theme: brand
      text: 阅读技术白皮书
      link: /zh/whitepaper/
    - theme: alt
      text: 查看架构设计
      link: /zh/architecture
    - theme: alt
      text: 查看性能证据
      link: /zh/benchmarks/

features:
  - icon: 🎓
    title: 架构先于堆料
    details: 不只是展示功能列表，而是先解释层次设计、优化路径、约束条件与工程判断。
  - icon: ⚡
    title: 用证据说话
    details: 性能、方法和参考来源会在 benchmark、whitepaper、papers 页面里串成一条完整证据链。
  - icon: 🔧
    title: 头文件内核库
    details: 保持 C++ 集成轻量，同时把 memory、feature detection、kernel 优化路径完整暴露出来。
  - icon: 🖥️
    title: 双语统一展示
    details: 英文和中文共享同一套信息架构，让整个项目对外表达更完整、更稳定。
---

<div class="showcase-band">
  <div class="showcase-metrics">
    <div class="showcase-metric">
      <strong>92%</strong>
      <span>A100 上 FP16 GEMM 相对 cuBLAS 的表现</span>
    </div>
    <div class="showcase-metric">
      <strong>SM70–SM100</strong>
      <span>覆盖 Volta 到 Blackwell 的编译时特性检测</span>
    </div>
    <div class="showcase-metric">
      <strong>白皮书 + API</strong>
      <span>项目叙事、实现证据与 operator 级参考并列呈现</span>
    </div>
  </div>
</div>

## 从最适合你的入口开始

<div class="showcase-grid">
  <div class="showcase-card">
    <p class="showcase-eyebrow">适合第一次了解项目的人</p>
    <h3>先读白皮书</h3>
    <p>先理解项目动机、架构思路、性能立场和方法论，再进入源码与 API 细节。</p>
    <div class="showcase-links">
      <a href="./whitepaper/">打开白皮书</a>
    </div>
  </div>
  <div class="showcase-card">
    <p class="showcase-eyebrow">适合看系统设计的人</p>
    <h3>再看架构</h3>
    <p>了解内核层、内存抽象、编译时能力检测和硬件支持是如何被组织成一个可讲清楚的系统。</p>
    <div class="showcase-links">
      <a href="./architecture">查看架构概览</a>
    </div>
  </div>
  <div class="showcase-card">
    <p class="showcase-eyebrow">适合评估可信度的人</p>
    <h3>检查证据链</h3>
    <p>不要只看宣传语，直接查看 benchmark 汇总、方法说明和论文 / 项目引用。</p>
    <div class="showcase-links">
      <a href="./benchmarks/">性能基准</a>
      <a href="./references/papers">论文引用</a>
    </div>
  </div>
  <div class="showcase-card">
    <p class="showcase-eyebrow">适合看实现细节的人</p>
    <h3>浏览 Kernel Atlas</h3>
    <p>从 operator 视角查看 GEMM、Attention、Normalization、Convolution、Sparse 和 Quantization 的接口与说明。</p>
    <div class="showcase-links">
      <a href="./api/gemm">打开 Kernel Atlas</a>
    </div>
  </div>
</div>

## 为什么这个项目值得被展示

TensorCraft-HPC 不是单纯把 CUDA kernel 放进仓库里，而是尝试把“如何设计现代 AI
内核”这件事讲清楚。这个站点因此不再只是文档入口，而是把仓库当作一个技术作品来呈现：
既有白皮书式的叙事，也有架构拆解、benchmark 证据和实现层面的参考入口。

这对两类读者都重要。对面试官和技术评审来说，项目需要把工程判断显性化；对学习者和贡献者
来说，项目需要把优化路径和设计取舍讲得比生产库更容易理解。

## 这个项目的亮点应该看什么

<div class="showcase-grid">
  <div class="showcase-card">
    <h3>渐进式优化路径</h3>
    <p>从朴素实现到分块、流水线、Tensor Core 版本，项目强调“为什么这样优化”，而不只是“已经优化好了”。</p>
  </div>
  <div class="showcase-card">
    <h3>证据驱动的展示</h3>
    <p>benchmark、图示、方法论说明和引用资源被当作一等资产，而不是零散附录。</p>
  </div>
  <div class="showcase-card">
    <h3>可讲清楚的工程架构</h3>
    <p>memory、feature detection、kernel 边界和硬件支持被组织成可以讲、可以学、可以评估的结构。</p>
  </div>
</div>
