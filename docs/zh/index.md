---
layout: home

hero:
  name: TensorCraft-HPC
  text: 技术白皮书 / 内核架构学院
  tagline: 这是一个用 C++ / CUDA 讲清现代 AI 内核如何设计、优化、验证与评估的项目站点。它的阅读路径更像系统论文，而不是普通文档首页。
  actions:
    - theme: brand
      text: 阅读技术白皮书
      link: /zh/whitepaper/
    - theme: alt
      text: 进入学院
      link: /zh/academy/
    - theme: alt
      text: 查看证据
      link: /zh/evidence/

features:
  - icon: ⛭
    title: 架构即叙事
    details: 先讲系统模型，再讲 operator 细节，让评审者更容易判断工程边界是否清晰。
  - icon: ▣
    title: 证据优先
    details: benchmark、方法、引用与结论并排出现，而不是藏在宣传语后面。
  - icon: ∿
    title: 双主题图示
    details: Logo、图表与说明图会被重建为深浅色都清晰可读的资产。
---

## 像读系统论文一样读这个项目

<ShowcaseBand
  eyebrow="架构叙事"
  title="先理解优化路径，再进入实现细节"
  description="从内核为什么存在、系统如何分层，到性能立场由哪些证据支撑，这个站点会把链路完整展开。"
>
  <div class="tc-home-columns">
    <ul class="tc-home-list">
      <li><strong>先读白皮书。</strong> 先看项目命题、优化哲学与约束，再进入源码细节。</li>
      <li><strong>再查证据。</strong> 把性能结论追溯到方法说明与引用来源。</li>
      <li><strong>最后看 Atlas。</strong> 在系统模型清楚之后，再按 operator 维度看接口与实现。</li>
    </ul>
    <div class="tc-band-grid">
      <div class="tc-metric">
        <strong>92%</strong>
        A100 上 FP16 GEMM 相对 cuBLAS 的结果
      </div>
      <div class="tc-metric">
        <strong>SM70–SM100</strong>
        从 Volta 到 Blackwell 的编译时能力覆盖
      </div>
      <div class="tc-metric">
        <strong>白皮书 + 学院</strong>
        面向面试评估与高级开发者审阅的阅读路径
      </div>
    </div>
  </div>
</ShowcaseBand>

<ShowcaseBand
  eyebrow="入口选择"
  title="按照你要验证的问题进入对应模块"
  description="每个模块在整套论证里承担不同角色，从概念 framing 到实现检查，各司其职。"
>
  <ul class="tc-link-list">
    <li><a href="./whitepaper/">技术白皮书</a></li>
    <li><a href="./academy/">学院</a></li>
    <li><a href="./evidence/">证据</a></li>
    <li><a href="./api/gemm">Kernel Atlas</a></li>
    <li><a href="./references/papers">参考</a></li>
  </ul>
</ShowcaseBand>

<ShowcaseBand
  eyebrow="评估视角"
  title="为什么 TensorCraft-HPC 值得被认真评估"
  description="真正有价值的不只是 kernel 快不快，而是优化路径能不能被讲清楚，系统边界是否明确，证据纪律是否可靠。"
>
  <ul class="tc-home-list">
    <li><strong>架构清晰度。</strong> memory abstraction、feature detection、kernel 边界、硬件支持被显性建模。</li>
    <li><strong>渐进式优化。</strong> 读者能从朴素实现一路追到 Tensor Core aware 版本。</li>
    <li><strong>研究素养。</strong> 站点会把实现判断和论文、库生态、竞品项目串联起来。</li>
  </ul>
</ShowcaseBand>

<GPUTimeline />
