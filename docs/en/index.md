---
layout: home

hero:
  name: TensorCraft-HPC
  text: Technical Whitepaper / Kernel Architecture Academy
  tagline: A header-only C++/CUDA project that explains how modern AI kernels are designed, optimized, benchmarked, and evaluated, with the reader journey shaped like a systems paper instead of a generic docs portal.
  actions:
    - theme: brand
      text: Read the whitepaper
      link: /en/whitepaper/
    - theme: alt
      text: Enter the academy
      link: /en/academy/
    - theme: alt
      text: Inspect evidence
      link: /en/evidence/

features:
  - icon: ⛭
    title: Architecture as narrative
    details: The site leads from system model to operator detail, so evaluators can understand why the repository is organized the way it is.
  - icon: ▣
    title: Evidence over slogans
    details: Benchmarks, methodology, and references sit next to each claim instead of being buried behind marketing language.
  - icon: ∿
    title: Dual-theme diagrams
    details: Logos, charts, and explanatory graphics are rebuilt to stay legible in both light and dark modes.
---

## Read this project like a systems paper

<ShowcaseBand
  eyebrow="Architecture narrative"
  title="A repository that explains the optimization path"
  description="Move from why the kernels exist, to how the system is layered, to what evidence supports the performance posture."
>
  <div class="tc-home-columns">
    <ul class="tc-home-list">
      <li><strong>Whitepaper first.</strong> Understand the project thesis, kernel philosophy, and constraints before reading implementation detail.</li>
      <li><strong>Evidence second.</strong> Trace performance claims back to methodology and cited work.</li>
      <li><strong>Atlas third.</strong> Drop into operator-level reference only after the system model is clear.</li>
    </ul>
    <div class="tc-band-grid">
      <div class="tc-metric">
        <strong>92%</strong>
        FP16 GEMM relative to cuBLAS on A100
      </div>
      <div class="tc-metric">
        <strong>SM70–SM100</strong>
        Compile-time coverage from Volta to Blackwell
      </div>
      <div class="tc-metric">
        <strong>Whitepaper + Academy</strong>
        Reader flow designed for interview and peer review
      </div>
    </div>
  </div>
</ShowcaseBand>

<ShowcaseBand
  eyebrow="Where to start"
  title="Choose the path that matches what you need to prove"
  description="Each section has a distinct job in the argument, from conceptual framing to implementation inspection."
>
  <ul class="tc-link-list">
    <li><a href="./whitepaper/">Whitepaper</a></li>
    <li><a href="./academy/">Academy</a></li>
    <li><a href="./evidence/">Evidence</a></li>
    <li><a href="./api/gemm">Kernel Atlas</a></li>
    <li><a href="./references/papers">References</a></li>
  </ul>
</ShowcaseBand>

<ShowcaseBand
  eyebrow="Evaluation lens"
  title="What makes TensorCraft-HPC worth evaluating"
  description="The project is designed to surface engineering judgment. The interesting question is not only whether a kernel is fast, but whether the optimization path, system boundaries, and supporting evidence remain legible."
>
  <ul class="tc-home-list">
    <li><strong>Architecture clarity.</strong> Memory abstractions, feature detection, kernel boundaries, and hardware support are explicit design objects.</li>
    <li><strong>Progressive optimization.</strong> Readers can trace kernels from naive baselines to Tensor Core aware implementations.</li>
    <li><strong>Research literacy.</strong> The site connects implementation choices to papers, libraries, and competitor surfaces worth studying.</li>
  </ul>
</ShowcaseBand>

<GPUTimeline />
