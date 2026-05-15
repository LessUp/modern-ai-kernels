---
layout: home

hero:
  name: TensorCraft-HPC
  text: Technical Whitepaper & Architecture Showcase
  tagline: A header-only C++/CUDA project that explains how modern AI kernels are designed, optimized, benchmarked, and reasoned about — from Volta to Blackwell.
  actions:
    - theme: brand
      text: Read the whitepaper
      link: /en/whitepaper/
    - theme: alt
      text: Review architecture
      link: /en/architecture
    - theme: alt
      text: Benchmark evidence
      link: /en/benchmarks/

features:
  - icon: 🎓
    title: Architecture-first learning
    details: The project shows why each kernel layer exists, how optimization stages build on each other, and where the design trade-offs live.
  - icon: ⚡
    title: Evidence over slogans
    details: Performance claims are backed by benchmark pages, methodology notes, and references to the libraries and papers being compared against.
  - icon: 🔧
    title: Header-only kernel library
    details: The C++ surface stays easy to integrate while still exposing optimization paths, memory primitives, and optional Python bindings.
  - icon: 🖥️
    title: Bilingual showcase
    details: English and Chinese routes mirror the same information architecture so the project reads like one coherent technical artifact.
---

<div class="showcase-band">
  <div class="showcase-metrics">
    <div class="showcase-metric">
      <strong>92%</strong>
      <span>FP16 GEMM vs cuBLAS on A100</span>
    </div>
    <div class="showcase-metric">
      <strong>SM70–SM100</strong>
      <span>Compile-time feature detection from Volta to Blackwell</span>
    </div>
    <div class="showcase-metric">
      <strong>Whitepaper + API</strong>
      <span>Project story, implementation evidence, and operator-level reference</span>
    </div>
  </div>
</div>

## Start with the path that matches your goal

<div class="showcase-grid">
  <div class="showcase-card">
    <p class="showcase-eyebrow">For first-time evaluators</p>
    <h3>Read the whitepaper</h3>
    <p>Understand the project motivation, architecture, performance posture, and methodology before diving into the code.</p>
    <div class="showcase-links">
      <a href="/en/whitepaper/">Open whitepaper</a>
    </div>
  </div>
  <div class="showcase-card">
    <p class="showcase-eyebrow">For system design review</p>
    <h3>Inspect the architecture</h3>
    <p>See how the library layers kernel launchers, memory abstractions, compile-time feature detection, and hardware support.</p>
    <div class="showcase-links">
      <a href="/en/architecture">View architecture overview</a>
    </div>
  </div>
  <div class="showcase-card">
    <p class="showcase-eyebrow">For technical credibility</p>
    <h3>Check the evidence</h3>
    <p>Review benchmark summaries, methodology notes, and source materials instead of relying on ungrounded claims.</p>
    <div class="showcase-links">
      <a href="/en/benchmarks/">Benchmarks</a>
      <a href="/en/references/papers">Papers & citations</a>
    </div>
  </div>
  <div class="showcase-card">
    <p class="showcase-eyebrow">For implementation detail</p>
    <h3>Browse the kernel atlas</h3>
    <p>Use the operator reference to trace GEMM, attention, normalization, convolution, sparse, and quantization surfaces.</p>
    <div class="showcase-links">
      <a href="/en/api/gemm">Open kernel atlas</a>
    </div>
  </div>
</div>

## Why this project exists

TensorCraft-HPC is meant to close the gap between "I use CUDA libraries" and "I can explain how a
modern AI kernel is structured, optimized, and validated." Instead of hiding the project behind a
generic docs homepage, this site now treats the repository as a technical artifact: part whitepaper,
part architecture walkthrough, part implementation guide.

That matters for two audiences. For interviewers and technical evaluators, the project should make
engineering judgment visible. For learners and contributors, it should make the optimization path
and design decisions easier to follow than a production-only library would.

## What makes the project worth evaluating

<div class="showcase-grid">
  <div class="showcase-card">
    <h3>Progressive optimization path</h3>
    <p>The library is organized to show how kernels evolve from naive implementations to tiled, pipelined, and Tensor Core aware variants.</p>
  </div>
  <div class="showcase-card">
    <h3>Evidence-driven presentation</h3>
    <p>Benchmarks, diagrams, methodology notes, and references are treated as first-class project assets instead of afterthoughts.</p>
  </div>
  <div class="showcase-card">
    <h3>Architecture that teaches</h3>
    <p>The repository surfaces memory management, feature detection, kernel boundaries, and hardware support as explicit design concerns.</p>
  </div>
</div>
