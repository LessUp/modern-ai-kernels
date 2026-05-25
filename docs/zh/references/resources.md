# 学习资源

精选的 CUDA 编程和 GPU 内核优化学习资源列表。

## 建议阅读顺序

如果你把 TensorCraft-HPC 当作学习项目或面试展示项目，比较推荐这样的阅读路径：

1. 先读项目的 [技术白皮书](/zh/whitepaper/)，理解仓库为什么这样设计。
2. 再用本页向外扩展，阅读 NVIDIA 官方资料和周边开源项目。
3. 最后带着这些外部语境回到仓库的 [架构概览](/zh/architecture) 和 [Kernel Atlas](/zh/api/gemm)。

这样更容易把本项目放到真实生态里去比较，而不是把它孤立地看成一个“只会跑起来”的 demo。

## NVIDIA 官方资源 {#nvidia}

### 文档

- [CUDA C++ 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — CUDA 编程的权威参考
- [CUDA 最佳实践指南](https://docs.nvidia.com/cuda/cuda-best-practices-guide/index.html) — 优化策略和常见陷阱
- [CUDA 分析器工具接口 (CUPTI)](https://docs.nvidia.com/cupti/index.html) — 用于分析 CUDA 应用程序

### 库

- [cuBLAS](https://docs.nvidia.com/cuda/cublas/) — 稠密线性代数
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/) — 深度学习原语
- [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) — 稀疏线性代数
- [NCCL](https://docs.nvidia.com/deeplearning/nccl/) — 多 GPU 通信

### 工具

- [Nsight Compute](https://docs.nvidia.com/nsight-compute/index.html) — 内核分析和剖析
- [Nsight Systems](https://docs.nvidia.com/nsight-systems/index.html) — 系统级分析
- [NVIDIA Visual Profiler](https://developer.nvidia.com/nvidia-visual-profiler) — 传统 GUI 分析器

---

## 开源项目 {#projects}

### 内核库

| 项目 | 重点 | 难度 |
|------|------|------|
| [CUTLASS](https://github.com/NVIDIA/cutlass) | GEMM, Tensor Core | 高级 |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | 注意力 | 高级 |
| [xFormers](https://github.com/facebookresearch/xformers) | 注意力, 内存 | 中级 |
| [Triton](https://github.com/openai/triton) | 内核 DSL | 中级 |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | 训练优化 | 高级 |

### 这些项目和 TensorCraft-HPC 的关系

| 项目 | 为什么值得对比 | TensorCraft-HPC 更强调什么 |
|------|----------------|-----------------------------|
| CUTLASS | CUDA GEMM / Tensor Core 工程的经典参考 | 更清晰的学习路径和优化叙事 |
| FlashAttention | 高质量的 attention 参考实现 | 更容易讲清楚分块和内存取舍 |
| Triton | 另一种 GPU kernel 编写范式 | 直接暴露 C++/CUDA 控制面，适合底层学习 |
| xFormers / DeepSpeed | 更接近真实训练系统的上下文 | 聚焦算子理解，而不是完整训练基础设施 |

### 教育性

| 项目 | 描述 |
|------|------|
| [CUDA Mode](https://github.com/cuda-mode) | CUDA 学习资源 |
| [GPU Mode](https://github.com/gpu-mode) | GPU 编程教程 |
| [Awesome CUDA](https://github.com/Erspamt/awesome-cuda) | CUDA 资源精选 |

---

## 书籍 {#books}

### GPU 编程

- **Programming Massively Parallel Processors** — David B. Kirk, Wen-mei W. Hwu
  - GPU 计算的经典教科书
- **CUDA by Example** — Jason Sanders, Edward Kandrot
  - CUDA 实践入门
- **Professional CUDA C Programming** — John Cheng, Max Grossman, Phil McGachey
  - 高级 CUDA 技术

### 计算机体系结构

- **Computer Architecture: A Quantitative Approach** — Hennessy & Patterson
  - 理解内存层次结构和并行性

---

## 在线课程 {#courses}

- [NVIDIA 深度学习学院](https://www.nvidia.com/en-us/training/) — NVIDIA 官方课程
- [CMU 15-418: Parallel Computer Architecture](http://15418.courses.cs.cmu.edu/) — 优秀的并行计算课程
- [MIT 6.172: Performance Engineering](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/) — 软件性能优化

---

## 关键概念 {#concepts}

### 内存层次结构

```mermaid
flowchart TB
    subgraph Memory["GPU 内存层次结构"]
        REG["寄存器<br/>(最快，每线程)"]
        SHMEM["共享内存<br/>(用户管理缓存)"]
        L1["L1 缓存<br/>(每个 SM)"]
        L2["L2 缓存<br/>(跨 SM 共享)"]
        HBM["HBM/GDDR<br/>(全局内存)"]
    end

    REG --> SHMEM
    SHMEM --> L1
    L1 --> L2
    L2 --> HBM

```

### 执行模型

```mermaid
flowchart LR
    GRID["Grid"] --> BLOCK1["Block 0"]
    GRID --> BLOCK2["Block 1"]
    GRID --> BLOCKN["Block N"]

    BLOCK1 --> WARP1["Warp 0<br/>(32 线程)"]
    BLOCK1 --> WARP2["Warp 1"]
    BLOCK1 --> WARPM["Warp M"]
```

### 优化优先级

1. **最大化并行性** — 足够的线程来隐藏延迟
2. **合并内存访问** — 相邻线程访问相邻内存
3. **共享内存使用** — 减少全局内存流量
4. **避免 Bank 冲突** — 确保共享内存效率
5. **占用率调优** — 平衡寄存器、共享内存、线程

## 这些资源最值得借鉴什么

后续扩展 TensorCraft-HPC 时，最值得吸收的不是“照搬功能”，而是这些项目背后的方法：

- **来自 CUTLASS**：严谨的 tiling 术语和 Tensor Core 分解方式
- **来自 FlashAttention**：围绕 IO 成本展开的叙事与分析方式
- **来自 Triton**：清晰的 operator 级 benchmark 习惯和紧凑示例
- **来自 Nsight 工具链**：用证据解释性能，而不是靠直觉猜测

---

## 哪些值得借鉴，哪些应该克制

让仓库变强的方式，不是盲目模仿风格，而是吸收方法。

### 值得借鉴

- 对 tiling、内存流量、硬件能力的严谨术语
- 会把限制条件和 baseline 一起公开的 benchmark 方法
- 能和 public API 干净映射的紧凑 operator 示例

### 应该克制

- 把优化路径完全藏起来的生产级抽象
- 让教学叙事变得模糊的功能膨胀
- 与工具链、工作负载形状、参考库脱钩的性能结论

这种张力是刻意保留的。TensorCraft-HPC 应该向更强的系统学习，但不应该因此变成另一套难以解释的生产黑盒。

---

## 性能指标 {#metrics}

| 指标 | 描述 | 目标 |
|------|------|------|
| **吞吐量** | 每秒操作数 | Roofline 极限 |
| **延迟** | 每次操作时间 | 最小化 |
| **占用率** | 活动 Warp / 最大 Warp | 50-100% |
| **内存带宽** | 每秒传输字节数 | ~90% 峰值 |
| **计算效率** | 实际 / 峰值 FLOPS | >80% 对于 GEMM |

---

## 常见陷阱 {#pitfalls}

::: warning 内存合并
非合并内存访问可能将带宽降低 10-32 倍。始终确保相邻线程访问相邻内存地址。
:::

::: warning 共享内存 Bank 冲突
当 Warp 中的多个线程访问同一个 Bank 时，访问会被串行化。使用填充或访问模式来避免。
:::

::: warning 分支分歧
Warp 内的分歧分支会顺序执行两条路径。最小化控制流分歧。
:::

::: tip 先分析
在优化之前始终进行分析。使用 Nsight Compute 识别实际瓶颈，而不是猜测。
:::
