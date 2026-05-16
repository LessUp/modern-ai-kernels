# 技术白皮书

::: abstract
**摘要**

TensorCraft-HPC 是一个仅头文件的 C++/CUDA 库，专为学习高性能 AI 内核实现而设计。本白皮书阐述了指导项目的架构决策、优化策略和性能分析。我们的目标是通过提供从朴素实现到生产级性能的清晰、渐进式优化路径，来解密 GPU 内核开发。

**关键成果**

- FP16 GEMM 在 Tensor Core 上达到 92% cuBLAS 性能
- FlashAttention 达到 85% cuDNN 性能
- 支持 NVIDIA SM70-SM100 架构
- 通过仅头文件设计实现零构建复杂度
:::

---

## 执行摘要

现代 AI 系统关键依赖于高性能 GPU 内核来执行矩阵乘法、注意力和归一化等操作。然而，从理解数学到实现生产级性能的路径往往被复杂性所遮蔽。

TensorCraft-HPC 通过以下方式解决这一差距：

1. **显式演进**：每个内核经历明确定义的优化阶段
2. **教育性清晰**：代码针对可读性优化，不仅是性能
3. **OpenSpec 治理**：规范驱动实现，确保正确性

---

## 项目哲学

### 为什么存在此项目

CUDA 生态系统有优秀的生产库（cuBLAS、cuDNN、CUTLASS），但它们针对部署而非学习进行优化。当开发者问"如何编写高效的 GEMM 内核？"时，答案往往指向数千行模板元编程代码。

TensorCraft-HPC 提供了另一种选择：从简单开始、逐步演进的内核，每一步优化都有理由和解释。

### 设计原则

| 原则 | 含义 |
|------|------|
| **可读性优先** | 代码注释解释*为什么*，不只是*是什么* |
| **渐进式复杂度** | 每个阶段都是完整、可运行的内核 |
| **规范驱动** | OpenSpec 文件在实现前定义契约 |
| **零构建摩擦** | C++ 仅头文件，Python 可选 pip |

---

## 核心贡献

### 1. 渐进式优化框架

每个内核遵循文档化的优化路径：

```
朴素 → 分块 → 双缓冲 → Tensor Core → 生产级性能
```

每个阶段：
- 是完整、可测试的实现
- 有明确的性能特征
- 演示特定的优化技术

### 2. 多架构支持

编译时特性检测启用：

```cpp
#if TENSORCRAFT_HAS_WMMA
    // Tensor Core 路径 (SM70+)
#elif TENSORCRAFT_HAS_FP8
    // FP8 路径 (SM90+)
#else
    // 后备路径
#endif
```

### 3. OpenSpec 工作流

`openspec/specs/` 中的规范定义：

- **需求 (Requirements)**：组件必须做什么
- **契约 (Contracts)**：API 保证和不变量
- **验收标准 (Acceptance Criteria)**：如何验证合规性

---

## 这个项目不是什么

TensorCraft-HPC **不是** 要取代 cuBLAS、cuDNN、CUTLASS 或 Triton 这类完整生产级内核栈，
也不是为了追求功能数量而不断堆范围。

它更在意的是一种更少见的组合：

- 代码在开始优化之后仍然可以读懂
- benchmark 数字始终和方法说明、限制条件绑在一起
- 架构设计能够在面试或设计评审里被讲清楚
- 文档不仅告诉你“怎么用”，还解释“为什么会这样实现”

## 应该从哪些维度评估这个仓库

| 维度 | 应该看什么 |
|------|------------|
| **架构设计** | kernel 层、memory abstraction、feature detection、public surface 是否边界清晰 |
| **实现质量** | 是否保留了渐进式优化过程，而不是只剩下难以解释的最终形态 |
| **证据纪律** | benchmark 数字是否和方法、引用、限制条件一起出现 |
| **项目一致性** | README、GitHub Pages、OpenSpec、workflow 是否在讲同一个故事 |

---

## 演进思考

TensorCraft-HPC 有意把“可讲清楚”放在“范围最大化”之前。项目的演进方向，不只是增加 kernel 种类，而是让优化路径更容易理解，让 benchmark 结论更可辩护，让架构边界更适合在设计评审或面试中被解释清楚。

## 相关开源项目探究

这个项目与以下重要表面形成对话关系：

- **CUTLASS**，代表高度优化、模板密集的 CUDA kernel 构建方式
- **Triton**，代表编译器驱动的 kernel authoring 与研究工作流
- **FlashAttention**，代表注意力算法与系统协同设计
- **cuBLAS / cuDNN**，代表生产级 baseline 与性能参考面

这里引用它们，不是为了营销式对比，而是为了帮助读者理解为什么 TensorCraft-HPC 会更强调教学性、架构可见性与 benchmark 纪律。

---

## 目标读者

本白皮书面向：

- **GPU 内核开发者**：寻求理解优化技术
- **ML 基础设施工程师**：评估内核实现
- **研究人员**：研究高性能计算模式
- **学生**：学习 CUDA 编程

---

## 文档结构

| 章节 | 内容 |
|------|------|
| [架构设计](/zh/whitepaper/architecture) | 系统设计、分层和扩展点 |
| [性能分析](/zh/whitepaper/performance) | 基准测试方法和分析 |
| [开发方法论](/zh/whitepaper/methodology) | OpenSpec 工作流和贡献指南 |
| [论文引用](/zh/references/papers) | 支撑设计判断的论文与生态参考 |

---

## 快速开始

::: code-group
```bash [克隆]
git clone https://github.com/AICL-Lab/modern-ai-kernels.git
cd modern-ai-kernels
```

```cpp [C++]
#include "tensorcraft/kernels/gemm.hpp"

tensorcraft::FloatTensor A({4096, 4096});
tensorcraft::FloatTensor B({4096, 4096});
tensorcraft::FloatTensor C({4096, 4096});

tensorcraft::kernels::gemm(A.data(), B.data(), C.data(), 4096, 4096, 4096);
```

```python [Python]
import tensorcraft_ops as tc
import numpy as np

A = np.random.randn(4096, 4096).astype(np.float16)
B = np.random.randn(4096, 4096).astype(np.float16)
C = tc.gemm(A, B)  # GPU 加速
```
:::

---

## 引用

如果您在学术工作中引用 TensorCraft-HPC：

```bibtex
@software{tensorcraft-hpc,
  title = {TensorCraft-HPC: Demystifying High-Performance AI Kernels
           with Modern C++ and CUDA},
  author = {TensorCraft-HPC Contributors},
  year = {2024},
  url = {https://github.com/AICL-Lab/modern-ai-kernels}
}
