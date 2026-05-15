# 架构概览

本文档描述 TensorCraft-HPC 的高层架构。

## 设计理念 {#philosophy}

TensorCraft-HPC 遵循三个核心原则：

1. **可读性优先** — 代码是为了阅读的。每个内核展示优化进程。
2. **仅头文件** — C++ 用户零构建复杂度。只需包含即可使用。
3. **OpenSpec 驱动** — `openspec/specs/` 中的规范是权威来源。

---

## 系统架构 {#system}

```mermaid
flowchart TB
    subgraph UserAPI["用户 API 层"]
        direction TB
        CPP["C++ 头文件<br/>(仅头文件)"]
        PY["Python 绑定<br/>(tensorcraft_ops)"]
    end

    subgraph KernelLayer["内核层"]
        direction TB
        GEMM["GEMM 内核<br/>(朴素 → Tensor Core)"]
        ATTN["Attention 内核<br/>(FlashAttention)"]
        NORM["归一化<br/>(LayerNorm, RMSNorm)"]
        CONV["卷积<br/>(Im2Col, Winograd)"]
        SPARSE["稀疏操作<br/>(CSR, CSC)"]
        QUANT["量化<br/>(INT8, FP8)"]
    end

    subgraph MemoryLayer["内存层"]
        direction TB
        TENSOR["FloatTensor<br/>(RAII GPU 内存)"]
        POOL["MemoryPool<br/>(可选池化)"]
        ALIGNED["AlignedVector<br/>(缓存友好)"]
    end

    subgraph CoreLayer["核心工具"]
        direction TB
        CUDA_CHECK["cuda_check.hpp<br/>(错误处理)"]
        FEATURES["features.hpp<br/>(编译时检测)"]
        TYPE_TRAITS["type_traits.hpp<br/>(类型工具)"]
        WARP_UTILS["warp_utils.hpp<br/>(Warp 原语)"]
    end

    subgraph Hardware["硬件抽象"]
        SM70["SM70<br/>(Volta)"]
        SM75["SM75<br/>(Turing)"]
        SM80["SM80<br/>(Ampere)"]
        SM90["SM90<br/>(Hopper)"]
        SM100["SM100<br/>(Blackwell)"]
    end

    CPP --> KernelLayer
    PY --> KernelLayer
    KernelLayer --> MemoryLayer
    MemoryLayer --> CoreLayer
    CoreLayer --> Hardware
```

---

## 目录结构 {#directories}

```
modern-ai-kernels/
├── include/tensorcraft/       # 仅头文件库
│   ├── core/                  # 工具 (错误处理, 类型特征)
│   │   ├── cuda_check.hpp     # CUDA 错误检查宏
│   │   ├── features.hpp       # 编译时 GPU 特性检测
│   │   ├── type_traits.hpp    # 类型操作工具
│   │   └── warp_utils.hpp     # Warp 级原语
│   ├── memory/                # 内存管理
│   │   ├── tensor.hpp         # RAII GPU 张量包装
│   │   ├── memory_pool.hpp    # 可选内存池化
│   │   └── aligned_vector.hpp # 缓存对齐向量
│   └── kernels/               # 所有计算内核
│       ├── gemm.hpp           # 矩阵乘法
│       ├── attention.hpp      # 注意力机制
│       ├── normalization.hpp  # LayerNorm, RMSNorm 等
│       ├── softmax.hpp        # Softmax 变体
│       ├── conv2d.hpp         # 2D 卷积
│       ├── sparse.hpp         # 稀疏操作
│       ├── fusion.hpp         # 融合内核
│       ├── elementwise.hpp    # ReLU, GeLU 等
│       ├── memory_ops.hpp     # 复制, 转置
│       └── fusion.hpp         # 融合算子与量化辅助能力
├── src/python_ops/            # Python 绑定 (pybind11)
├── tests/                     # 单元测试 (GoogleTest)
├── benchmarks/                # 性能基准
├── examples/                  # 使用示例
├── docs/                      # VitePress 文档
└── openspec/                  # 规范工作流
    ├── specs/                 # 已接受规范
    ├── changes/               # 活动变更提案
    └── archive/               # 已完成变更
```

---

## GEMM 优化路径 {#gemm-path}

GEMM 内核演示了渐进式优化方法：

```mermaid
flowchart LR
    A["朴素<br/>(O(N³) 全局内存)"]
    B["分块<br/>(共享内存)"]
    C["双缓冲<br/>(重叠复制/计算)"]
    D["Tensor Core<br/>(WMMA)"]
    E["cuBLAS 同等<br/>(85-95%)"]

    A -->|"分块以<br/>重用"| B
    B -->|"双缓冲<br/>以重叠"| C
    C -->|"使用 Tensor Core<br/>(WMMA)"| D
    D -->|"精细调整<br/>参数"| E

    style A fill:#F4F7F1,stroke:#2E7D32,color:#1A1A1A
    style B fill:#F4F7F1,stroke:#2E7D32,color:#1A1A1A
    style C fill:#F4F7F1,stroke:#2E7D32,color:#1A1A1A
    style D fill:#76B900,stroke:#5A9100,color:#000
    style E fill:#76B900,stroke:#5A9100,color:#000
```

### 性能特征

| 阶段 | 内存流量 | 计算效率 | 相对速度 |
|------|----------|----------|----------|
| 朴素 | O(N³) 全局 | ~1% | 1x |
| 分块 | O(N²) 全局 | ~10% | 10x |
| 双缓冲 | O(N²) 全局 | ~30% | 30x |
| Tensor Core | O(N²) 全局 | ~80% | 80x |

---

## FlashAttention 实现 {#flash-attention}

```mermaid
sequenceDiagram
    participant Host as 主机
    participant SRAM as 共享内存
    participant Reg as 寄存器
    participant DRAM as 全局内存

    Note over Host,DRAM: FlashAttention 分块

    Host->>DRAM: 分配 Q, K, V, O
    loop 对每个分块
        DRAM->>SRAM: 加载 Q_tile, K_tile, V_tile
        SRAM->>Reg: 加载到寄存器
        Reg->>Reg: 计算 QK^T (部分)
        Reg->>Reg: 更新 softmax 状态
        Reg->>Reg: 累积输出
        Reg->>SRAM: 存储部分 O
    end
    SRAM->>DRAM: 写入最终 O

    Note over Host,DRAM: 内存: O(N²) 而非 O(N²d)
```

### 关键创新

1. **分块** — 处理适合 SRAM 的注意力分块
2. **在线 Softmax** — 增量更新 softmax 统计信息
3. **重计算** — 重计算注意力权重而非存储

---

## 内存管理 {#memory}

### RAII 模式

```cpp
// 自动内存管理
{
    tensorcraft::FloatTensor A({4096, 4096});
    // 使用 A...
} // 作用域退出时自动释放
```

### 内存池 (可选)

```mermaid
flowchart LR
    REQ["内核请求"] --> POOL{"池可用?"}
    POOL -->|"是"| ALLOC["从池返回"]
    POOL -->|"否"| NEW["cudaMalloc"]
    NEW --> POOL
    ALLOC --> KERNEL["内核执行"]
    KERNEL --> RET["返回池"]
    RET --> POOL
```

---

## 编译时特性检测 {#features}

`features.hpp` 头文件提供编译时 GPU 能力检测：

```cpp
// 编译时自动检测
#if TENSORCRAFT_HAS_WMMA
    // 使用 Tensor Core (SM70+)
#endif

#if TENSORCRAFT_HAS_FP8
    // 使用 FP8 类型 (SM90+)
#endif

#if TENSORCRAFT_HAS_TMA
    // 使用张量内存加速器 (SM90+)
#endif
```

---

## OpenSpec 工作流 {#openspec}

```mermaid
flowchart TB
    IDEA["新想法"] --> PROPOSAL["创建提案<br/>openspec/changes/"]
    PROPOSAL --> REVIEW["审查和讨论"]
    REVIEW -->|"接受"| SPEC["移至<br/>openspec/specs/"]
    REVIEW -->|"拒绝"| ARCHIVE["归档并<br/>附理由"]
    SPEC --> IMPL["实现"]
    IMPL --> VERIFY["验证规范"]
    VERIFY --> DONE["完成"]
```

### 规范结构

`openspec/specs/` 中的每个规范包含：
- **需求** — 组件必须做什么
- **契约** — API 保证和不变量
- **验收标准** — 如何验证合规性

---

## 测试策略 {#testing}

| 级别 | 工具 | 目的 |
|------|------|------|
| 单元 | GoogleTest | 每个内核正确性 |
| 集成 | pytest | Python 绑定 |
| 基准 | Google Benchmark | 性能回归 |
| 验证 | 自定义 | 数值精度 |

### 运行测试

```bash
# 所有测试
ctest --preset dev --output-on-failure

# 特定内核
ctest --preset dev -R gemm

# 基准测试
cmake --preset release
cmake --build --preset release --parallel 2
./build/release/benchmarks/gemm_benchmark
```
