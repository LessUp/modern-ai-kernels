# 快速开始

欢迎使用 TensorCraft-HPC！本指南将帮助你快速上手。

## 前置条件 {#prerequisites}

- **C++17** 或更高版本的兼容编译器
- **CUDA Toolkit 11.0+** (推荐 12.0+ 以支持 FP8)
- **CMake 3.20+**
- **Python 3.9+** (可选，用于 Python 绑定)

### 硬件要求

| GPU 架构 | 计算能力 | 最低 CUDA 版本 |
|----------|----------|----------------|
| Volta | SM70 (7.0) | 9.0 |
| Turing | SM75 (7.5) | 10.0 |
| Ampere | SM80/SM86 (8.0/8.6) | 11.0 |
| Ada Lovelace | SM89 (8.9) | 11.8 |
| Hopper | SM90 (9.0) | 12.0 |
| Blackwell | SM100 (10.0) | 12.4 |

---

## 安装 {#installation}

### 仅头文件 (推荐)

TensorCraft-HPC 是一个仅头文件库。只需将头文件包含到你的项目中：

```bash
# 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git

# 复制头文件到你的项目
cp -r modern-ai-kernels/include/tensorcraft your_project/include/
```

```cpp
// 在你的代码中
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"
```

### CMake 集成

对于使用 CMake 的项目：

```cmake
# 添加包含目录
target_include_directories(your_target PRIVATE
    path/to/modern-ai-kernels/include
)

# 链接 CUDA
find_package(CUDA REQUIRED)
target_link_libraries(your_target CUDA::cudart)
```

### Python 绑定

```bash
# 构建并安装 Python 包
cd modern-ai-kernels
pip install -e .
```

---

## 快速示例 {#examples}

### GEMM (矩阵乘法)

::: code-group
```cpp [C++]
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

int main() {
    // 创建张量 (RAII 管理的 GPU 内存)
    tensorcraft::FloatTensor A({1024, 1024});
    tensorcraft::FloatTensor B({1024, 1024});
    tensorcraft::FloatTensor C({1024, 1024});

    // 用随机数据初始化
    A.random_fill();
    B.random_fill();

    // 执行 GEMM: C = A × B
    tensorcraft::kernels::gemm(
        A.data(), B.data(), C.data(),
        1024, 1024, 1024  // M, N, K
    );

    return 0;
}
```

```python [Python]
import tensorcraft_ops as tc
import numpy as np

# 创建矩阵
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

# GPU 加速 GEMM
C = tc.gemm(A, B)
```
:::

### FlashAttention

::: code-group
```cpp [C++]
#include "tensorcraft/kernels/attention.hpp"
#include "tensorcraft/memory/tensor.hpp"

int main() {
    // 批大小、序列长度、头维度
    int batch = 32, seq_len = 128, head_dim = 64;

    // Q, K, V 张量
    tensorcraft::FloatTensor Q({batch, seq_len, head_dim});
    tensorcraft::FloatTensor K({batch, seq_len, head_dim});
    tensorcraft::FloatTensor V({batch, seq_len, head_dim});
    tensorcraft::FloatTensor O({batch, seq_len, head_dim});

    // FlashAttention
    tensorcraft::kernels::flash_attention(
        Q.data(), K.data(), V.data(), O.data(),
        batch, seq_len, head_dim
    );

    return 0;
}
```

```python [Python]
import tensorcraft_ops as tc
import numpy as np

batch, seq_len, head_dim = 32, 128, 64

Q = np.random.randn(batch, seq_len, head_dim).astype(np.float32)
K = np.random.randn(batch, seq_len, head_dim).astype(np.float32)
V = np.random.randn(batch, seq_len, head_dim).astype(np.float32)

# FlashAttention
output = tc.flash_attention(Q, K, V)
```
:::

---

## 构建预设 {#presets}

TensorCraft-HPC 包含常用配置的 CMake 预设：

```bash
# 仅 CPU 冒烟测试 (无需 GPU)
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke

# 带 CUDA 的开发构建
cmake --preset dev
cmake --build --preset dev --parallel 4

# 包含所有优化的发布构建
cmake --preset release
cmake --build --preset release

# 运行测试
ctest --preset dev --output-on-failure
```

---

## 下一步 {#next-steps}

- 阅读 [架构概览](/zh/architecture) 了解设计理念
- 浏览 [API 参考](/zh/api/gemm) 获取详细内核文档
- 查看 [论文引用](/zh/references/papers) 获取学术参考
- 参考 [学习资源](/zh/references/resources) 获取 CUDA 教程