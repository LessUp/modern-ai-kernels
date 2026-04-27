---
title: 安装指南
parent: 快速开始
nav_order: 1
---

# 安装指南

TensorCraft-HPC 在不同使用场景下的完整安装说明。

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 内存 | 4 GB | 16 GB |
| 磁盘 | 2 GB 可用空间 | 10 GB 可用空间 |
| GPU | NVIDIA (计算能力 70+) | NVIDIA Hopper (SM 90) |

### 软件要求

#### 必需

| 工具 | 版本 | 用途 |
|------|------|------|
| **CUDA Toolkit** | 12.0+ | GPU 内核编译和运行时 |
| **CMake** | 3.20+ | 构建系统 |
| **C++ 编译器** | 支持 C++17 | 主机代码编译 |
| **NVIDIA 驱动** | 520+ | GPU 运行时支持 |

#### 可选

| 工具 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.8+ | Python 绑定 |
| **Ninja** | 1.10+ | 更快的构建生成 |
| **GoogleTest** | 1.10+ | 单元测试（自动获取） |
| **pybind11** | 2.10+ | Python 绑定（自动获取） |

## 按平台安装

### Ubuntu / Debian Linux

#### 1. 安装依赖

```bash
# 更新软件包列表
sudo apt update

# 安装 CUDA Toolkit（如尚未安装）
# 方式 A: 官方 NVIDIA .deb 仓库
# 参见: https://developer.nvidia.com/cuda-downloads

# 方式 B: 直接安装
sudo apt install -y cuda-toolkit-12-8

# 安装构建工具
sudo apt install -y cmake build-essential

# 安装 Python（可选，用于绑定）
sudo apt install -y python3 python3-pip python3-dev
```

#### 2. 验证安装

```bash
# 检查 CUDA
nvcc --version
# 应显示: Cuda compilation tools, release 12.x

# 检查 CMake
cmake --version
# 应显示: cmake version 3.20 或更高

# 检查 Python（如已安装）
python3 --version
# 应显示: Python 3.8 或更高
```

#### 3. 构建 TensorCraft-HPC

```bash
# 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 配置并构建
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)

# 运行测试
ctest --preset dev --output-on-failure
```

### macOS（仅 CPU）

{: .warning }
TensorCraft-HPC 是基于 CUDA 的库。在 macOS 上，只能使用 CPU-only 验证和文档构建。

```bash
# 克隆仓库
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# 安装依赖
brew install cmake

# CPU-only 验证
cmake --preset cpu-smoke
cmake --build build/cpu-smoke --parallel $(sysctl -n hw.ncpu)
```

### Windows

{: .warning }
Windows 需要安装 Visual Studio 和 CUDA Toolkit。推荐使用 WSL2。

#### 使用 WSL2（推荐）

```bash
# 安装带 Ubuntu 的 WSL2
wsl --install

# 在 WSL2 内，按照上述 Ubuntu 说明操作
```

#### 原生 Windows（Visual Studio）

1. 安装 [Visual Studio 2022](https://visualstudio.microsoft.com/)
2. 安装 [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads?target_os=Windows)
3. 打开开发者命令提示符

```cmd
cmake -B build -G "Visual Studio 17 2022" ^
  -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -DTC_BUILD_TESTS=ON

cmake --build build --config Release --parallel
```

## 构建预设说明

TensorCraft-HPC 为不同使用场景提供了多个 CMake 预设：

### `dev` - 开发（推荐）

**适用于**：日常 CUDA 开发

```bash
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)
ctest --preset dev --output-on-failure
```

**包含内容**：

- ✅ 所有 GPU 内核
- ✅ 单元测试
- ✅ 调试符号
- ❌ 基准测试（节省构建时间）

### `python-dev` - Python 开发

**适用于**：专注于 Python 绑定

```bash
cmake --preset python-dev
cmake --build --preset python-dev --parallel $(nproc)
python3 -m pip install -e .
python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

**包含内容**：

- ✅ Python 绑定
- ✅ Python API 所需的核心 GPU 内核
- ❌ 完整测试套件
- ❌ 基准测试

### `release` - 完整发布

**适用于**：包含基准测试的完整构建

```bash
cmake --preset release
cmake --build --preset release --parallel $(nproc)
ctest --test-dir build/release --output-on-failure
./build/release/benchmarks/gemm_benchmark
```

**包含内容**：

- ✅ `dev` 中的所有内容
- ✅ 性能基准测试
- ✅ 优化构建 (RelWithDebInfo)

### `debug` - 调试构建

**适用于**：调试问题

```bash
cmake --preset debug
cmake --build --preset debug --parallel $(nproc)
```

**包含内容**：

- ✅ 完整调试符号
- ✅ 无优化
- ✅ 启用运行时检查

### `cpu-smoke` - CPU-only 验证

**适用于**：无 CUDA 环境，测试构建基础设施

```bash
cmake --preset cpu-smoke
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
```

**包含内容**：

- ✅ 构建系统验证
- ✅ 安装流程
- ❌ GPU 功能（已禁用）
- ❌ 测试、基准测试、Python 绑定

## 手动配置

高级用户可使用自定义配置：

```bash
cmake -B build/manual -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DTC_BUILD_TESTS=ON \
  -DTC_BUILD_BENCHMARKS=ON \
  -DTC_BUILD_PYTHON=ON \
  -DTC_PYTHON_EXECUTABLE=$(which python3)

cmake --build build/manual --parallel $(nproc)
ctest --test-dir build/manual --output-on-failure
```

### 关键 CMake 选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `CMAKE_CUDA_ARCHITECTURES` | 75 | 目标 GPU 架构 |
| `TC_BUILD_TESTS` | ON | 构建单元测试 |
| `TC_BUILD_BENCHMARKS` | ON（仅 release） | 构建性能基准测试 |
| `TC_BUILD_PYTHON` | AUTO | 如果找到 Python 则构建 Python 绑定 |
| `TC_PYTHON_EXECUTABLE` | 自动检测 | Python 可执行文件路径 |
| `CUDA_TOOLKIT_ROOT_DIR` | /usr/local/cuda | CUDA 安装路径 |

## Python 绑定

### 安装

```bash
# 从仓库根目录
python3 -m pip install -e .
```

### 验证

```python
import tensorcraft_ops as tc

# 检查版本
print(f"TensorCraft version: {tc.__version__}")

# 创建张量
a = tc.tensor([[1.0, 2.0], [3.0, 4.0]])
b = tc.tensor([[5.0, 6.0], [7.0, 8.0]])

# 矩阵乘法
c = tc.matmul(a, b)
print(f"Result: {c.numpy()}")
```

### 可用 Python API

| API | 描述 | 示例 |
|-----|------|------|
| `tensor(data)` | 从列表创建张量 | `tc.tensor([[1,2],[3,4]])` |
| `matmul(a, b)` | 矩阵乘法 | `tc.matmul(a, b)` |
| `softmax(x)` | Softmax 操作 | `tc.softmax(x, dim=-1)` |
| `layer_norm(x)` | 层归一化 | `tc.layer_norm(x)` |

## CUDA 架构配置

TensorCraft-HPC 默认使用 CUDA 架构 75（Turing）以获得广泛兼容性。可根据您的 GPU 进行配置：

### 查找您的 GPU 架构

| GPU 系列 | 架构 | SM 值 |
|----------|------|-------|
| V100 | Volta | 70 |
| RTX 2000 | Turing | 75 |
| RTX 3000 / A100 | Ampere | 80 |
| RTX 4000 | Ada Lovelace | 89 |
| H100 | Hopper | 90 |

### 为您的 GPU 配置

```bash
# 单一架构（更快构建）
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=80

# 多个架构
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES="75;80;90"

# 所有支持的架构（构建较慢，通用二进制）
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES="70;75;80;89;90"
```

## 故障排除快速参考

| 问题 | 解决方案 |
|------|----------|
| `nvcc not found` | 安装 CUDA Toolkit 或检查 PATH |
| `CUDA architecture mismatch` | 设置 `CMAKE_CUDA_ARCHITECTURES` |
| `CMake version too old` | 升级 CMake 到 3.20+ |
| `Python import fails` | 从仓库根目录运行 `python3 -m pip install -e .` |
| `Tests fail on GPU` | 检查 GPU 驱动，运行 `nvidia-smi` |

详细故障排除请参见 [故障排除指南](troubleshooting.md)。

## 下一步

成功安装后：

1. **运行示例** → [示例章节](../examples/)
2. **了解架构** → [架构指南](../guides/architecture.md)
3. **优化性能** → [优化指南](../guides/optimization.md)
4. **API 参考** → [API 文档](../api/)

## 需要帮助？

- 🐛 [报告构建问题](https://github.com/LessUp/modern-ai-kernels/issues)
- 💬 [提问讨论](https://github.com/LessUp/modern-ai-kernels/discussions)
- 📖 [完整文档](https://lessup.github.io/modern-ai-kernels/)
