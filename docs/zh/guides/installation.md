# 安装说明

详细的安装指南。

## 系统要求 {#requirements}

### 硬件

- NVIDIA GPU，计算能力 7.0+（Volta 或更新）
- 最低 8GB GPU 内存用于基准测试
- 推荐系统内存 16GB+

### 软件

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| CUDA Toolkit | 11.0 | 12.4 |
| C++ 编译器 | GCC 9 / Clang 12 | GCC 12 / Clang 15 |
| CMake | 3.20 | 3.28 |
| Python | 3.9 | 3.11 |

## 安装方法 {#methods}

### 方法 1: 仅头文件（推荐）

```bash
git clone https://github.com/AICL-Lab/modern-ai-kernels.git
cp -r modern-ai-kernels/include/tensorcraft your_project/include/
```

### 方法 2: CMake 子目录

```cmake
add_subdirectory(modern-ai-kernels)
target_link_libraries(your_target PRIVATE tensorcraft::tensorcraft)
```

### 方法 3: Python 包

```bash
pip install -e .
```