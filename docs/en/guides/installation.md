# Installation

Detailed installation instructions for different platforms and use cases.

## System Requirements {#requirements}

### Hardware

- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- Minimum 8GB GPU memory for benchmarks
- 16GB+ system RAM recommended

### Software

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | 11.0 | 12.4 |
| C++ Compiler | GCC 9 / Clang 12 / MSVC 2019 | GCC 12 / Clang 15 |
| CMake | 3.20 | 3.28 |
| Python | 3.9 | 3.11 |

---

## Installation Methods {#methods}

### Method 1: Header-Only (Recommended for C++)

The simplest approach—just include the headers:

```bash
# Clone
git clone https://github.com/AICL-Lab/modern-ai-kernels.git

# Use in your project
cp -r modern-ai-kernels/include/tensorcraft your_project/include/
```

In your C++ code:

```cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

// That's it! No linking required.
```

### Method 2: CMake Subdirectory

For better integration with CMake projects:

```cmake
# In your CMakeLists.txt
add_subdirectory(modern-ai-kernels)

target_link_libraries(your_target PRIVATE tensorcraft::tensorcraft)
```

### Method 3: Python Package

```bash
# From source
git clone https://github.com/AICL-Lab/modern-ai-kernels.git
cd modern-ai-kernels
pip install -e .

# Or from PyPI (when available)
pip install tensorcraft-ops
```

---

## Platform-Specific Notes {#platforms}

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake ninja-build
sudo apt install nvidia-cuda-toolkit  # or use NVIDIA runfile

# Verify CUDA
nvcc --version
nvidia-smi
```

### Windows

1. Install Visual Studio 2019+ with C++ development tools
2. Install CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
3. Install CMake from [cmake.org](https://cmake.org/download/)

```powershell
# Verify
nvcc --version
cmake --version
```

### macOS

::: warning
macOS does not support NVIDIA GPUs. Use the CPU-only preset for development:
:::

```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke
```

---

## Build Presets {#presets}

TensorCraft-HPC provides CMake presets for common configurations:

### cpu-smoke

CPU-only validation build (no GPU required):

```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke
```

### dev

Development build with CUDA:

```bash
cmake --preset dev
cmake --build --preset dev --parallel 4
ctest --preset dev --output-on-failure
```

### release

Optimized release build:

```bash
cmake --preset release
cmake --build --preset release
```

### python-dev

Python bindings development:

```bash
cmake --preset python-dev
cmake --build --preset python-dev
pip install -e .
```

---

## Troubleshooting {#troubleshooting}

### CUDA Not Found

```bash
# Set CUDA path explicitly
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
cmake --preset dev -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

### Out of Memory During Build

```bash
# Reduce parallelism
cmake --build --preset dev --parallel 1
```

### Python Import Error

```python
# Check installation
import tensorcraft_ops
print(tensorcraft_ops.__file__)

# Reinstall if needed
pip uninstall tensorcraft-ops
pip install -e .
```