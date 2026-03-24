# Installation Guide

This guide provides detailed instructions for installing TensorCraft-HPC on various platforms.

## Prerequisites

### Required

- **CUDA Toolkit**: 11.0 or later (12.x recommended for best performance)
- **CMake**: 3.20 or later
- **C++ Compiler**: C++17 compatible
  - GCC 9+ (Linux)
  - Clang 10+ (Linux/macOS)
  - MSVC 2019+ (Windows)
- **NVIDIA GPU**: Compute Capability 7.0+ (Volta, Turing, Ampere, Hopper)

### Optional

- **Ninja**: For faster builds
- **pybind11**: For Python bindings
- **Python**: 3.8+ (for Python bindings)
- **NumPy**: Installed automatically for the `tensorcraft-ops` Python package

## Quick Start

```bash
# Clone the repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Configure and build
cmake --preset release
cmake --build build/release --parallel

# Run tests
ctest --test-dir build/release --output-on-failure

# Build and install Python bindings
pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

Python bindings currently require CUDA. If CUDA is unavailable, CMake disables tests, benchmarks, and Python bindings automatically.

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

#### 1. Install CUDA Toolkit

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA (choose your version)
sudo apt-get install cuda-toolkit-12-2
```

#### 2. Install Build Tools

```bash
sudo apt-get install build-essential cmake ninja-build git
```

#### 3. Set Environment Variables

Add to your `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 4. Build TensorCraft-HPC

```bash
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Using CMake presets (recommended)
cmake --preset release
cmake --build build/release --parallel

# Or manual configuration
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"
cmake --build build --parallel
```

### Linux (CentOS/RHEL)

#### 1. Install CUDA Toolkit

```bash
# Add NVIDIA repository
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install CUDA
sudo yum install cuda-toolkit-12-2
```

#### 2. Install Build Tools

```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 ninja-build git

# Use cmake3 instead of cmake on older systems
alias cmake=cmake3
```

#### 3. Build

Follow the same build steps as Ubuntu.

### Windows

#### 1. Install CUDA Toolkit

1. Download CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow the prompts
3. Ensure Visual Studio integration is selected

#### 2. Install Build Tools

1. Install [Visual Studio 2019/2022](https://visualstudio.microsoft.com/) with C++ workload
2. Install [CMake](https://cmake.org/download/) (or use the one bundled with VS)
3. Install [Git for Windows](https://git-scm.com/download/win)

#### 3. Build TensorCraft-HPC

Open "x64 Native Tools Command Prompt for VS":

```cmd
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

cmake -B build -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --parallel
```

Or using Ninja:

```cmd
cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe"

cmake --build build --parallel
```

### macOS (Apple Silicon with External GPU)

> Note: Native CUDA support on macOS is limited. Consider using Linux or Windows for best results.

For development without GPU:

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake via Homebrew
brew install cmake ninja

# Build (CPU-only configuration for docs/install-rule validation)
cmake --preset cpu-smoke
cmake --build build/cpu-smoke --parallel
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Debug, Release, RelWithDebInfo) |
| `CMAKE_CUDA_ARCHITECTURES` | `70;75;80;86;89;90` | Target GPU architectures |
| `TC_ENABLE_CUDA` | `ON` | Enable CUDA language support and CUDA-dependent targets |
| `TC_BUILD_TESTS` | `ON` | Build CUDA unit tests |
| `TC_BUILD_BENCHMARKS` | `ON` | Build CUDA benchmarks |
| `TC_BUILD_PYTHON` | `ON` | Build pybind11 Python bindings |
| `TC_BUILD_DOCS` | `OFF` | Reserved documentation build toggle |

### Example Configurations

```bash
# Debug build with tests
cmake --preset debug

# Release build with Python bindings
cmake --preset release -DTC_BUILD_PYTHON=ON

# Specific GPU architectures
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

## GPU Architecture Reference

| Architecture | Compute Capability | GPUs |
|--------------|-------------------|------|
| Volta | 70 | V100 |
| Turing | 75 | RTX 2080, T4 |
| Ampere | 80, 86 | A100, RTX 3090 |
| Ada Lovelace | 89 | RTX 4090, L40 |
| Hopper | 90 | H100 |

## Python Bindings

### Installation

```bash
# Editable install from repository root
pip install -e .
```

This builds and installs the current pybind11 extension module as `tensorcraft_ops`.

### Usage

```python
import numpy as np
import tensorcraft_ops as tc

x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
y = tc.softmax(x)
z = tc.transpose(x)
```

## Verification

After installation, verify everything works:

```bash
# Run tests
ctest --test-dir build/release --output-on-failure

# Run a benchmark
./build/release/benchmarks/gemm_benchmark

# Check Python module import
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

On systems without CUDA, validate configure/install behavior instead:

```bash
cmake --preset cpu-smoke
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
```

Tests, benchmarks, and Python bindings are intentionally disabled in that mode.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## Updating

```bash
cd modern-ai-kernels
git pull origin main
cmake --build build --parallel
```

## Uninstallation

TensorCraft-HPC ships headers plus an optional Python extension. To uninstall:

1. Remove the cloned repository
2. Remove any installed Python package: `pip uninstall tensorcraft-ops`
