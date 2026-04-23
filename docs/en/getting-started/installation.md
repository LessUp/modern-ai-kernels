---
title: Installation Guide
parent: Getting Started
nav_order: 1
---

# Installation Guide

Complete setup instructions for TensorCraft-HPC across different use cases.

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 16 GB |
| Disk | 2 GB free | 10 GB free |
| GPU | NVIDIA (Compute 70+) | NVIDIA Hopper (SM 90) |

### Software Requirements

#### Required

| Tool | Version | Purpose |
|------|---------|---------|
| **CUDA Toolkit** | 12.0+ | GPU kernel compilation and runtime |
| **CMake** | 3.20+ | Build system |
| **C++ Compiler** | C++17-capable | Host code compilation |
| **NVIDIA Driver** | 520+ | GPU runtime support |

#### Optional

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.8+ | Python bindings |
| **Ninja** | 1.10+ | Faster build generation |
| **GoogleTest** | 1.10+ | Unit testing (auto-fetched) |
| **pybind11** | 2.10+ | Python bindings (auto-fetched) |

## Installation by Platform

### Ubuntu / Debian Linux

#### 1. Install Prerequisites

```bash
# Update package lists
sudo apt update

# Install CUDA Toolkit (if not already installed)
# Option A: Official NVIDIA .deb repository
# See: https://developer.nvidia.com/cuda-downloads

# Option B: Direct installation
sudo apt install -y cuda-toolkit-12-8

# Install build tools
sudo apt install -y cmake build-essential

# Install Python (optional, for bindings)
sudo apt install -y python3 python3-pip python3-dev
```

#### 2. Verify Installation

```bash
# Check CUDA
nvcc --version
# Should show: Cuda compilation tools, release 12.x

# Check CMake
cmake --version
# Should show: cmake version 3.20 or higher

# Check Python (if installed)
python3 --version
# Should show: Python 3.8 or higher
```

#### 3. Build TensorCraft-HPC

```bash
# Clone repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Configure and build
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)

# Run tests
ctest --preset dev --output-on-failure
```

### macOS (CPU-Only)

{: .warning }
TensorCraft-HPC is a CUDA-based library. On macOS, you can only use CPU-only validation and documentation builds.

```bash
# Clone repository
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Install dependencies
brew install cmake

# CPU-only validation
cmake --preset cpu-smoke
cmake --build build/cpu-smoke --parallel $(sysctl -n hw.ncpu)
```

### Windows

{: .warning }
Windows requires Visual Studio and CUDA Toolkit installed. WSL2 is recommended.

#### Using WSL2 (Recommended)

```bash
# Install WSL2 with Ubuntu
wsl --install

# Inside WSL2, follow Ubuntu instructions above
```

#### Native Windows (Visual Studio)

1. Install [Visual Studio 2022](https://visualstudio.microsoft.com/)
2. Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads?target_os=Windows)
3. Open Developer Command Prompt

```cmd
cmake -B build -G "Visual Studio 17 2022" ^
  -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -DTC_BUILD_TESTS=ON

cmake --build build --config Release --parallel
```

## Build Presets Explained

TensorCraft-HPC provides several CMake presets for different use cases:

### `dev` - Development (Recommended)

**Use when**: Daily development with CUDA support

```bash
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)
ctest --preset dev --output-on-failure
```

**Includes**:

- ✅ All GPU kernels
- ✅ Unit tests
- ✅ Debug symbols
- ❌ Benchmarks (to save build time)

### `python-dev` - Python Development

**Use when**: Focusing on Python bindings

```bash
cmake --preset python-dev
cmake --build --preset python-dev --parallel $(nproc)
python3 -m pip install -e .
python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

**Includes**:

- ✅ Python bindings
- ✅ Core GPU kernels needed by Python API
- ❌ Full test suite
- ❌ Benchmarks

### `release` - Full Release

**Use when**: Complete build with benchmarks

```bash
cmake --preset release
cmake --build --preset release --parallel $(nproc)
ctest --test-dir build/release --output-on-failure
./build/release/benchmarks/gemm_benchmark
```

**Includes**:

- ✅ Everything in `dev`
- ✅ Performance benchmarks
- ✅ Optimized build (RelWithDebInfo)

### `debug` - Debug Build

**Use when**: Debugging issues

```bash
cmake --preset debug
cmake --build --preset debug --parallel $(nproc)
```

**Includes**:

- ✅ Full debug symbols
- ✅ No optimizations
- ✅ Runtime checks enabled

### `cpu-smoke` - CPU-Only Validation

**Use when**: No CUDA available, testing build infrastructure

```bash
cmake --preset cpu-smoke
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
```

**Includes**:

- ✅ Build system validation
- ✅ Installation flow
- ❌ GPU features (disabled)
- ❌ Tests, benchmarks, Python bindings

## Manual Configuration

For advanced users who need custom configurations:

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

### Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_CUDA_ARCHITECTURES` | 75 | Target GPU architectures |
| `TC_BUILD_TESTS` | ON | Build unit tests |
| `TC_BUILD_BENCHMARKS` | ON (release only) | Build performance benchmarks |
| `TC_BUILD_PYTHON` | AUTO | Build Python bindings if Python found |
| `TC_PYTHON_EXECUTABLE` | auto-detected | Python executable path |
| `CUDA_TOOLKIT_ROOT_DIR` | /usr/local/cuda | CUDA installation path |

## Python Bindings

### Installation

```bash
# From repository root
python3 -m pip install -e .
```

### Verification

```python
import tensorcraft_ops as tc

# Check version
print(f"TensorCraft version: {tc.__version__}")

# Create tensors
a = tc.tensor([[1.0, 2.0], [3.0, 4.0]])
b = tc.tensor([[5.0, 6.0], [7.0, 8.0]])

# Matrix multiplication
c = tc.matmul(a, b)
print(f"Result: {c.numpy()}")
```

### Available Python APIs

| API | Description | Example |
|-----|-------------|---------|
| `tensor(data)` | Create tensor from list | `tc.tensor([[1,2],[3,4]])` |
| `matmul(a, b)` | Matrix multiplication | `tc.matmul(a, b)` |
| `softmax(x)` | Softmax operation | `tc.softmax(x, dim=-1)` |
| `layer_norm(x)` | Layer normalization | `tc.layer_norm(x)` |

## CUDA Architecture Configuration

TensorCraft-HPC defaults to CUDA architecture 75 (Turing) for broad compatibility. Configure for your specific GPU:

### Find Your GPU Architecture

| GPU Series | Architecture | SM Value |
|------------|--------------|----------|
| V100 | Volta | 70 |
| RTX 2000 | Turing | 75 |
| RTX 3000 / A100 | Ampere | 80 |
| RTX 4000 | Ada Lovelace | 89 |
| H100 | Hopper | 90 |

### Configure for Your GPU

```bash
# Single architecture (faster builds)
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=80

# Multiple architectures
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES="75;80;90"

# All supported architectures (slower builds, universal binaries)
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES="70;75;80;89;90"
```

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| `nvcc not found` | Install CUDA Toolkit or check PATH |
| `CUDA architecture mismatch` | Set `CMAKE_CUDA_ARCHITECTURES` |
| `CMake version too old` | Upgrade CMake to 3.20+ |
| `Python import fails` | Run `python3 -m pip install -e .` from repo root |
| `Tests fail on GPU` | Check GPU driver, run `nvidia-smi` |

For detailed troubleshooting, see [Troubleshooting Guide](troubleshooting.md).

## Next Steps

After successful installation:

1. **Run Examples** → [Examples Section](../examples/)
2. **Learn Architecture** → [Architecture Guide](../guides/architecture.md)
3. **Optimize Performance** → [Optimization Guide](../guides/optimization.md)
4. **Reference API** → [API Documentation](../api/)

## Need Help?

- 🐛 [Report Build Issues](https://github.com/LessUp/modern-ai-kernels/issues)
- 💬 [Ask Questions](https://github.com/LessUp/modern-ai-kernels/discussions)
- 📖 [Full Documentation](https://lessup.github.io/modern-ai-kernels/)
