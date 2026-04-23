---
title: Troubleshooting
parent: Getting Started
nav_order: 2
---

# Troubleshooting

Common issues and their solutions when building and running TensorCraft-HPC.

## Build Issues

### CMake Configuration Failures

#### Issue: "CMake version too old"

**Error Message**:

```
CMake 3.20 or higher is required. You are running version 3.18
```

**Solution**:

```bash
# Ubuntu/Debian
sudo apt remove cmake
pip3 install cmake

# Or use snap
sudo snap install cmake --classic

# Verify
cmake --version  # Should show 3.20+
```

#### Issue: "CUDA toolkit not found"

**Error Message**:

```
Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR)
```

**Solution**:

```bash
# Verify CUDA installation
nvcc --version

# If not found, add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or specify explicitly
cmake --preset dev -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

#### Issue: "No CUDA capable devices found"

**Error Message**:

```
No CUDA capable devices found or CUDA driver error
```

**Solution**:

```bash
# Check NVIDIA driver
nvidia-smi

# If driver not loaded, reinstall
sudo apt install nvidia-driver-535  # or latest version

# Reboot if needed
sudo reboot
```

### Compilation Errors

#### Issue: "Unsupported GPU architecture"

**Error Message**:

```
nvcc fatal: Unsupported gpu architecture 'compute_90'
```

**Solution**:

```bash
# Use architecture compatible with your CUDA version
# CUDA 11.x: Use up to 86 (Ampere)
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=80

# CUDA 12.x: Supports up to 90 (Hopper)
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=86

# Check your CUDA version
nvcc --version
```

#### Issue: "C++17 features not supported"

**Error Message**:

```
target feature 'c++17' not supported by your implementation
```

**Solution**:

```bash
# Use a newer compiler
sudo apt install g++-11
export CXX=g++-11

# Or specify in CMake
cmake --preset dev -DCMAKE_CXX_COMPILER=g++-11
```

#### Issue: "pybind11 not found" (Python bindings)

**Error Message**:

```
Could NOT find pybind11 (missing: pybind11_DIR)
```

**Solution**:

```bash
# Install pybind11
pip3 install pybind11

# Or disable Python bindings
cmake --preset python-dev -DTC_BUILD_PYTHON=OFF
```

### Linker Errors

#### Issue: "Undefined reference to CUDA functions"

**Error Message**:

```
undefined reference to `cudaMalloc'
undefined reference to `cudaMemcpy'
```

**Solution**:

```bash
# Clean and reconfigure
rm -rf build
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)

# Check CUDA installation
nvcc --version
nvidia-smi
```

## Runtime Issues

### Test Failures

#### Issue: "Tests fail with CUDA errors"

**Error Message**:

```
CUDA error: initialization error
```

**Solution**:

```bash
# Check GPU is accessible
nvidia-smi

# Test with simple CUDA program
cat << 'EOF' > test_cuda.cu
#include <stdio.h>
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);
    return 0;
}
EOF

nvcc test_cuda.cu -o test_cuda && ./test_cuda
```

#### Issue: "Numerical precision failures"

**Symptoms**: Tests fail due to small numerical differences

**Solution**:

```bash
# This can happen with different GPU architectures
# Tests have tolerance thresholds, some are stricter

# Run with verbose output
ctest --preset dev --output-on-failure --verbose

# If specific test fails, check if it's known issue
# See: https://github.com/LessUp/modern-ai-kernels/issues
```

### Python Binding Issues

#### Issue: "ImportError: No module named tensorcraft_ops"

**Error Message**:

```python
ImportError: No module named 'tensorcraft_ops'
```

**Solution**:

```bash
# Install in editable mode
python3 -m pip install -e .

# Verify installation
pip list | grep tensorcraft

# Check Python path matches installation
which python
python3 -c "import sys; print(sys.executable)"
```

#### Issue: "ModuleNotFoundError: tensorcraft_ops.so undefined symbol"

**Error Message**:

```python
ModuleNotFoundError: /path/to/tensorcraft_ops.so: undefined symbol: _Z14cudaMallocPvmm
```

**Solution**:

```bash
# Rebuild Python bindings
rm -rf build
cmake --preset python-dev
cmake --build --preset python-dev --parallel $(nproc)
python3 -m pip install -e . --force-reinstall

# Check CUDA libraries are linked
ldd $(python3 -c "import tensorcraft_ops; print(tensorcraft_ops.__file__)")
```

#### Issue: "Import works but functions fail"

**Symptoms**: Can import module but functions throw errors

**Solution**:

```python
# Check if GPU is accessible from Python
import tensorcraft_ops as tc
import torch  # if available

# Verify CUDA context
print(tc.__version__)

# Test simple operation
try:
    a = tc.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("Tensor created successfully")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Issues

### Slow Build Times

#### Issue: "Build takes too long"

**Symptoms**: Compilation takes 30+ minutes

**Solution**:

```bash
# Build only what you need
# For development (no benchmarks)
cmake --preset dev

# For specific architecture (faster than all)
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=80

# Use Ninja for faster builds
sudo apt install ninja-build
cmake --preset dev -G Ninja
cmake --build --preset dev --parallel $(nproc)
```

#### Issue: "CUDA compilation is slow"

**Symptoms**: Each rebuild takes several minutes

**Solution**:

```bash
# Use CMake's incremental build features
cmake --preset dev
cmake --build --preset dev --parallel $(nproc) --target tensorcraft

# Only build changed files
touch include/tensorcraft/kernels/gemm.cuh
cmake --build --preset dev --parallel $(nproc)
```

### Memory Issues

#### Issue: "Out of memory during build"

**Symptoms**: Build fails with memory errors

**Solution**:

```bash
# Reduce parallel jobs
cmake --build --preset dev --parallel 2  # instead of $(nproc)

# Close other applications
# CUDA compilation is memory-intensive (2-4GB per job)

# Build in stages
cmake --preset dev
cmake --build --preset dev --target tensorcraft_core
cmake --build --preset dev --target tensorcraft_kernels
```

## Environment Issues

### Multiple CUDA Versions

#### Issue: "Wrong CUDA version being used"

**Symptoms**: Build uses unexpected CUDA version

**Solution**:

```bash
# Check which nvcc is used
which nvcc
nvcc --version

# Specify CUDA version explicitly
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Or in CMake
cmake --preset dev -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8
```

### WSL2 Issues

#### Issue: "CUDA not available in WSL2"

**Solution**:

```bash
# Install WSL2 CUDA driver
# See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

# In WSL2, install CUDA toolkit (not driver)
sudo apt install cuda-toolkit-12-8

# Verify
nvidia-smi  # Should show GPU
nvcc --version  # Should show CUDA 12.x
```

#### Issue: "WSL2 performance is poor"

**Solution**:

```bash
# Use WSL2 with GPU passthrough (Windows 11)
# Ensure Windows GPU driver is up to date

# Check CUDA is using GPU
nvidia-smi

# If using CPU fallback, check WSL2 configuration
```

## Known Issues

### GitHub Issues

Check current known issues:

- 🔍 [Open Issues](https://github.com/LessUp/modern-ai-kernels/issues)
- ✅ [Fixed in Latest](https://github.com/LessUp/modern-ai-kernels/issues?q=is%3Aissue+is%3Aclosed)

### Platform-Specific Limitations

| Platform | Limitation | Workaround |
|----------|-----------|------------|
| macOS | No CUDA support | CPU-only builds |
| Windows Native | Visual Studio + CUDA complexity | Use WSL2 |
| Docker | GPU passthrough needed | Use `--gpus all` flag |
| WSL1 | No CUDA support | Upgrade to WSL2 |

## Getting Help

If none of the above solutions work:

1. **Gather Information**:

   ```bash
   # System info
   uname -a
   nvidia-smi
   nvcc --version
   cmake --version
   g++ --version
   
   # Build logs
   cmake --preset dev 2>&1 | tee cmake_output.log
   cmake --build --preset dev 2>&1 | tee build.log
   ```

2. **Check Existing Issues**:
   - [Search Open Issues](https://github.com/LessUp/modern-ai-kernels/issues)
   - [Search Closed Issues](https://github.com/LessUp/modern-ai-kernels/issues?q=is%3Aissue+is%3Aclosed)

3. **Create New Issue**:
   - [New Issue](https://github.com/LessUp/modern-ai-kernels/issues/new)
   - Include system info, build logs, and reproduction steps

## Quick Diagnostic Script

Run this script to check your environment:

```bash
#!/bin/bash
echo "=== TensorCraft-HPC Environment Check ==="
echo ""

echo "OS:"
uname -a
echo ""

echo "GPU:"
nvidia-smi 2>/dev/null || echo "  No NVIDIA GPU detected"
echo ""

echo "CUDA:"
nvcc --version 2>/dev/null || echo "  CUDA not found"
echo ""

echo "CMake:"
cmake --version 2>/dev/null || echo "  CMake not found"
echo ""

echo "C++ Compiler:"
g++ --version 2>/dev/null || echo "  g++ not found"
echo ""

echo "Python:"
python3 --version 2>/dev/null || echo "  Python 3 not found"
echo ""

echo "=== Recommendations ==="
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    echo "✅ CUDA $CUDA_VERSION found"
else
    echo "❌ CUDA not installed - required for GPU features"
fi

if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    if [[ $(echo $CMAKE_VERSION | cut -d'.' -f1) -ge 3 ]] && [[ $(echo $CMAKE_VERSION | cut -d'.' -f2) -ge 20 ]]; then
        echo "✅ CMake $CMAKE_VERSION meets requirements"
    else
        echo "❌ CMake $CMAKE_VERSION too old - need 3.20+"
    fi
else
    echo "❌ CMake not installed"
fi
```

Save as `check_env.sh`, run `bash check_env.sh` to diagnose your setup.
