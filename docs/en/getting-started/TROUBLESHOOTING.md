# Troubleshooting Guide

This guide helps resolve common issues when building and using TensorCraft-HPC.

## Build Issues

### CUDA Compiler Not Found

**Symptom**: CMake fails with "Failed to find nvcc (CUDA compiler)"

**Solutions**:

1. Ensure CUDA Toolkit is installed:
   ```bash
   nvcc --version
   ```

2. Set the CUDA path explicitly:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

3. Or specify in CMake:
   ```bash
   cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
   ```

### Unsupported CUDA Architecture

**Symptom**: `error: no kernel image is available for execution on the device`

**Solutions**:

1. Check your GPU compute capability:
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```

2. Set the correct architecture in CMake:
   ```bash
   cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86  # Example for RTX 3090
   ```

   | GPU Series | Architecture |
   |------------|--------------|
   | GTX 10xx | 61 |
   | RTX 20xx | 75 |
   | RTX 30xx | 86 |
   | RTX 40xx | 89 |
   | H100 | 90 |

### Out of Memory During Build

**Symptom**: Build fails with memory errors or system becomes unresponsive

**Solutions**:

1. Reduce parallel build jobs:
   ```bash
   cmake --build build --parallel 1
   ```

2. Use Ninja generator (more memory efficient):
   ```bash
   cmake -B build -G Ninja
   ninja -C build -j 2
   ```

### CMake Version Too Old

**Symptom**: `cmake_minimum_required` version error

**Solution**: Install CMake 3.20 or later:
```bash
# Ubuntu/Debian
pip install cmake --upgrade

# Or download from https://cmake.org/download/
```

## CUDA Runtime Issues

### CUDA Out of Memory

**Symptom**: `cudaErrorMemoryAllocation` during kernel execution

**Solutions**:

1. Reduce batch size or sequence length
2. Use smaller tile sizes if available
3. Check GPU memory usage:
   ```bash
   nvidia-smi
   watch -n 1 nvidia-smi
   ```

### Kernel Launch Failure

**Symptom**: `cudaErrorLaunchFailure` or unspecified launch failure

**Solutions**:

1. Check for invalid parameters (negative sizes, null pointers)
2. Ensure GPU is not in a bad state:
   ```bash
   sudo nvidia-smi --gpu-reset -i 0
   ```

3. Run with CUDA error checking enabled:
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   ```

### Incorrect Results

**Symptom**: Output values are NaN or obviously wrong

**Solutions**:

1. Check input data for NaN/Inf:
   ```python
   import numpy as np
   assert not np.isnan(input).any()
   assert not np.isinf(input).any()
   ```

2. Verify tensor shapes match the kernel expectations
3. Try with `CUDA_LAUNCH_BLOCKING=1` for easier debugging

## Python Binding Issues

### Import Error

**Symptom**: `ImportError: cannot import name 'tensorcraft_ops'`

**Solutions**:

1. Rebuild and reinstall:
   ```bash
   pip uninstall tensorcraft-ops
   cmake --preset python-dev
   cmake --build --preset python-dev
   pip install -e .
   ```

2. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

### Version Mismatch

**Symptom**: Installed version doesn't match expected

**Solution**:
```bash
pip show tensorcraft-ops
pip install -e . --force-reinstall
```

### NumPy Compatibility

**Symptom**: Errors related to NumPy array types

**Solution**: Ensure NumPy 1.24 or later:
```bash
pip install "numpy>=1.24"
```

## Performance Issues

### Slow Kernel Execution

**Solutions**:

1. Use release build:
   ```bash
   cmake --preset release
   ```

2. Check GPU utilization:
   ```bash
   nvidia-smi dmon -s u
   ```

3. Profile with Nsight:
   ```bash
   nsys profile python your_script.py
   ncu python your_script.py
   ```

### Slow Build Times

**Solutions**:

1. Use Ninja generator
2. Enable ccache:
   ```bash
   export CMAKE_CXX_COMPILER_LAUNCHER=ccache
   export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
   ```

3. Use precompiled headers (CMake 3.16+)

## Common Error Messages

### "static assertion failed"

Usually indicates template parameter mismatch. Check:
- Input types match expected (float vs half)
- Tensor dimensions are correct
- Tile sizes are valid

### "illegal memory access"

Usually indicates out-of-bounds access:
- Check array sizes match kernel parameters
- Verify indices are within bounds
- Run with `compute-sanitizer`:
   ```bash
   compute-sanitizer python your_script.py
   ```

### "too many resources requested"

Kernel uses too many registers:
- Reduce block size
- Use `__launch_bounds__` to limit registers

## Getting Help

If you can't resolve an issue:

1. Search existing [GitHub Issues](https://github.com/LessUp/modern-ai-kernels/issues)
2. Create a new issue with:
   - Full error message
   - Environment details (OS, CUDA version, GPU)
   - Minimal reproduction code
   - CMake configuration output
