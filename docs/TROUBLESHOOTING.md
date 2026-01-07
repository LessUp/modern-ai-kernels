# Troubleshooting Guide

This guide covers common issues and their solutions when building or using TensorCraft-HPC.

## Build Issues

### CUDA Not Found

**Error:**
```
CMake Error: Could not find CUDA
```

**Solutions:**

1. Ensure CUDA Toolkit is installed:
   ```bash
   nvcc --version
   ```

2. Set CUDA path explicitly:
   ```bash
   cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
   ```

3. Set environment variables:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   ```

### Unsupported GPU Architecture

**Error:**
```
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```

**Solution:**

Specify supported architectures for your CUDA version:

```bash
# CUDA 11.x
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"

# CUDA 12.x
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90"
```

### C++17 Not Supported

**Error:**
```
error: 'if constexpr' is a C++17 extension
```

**Solutions:**

1. Use a newer compiler:
   ```bash
   # Ubuntu
   sudo apt install g++-11
   cmake -B build -DCMAKE_CXX_COMPILER=g++-11
   ```

2. Explicitly set C++ standard:
   ```bash
   cmake -B build -DCMAKE_CXX_STANDARD=17
   ```

### Ninja Not Found

**Error:**
```
CMake Error: CMake was unable to find a build program corresponding to "Ninja"
```

**Solution:**

Install Ninja or use Make:

```bash
# Install Ninja
sudo apt install ninja-build  # Ubuntu
brew install ninja            # macOS

# Or use Make instead
cmake -B build -G "Unix Makefiles"
```

### Out of Memory During Compilation

**Error:**
```
nvcc error: ran out of memory during compilation
```

**Solutions:**

1. Reduce parallel jobs:
   ```bash
   cmake --build build --parallel 2
   ```

2. Reduce target architectures:
   ```bash
   cmake -B build -DCMAKE_CUDA_ARCHITECTURES="86"
   ```

3. Use `--threads` flag for nvcc:
   ```bash
   cmake -B build -DCMAKE_CUDA_FLAGS="--threads 2"
   ```

## Runtime Issues

### CUDA Driver Version Mismatch

**Error:**
```
CUDA driver version is insufficient for CUDA runtime version
```

**Solutions:**

1. Update NVIDIA driver:
   ```bash
   # Ubuntu
   sudo apt update
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

2. Check compatibility:
   ```bash
   nvidia-smi  # Shows driver version
   nvcc --version  # Shows CUDA toolkit version
   ```

### GPU Not Detected

**Error:**
```
CUDA error: no CUDA-capable device is detected
```

**Solutions:**

1. Check GPU is visible:
   ```bash
   nvidia-smi
   lspci | grep -i nvidia
   ```

2. Check CUDA device:
   ```bash
   # In your code
   int deviceCount;
   cudaGetDeviceCount(&deviceCount);
   printf("Found %d CUDA devices\n", deviceCount);
   ```

3. Ensure correct permissions:
   ```bash
   sudo usermod -aG video $USER
   # Log out and back in
   ```

### Illegal Memory Access

**Error:**
```
CUDA error: an illegal memory access was encountered
```

**Solutions:**

1. Enable debug mode:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Debug
   ```

2. Use cuda-memcheck:
   ```bash
   compute-sanitizer ./your_program
   ```

3. Check array bounds and pointer validity in your kernel code.

### Kernel Launch Failure

**Error:**
```
CUDA error: invalid configuration argument
```

**Common Causes:**

1. Block size exceeds maximum (1024 threads)
2. Shared memory exceeds limit
3. Grid dimensions exceed maximum

**Solution:**

Check your launch configuration:
```cpp
// Query device limits
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max shared memory: %zu\n", prop.sharedMemPerBlock);
```

## Performance Issues

### Slow Kernel Execution

**Possible Causes and Solutions:**

1. **Memory-bound kernel**: Optimize memory access patterns
   - Use coalesced memory access
   - Utilize shared memory for data reuse

2. **Low occupancy**: Adjust block size
   ```cpp
   // Use occupancy calculator
   int blockSize;
   int minGridSize;
   cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel);
   ```

3. **Wrong architecture**: Ensure you're targeting your GPU
   ```bash
   # Check your GPU's compute capability
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```

### Memory Allocation Failures

**Error:**
```
CUDA error: out of memory
```

**Solutions:**

1. Check available memory:
   ```bash
   nvidia-smi
   ```

2. Use memory pool:
   ```cpp
   #include <tensorcraft/memory/memory_pool.hpp>
   // Use pooled allocations to reduce fragmentation
   ```

3. Process data in smaller batches.

## CUDA Version Compatibility

### Minimum CUDA Versions

| Feature | Minimum CUDA |
|---------|--------------|
| Basic kernels | 11.0 |
| Tensor Cores (FP16) | 11.0 |
| Tensor Cores (BF16) | 11.0 |
| Tensor Cores (FP8) | 11.8 |
| Hopper features | 12.0 |

### Driver Compatibility

| CUDA Toolkit | Minimum Driver |
|--------------|----------------|
| CUDA 11.0 | 450.36 |
| CUDA 11.8 | 520.61 |
| CUDA 12.0 | 525.60 |
| CUDA 12.2 | 535.54 |

## Getting Help

If you're still experiencing issues:

1. **Search existing issues**: Check [GitHub Issues](https://github.com/username/tensorcraft-hpc/issues)

2. **Create a new issue**: Include:
   - Operating system and version
   - CUDA version (`nvcc --version`)
   - GPU model (`nvidia-smi`)
   - Driver version
   - Complete error message
   - Minimal reproducible example

3. **Check NVIDIA forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

## Useful Commands

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check CUDA samples
cuda-install-samples-12.2.sh ~/cuda-samples
cd ~/cuda-samples/Samples/1_Utilities/deviceQuery
make
./deviceQuery

# Profile a kernel
nsys profile ./your_program
ncu ./your_program
```
