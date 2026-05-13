# Quick Start Guide

Get up and running with TensorCraft-HPC in 5 minutes.

## 1. Verify Your Setup {#verify}

```bash
# Check CUDA is available
nvidia-smi
nvcc --version

# Check Python (optional)
python --version
```

## 2. Build the Library {#build}

```bash
git clone https://github.com/LessUp/modern-ai-kernels.git
cd modern-ai-kernels

# Quick CPU test
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke

# Full CUDA build
cmake --preset dev
cmake --build --preset dev --parallel 4
```

## 3. Run Tests {#test}

```bash
# Run all tests
ctest --preset dev --output-on-failure

# Run specific test
ctest --preset dev -R gemm_test
```

## 4. Try an Example {#example}

### C++ Example

```cpp
// example.cpp
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"
#include <iostream>

int main() {
    // Create 1024x1024 matrices
    tensorcraft::FloatTensor A({1024, 1024});
    tensorcraft::FloatTensor B({1024, 1024});
    tensorcraft::FloatTensor C({1024, 1024});

    // Initialize with random data
    A.random_fill();
    B.random_fill();

    // Perform GEMM
    tensorcraft::kernels::gemm(
        A.data(), B.data(), C.data(),
        1024, 1024, 1024
    );

    std::cout << "GEMM completed successfully!" << std::endl;
    return 0;
}
```

Compile and run:

```bash
# Using nvcc directly
nvcc -std=c++17 -I include example.cpp -o example
./example

# Or use CMake
# Add to examples/CMakeLists.txt
```

### Python Example

```python
import tensorcraft_ops as tc
import numpy as np

# Create matrices
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

# GPU-accelerated GEMM
C = tc.gemm(A, B)

print(f"Result shape: {C.shape}")
print(f"Sample value: {C[0, 0]}")
```

## 5. Run Benchmarks {#benchmark}

```bash
# Build benchmarks
cmake --preset release
cmake --build --preset release

# Run GEMM benchmark
./build/benchmarks/gemm_benchmark

# Run attention benchmark
./build/benchmarks/attention_benchmark
```

## Next Steps {#next}

- [Architecture Overview](/en/architecture) — Understand the design
- [API Reference](/en/api/gemm) — Detailed kernel documentation
- [Papers & Citations](/en/references/papers) — Academic references