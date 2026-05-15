# Getting Started

Welcome to TensorCraft-HPC! This guide will help you get up and running quickly.

## Prerequisites {#prerequisites}

- **C++17** or later compatible compiler
- **CUDA Toolkit 11.0+** (12.0+ recommended for FP8 support)
- **CMake 3.20+**
- **Python 3.9+** (optional, for Python bindings)

### Hardware Requirements

| GPU Architecture | Compute Capability | Minimum CUDA |
|------------------|-------------------|--------------|
| Volta | SM70 (7.0) | 9.0 |
| Turing | SM75 (7.5) | 10.0 |
| Ampere | SM80/SM86 (8.0/8.6) | 11.0 |
| Ada Lovelace | SM89 (8.9) | 11.8 |
| Hopper | SM90 (9.0) | 12.0 |
| Blackwell | SM100 (10.0) | 12.4 |

---

## Installation {#installation}

### Header-Only (Recommended)

TensorCraft-HPC is a header-only library. Simply include the headers in your project:

```bash
# Clone the repository
git clone https://github.com/AICL-Lab/modern-ai-kernels.git

# Copy headers to your project
cp -r modern-ai-kernels/include/tensorcraft your_project/include/
```

```cpp
// In your code
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"
```

### CMake Integration

For projects using CMake:

```cmake
# Add the include directory
target_include_directories(your_target PRIVATE
    path/to/modern-ai-kernels/include
)

# Link against CUDA
find_package(CUDA REQUIRED)
target_link_libraries(your_target CUDA::cudart)
```

### Python Bindings

```bash
# Build and install Python package
cd modern-ai-kernels
pip install -e .
```

---

## Quick Examples {#examples}

### GEMM (Matrix Multiplication)

::: code-group
```cpp [C++]
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

int main() {
    // Create tensors (RAII-managed GPU memory)
    tensorcraft::FloatTensor A({1024, 1024});
    tensorcraft::FloatTensor B({1024, 1024});
    tensorcraft::FloatTensor C({1024, 1024});

    // Initialize with random data
    A.random_fill();
    B.random_fill();

    // Perform GEMM: C = A × B
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

# Create matrices
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

# GPU-accelerated GEMM
C = tc.gemm(A, B)
```
:::

### FlashAttention

::: code-group
```cpp [C++]
#include "tensorcraft/kernels/attention.hpp"
#include "tensorcraft/memory/tensor.hpp"

int main() {
    // Batch size, sequence length, head dimension
    int batch = 32, seq_len = 128, head_dim = 64;

    // Q, K, V tensors
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

## Build Presets {#presets}

TensorCraft-HPC includes CMake presets for common configurations:

```bash
# CPU-only smoke test (no GPU required)
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke

# Development build with CUDA
cmake --preset dev
cmake --build --preset dev --parallel 4

# Release build with all optimizations
cmake --preset release
cmake --build --preset release

# Run tests
ctest --preset dev --output-on-failure
```

---

## Next Steps {#next-steps}

- Read the [Architecture Overview](/en/architecture) to understand the design
- Explore [API Reference](/en/api/gemm) for detailed kernel documentation
- Check [Papers & Citations](/en/references/papers) for academic references
- See [Learning Resources](/en/references/resources) for CUDA tutorials