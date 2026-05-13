# Python Bindings

::: info Coming Soon
This guide is under development. Check back soon for comprehensive Python binding documentation.
:::

## Overview

TensorCraft-HPC provides Python bindings via pybind11, allowing you to use optimized kernels from Python code.

## Installation

```bash
pip install tensorcraft-ops
```

## Basic Usage

```python
import tensorcraft_ops as tc
import numpy as np

# Create matrices
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

# GPU-accelerated GEMM
C = tc.gemm(A, B)
```