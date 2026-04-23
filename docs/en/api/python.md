---
title: Python API Reference
lang: en
---

# Python API Reference

TensorCraft-HPC provides Python bindings via pybind11. The module is named `tensorcraft_ops`.

## Installation

```bash
python3 -m pip install -e .
python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

## Module Overview

```python
import tensorcraft_ops as tc

# Available functions
tc.relu(input)           # ReLU activation
tc.silu(input)           # SiLU activation
tc.gelu(input)           # GeLU activation
tc.sigmoid(input)        # Sigmoid activation
tc.vector_add(a, b)      # Element-wise addition
tc.softmax(input)        # Softmax along last dimension
tc.layernorm(input, gamma, beta, eps=1e-5)  # Layer normalization
tc.rmsnorm(input, weight, eps=1e-6)         # RMS normalization
tc.gemm(A, B, alpha=1.0, beta=0.0, version='tiled')  # Matrix multiplication
tc.transpose(input)      # Matrix transpose
```

---

## Activation Functions

All activation functions take a NumPy array and return a new array with the same shape.

### relu

```python
def relu(input: np.ndarray) -> np.ndarray
```

Apply ReLU activation: `max(0, x)`

**Parameters:**

- `input`: Input array of any shape

**Returns:** Output array with same shape

```python
import tensorcraft_ops as tc
import numpy as np

x = np.array([[-1.0, 2.0], [0.5, -0.5]], dtype=np.float32)
y = tc.relu(x)
# [[0.0, 2.0], [0.5, 0.0]]
```

### silu

```python
def silu(input: np.ndarray) -> np.ndarray
```

Apply SiLU (Swish) activation: `x * sigmoid(x)`

```python
y = tc.silu(x)
```

### gelu

```python
def gelu(input: np.ndarray) -> np.ndarray
```

Apply GeLU activation (approximate): `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`

```python
y = tc.gelu(x)
```

### sigmoid

```python
def sigmoid(input: np.ndarray) -> np.ndarray
```

Apply sigmoid activation: `1 / (1 + exp(-x))`

```python
y = tc.sigmoid(x)
```

---

## Vector Operations

### vector_add

```python
def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray
```

Element-wise addition of two arrays.

**Parameters:**

- `a`, `b`: Input arrays (must have same shape)

**Returns:** Element-wise sum

```python
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
c = tc.vector_add(a, b)
# [5.0, 7.0, 9.0]
```

---

## Softmax

### softmax

```python
def softmax(input: np.ndarray) -> np.ndarray
```

Apply softmax along the last dimension.

**Parameters:**

- `input`: Input array of any shape

**Returns:** Output with softmax applied along last dimension

```python
# 2D example
x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
y = tc.softmax(x)
# Each row sums to 1.0

# 3D example
x = np.random.randn(2, 3, 4).astype(np.float32)
y = tc.softmax(x)  # Softmax along dimension 4
```

---

## Normalization

### layernorm

```python
def layernorm(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray
```

Apply layer normalization.

**Parameters:**

- `input`: Input array of shape `(..., hidden_size)`
- `gamma`: Scale parameter of shape `(hidden_size,)`
- `beta`: Shift parameter of shape `(hidden_size,)`
- `eps`: Small constant for numerical stability

**Returns:** Normalized output with same shape as input

```python
batch_size, hidden_size = 32, 256

x = np.random.randn(batch_size, hidden_size).astype(np.float32)
gamma = np.ones(hidden_size, dtype=np.float32)
beta = np.zeros(hidden_size, dtype=np.float32)

y = tc.layernorm(x, gamma, beta)
```

### rmsnorm

```python
def rmsnorm(
    input: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray
```

Apply RMS normalization.

**Parameters:**

- `input`: Input array of shape `(..., hidden_size)`
- `weight`: Scale parameter of shape `(hidden_size,)`
- `eps`: Small constant for numerical stability

```python
x = np.random.randn(batch_size, hidden_size).astype(np.float32)
weight = np.ones(hidden_size, dtype=np.float32)

y = tc.rmsnorm(x, weight)
```

---

## Matrix Operations

### gemm

```python
def gemm(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    version: str = 'tiled'
) -> np.ndarray
```

General matrix multiplication: `C = alpha * A @ B + beta * C`

**Parameters:**

- `A`: Input matrix of shape `(M, K)`
- `B`: Input matrix of shape `(K, N)`
- `alpha`: Scalar multiplier for `A @ B`
- `beta`: Scalar multiplier for accumulator (currently ignored, output is initialized to 0)
- `version`: GEMM implementation to use
  - `'naive'`: Basic implementation
  - `'tiled'`: Shared memory tiling (default, recommended)
  - `'double_buffer'`: Double buffering optimization

**Returns:** Output matrix of shape `(M, N)`

```python
M, N, K = 128, 256, 64

A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

# Basic usage
C = tc.gemm(A, B)

# With specific version
C = tc.gemm(A, B, version='double_buffer')
```

### transpose

```python
def transpose(input: np.ndarray) -> np.ndarray
```

Transpose a 2D matrix.

**Parameters:**

- `input`: 2D input matrix of shape `(rows, cols)`

**Returns:** Transposed matrix of shape `(cols, rows)`

```python
A = np.random.randn(3, 5).astype(np.float32)
B = tc.transpose(A)
# B.shape == (5, 3)
```

---

## Error Handling

All operations may raise `RuntimeError` if a CUDA error occurs:

```python
import tensorcraft_ops as tc

try:
    result = tc.gemm(A, B)
except RuntimeError as e:
    print(f"CUDA error: {e}")
```

---

## Data Type Support

Currently, only `float32` is fully supported. Input arrays are automatically converted to `float32`:

```python
# Works - float32
x = np.array([1.0, 2.0], dtype=np.float32)
y = tc.relu(x)

# May lose precision - float64
x = np.array([1.0, 2.0], dtype=np.float64)
y = tc.relu(x)  # Converted to float32 internally
```

---

## Performance Tips

1. **Use contiguous arrays**: NumPy arrays should be C-contiguous for best performance

   ```python
   x = np.ascontiguousarray(x)
   ```

2. **Reuse memory**: For repeated operations, pre-allocate output buffers

3. **Batch operations**: Prefer larger batch sizes for better GPU utilization

4. **Choose the right GEMM version**:
   - `'tiled'`: Good balance of performance and compatibility
   - `'double_buffer'`: May be faster for very large matrices
   - `'naive'`: Useful for debugging/verification

---

## Complete Example

```python
import numpy as np
import tensorcraft_ops as tc

# Create sample data
batch_size, seq_len, hidden_size = 4, 128, 256

# Input tensor
x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

# Normalization parameters
gamma = np.ones(hidden_size, dtype=np.float32)
beta = np.zeros(hidden_size, dtype=np.float32)

# Apply operations
x_norm = tc.layernorm(x, gamma, beta)
x_act = tc.gelu(x_norm)
x_out = tc.softmax(x_act)

print(f"Output shape: {x_out.shape}")
print(f"Row sums (should be ~1.0): {x_out[0, 0, :].sum()}")

# GEMM example
M, N, K = 256, 512, 128
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
C = tc.gemm(A, B, version='tiled')
print(f"GEMM output shape: {C.shape}")
```
