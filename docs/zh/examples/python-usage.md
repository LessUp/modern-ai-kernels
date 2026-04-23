---
title: Python Usage Examples
lang: zh
---

# Python Usage Examples

Complete examples for using TensorCraft-HPC Python bindings.

## Installation

```bash
# From source
python3 -m pip install -e .

# Verify installation
python3 -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

---

## Quick Start

```python
import numpy as np
import tensorcraft_ops as tc

# Create sample data
x = np.random.randn(32, 256).astype(np.float32)

# Activation functions
x_relu = tc.relu(x)
x_gelu = tc.gelu(x)
x_silu = tc.silu(x)

# Normalization
gamma = np.ones(256, dtype=np.float32)
beta = np.zeros(256, dtype=np.float32)
x_norm = tc.layernorm(x, gamma, beta)

# GEMM
A = np.random.randn(128, 64).astype(np.float32)
B = np.random.randn(64, 256).astype(np.float32)
C = tc.gemm(A, B)
```

---

## Complete Neural Network Layer Example

```python
import numpy as np
import tensorcraft_ops as tc

class SimpleTransformerLayer:
    """A simple transformer layer using TensorCraft-HPC kernels."""
    
    def __init__(self, hidden_size, num_heads, intermediate_size):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        
        # Initialize weights (simplified)
        scale = np.sqrt(2.0 / hidden_size)
        
        # Attention weights
        self.Wq = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.Wk = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.Wv = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        self.Wo = np.random.randn(hidden_size, hidden_size).astype(np.float32) * scale
        
        # FFN weights
        self.W1 = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * scale
        self.W2 = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * scale
        
        # LayerNorm parameters
        self.ln1_gamma = np.ones(hidden_size, dtype=np.float32)
        self.ln1_beta = np.zeros(hidden_size, dtype=np.float32)
        self.ln2_gamma = np.ones(hidden_size, dtype=np.float32)
        self.ln2_beta = np.zeros(hidden_size, dtype=np.float32)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Output tensor of same shape
        """
        batch_size, seq_len, _ = x.shape
        
        # Pre-attention LayerNorm
        x_norm = tc.layernorm(x, self.ln1_gamma, self.ln1_beta)
        
        # Compute Q, K, V projections
        Q = tc.gemm(x_norm.reshape(-1, self.hidden_size), self.Wq)
        K = tc.gemm(x_norm.reshape(-1, self.hidden_size), self.Wk)
        V = tc.gemm(x_norm.reshape(-1, self.hidden_size), self.Wv)
        
        # Reshape for attention (simplified - using GEMM for demonstration)
        # In practice, you would use tc FlashAttention here
        
        # Output projection
        attn_out = tc.gemm(Q, self.Wo)
        
        # Residual connection
        x = x + attn_out.reshape(batch_size, seq_len, self.hidden_size)
        
        # Pre-FFN LayerNorm
        x_norm = tc.layernorm(x, self.ln2_gamma, self.ln2_beta)
        
        # FFN with GELU
        ffn_hidden = tc.gemm(x_norm.reshape(-1, self.hidden_size), self.W1)
        ffn_hidden = tc.gelu(ffn_hidden)
        ffn_out = tc.gemm(ffn_hidden, self.W2)
        
        # Residual connection
        x = x + ffn_out.reshape(batch_size, seq_len, self.hidden_size)
        
        return x


# Usage example
if __name__ == "__main__":
    # Create layer
    layer = SimpleTransformerLayer(
        hidden_size=256,
        num_heads=8,
        intermediate_size=512
    )
    
    # Create input
    batch_size, seq_len = 4, 128
    x = np.random.randn(batch_size, seq_len, 256).astype(np.float32)
    
    # Forward pass
    output = layer.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

---

## Batch Processing Example

```python
import numpy as np
import tensorcraft_ops as tc

def process_batch(inputs, gamma, beta, weight):
    """
    Process a batch of sequences with normalization and activation.
    
    Args:
        inputs: List of input arrays, each of shape (seq_len, hidden_size)
        gamma, beta: LayerNorm parameters
        weight: Output projection weight
    
    Returns:
        List of processed outputs
    """
    outputs = []
    
    for x in inputs:
        # LayerNorm
        x_norm = tc.layernorm(x, gamma, beta)
        
        # Activation
        x_act = tc.gelu(x_norm)
        
        # Projection
        out = tc.gemm(x_act, weight)
        
        outputs.append(out)
    
    return outputs


# Usage
batch_size = 8
seq_len = 64
hidden_size = 256

inputs = [np.random.randn(seq_len, hidden_size).astype(np.float32) 
          for _ in range(batch_size)]
gamma = np.ones(hidden_size, dtype=np.float32)
beta = np.zeros(hidden_size, dtype=np.float32)
weight = np.random.randn(hidden_size, 128).astype(np.float32)

outputs = process_batch(inputs, gamma, beta, weight)
print(f"Processed {len(outputs)} sequences")
```

---

## Softmax Example

```python
import numpy as np
import tensorcraft_ops as tc

def attention_scores(Q, K):
    """
    Compute attention scores (without masking).
    
    Args:
        Q: Query tensor (batch, heads, seq, head_dim)
        K: Key tensor (batch, heads, seq, head_dim)
    
    Returns:
        Attention weights (batch, heads, seq, seq)
    """
    batch, heads, seq, head_dim = Q.shape
    
    # Compute QK^T (simplified with reshape)
    Q_flat = Q.reshape(-1, head_dim)
    K_flat = K.reshape(-1, head_dim)
    
    # Scale factor
    scale = 1.0 / np.sqrt(head_dim)
    
    # Compute attention scores (simplified)
    scores = tc.gemm(Q_flat, K_flat.T) * scale
    scores = scores.reshape(batch * heads, seq, seq)
    
    # Apply softmax to each row
    attn_weights = tc.softmax(scores)
    
    return attn_weights.reshape(batch, heads, seq, seq)


# Verify softmax properties
def test_softmax():
    x = np.random.randn(10, 100).astype(np.float32)
    y = tc.softmax(x)
    
    # Check 1: All values positive
    assert np.all(y >= 0), "Softmax output should be non-negative"
    
    # Check 2: Rows sum to 1
    row_sums = y.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Softmax rows should sum to 1"
    
    print("Softmax tests passed!")


test_softmax()
```

---

## Comparing GEMM Versions

```python
import numpy as np
import tensorcraft_ops as tc
import time

def benchmark_gemm(M, N, K, iterations=100):
    """Compare different GEMM implementations."""
    
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    versions = ['naive', 'tiled', 'double_buffer']
    results = {}
    
    for version in versions:
        # Warmup
        for _ in range(10):
            _ = tc.gemm(A, B, version=version)
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            C = tc.gemm(A, B, version=version)
        end = time.perf_counter()
        
        ms_per_iter = (end - start) / iterations * 1000
        gflops = 2.0 * M * N * K / (ms_per_iter * 1e6)
        
        results[version] = {
            'ms': ms_per_iter,
            'gflops': gflops,
            'result': C
        }
        
        print(f"{version}: {ms_per_iter:.3f} ms, {gflops:.1f} GFLOPS")
    
    # Verify correctness
    ref = results['tiled']['result']
    for version in ['naive', 'double_buffer']:
        diff = np.abs(results[version]['result'] - ref).max()
        print(f"{version} vs tiled max diff: {diff:.2e}")
    
    return results


# Run benchmark
benchmark_gemm(M=256, N=512, K=128)
```

---

## End-to-End Example: Simple MLP

```python
import numpy as np
import tensorcraft_ops as tc

class SimpleMLP:
    """A simple 2-layer MLP using TensorCraft-HPC kernels."""
    
    def __init__(self, input_size, hidden_size, output_size):
        scale = np.sqrt(2.0 / input_size)
        
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * scale
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, output_size).astype(np.float32) * scale
        self.b2 = np.zeros(output_size, dtype=np.float32)
    
    def forward(self, x):
        # First layer
        h = tc.gemm(x, self.W1)
        h = h + self.b1  # Add bias
        h = tc.gelu(h)   # Activation
        
        # Second layer
        out = tc.gemm(h, self.W2)
        out = out + self.b2
        
        return out
    
    def __call__(self, x):
        return self.forward(x)


# Usage
mlp = SimpleMLP(input_size=784, hidden_size=256, output_size=10)

# Simulate a batch of MNIST-like inputs
batch_size = 32
x = np.random.randn(batch_size, 784).astype(np.float32)

# Forward pass
logits = mlp(x)
probs = tc.softmax(logits)

print(f"Input: {x.shape}")
print(f"Logits: {logits.shape}")
print(f"Probabilities: {probs.shape}")
print(f"Sum of probs: {probs.sum(axis=1)}")  # Should be close to 1.0
```

---

## Error Handling

```python
import tensorcraft_ops as tc
import numpy as np

try:
    # This might fail if dimensions don't match
    A = np.random.randn(100, 50).astype(np.float32)
    B = np.random.randn(60, 80).astype(np.float32)  # Wrong dimension
    C = tc.gemm(A, B)
except RuntimeError as e:
    print(f"CUDA error: {e}")
except ValueError as e:
    print(f"Input error: {e}")
```

---

## Performance Tips

1. **Use float32**: The library is optimized for float32

   ```python
   x = x.astype(np.float32)  # Convert if needed
   ```

2. **Use contiguous arrays**:

   ```python
   x = np.ascontiguousarray(x)
   ```

3. **Batch operations**:

   ```python
   # Instead of processing one at a time
   for item in items:
       result = tc.relu(item)
   
   # Process all at once
   all_items = np.stack(items)
   results = tc.relu(all_items)
   ```

4. **Choose the right GEMM version**:
   - `'tiled'`: Best for most cases
   - `'double_buffer'`: May help for very large matrices
   - `'naive'`: Only for debugging
