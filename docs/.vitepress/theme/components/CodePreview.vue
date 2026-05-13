<script setup lang="ts">
import { ref } from 'vue'

const activeLang = ref<'cpp' | 'python'>('cpp')
const copied = ref(false)

const cppCode = `// C++: High-performance GEMM with TensorCraft
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/memory/tensor.hpp"

// Create tensors (RAII-managed GPU memory)
tensorcraft::FloatTensor A({4096, 4096});
tensorcraft::FloatTensor B({4096, 4096});
tensorcraft::FloatTensor C({4096, 4096});

// Optimized GEMM: C = α(A × B) + βC
tensorcraft::kernels::gemm(
    A.data(), B.data(), C.data(),
    4096, 4096, 4096);`

const pythonCode = `# Python: Simple NumPy-compatible API
import tensorcraft_ops as tc
import numpy as np

# Matrix multiplication
A = np.random.randn(4096, 4096).astype(np.float32)
B = np.random.randn(4096, 4096).astype(np.float32)

# FlashAttention-style operation
Q = np.random.randn(32, 128, 64).astype(np.float32)
K = np.random.randn(32, 128, 64).astype(np.float32)
V = np.random.randn(32, 128, 64).astype(np.float32)

output = tc.flash_attention(Q, K, V)`

const currentCode = computed(() =>
  activeLang.value === 'cpp' ? cppCode : pythonCode
)

function copyCode() {
  navigator.clipboard.writeText(currentCode.value)
  copied.value = true
  setTimeout(() => {
    copied.value = false
  }, 2000)
}
</script>

<template>
  <div class="code-preview">
    <div class="code-window">
      <div class="code-header">
        <div class="code-dots">
          <span class="dot red"></span>
          <span class="dot yellow"></span>
          <span class="dot green"></span>
        </div>
        <div class="code-tabs">
          <button
            class="code-tab"
            :class="{ active: activeLang === 'cpp' }"
            @click="activeLang = 'cpp'"
          >
            C++
          </button>
          <button
            class="code-tab"
            :class="{ active: activeLang === 'python' }"
            @click="activeLang = 'python'"
          >
            Python
          </button>
        </div>
        <button class="copy-btn" @click="copyCode">
          {{ copied ? '✓ Copied!' : 'Copy' }}
        </button>
      </div>
      <div class="code-content">
        <pre><code>{{ currentCode }}</code></pre>
      </div>
    </div>
  </div>
</template>

<style scoped>
.code-preview {
  margin: 2rem 0;
}

.code-window {
  background: var(--vp-c-black-mute);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  overflow: hidden;
}

.code-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem 1rem;
  background: var(--vp-c-gray);
  border-bottom: 1px solid var(--vp-c-border);
}

.code-dots {
  display: flex;
  gap: 6px;
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.dot.red {
  background: #ff5f56;
}

.dot.yellow {
  background: #ffbd2e;
}

.dot.green {
  background: #27c93f;
}

.code-tabs {
  display: flex;
  gap: 0.5rem;
}

.code-tab {
  padding: 0.375rem 0.75rem;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.code-tab:hover {
  color: var(--vp-c-text-2);
}

.code-tab.active {
  background: var(--vp-c-gray-light);
  border-color: var(--nvidia-green);
  color: var(--nvidia-green);
}

.copy-btn {
  padding: 0.375rem 0.75rem;
  background: transparent;
  border: 1px solid var(--vp-c-border);
  border-radius: 6px;
  color: var(--vp-c-text-3);
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.copy-btn:hover {
  border-color: var(--nvidia-green);
  color: var(--nvidia-green);
}

.code-content {
  padding: 1rem 1.5rem;
  overflow-x: auto;
}

.code-content pre {
  margin: 0;
}

.code-content code {
  font-family: var(--vp-font-family-mono);
  font-size: 0.875rem;
  line-height: 1.6;
  color: var(--vp-c-text-2);
}

@media (max-width: 768px) {
  .code-header {
    flex-wrap: wrap;
    gap: 0.75rem;
  }

  .code-dots {
    order: 1;
  }

  .code-tabs {
    order: 2;
    flex: 1;
    justify-content: center;
  }

  .copy-btn {
    order: 3;
  }

  .code-content {
    padding: 1rem;
  }

  .code-content code {
    font-size: 0.75rem;
  }
}
</style>