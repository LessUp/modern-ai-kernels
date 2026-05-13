<script setup lang="ts">
import { ref, onMounted } from 'vue'

interface GPUArchitecture {
  name: string
  codeName: string
  year: number
  computeCapability: string
  sm: string
  features: string[]
  supported: boolean
}

const gpuArchitectures: GPUArchitecture[] = [
  {
    name: 'Volta',
    codeName: 'GV100',
    year: 2017,
    computeCapability: '7.0',
    sm: 'SM70',
    features: ['Tensor Cores (1st Gen)', 'NVLink 2.0', 'HBM2'],
    supported: true
  },
  {
    name: 'Turing',
    codeName: 'TU102',
    year: 2018,
    computeCapability: '7.5',
    sm: 'SM75',
    features: ['Tensor Cores (2nd Gen)', 'RT Cores', 'GDDR6'],
    supported: true
  },
  {
    name: 'Ampere',
    codeName: 'GA100',
    year: 2020,
    computeCapability: '8.0/8.6',
    sm: 'SM80/SM86',
    features: ['Tensor Cores (3rd Gen)', 'BF16', 'TF32', 'Sparsity'],
    supported: true
  },
  {
    name: 'Hopper',
    codeName: 'GH100',
    year: 2022,
    computeCapability: '9.0',
    sm: 'SM90',
    features: ['Tensor Cores (4th Gen)', 'FP8', 'Transformer Engine', 'DPX'],
    supported: true
  },
  {
    name: 'Blackwell',
    codeName: 'GB100',
    year: 2024,
    computeCapability: '10.0',
    sm: 'SM100',
    features: ['Tensor Cores (5th Gen)', 'NVLink 5.0', 'FP4/FP8', 'RAS'],
    supported: true
  }
]

const isVisible = ref(false)
const timelineRef = ref<HTMLElement | null>(null)

onMounted(() => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        isVisible.value = true
      }
    })
  }, { threshold: 0.2 })

  if (timelineRef.value) {
    observer.observe(timelineRef.value)
  }
})
</script>

<template>
  <div class="gpu-timeline" ref="timelineRef">
    <h2 class="timeline-title">
      <span class="nvidia-text">GPU Architecture</span> Support
    </h2>
    <p class="timeline-description">
      TensorCraft-HPC supports NVIDIA GPUs from Volta (2017) to Blackwell (2024),
      covering CUDA Compute Capability 7.0-10.0.
    </p>

    <div class="timeline-container">
      <div
        v-for="(gpu, index) in gpuArchitectures"
        :key="gpu.name"
        class="gpu-card"
        :class="{ visible: isVisible, supported: gpu.supported }"
        :style="{ animationDelay: `${index * 0.1}s` }"
      >
        <div class="gpu-header">
          <span class="gpu-year">{{ gpu.year }}</span>
          <h3 class="gpu-name">{{ gpu.name }}</h3>
          <span class="gpu-code">{{ gpu.codeName }}</span>
        </div>

        <div class="gpu-specs">
          <div class="spec-item">
            <span class="spec-label">SM</span>
            <span class="spec-value">{{ gpu.sm }}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">CC</span>
            <span class="spec-value">{{ gpu.computeCapability }}</span>
          </div>
        </div>

        <div class="gpu-features">
          <span
            v-for="feature in gpu.features.slice(0, 3)"
            :key="feature"
            class="feature-tag"
          >
            {{ feature }}
          </span>
        </div>

        <div class="gpu-status" :class="{ active: gpu.supported }">
          {{ gpu.supported ? '✓ Supported' : '○ Planned' }}
        </div>
      </div>
    </div>

    <div class="cuda-badge">
      <span class="badge-icon">⚡</span>
      <span>CUDA 11.0 - 13.1</span>
    </div>
  </div>
</template>

<style scoped>
.gpu-timeline {
  padding: 4rem 0;
  max-width: 1200px;
  margin: 0 auto;
}

.timeline-title {
  font-size: 2rem;
  text-align: center;
  margin-bottom: 1rem;
  color: var(--vp-c-text-1);
}

.nvidia-text {
  color: var(--nvidia-green);
}

.timeline-description {
  text-align: center;
  color: var(--vp-c-text-2);
  margin-bottom: 3rem;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
}

.timeline-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.gpu-card {
  background: var(--vp-c-gray);
  border: 1px solid var(--vp-c-border);
  border-radius: 12px;
  padding: 1.5rem;
  opacity: 0;
  transform: translateY(20px);
  transition: all 0.4s ease;
}

.gpu-card.visible {
  opacity: 1;
  transform: translateY(0);
}

.gpu-card:hover {
  border-color: var(--nvidia-green);
  box-shadow: 0 0 30px var(--nvidia-green-glow);
  transform: translateY(-4px);
}

.gpu-card.supported {
  border-left: 3px solid var(--nvidia-green);
}

.gpu-header {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin-bottom: 1rem;
}

.gpu-year {
  font-size: 0.75rem;
  color: var(--nvidia-green);
  font-weight: 600;
  letter-spacing: 0.05em;
}

.gpu-name {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(135deg, #fff, var(--nvidia-green));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.gpu-code {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}

.gpu-specs {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
}

.spec-item {
  display: flex;
  flex-direction: column;
  gap: 0.125rem;
}

.spec-label {
  font-size: 0.625rem;
  color: var(--vp-c-text-4);
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.spec-value {
  font-size: 0.875rem;
  font-family: var(--vp-font-family-mono);
  color: var(--nvidia-green);
}

.gpu-features {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.feature-tag {
  font-size: 0.625rem;
  padding: 0.25rem 0.5rem;
  background: rgba(118, 185, 0, 0.1);
  border: 1px solid rgba(118, 185, 0, 0.2);
  border-radius: 4px;
  color: var(--nvidia-green);
  font-family: var(--vp-font-family-mono);
}

.gpu-status {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  padding-top: 0.75rem;
  border-top: 1px solid var(--vp-c-border);
}

.gpu-status.active {
  color: var(--nvidia-green);
}

.cuda-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  background: var(--vp-c-gray);
  border: 1px solid var(--nvidia-green);
  border-radius: 100px;
  width: fit-content;
  margin: 0 auto;
  font-family: var(--vp-font-family-mono);
  font-size: 0.875rem;
  color: var(--nvidia-green);
}

.badge-icon {
  font-size: 1rem;
}

@media (max-width: 768px) {
  .timeline-container {
    grid-template-columns: 1fr;
  }

  .timeline-title {
    font-size: 1.5rem;
  }
}
</style>