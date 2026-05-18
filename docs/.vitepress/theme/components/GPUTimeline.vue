<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useData } from 'vitepress'

const { isDark } = useData()

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
  }, { threshold: 0.15 })

  if (timelineRef.value) {
    observer.observe(timelineRef.value)
  }
})
</script>

<template>
  <div class="gpu-timeline" ref="timelineRef" :class="{ dark: isDark }">
    <h2 class="timeline-title">
      <span class="accent-text">GPU Architecture</span> Support
    </h2>
    <p class="timeline-description">
      TensorCraft-HPC supports NVIDIA GPUs from Volta (2017) to Blackwell (2024),
      covering CUDA Compute Capability 7.0-10.0.
    </p>

    <div class="timeline-track">
      <div class="track-line" :class="{ visible: isVisible }"></div>
    </div>

    <div class="timeline-container">
      <div
        v-for="(gpu, index) in gpuArchitectures"
        :key="gpu.name"
        class="gpu-card"
        :class="{ visible: isVisible, supported: gpu.supported }"
        :style="{ animationDelay: `${index * 0.12}s` }"
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
            v-for="feature in gpu.features"
            :key="feature"
            class="feature-tag"
          >
            {{ feature }}
          </span>
        </div>

        <div class="gpu-status" :class="{ active: gpu.supported }">
          {{ gpu.supported ? 'Supported' : 'Planned' }}
        </div>
      </div>
    </div>

    <div class="cuda-badge">
      <span class="badge-icon">&#9889;</span>
      <span>CUDA 11.0 - 13.1</span>
    </div>
  </div>
</template>

<style scoped>
.gpu-timeline {
  padding: 3rem 0 2rem;
  max-width: 1200px;
  margin: 0 auto;
}

.timeline-title {
  font-family: 'Fraunces', serif;
  font-size: 2rem;
  text-align: center;
  margin-bottom: 0.6rem;
  color: var(--vp-c-text-1);
  font-weight: 700;
}

.accent-text {
  color: var(--tc-accent-primary);
}

.timeline-description {
  text-align: center;
  color: var(--vp-c-text-2);
  margin-bottom: 2.5rem;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;
  font-size: 0.95rem;
}

/* Track */
.timeline-track {
  position: relative;
  height: 4px;
  margin-bottom: -2px;
  z-index: 1;
  padding: 0 1rem;
}

.track-line {
  height: 100%;
  background: var(--vp-c-divider);
  border-radius: 2px;
  position: relative;
}

.track-line.visible::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  width: 100%;
  background: linear-gradient(90deg, var(--tc-accent-primary), var(--tc-accent-success));
  border-radius: 2px;
  animation: trackGrow 1.2s ease-out forwards;
}

@keyframes trackGrow {
  from { width: 0; }
  to { width: 100%; }
}

/* Cards */
.timeline-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 1.2rem;
  margin-bottom: 2rem;
  position: relative;
  z-index: 2;
}

.gpu-card {
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--tc-radius-md);
  padding: 1.3rem 1.1rem;
  opacity: 0;
  transform: translateY(18px);
  transition: all 0.5s cubic-bezier(0.22, 1, 0.36, 1);
  position: relative;
}

.gpu-card.visible {
  opacity: 1;
  transform: translateY(0);
}

.gpu-card:hover {
  border-color: var(--tc-accent-primary);
  box-shadow: var(--tc-shadow-glow), var(--tc-shadow-md);
  transform: translateY(-4px);
}

.gpu-card.supported::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--tc-accent-primary), var(--tc-accent-success));
  border-radius: var(--tc-radius-md) var(--tc-radius-md) 0 0;
}

.gpu-header {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  margin-bottom: 0.9rem;
}

.gpu-year {
  font-size: 0.72rem;
  color: var(--tc-accent-primary);
  font-weight: 700;
  letter-spacing: 0.05em;
  font-family: var(--vp-font-family-mono);
}

.gpu-name {
  font-size: 1.35rem;
  font-weight: 800;
  margin: 0;
  font-family: 'Fraunces', serif;
  color: var(--vp-c-text-1);
}

.gpu-code {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}

.gpu-specs {
  display: flex;
  gap: 1rem;
  margin-bottom: 0.9rem;
}

.spec-item {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}

.spec-label {
  font-size: 0.6rem;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-weight: 700;
}

.spec-value {
  font-size: 0.82rem;
  font-family: var(--vp-font-family-mono);
  color: var(--tc-accent-primary);
  font-weight: 600;
}

.gpu-features {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 0.9rem;
}

.feature-tag {
  font-size: 0.68rem;
  padding: 0.22rem 0.5rem;
  background: var(--tc-blue-soft);
  border: 1px solid color-mix(in srgb, var(--tc-accent-primary) 15%, var(--vp-c-divider));
  border-radius: 4px;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
  font-weight: 500;
}

.gpu-status {
  font-size: 0.72rem;
  color: var(--vp-c-text-3);
  padding-top: 0.7rem;
  border-top: 1px solid var(--vp-c-divider);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.gpu-status.active {
  color: var(--tc-accent-success);
}

/* Badge */
.cuda-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.7rem 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 100px;
  width: fit-content;
  margin: 0 auto;
  font-family: var(--vp-font-family-mono);
  font-size: 0.82rem;
  color: var(--vp-c-text-2);
  font-weight: 600;
}

.badge-icon {
  font-size: 1rem;
  color: var(--tc-accent-primary);
}

@media (max-width: 768px) {
  .timeline-container {
    grid-template-columns: repeat(2, 1fr);
  }

  .timeline-title {
    font-size: 1.5rem;
  }

  .track-line.visible::after {
    animation: none;
    width: 100%;
  }
}

@media (max-width: 480px) {
  .timeline-container {
    grid-template-columns: 1fr;
  }
}
</style>
