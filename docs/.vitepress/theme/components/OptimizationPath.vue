<script setup lang="ts">
import { ref, onMounted } from 'vue'

interface OptimizationStep {
  name: string
  description: string
  techniques: string[]
  performance: number
  color: string
}

const steps: OptimizationStep[] = [
  {
    name: 'Naive',
    description: 'Direct triple loop implementation',
    techniques: ['Global memory access', 'No parallelism optimization'],
    performance: 5,
    color: '#6e7681'
  },
  {
    name: 'Tiled',
    description: 'Shared memory blocking',
    techniques: ['Block-level tiling', 'Shared memory reuse', 'Coalesced access'],
    performance: 45,
    color: '#5A9100'
  },
  {
    name: 'Double Buffer',
    description: 'Pipeline memory access',
    techniques: ['Prefetching', 'Latency hiding', 'Warp synchronization'],
    performance: 75,
    color: '#76B900'
  },
  {
    name: 'Tensor Core',
    description: 'WMMA hardware acceleration',
    techniques: ['WMMA instructions', 'Mixed precision', 'Maximum throughput'],
    performance: 92,
    color: '#8ED000'
  }
]

const isVisible = ref(false)
const containerRef = ref<HTMLElement | null>(null)

onMounted(() => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        isVisible.value = true
      }
    })
  }, { threshold: 0.2 })

  if (containerRef.value) {
    observer.observe(containerRef.value)
  }
})

function getPerformanceColor(percentage: number): string {
  if (percentage < 30) return 'var(--vp-c-text-3)'
  if (percentage < 60) return '#FFB800'
  return 'var(--vp-c-brand-1)'
}
</script>

<template>
  <div class="optimization-path" ref="containerRef">
    <h3 class="path-title">
      <span class="brand-text">Optimization</span> Path
    </h3>
    <p class="path-description">
      Progressive optimization from naive to Tensor Core implementation
    </p>

    <div class="path-container">
      <!-- Progress line -->
      <div class="progress-line">
        <div
          class="progress-fill"
          :class="{ visible: isVisible }"
        ></div>

        <!-- Step markers -->
        <div
          v-for="(step, index) in steps"
          :key="step.name"
          class="step-marker"
          :style="{ left: `${index * 33.33}%` }"
          :class="{ active: isVisible }"
        >
          <div class="marker-dot" :style="{ backgroundColor: step.color }">
            <span class="marker-number">{{ index + 1 }}</span>
          </div>
          <span class="marker-label">{{ step.name }}</span>
        </div>
      </div>

      <!-- Step cards -->
      <div class="steps-grid">
        <div
          v-for="(step, index) in steps"
          :key="step.name"
          class="step-card"
          :class="{ visible: isVisible }"
          :style="{
            animationDelay: `${index * 0.15}s`,
            borderColor: step.color
          }"
        >
          <div class="card-header">
            <span class="step-number" :style="{ color: step.color }">{{ index + 1 }}</span>
            <h4 class="step-name">{{ step.name }}</h4>
          </div>

          <p class="step-desc">{{ step.description }}</p>

          <div class="techniques">
            <span
              v-for="tech in step.techniques"
              :key="tech"
              class="tech-tag"
            >
              {{ tech }}
            </span>
          </div>

          <div class="performance-bar">
            <div class="bar-bg">
              <div
                class="bar-fill"
                :class="{ animate: isVisible }"
                :style="{
                  width: isVisible ? `${step.performance}%` : '0%',
                  backgroundColor: step.color,
                  animationDelay: `${index * 0.2}s`
                }"
              >
                <span class="perf-label">{{ step.performance }}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="path-footer">
      <span class="footer-item">
        <span class="footer-icon">📊</span>
        Performance relative to cuBLAS on A100 80GB
      </span>
    </div>
  </div>
</template>

<style scoped>
.optimization-path {
  padding: 3rem 0;
  max-width: 900px;
  margin: 0 auto;
}

.path-title {
  font-size: 1.5rem;
  text-align: center;
  margin-bottom: 0.5rem;
  color: var(--vp-c-text-1);
}

.brand-text {
  color: var(--vp-c-brand-1);
}

.path-description {
  text-align: center;
  color: var(--vp-c-text-2);
  margin-bottom: 2rem;
}

.path-container {
  position: relative;
}

/* Progress line */
.progress-line {
  position: relative;
  height: 60px;
  margin-bottom: 1rem;
  background: var(--vp-c-bg-soft);
  border-radius: var(--radius-md);
  padding: 0 20px;
}

.progress-fill {
  position: absolute;
  top: 50%;
  left: 20px;
  right: 20px;
  height: 4px;
  background: var(--vp-c-border);
  border-radius: 2px;
}

.progress-fill.visible {
  background: linear-gradient(90deg, #5A9100, #76B900, #8ED000);
}

/* Step markers */
.step-marker {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  opacity: 0;
  transition: opacity 0.5s ease;
}

.step-marker.active {
  opacity: 1;
}

.marker-dot {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 3px solid var(--vp-c-bg);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.marker-number {
  font-size: 14px;
  font-weight: 700;
  color: #000;
}

.marker-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--vp-c-text-2);
  white-space: nowrap;
}

/* Step cards */
.steps-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.step-card {
  background: var(--vp-c-bg-soft);
  border-radius: var(--radius-md);
  padding: 16px;
  border-width: 2px;
  border-style: solid;
  opacity: 0;
  transform: translateY(20px);
  transition: all 0.5s ease;
}

.step-card.visible {
  opacity: 1;
  transform: translateY(0);
}

.step-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.step-number {
  font-size: 24px;
  font-weight: 700;
}

.step-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0;
}

.step-desc {
  font-size: 12px;
  color: var(--vp-c-text-2);
  margin-bottom: 12px;
  line-height: 1.4;
}

.techniques {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 12px;
}

.tech-tag {
  font-size: 10px;
  padding: 2px 6px;
  background: var(--vp-c-bg-mute);
  border-radius: 4px;
  color: var(--vp-c-text-2);
}

.performance-bar {
  margin-top: auto;
}

.bar-bg {
  height: 24px;
  background: var(--vp-c-bg-mute);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}

.bar-fill {
  height: 100%;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 8px;
  transition: width 1s ease-out;
}

.bar-fill.animate {
  transition: width 1.5s ease-out;
}

.perf-label {
  font-size: 11px;
  font-weight: 600;
  color: #000;
}

.path-footer {
  display: flex;
  justify-content: center;
  margin-top: 1.5rem;
}

.footer-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--vp-c-text-3);
  padding: 8px 16px;
  background: var(--vp-c-bg-soft);
  border-radius: var(--radius-md);
}

.footer-icon {
  font-size: 14px;
}

/* Responsive */
@media (max-width: 768px) {
  .steps-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .progress-line {
    display: none;
  }

  .step-marker {
    display: none;
  }
}

@media (max-width: 480px) {
  .steps-grid {
    grid-template-columns: 1fr;
  }
}
</style>