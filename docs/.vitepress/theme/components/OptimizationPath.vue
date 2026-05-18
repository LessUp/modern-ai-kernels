<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useData } from 'vitepress'

const { isDark } = useData()

interface OptimizationStep {
  name: string
  description: string
  techniques: string[]
  performance: number
}

const steps: OptimizationStep[] = [
  {
    name: 'Naive',
    description: 'Direct triple loop implementation',
    techniques: ['Global memory access', 'No parallelism optimization'],
    performance: 5
  },
  {
    name: 'Tiled',
    description: 'Shared memory blocking',
    techniques: ['Block-level tiling', 'Shared memory reuse', 'Coalesced access'],
    performance: 45
  },
  {
    name: 'Double Buffer',
    description: 'Pipeline memory access',
    techniques: ['Prefetching', 'Latency hiding', 'Warp synchronization'],
    performance: 75
  },
  {
    name: 'Tensor Core',
    description: 'WMMA hardware acceleration',
    techniques: ['WMMA instructions', 'Mixed precision', 'Maximum throughput'],
    performance: 92
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
</script>

<template>
  <div class="optimization-path" ref="containerRef">
    <h3 class="path-title">
      <span class="accent-text">Optimization</span> Path
    </h3>
    <p class="path-description">
      Progressive optimization from naive to Tensor Core implementation
    </p>

    <div class="path-container">
      <!-- Progress line -->
      <div class="progress-line">
        <div class="progress-fill" :class="{ visible: isVisible }"></div>

        <!-- Step markers -->
        <div
          v-for="(step, index) in steps"
          :key="step.name"
          class="step-marker"
          :style="{ left: `${index * 33.33}%` }"
          :class="{ active: isVisible }"
        >
          <div class="marker-dot">
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
          :style="{ animationDelay: `${index * 0.15}s` }"
        >
          <div class="card-header">
            <span class="step-number">{{ index + 1 }}</span>
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
        <span class="footer-icon">&#128202;</span>
        Performance relative to cuBLAS on A100 80GB
      </span>
    </div>
  </div>
</template>

<style scoped>
.optimization-path {
  padding: 2.5rem 0;
  max-width: 900px;
  margin: 0 auto;
}

.path-title {
  font-family: 'Fraunces', serif;
  font-size: 1.4rem;
  text-align: center;
  margin-bottom: 0.4rem;
  color: var(--vp-c-text-1);
  font-weight: 700;
}

.accent-text {
  color: var(--tc-accent-primary);
}

.path-description {
  text-align: center;
  color: var(--vp-c-text-2);
  margin-bottom: 2rem;
  font-size: 0.92rem;
}

.path-container {
  position: relative;
}

/* Progress line */
.progress-line {
  position: relative;
  height: 56px;
  margin-bottom: 1.2rem;
  background: var(--vp-c-bg-soft);
  border-radius: var(--tc-radius-md);
  padding: 0 20px;
  border: 1px solid var(--vp-c-divider);
}

.progress-fill {
  position: absolute;
  top: 50%;
  left: 20px;
  right: 20px;
  height: 4px;
  background: var(--vp-c-divider);
  border-radius: 2px;
  transform: translateY(-50%);
}

.progress-fill.visible {
  background: linear-gradient(90deg, var(--tc-accent-primary), var(--tc-accent-success));
  animation: lineGrow 1.2s ease-out forwards;
}

@keyframes lineGrow {
  from { transform: translateY(-50%) scaleX(0); transform-origin: left; }
  to { transform: translateY(-50%) scaleX(1); transform-origin: left; }
}

/* Step markers */
.step-marker {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  opacity: 0;
  transition: opacity 0.5s ease;
}

.step-marker.active {
  opacity: 1;
}

.marker-dot {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px solid var(--vp-c-bg);
  background: var(--tc-accent-primary);
  box-shadow: 0 2px 8px color-mix(in srgb, var(--tc-accent-primary) 30%, transparent);
}

.marker-number {
  font-size: 13px;
  font-weight: 700;
  color: var(--tc-paper);
  font-family: var(--vp-font-family-mono);
}

.marker-label {
  font-size: 10px;
  font-weight: 700;
  color: var(--vp-c-text-2);
  white-space: nowrap;
  font-family: var(--vp-font-family-mono);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

/* Step cards */
.steps-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.step-card {
  background: var(--vp-c-bg-soft);
  border-radius: var(--tc-radius-md);
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  opacity: 0;
  transform: translateY(20px);
  transition: all 0.5s cubic-bezier(0.22, 1, 0.36, 1);
}

.step-card.visible {
  opacity: 1;
  transform: translateY(0);
}

.step-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--tc-shadow-md);
  border-color: color-mix(in srgb, var(--tc-accent-primary) 30%, var(--vp-c-divider));
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.step-number {
  font-size: 22px;
  font-weight: 800;
  color: var(--tc-accent-primary);
  font-family: 'Fraunces', serif;
}

.step-name {
  font-size: 14px;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin: 0;
  font-family: 'Fraunces', serif;
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
  padding: 2px 7px;
  background: var(--vp-c-bg-mute);
  border-radius: 4px;
  color: var(--vp-c-text-2);
  font-family: var(--vp-font-family-mono);
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
  background: linear-gradient(90deg, var(--tc-accent-primary), var(--tc-accent-success));
  transition: width 1.5s cubic-bezier(0.22, 1, 0.36, 1);
}

.bar-fill.animate {
  transition: width 1.5s cubic-bezier(0.22, 1, 0.36, 1);
}

.perf-label {
  font-size: 11px;
  font-weight: 700;
  color: var(--tc-paper);
  font-family: var(--vp-font-family-mono);
  text-shadow: 0 1px 2px rgba(0,0,0,0.25);
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
  font-size: 11px;
  color: var(--vp-c-text-3);
  padding: 8px 16px;
  background: var(--vp-c-bg-soft);
  border-radius: var(--tc-radius-md);
  font-family: var(--vp-font-family-mono);
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
