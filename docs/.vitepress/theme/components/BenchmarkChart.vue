<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useData } from 'vitepress'

const { isDark } = useData()

interface Benchmark {
  name: string
  target: string
  percentage: number
  details?: string
}

const benchmarks: Benchmark[] = [
  { name: 'GEMM (FP16)', target: 'cuBLAS', percentage: 92, details: 'Tensor Core enabled' },
  { name: 'FlashAttention', target: 'cuDNN', percentage: 85, details: 'Memory-efficient tiling' },
  { name: 'LayerNorm', target: 'cuDNN', percentage: 95, details: 'Fused kernel' },
  { name: 'Conv2D', target: 'cuDNN', percentage: 78, details: 'Im2Col optimization' },
  { name: 'SpMV (CSR)', target: 'cuSPARSE', percentage: 88, details: 'CSR format' }
]

const isVisible = ref(false)
const chartRef = ref<HTMLElement | null>(null)
const selectedBenchmark = ref<Benchmark | null>(null)

onMounted(() => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        isVisible.value = true
      }
    })
  }, { threshold: 0.2 })

  if (chartRef.value) {
    observer.observe(chartRef.value)
  }
})

function getBarColor(percentage: number): string {
  // Use CSS variable-driven colors instead of hardcoded NVIDIA greens
  if (percentage >= 90) return 'var(--tc-accent-success)'
  if (percentage >= 80) return 'var(--tc-accent-primary)'
  if (percentage >= 70) return 'var(--tc-accent-warning)'
  return '#94a3b8'
}

function selectBenchmark(bench: Benchmark) {
  selectedBenchmark.value = selectedBenchmark.value === bench ? null : bench
}
</script>

<template>
  <div class="benchmark-chart" ref="chartRef">
    <h2 class="chart-title">
      <span class="accent-text">Performance</span> Benchmarks
    </h2>
    <p class="chart-description">
      Relative performance compared to NVIDIA libraries on A100 80GB (FP16 Tensor Core)
    </p>

    <div class="chart-container">
      <div
        v-for="(bench, index) in benchmarks"
        :key="bench.name"
        class="benchmark-row"
        :class="{ selected: selectedBenchmark === bench }"
        :style="{ animationDelay: `${index * 0.08}s` }"
        @click="selectBenchmark(bench)"
      >
        <div class="benchmark-info">
          <div class="benchmark-header">
            <span class="benchmark-name">{{ bench.name }}</span>
            <span class="benchmark-target">vs {{ bench.target }}</span>
          </div>
          <span v-if="bench.details" class="benchmark-details">{{ bench.details }}</span>
        </div>

        <div class="benchmark-bar-container">
          <div class="bar-background">
            <div
              class="benchmark-bar"
              :class="{ visible: isVisible }"
              :style="{
                width: isVisible ? `${bench.percentage}%` : '0%',
                background: `linear-gradient(90deg, ${getBarColor(bench.percentage)} 0%, color-mix(in srgb, ${getBarColor(bench.percentage)} 70%, white) 100%)`,
                animationDelay: `${index * 0.12}s`
              }"
            >
              <span class="benchmark-value">{{ bench.percentage }}%</span>
            </div>
          </div>

          <div class="reference-line">
            <span class="ref-label">100%</span>
          </div>
        </div>
      </div>
    </div>

    <div class="chart-summary">
      <div class="summary-item">
        <span class="summary-value">{{ Math.round(benchmarks.reduce((a, b) => a + b.percentage, 0) / benchmarks.length) }}%</span>
        <span class="summary-label">Average</span>
      </div>
      <div class="summary-item">
        <span class="summary-value">{{ Math.max(...benchmarks.map(b => b.percentage)) }}%</span>
        <span class="summary-label">Best</span>
      </div>
      <div class="summary-item">
        <span class="summary-value">5</span>
        <span class="summary-label">Kernels</span>
      </div>
    </div>

    <div class="chart-note">
      <span class="note-icon">&#128202;</span>
      <span>Benchmarks run on A100 80GB, CUDA 12.4, Tensor Core enabled</span>
    </div>
  </div>
</template>

<style scoped>
.benchmark-chart {
  padding: 3rem 0;
  max-width: 800px;
  margin: 0 auto;
}

.chart-title {
  font-family: 'Fraunces', serif;
  font-size: 1.8rem;
  text-align: center;
  margin-bottom: 0.6rem;
  color: var(--vp-c-text-1);
  font-weight: 700;
}

.accent-text {
  color: var(--tc-accent-primary);
}

.chart-description {
  text-align: center;
  color: var(--vp-c-text-2);
  margin-bottom: 2.5rem;
  font-size: 0.92rem;
}

.chart-container {
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
}

.benchmark-row {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem 1.2rem;
  background: var(--vp-c-bg-soft);
  border-radius: var(--tc-radius-md);
  border: 1px solid var(--vp-c-divider);
  cursor: pointer;
  transition: all 0.2s ease;
  opacity: 0;
  transform: translateX(-10px);
  animation: slideIn 0.5s ease forwards;
}

@keyframes slideIn {
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.benchmark-row:hover {
  border-color: color-mix(in srgb, var(--tc-accent-primary) 40%, var(--vp-c-divider));
  box-shadow: var(--tc-shadow-md);
  transform: translateX(4px);
}

.benchmark-row.selected {
  border-color: var(--tc-accent-primary);
  background: color-mix(in srgb, var(--tc-blue-soft) 30%, var(--vp-c-bg-soft));
}

.benchmark-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.benchmark-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.benchmark-name {
  font-weight: 700;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
}

.benchmark-target {
  font-size: 11px;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}

.benchmark-details {
  font-size: 11px;
  color: var(--vp-c-text-3);
  font-style: italic;
}

.benchmark-bar-container {
  position: relative;
  height: 34px;
}

.bar-background {
  height: 100%;
  background: var(--vp-c-bg-mute);
  border-radius: var(--tc-radius-sm);
  overflow: hidden;
  position: relative;
}

.benchmark-bar {
  height: 100%;
  border-radius: var(--tc-radius-sm);
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 1rem;
  transition: width 1.2s cubic-bezier(0.22, 1, 0.36, 1);
  position: relative;
}

.benchmark-value {
  font-weight: 700;
  color: var(--tc-paper-elevated);
  font-size: 12px;
  font-family: var(--vp-font-family-mono);
  z-index: 1;
  text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

.reference-line {
  position: absolute;
  top: 0;
  right: 0;
  height: 100%;
  width: 2px;
  background: var(--vp-c-border);
}

.ref-label {
  position: absolute;
  top: -18px;
  right: 0;
  font-size: 10px;
  color: var(--vp-c-text-3);
  transform: translateX(50%);
  font-family: var(--vp-font-family-mono);
}

/* Summary */
.chart-summary {
  display: flex;
  justify-content: center;
  gap: 2.5rem;
  margin: 2rem 0;
  padding: 1.3rem;
  background: var(--vp-c-bg-soft);
  border-radius: var(--tc-radius-lg);
  border: 1px solid var(--vp-c-divider);
}

.summary-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.2rem;
}

.summary-value {
  font-size: 1.4rem;
  font-weight: 800;
  color: var(--tc-accent-primary);
  font-family: var(--vp-font-family-mono);
}

.summary-label {
  font-size: 11px;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 600;
}

.chart-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
  padding: 0.7rem 1.3rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--tc-radius-md);
  font-size: 11px;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}

.note-icon {
  font-size: 1rem;
}

@media (max-width: 768px) {
  .chart-title {
    font-size: 1.4rem;
  }

  .benchmark-bar-container {
    height: 28px;
  }

  .benchmark-value {
    font-size: 10px;
  }

  .chart-summary {
    gap: 1.8rem;
  }

  .summary-value {
    font-size: 1.2rem;
  }
}
</style>
