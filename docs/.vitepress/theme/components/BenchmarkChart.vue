<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'

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
  if (percentage >= 90) return '#8ED000'
  if (percentage >= 80) return '#76B900'
  if (percentage >= 70) return '#5A9100'
  return '#FFB800'
}

function selectBenchmark(bench: Benchmark) {
  selectedBenchmark.value = selectedBenchmark.value === bench ? null : bench
}
</script>

<template>
  <div class="benchmark-chart" ref="chartRef">
    <h2 class="chart-title">
      <span class="brand-text">Performance</span> Benchmarks
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
        :style="{ animationDelay: `${index * 0.1}s` }"
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
                backgroundColor: getBarColor(bench.percentage),
                animationDelay: `${index * 0.15}s`
              }"
            >
              <span class="benchmark-value">{{ bench.percentage }}%</span>
            </div>
          </div>

          <!-- Reference line at 100% -->
          <div class="reference-line">
            <span class="ref-label">100%</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Summary stats -->
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
      <span class="note-icon">📊</span>
      <span>Benchmarks run on A100 80GB, CUDA 12.4, Tensor Core enabled</span>
    </div>
  </div>
</template>

<style scoped>
.benchmark-chart {
  padding: 4rem 0;
  max-width: 800px;
  margin: 0 auto;
}

.chart-title {
  font-size: 2rem;
  text-align: center;
  margin-bottom: 1rem;
  color: var(--vp-c-text-1);
}

.brand-text {
  color: var(--vp-c-brand-1);
}

.chart-description {
  text-align: center;
  color: var(--vp-c-text-2);
  margin-bottom: 3rem;
}

.chart-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.benchmark-row {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  background: var(--vp-c-bg-soft);
  border-radius: var(--radius-md);
  border: 1px solid var(--vp-c-border);
  cursor: pointer;
  transition: all 0.2s ease;
}

.benchmark-row:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateX(4px);
}

.benchmark-row.selected {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-mute);
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
  font-weight: 600;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
}

.benchmark-target {
  font-size: 12px;
  color: var(--vp-c-text-3);
}

.benchmark-details {
  font-size: 11px;
  color: var(--vp-c-text-3);
  font-style: italic;
}

.benchmark-bar-container {
  position: relative;
  height: 36px;
}

.bar-background {
  height: 100%;
  background: var(--vp-c-bg-mute);
  border-radius: var(--radius-sm);
  overflow: hidden;
  position: relative;
}

.benchmark-bar {
  height: 100%;
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 1rem;
  transition: width 1s ease-out;
  position: relative;
}

.benchmark-bar::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15));
  border-radius: var(--radius-sm);
}

.benchmark-value {
  font-weight: 700;
  color: #000;
  font-size: 13px;
  font-family: var(--vp-font-family-mono);
  z-index: 1;
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
}

/* Summary stats */
.chart-summary {
  display: flex;
  justify-content: center;
  gap: 3rem;
  margin: 2rem 0;
  padding: 1.5rem;
  background: var(--vp-c-bg-soft);
  border-radius: var(--radius-lg);
  border: 1px solid var(--vp-c-border);
}

.summary-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.summary-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  font-family: var(--vp-font-family-mono);
}

.summary-label {
  font-size: 12px;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.chart-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 1rem;
  padding: 0.75rem 1.5rem;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-border);
  border-radius: var(--radius-md);
  font-size: 12px;
  color: var(--vp-c-text-3);
}

.note-icon {
  font-size: 1rem;
}

@media (max-width: 768px) {
  .chart-title {
    font-size: 1.5rem;
  }

  .benchmark-bar-container {
    height: 28px;
  }

  .benchmark-value {
    font-size: 11px;
  }

  .chart-summary {
    gap: 2rem;
  }

  .summary-value {
    font-size: 1.25rem;
  }
}
</style>