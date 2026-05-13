<script setup lang="ts">
import { ref, onMounted } from 'vue'

interface Benchmark {
  name: string
  target: string
  percentage: number
  color: string
}

const benchmarks: Benchmark[] = [
  { name: 'GEMM (FP16)', target: 'cuBLAS', percentage: 92, color: '#76B900' },
  { name: 'FlashAttention', target: 'cuDNN', percentage: 85, color: '#76B900' },
  { name: 'LayerNorm', target: 'cuDNN', percentage: 95, color: '#76B900' },
  { name: 'Conv2D', target: 'cuDNN', percentage: 78, color: '#76B900' },
  { name: 'SpMV (CSR)', target: 'cuSPARSE', percentage: 88, color: '#76B900' }
]

const isVisible = ref(false)
const chartRef = ref<HTMLElement | null>(null)

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
</script>

<template>
  <div class="benchmark-chart" ref="chartRef">
    <h2 class="chart-title">
      <span class="nvidia-text">Performance</span> Benchmarks
    </h2>
    <p class="chart-description">
      Relative performance compared to NVIDIA libraries on A100 (FP16 Tensor Core).
      Use as a learning and regression tool.
    </p>

    <div class="chart-container">
      <div
        v-for="(bench, index) in benchmarks"
        :key="bench.name"
        class="benchmark-row"
        :style="{ animationDelay: `${index * 0.1}s` }"
      >
        <div class="benchmark-info">
          <span class="benchmark-name">{{ bench.name }}</span>
          <span class="benchmark-target">vs {{ bench.target }}</span>
        </div>
        <div class="benchmark-bar-container">
          <div
            class="benchmark-bar"
            :class="{ visible: isVisible }"
            :style="{
              width: isVisible ? `${bench.percentage}%` : '0%',
              backgroundColor: bench.color,
              animationDelay: `${index * 0.15}s`
            }"
          >
            <span class="benchmark-value">{{ bench.percentage }}%</span>
          </div>
        </div>
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

.nvidia-text {
  color: var(--nvidia-green);
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
}

.benchmark-info {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.benchmark-name {
  font-weight: 600;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
}

.benchmark-target {
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
}

.benchmark-bar-container {
  height: 32px;
  background: var(--vp-c-gray);
  border-radius: 8px;
  overflow: hidden;
  position: relative;
}

.benchmark-bar {
  height: 100%;
  border-radius: 8px;
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
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1));
  border-radius: 8px;
}

.benchmark-value {
  font-weight: 700;
  color: #000;
  font-size: 0.875rem;
  font-family: var(--vp-font-family-mono);
  z-index: 1;
}

.chart-note {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 2rem;
  padding: 0.75rem 1.5rem;
  background: var(--vp-c-gray);
  border: 1px solid var(--vp-c-border);
  border-radius: 8px;
  font-size: 0.75rem;
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
    height: 24px;
  }

  .benchmark-value {
    font-size: 0.75rem;
  }
}
</style>