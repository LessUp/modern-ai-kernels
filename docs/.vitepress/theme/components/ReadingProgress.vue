<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const progress = ref(0)

function updateProgress() {
  const scrollTop = window.scrollY || document.documentElement.scrollTop
  const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight
  progress.value = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0
}

let rafId: number | null = null

function onScroll() {
  if (rafId) cancelAnimationFrame(rafId)
  rafId = requestAnimationFrame(updateProgress)
}

onMounted(() => {
  window.addEventListener('scroll', onScroll, { passive: true })
  updateProgress()
})

onUnmounted(() => {
  window.removeEventListener('scroll', onScroll)
  if (rafId) cancelAnimationFrame(rafId)
})
</script>

<template>
  <div class="tc-reading-progress" aria-hidden="true">
    <div
      class="tc-reading-progress-bar"
      :style="{ width: `${progress}%` }"
    ></div>
  </div>
</template>

<style scoped>
.tc-reading-progress {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 3px;
  background: transparent;
  z-index: 1000;
  pointer-events: none;
}

.tc-reading-progress-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--tc-accent-primary), var(--tc-accent-success));
  width: 0%;
  transition: width 0.05s linear;
}
</style>
