<script setup lang="ts">
interface Props {
  name: string
  complexity?: string
  lines?: string[]
}

const props = defineProps<Props>()
</script>

<template>
  <div class="tc-algorithm">
    <div class="tc-algorithm-header">
      <span class="tc-algorithm-name">Algorithm: {{ name }}</span>
      <span v-if="complexity" class="tc-algorithm-complexity">{{ complexity }}</span>
    </div>
    <div v-if="lines" class="tc-algorithm-body">
      <div
        v-for="(line, idx) in lines"
        :key="idx"
        class="tc-algorithm-line"
      >
        <span class="tc-algorithm-line-num">{{ idx + 1 }}</span>
        <span class="tc-algorithm-code" v-html="line"></span>
      </div>
    </div>
    <div v-else class="tc-algorithm-body">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.tc-algorithm {
  margin: 1.5rem 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--tc-radius-md);
  background: var(--vp-c-bg-soft);
  overflow: hidden;
}

.tc-algorithm-header {
  padding: 0.7rem 1rem;
  background: color-mix(in srgb, var(--tc-accent-primary) 7%, var(--vp-c-bg-soft));
  border-bottom: 1px solid var(--vp-c-divider);
  font-family: 'Fraunces', serif;
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--vp-c-text-1);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tc-algorithm-complexity {
  font-family: var(--vp-font-family-mono);
  font-size: 0.78rem;
  color: var(--tc-accent-primary);
  background: var(--tc-blue-soft);
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-weight: 600;
}

.tc-algorithm-body {
  padding: 1rem 1.2rem;
  font-family: var(--vp-font-family-mono);
  font-size: 0.84rem;
  line-height: 1.8;
  color: var(--vp-c-text-2);
}

.tc-algorithm-line {
  display: flex;
  gap: 0.8rem;
  padding: 0.1rem 0;
}

.tc-algorithm-line-num {
  color: var(--tc-ink-quaternary);
  min-width: 1.8rem;
  text-align: right;
  user-select: none;
  font-weight: 500;
}

.tc-algorithm-code :deep(.kw) {
  color: var(--tc-accent-primary);
  font-weight: 600;
}

.tc-algorithm-code :deep(.comment) {
  color: var(--vp-c-text-3);
  font-style: italic;
}

.tc-algorithm-code :deep(.var) {
  color: var(--tc-accent-warning);
}
</style>
