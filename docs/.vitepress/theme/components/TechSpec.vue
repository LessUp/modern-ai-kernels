<script setup lang="ts">
interface SpecItem {
  label: string
  value: string
  highlight?: boolean
}

interface Props {
  title?: string
  specs: SpecItem[]
  columns?: number
}

const props = withDefaults(defineProps<Props>(), {
  title: undefined,
  columns: 2
})
</script>

<template>
  <div class="tc-tech-spec">
    <div v-if="title" class="tc-tech-spec-title">{{ title }}</div>
    <div class="tc-tech-spec-grid" :style="{ gridTemplateColumns: `repeat(${columns}, 1fr)` }">
      <div
        v-for="(spec, idx) in specs"
        :key="idx"
        class="tc-tech-spec-item"
        :class="{ highlight: spec.highlight }"
      >
        <span class="tc-tech-spec-label">{{ spec.label }}</span>
        <span class="tc-tech-spec-value">{{ spec.value }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tc-tech-spec {
  margin: 1.2rem 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: var(--tc-radius-md);
  background: var(--vp-c-bg-alt);
  overflow: hidden;
}

.tc-tech-spec-title {
  padding: 0.7rem 1rem;
  background: color-mix(in srgb, var(--tc-accent-primary) 6%, var(--vp-c-bg-soft));
  border-bottom: 1px solid var(--vp-c-divider);
  font-family: 'Fraunces', serif;
  font-weight: 700;
  font-size: 0.92rem;
  color: var(--vp-c-text-1);
}

.tc-tech-spec-grid {
  display: grid;
  gap: 1px;
  background: var(--vp-c-divider);
}

.tc-tech-spec-item {
  padding: 0.65rem 0.9rem;
  background: var(--vp-c-bg-alt);
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  transition: background 0.15s ease;
}

.tc-tech-spec-item:hover {
  background: color-mix(in srgb, var(--tc-accent-primary) 3%, var(--vp-c-bg-alt));
}

.tc-tech-spec-item.highlight {
  background: color-mix(in srgb, var(--tc-blue-soft) 40%, var(--vp-c-bg-alt));
}

.tc-tech-spec-item.highlight .tc-tech-spec-value {
  color: var(--tc-accent-primary);
  font-weight: 700;
}

.tc-tech-spec-label {
  font-size: 0.68rem;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 700;
}

.tc-tech-spec-value {
  font-size: 0.9rem;
  color: var(--vp-c-text-1);
  font-weight: 600;
  font-family: var(--vp-font-family-mono);
}

@media (max-width: 768px) {
  .tc-tech-spec-grid {
    grid-template-columns: 1fr !important;
  }
}
</style>
