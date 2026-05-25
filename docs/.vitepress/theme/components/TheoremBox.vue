<script setup lang="ts">
interface Props {
  type?: 'theorem' | 'lemma' | 'corollary' | 'proposition' | 'definition'
  label?: string
}

const props = withDefaults(defineProps<Props>(), {
  type: 'theorem',
  label: undefined
})

const labels: Record<string, string> = {
  theorem: 'Theorem',
  lemma: 'Lemma',
  corollary: 'Corollary',
  proposition: 'Proposition',
  definition: 'Definition'
}

const icons: Record<string, string> = {
  theorem: '&#9670;',
  lemma: '&#9671;',
  corollary: '&#9672;',
  proposition: '&#9632;',
  definition: '&#9633;'
}
</script>

<template>
  <div class="tc-theorem" :class="`tc-theorem--${type}`">
    <div class="tc-theorem-header">
      <span class="tc-theorem-icon" v-html="icons[type]"></span>
      <span class="tc-theorem-label">{{ labels[type] }}</span>
      <span v-if="label" class="tc-theorem-id">{{ label }}</span>
    </div>
    <div class="tc-theorem-body">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.tc-theorem {
  margin: 1.5rem 0;
  padding: 1.3rem 1.5rem;
  border-radius: var(--tc-radius-md);
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  position: relative;
}

.tc-theorem::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  border-radius: var(--tc-radius-md) 0 0 var(--tc-radius-md);
}

.tc-theorem--theorem::before {
  background: linear-gradient(180deg, var(--tc-accent-primary), var(--tc-accent-success));
}

.tc-theorem--lemma::before {
  background: linear-gradient(180deg, var(--tc-accent-success), var(--tc-accent-primary));
}

.tc-theorem--corollary::before {
  background: var(--tc-accent-primary);
}

.tc-theorem--proposition::before {
  background: var(--tc-accent-warning);
}

.tc-theorem--definition::before {
  background: var(--tc-ink-tertiary);
}

.tc-theorem-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.7rem;
  font-family: 'Fraunces', serif;
  font-weight: 700;
  font-size: 1rem;
  color: var(--vp-c-text-1);
}

.tc-theorem-icon {
  color: var(--tc-accent-primary);
  font-size: 0.85rem;
}

.tc-theorem-label {
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.tc-theorem-id {
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
  font-size: 0.82rem;
  font-weight: 600;
  margin-left: auto;
}

.tc-theorem-body {
  color: var(--vp-c-text-2);
  line-height: 1.75;
}

.tc-theorem-body :deep(p):last-child {
  margin-bottom: 0;
}

.tc-theorem-body :deep(p):first-child {
  margin-top: 0;
}
</style>
