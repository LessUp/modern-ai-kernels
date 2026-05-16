<script setup lang="ts">
defineProps<{
  claim: string
  method: string
  source: string
  methodLabel?: string
  sourceLabel?: string
}>()
</script>

<template>
  <div class="tc-evidence-strip">
    <strong>{{ claim }}</strong>
    <span><span class="tc-mini-label">{{ methodLabel ?? 'Method' }}</span> {{ method }}</span>
    <span><span class="tc-mini-label">{{ sourceLabel ?? 'Source' }}</span> {{ source }}</span>
  </div>
</template>
