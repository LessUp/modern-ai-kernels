<script setup lang="ts">
defineProps<{
  eyebrow: string
  title: string
  description: string
}>()
</script>

<template>
  <section class="tc-band">
    <header class="tc-band-header">
      <span class="tc-band-eyebrow">{{ eyebrow }}</span>
      <h2 class="tc-band-title">{{ title }}</h2>
      <p class="tc-band-description">{{ description }}</p>
    </header>
    <div class="tc-band-body">
      <slot />
    </div>
  </section>
</template>
