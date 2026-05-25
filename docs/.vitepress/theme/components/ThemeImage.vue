<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

const props = defineProps<{
  light: string
  dark: string
  alt: string
  className?: string
}>()

const { isDark } = useData()

const src = computed(() => withBase(isDark.value ? props.dark : props.light))
</script>

<template>
  <img :src="src" :alt="alt" :class="className">
</template>
