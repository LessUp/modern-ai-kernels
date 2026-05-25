---
layout: home
---

# Welcome to TensorCraft-HPC

<script setup>
import { onMounted } from 'vue'
import { useRouter, withBase } from 'vitepress'

onMounted(() => {
  const router = useRouter()
  const lang = navigator.language || navigator.userLanguage
  if (lang.startsWith('zh')) {
    router.go(withBase('/zh/'))
  } else {
    router.go(withBase('/en/'))
  }
})
</script>
