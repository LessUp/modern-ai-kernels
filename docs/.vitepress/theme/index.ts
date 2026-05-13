import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

// Import NVIDIA-style CSS
import './style.css'

// Import custom components
import GPUTimeline from './components/GPUTimeline.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'
import CodePreview from './components/CodePreview.vue'

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // Custom slots for landing page
      'home-features-after': () => h('div', { class: 'nvidia-home-extra' }, [
        h(GPUTimeline),
        h(BenchmarkChart)
      ])
    })
  },
  enhanceApp({ app }) {
    // Register global components
    app.component('GPUTimeline', GPUTimeline)
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('CodePreview', CodePreview)
  }
} satisfies Theme