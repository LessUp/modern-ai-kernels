import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

// Technical whitepaper style
import './style.css'

// Custom components (available for use in markdown)
import GPUTimeline from './components/GPUTimeline.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'
import OptimizationPath from './components/OptimizationPath.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Register global components for use in markdown
    app.component('GPUTimeline', GPUTimeline)
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('OptimizationPath', OptimizationPath)
  }
} satisfies Theme