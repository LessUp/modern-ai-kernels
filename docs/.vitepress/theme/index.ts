import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

import './style.css'

import BenchmarkChart from './components/BenchmarkChart.vue'
import EvidenceStrip from './components/EvidenceStrip.vue'
import GPUTimeline from './components/GPUTimeline.vue'
import OptimizationPath from './components/OptimizationPath.vue'
import ShowcaseBand from './components/ShowcaseBand.vue'
import ThemeImage from './components/ThemeImage.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('EvidenceStrip', EvidenceStrip)
    app.component('GPUTimeline', GPUTimeline)
    app.component('OptimizationPath', OptimizationPath)
    app.component('ShowcaseBand', ShowcaseBand)
    app.component('ThemeImage', ThemeImage)
  }
} satisfies Theme
