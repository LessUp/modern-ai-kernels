import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'

import './style.css'

import AlgorithmBlock from './components/AlgorithmBlock.vue'
import BenchmarkChart from './components/BenchmarkChart.vue'
import CitationCard from './components/CitationCard.vue'
import EvidenceStrip from './components/EvidenceStrip.vue'
import GPUTimeline from './components/GPUTimeline.vue'
import OptimizationPath from './components/OptimizationPath.vue'
import ReadingProgress from './components/ReadingProgress.vue'
import ShowcaseBand from './components/ShowcaseBand.vue'
import TechSpec from './components/TechSpec.vue'
import ThemeImage from './components/ThemeImage.vue'
import TheoremBox from './components/TheoremBox.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('AlgorithmBlock', AlgorithmBlock)
    app.component('BenchmarkChart', BenchmarkChart)
    app.component('CitationCard', CitationCard)
    app.component('EvidenceStrip', EvidenceStrip)
    app.component('GPUTimeline', GPUTimeline)
    app.component('OptimizationPath', OptimizationPath)
    app.component('ReadingProgress', ReadingProgress)
    app.component('ShowcaseBand', ShowcaseBand)
    app.component('TechSpec', TechSpec)
    app.component('ThemeImage', ThemeImage)
    app.component('TheoremBox', TheoremBox)
  }
} satisfies Theme
