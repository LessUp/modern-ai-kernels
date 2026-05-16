import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/modern-ai-kernels/'

export default withMermaid(defineConfig({
  base,
  title: 'TensorCraft-HPC',
  description: 'Technical whitepaper, architecture showcase, and academy for modern AI kernels',
  cleanUrls: true,
  lastUpdated: true,

  head: [
    ['link', { rel: 'icon', href: `${base}images/favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#c96e32' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'TensorCraft-HPC' }],
    ['meta', { property: 'og:description', content: 'Technical whitepaper and architecture showcase for modern AI kernels' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', {
      href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Inter+Tight:wght@500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap',
      rel: 'stylesheet'
    }]
  ],

  vite: {
    plugins: [
      llmstxt({
        domain: 'https://aicl-lab.github.io'
      })
    ]
  },

  mermaid: {
    theme: 'base',
    themeVariables: {
      primaryColor: '#e7e0d7',
      primaryTextColor: '#1d2733',
      primaryBorderColor: '#c96e32',
      lineColor: '#506071',
      secondaryColor: '#f3efe8',
      tertiaryColor: '#171e26',
      background: '#f3efe8',
      mainBkg: '#e7e0d7',
      clusterBkg: '#e7e0d7',
      clusterBorder: '#8ea0b4',
      titleColor: '#1d2733',
      edgeLabelBackground: '#f3efe8',
      textColor: '#1d2733'
    }
  },

  themeConfig: {
    logo: {
      light: '/images/logo.svg',
      dark: '/images/logo-dark.svg',
      alt: 'TensorCraft-HPC'
    },
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/AICL-Lab/modern-ai-kernels' }
    ],
    editLink: {
      pattern: 'https://github.com/AICL-Lab/modern-ai-kernels/edit/master/docs/:path',
      text: 'Edit this page on GitHub'
    },
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present TensorCraft-HPC Contributors'
    },
    outline: {
      level: [2, 3],
      label: 'On this page'
    }
  },

  locales: {
    root: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      themeConfig: {
        nav: nav(),
        sidebar: sidebarEn(),
        docFooter: { prev: 'Previous', next: 'Next' },
        lastUpdated: { text: 'Last updated' },
        outline: { label: 'On this page' }
      }
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: navZh(),
        sidebar: sidebarZh(),
        docFooter: { prev: '上一页', next: '下一页' },
        lastUpdated: { text: '最后更新' },
        editLink: {
          pattern: 'https://github.com/AICL-Lab/modern-ai-kernels/edit/master/docs/:path',
          text: '在 GitHub 上编辑此页'
        },
        outline: { label: '本页目录' }
      }
    }
  }
}))

function nav() {
  return [
    { text: 'Home', link: '/en/' },
    { text: 'Whitepaper', link: '/en/whitepaper/' },
    { text: 'Academy', link: '/en/academy/' },
    { text: 'Kernel Atlas', link: '/en/api/gemm', activeMatch: '/en/api/' },
    { text: 'Evidence', link: '/en/evidence/' },
    { text: 'References', link: '/en/references/papers', activeMatch: '/en/references/' }
  ]
}

function navZh() {
  return [
    { text: '首页', link: '/zh/' },
    { text: '技术白皮书', link: '/zh/whitepaper/' },
    { text: '学院', link: '/zh/academy/' },
    { text: 'Kernel Atlas', link: '/zh/api/gemm', activeMatch: '/zh/api/' },
    { text: '证据', link: '/zh/evidence/' },
    { text: '参考', link: '/zh/references/papers', activeMatch: '/zh/references/' }
  ]
}

function sidebarEn() {
  return {
    '/en/whitepaper/': [
      {
        text: 'Technical whitepaper',
        items: [
          { text: 'Overview', link: '/en/whitepaper/' },
          { text: 'Architecture', link: '/en/whitepaper/architecture' },
          { text: 'Performance', link: '/en/whitepaper/performance' },
          { text: 'Methodology', link: '/en/whitepaper/methodology' }
        ]
      }
    ],
    '/en/academy/': [
      {
        text: 'Academy',
        items: [
          { text: 'Overview', link: '/en/academy/' },
          { text: 'Quick start', link: '/en/guides/quick-start' },
          { text: 'Architecture lessons', link: '/en/guides/architecture' },
          { text: 'Modern C++ and CUDA', link: '/en/guides/modern-cpp-cuda' },
          { text: 'Benchmarking', link: '/en/guides/benchmarking' },
          { text: 'Examples', link: '/en/examples/' }
        ]
      }
    ],
    '/en/evidence/': [
      {
        text: 'Evidence',
        items: [
          { text: 'Overview', link: '/en/evidence/' },
          { text: 'Benchmarks', link: '/en/benchmarks/' },
          { text: 'Papers and citations', link: '/en/references/papers' },
          { text: 'Related resources', link: '/en/references/resources' }
        ]
      }
    ],
    '/en/api/': [
      {
        text: 'Kernel Atlas',
        items: [
          { text: 'GEMM', link: '/en/api/gemm' },
          { text: 'Attention', link: '/en/api/attention' },
          { text: 'Normalization', link: '/en/api/normalization' },
          { text: 'Convolution', link: '/en/api/convolution' },
          { text: 'Sparse', link: '/en/api/sparse' },
          { text: 'Quantization', link: '/en/api/quantization' }
        ]
      }
    ],
    '/en/references/': [
      {
        text: 'References',
        items: [
          { text: 'Papers and citations', link: '/en/references/papers' },
          { text: 'Related resources', link: '/en/references/resources' }
        ]
      }
    ],
    '/en/': [
      {
        text: 'Project journey',
        items: [
          { text: 'Home', link: '/en/' },
          { text: 'Whitepaper', link: '/en/whitepaper/' },
          { text: 'Academy', link: '/en/academy/' },
          { text: 'Evidence', link: '/en/evidence/' },
          { text: 'Kernel Atlas', link: '/en/api/gemm' },
          { text: 'References', link: '/en/references/papers' }
        ]
      }
    ]
  }
}

function sidebarZh() {
  return {
    '/zh/whitepaper/': [
      {
        text: '技术白皮书',
        items: [
          { text: '概述', link: '/zh/whitepaper/' },
          { text: '架构设计', link: '/zh/whitepaper/architecture' },
          { text: '性能分析', link: '/zh/whitepaper/performance' },
          { text: '开发方法论', link: '/zh/whitepaper/methodology' }
        ]
      }
    ],
    '/zh/academy/': [
      {
        text: '学院',
        items: [
          { text: '总览', link: '/zh/academy/' },
          { text: '快速开始', link: '/zh/guides/quick-start' },
          { text: '架构解读', link: '/zh/guides/architecture' },
          { text: '现代 C++ 与 CUDA', link: '/zh/guides/modern-cpp-cuda' },
          { text: '基准测试', link: '/zh/guides/benchmarking' },
          { text: '示例', link: '/zh/examples/' }
        ]
      }
    ],
    '/zh/evidence/': [
      {
        text: '证据',
        items: [
          { text: '总览', link: '/zh/evidence/' },
          { text: '性能基准', link: '/zh/benchmarks/' },
          { text: '论文引用', link: '/zh/references/papers' },
          { text: '相关资源', link: '/zh/references/resources' }
        ]
      }
    ],
    '/zh/api/': [
      {
        text: 'Kernel Atlas',
        items: [
          { text: 'GEMM', link: '/zh/api/gemm' },
          { text: 'Attention', link: '/zh/api/attention' },
          { text: '归一化', link: '/zh/api/normalization' },
          { text: '卷积', link: '/zh/api/convolution' },
          { text: '稀疏', link: '/zh/api/sparse' },
          { text: '量化', link: '/zh/api/quantization' }
        ]
      }
    ],
    '/zh/references/': [
      {
        text: '参考',
        items: [
          { text: '论文引用', link: '/zh/references/papers' },
          { text: '相关资源', link: '/zh/references/resources' }
        ]
      }
    ],
    '/zh/': [
      {
        text: '阅读路径',
        items: [
          { text: '首页', link: '/zh/' },
          { text: '技术白皮书', link: '/zh/whitepaper/' },
          { text: '学院', link: '/zh/academy/' },
          { text: '证据', link: '/zh/evidence/' },
          { text: 'Kernel Atlas', link: '/zh/api/gemm' },
          { text: '参考', link: '/zh/references/papers' }
        ]
      }
    ]
  }
}
