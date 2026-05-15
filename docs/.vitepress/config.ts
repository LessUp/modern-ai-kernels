import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

// Dynamic base path for GitHub Pages
// Supports both aicl-lab and lessup organization repos
const base = process.env.VITEPRESS_BASE || '/modern-ai-kernels/'

export default withMermaid(defineConfig({
  base,
  title: 'TensorCraft-HPC',
  description: 'Technical whitepaper and architecture showcase for high-performance AI kernels',

  // Clean URLs without .html
  cleanUrls: true,

  // Last updated timestamp
  lastUpdated: true,

  // Head tags for fonts and meta
  head: [
    ['link', { rel: 'icon', href: '/images/favicon.svg' }],
    ['meta', { name: 'theme-color', content: '#2E7D32' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'TensorCraft-HPC' }],
    ['meta', { property: 'og:description', content: 'High-Performance AI Kernels Technical Whitepaper' }],
    // Google Fonts - Source Serif for academic feel, JetBrains Mono for code
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', {
      href: 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,400&display=swap',
      rel: 'stylesheet'
    }]
  ],

  // Vite configuration for plugins
  vite: {
    plugins: [
      llmstxt({
        domain: 'https://aicl-lab.github.io/modern-ai-kernels'
      })
    ]
  },

  // Mermaid configuration - neutral palette that stays readable in both themes
  mermaid: {
    theme: 'base',
    themeVariables: {
      primaryColor: '#EAF4E0',
      primaryTextColor: '#1A1A1A',
      primaryBorderColor: '#2E7D32',
      lineColor: '#2E7D32',
      secondaryColor: '#F4F7F1',
      tertiaryColor: '#FFFFFF',
      background: '#FFFFFF',
      mainBkg: '#F8FBF6',
      nodeBorder: '#2E7D32',
      clusterBkg: '#F8FBF6',
      clusterBorder: '#D0E0CF',
      titleColor: '#1A1A1A',
      edgeLabelBackground: '#F8FBF6',
      textColor: '#1A1A1A'
    }
  },

  // Theme configuration
  themeConfig: {
    logo: {
      light: '/images/logo.svg',
      dark: '/images/logo-dark.svg',
      alt: 'TensorCraft-HPC'
    },

    // Navigation
    nav: nav(),

    // Social links
    socialLinks: [
      { icon: 'github', link: 'https://github.com/AICL-Lab/modern-ai-kernels' }
    ],

    // Edit link
    editLink: {
      pattern: 'https://github.com/AICL-Lab/modern-ai-kernels/edit/master/docs/:path',
      text: 'Edit this page on GitHub'
    },

    // Footer
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present TensorCraft-HPC Contributors'
    },

    // Local search
    search: {
      provider: 'local',
      options: {
        detailedView: true,
        miniSearch: {
          searchOptions: {
            fuzzy: 0.2,
            prefix: true,
            boost: { title: 4, text: 2, titles: 1 }
          }
        }
      }
    },

    // Outline
    outline: {
      level: [2, 3],
      label: 'On this page'
    }
  },

  // Internationalization
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      link: '/en/',
      themeConfig: {
        nav: nav(),
        sidebar: sidebarEn(),
        docFooter: {
          prev: 'Previous',
          next: 'Next'
        },
        lastUpdated: {
          text: 'Last updated',
          formatOptions: {
            dateStyle: 'medium',
            timeStyle: 'short'
          }
        },
        outline: {
          label: 'On this page'
        }
      }
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: navZh(),
        sidebar: sidebarZh(),
        docFooter: {
          prev: '上一页',
          next: '下一页'
        },
        lastUpdated: {
          text: '最后更新',
          formatOptions: {
            dateStyle: 'medium',
            timeStyle: 'short'
          }
        },
        editLink: {
          pattern: 'https://github.com/AICL-Lab/modern-ai-kernels/edit/master/docs/:path',
          text: '在 GitHub 上编辑此页'
        },
        outline: {
          label: '本页目录'
        }
      }
    }
  }
}))

// Navigation items - English
function nav() {
  return [
    { text: 'Home', link: '/en/' },
    {
      text: 'Whitepaper',
      link: '/en/whitepaper/',
      activeMatch: '/en/whitepaper/'
    },
    {
      text: 'Architecture',
      link: '/en/architecture',
      activeMatch: '/en/architecture'
    },
    {
      text: 'Kernel Atlas',
      link: '/en/api/gemm',
      activeMatch: '/en/api/'
    },
    {
      text: 'Evidence',
      items: [
        { text: 'Benchmarks', link: '/en/benchmarks/' },
        { text: 'Papers & Citations', link: '/en/references/papers' },
        { text: 'Related Resources', link: '/en/references/resources' }
      ]
    },
    {
      text: 'Learn',
      items: [
        { text: 'Getting Started', link: '/en/getting-started' },
        { text: 'Examples', link: '/en/examples/' },
        { text: 'FAQ', link: '/en/faq' },
        { text: 'Changelog', link: '/en/changelog' }
      ]
    }
  ]
}

// Navigation items - Chinese
function navZh() {
  return [
    { text: '首页', link: '/zh/' },
    {
      text: '技术白皮书',
      link: '/zh/whitepaper/',
      activeMatch: '/zh/whitepaper/'
    },
    {
      text: '架构',
      link: '/zh/architecture',
      activeMatch: '/zh/architecture'
    },
    {
      text: 'Kernel Atlas',
      link: '/zh/api/gemm',
      activeMatch: '/zh/api/'
    },
    {
      text: '证据',
      items: [
        { text: '性能基准', link: '/zh/benchmarks/' },
        { text: '论文引用', link: '/zh/references/papers' },
        { text: '相关资源', link: '/zh/references/resources' }
      ]
    },
    {
      text: '学习',
      items: [
        { text: '快速开始', link: '/zh/getting-started' },
        { text: '示例', link: '/zh/examples/' },
        { text: '常见问题', link: '/zh/faq' },
        { text: '更新日志', link: '/zh/changelog' }
      ]
    }
  ]
}

// English sidebar
function sidebarEn() {
  return {
    '/en/whitepaper/': [
      {
        text: 'Technical Whitepaper',
        items: [
          { text: 'Overview', link: '/en/whitepaper/' },
          { text: 'Architecture', link: '/en/whitepaper/architecture' },
          { text: 'Performance', link: '/en/whitepaper/performance' },
          { text: 'Methodology', link: '/en/whitepaper/methodology' }
        ]
      }
    ],
    '/en/examples/': [
      {
        text: 'Examples',
        items: [
          { text: 'Overview', link: '/en/examples/' },
          { text: 'GEMM Tutorial', link: '/en/examples/gemm-tutorial' },
          { text: 'FlashAttention', link: '/en/examples/flash-attention' },
          { text: 'Python Bindings', link: '/en/examples/python-bindings' }
        ]
      }
    ],
    '/en/benchmarks/': [
      {
        text: 'Benchmarks',
        items: [
          { text: 'Overview', link: '/en/benchmarks/' },
          { text: 'GEMM Performance', link: '/en/benchmarks/gemm' },
          { text: 'Attention Performance', link: '/en/benchmarks/attention' }
        ]
      }
    ],
    '/en/api/': [
      {
        text: 'Kernel Atlas',
        items: [
          { text: 'GEMM Kernels', link: '/en/api/gemm' },
          { text: 'Attention Kernels', link: '/en/api/attention' },
          { text: 'Normalization', link: '/en/api/normalization' },
          { text: 'Convolution', link: '/en/api/convolution' },
          { text: 'Sparse Operations', link: '/en/api/sparse' },
          { text: 'Quantization', link: '/en/api/quantization' }
        ]
      }
    ],
    '/en/references/': [
      {
        text: 'Evidence & references',
        items: [
          { text: 'Papers & Citations', link: '/en/references/papers' },
          { text: 'Related Resources', link: '/en/references/resources' }
        ]
      }
    ],
    '/en/': [
      {
        text: 'Start here',
        items: [
          { text: 'Showcase Overview', link: '/en/' },
          { text: 'Getting Started', link: '/en/getting-started' },
          { text: 'Architecture Overview', link: '/en/architecture' }
        ]
      },
      {
        text: 'Technical Whitepaper',
        items: [
          { text: 'Overview', link: '/en/whitepaper/' },
          { text: 'Architecture', link: '/en/whitepaper/architecture' },
          { text: 'Performance', link: '/en/whitepaper/performance' },
          { text: 'Methodology', link: '/en/whitepaper/methodology' }
        ]
      },
      {
        text: 'Evidence',
        items: [
          { text: 'Benchmarks', link: '/en/benchmarks/' },
          { text: 'Papers & Citations', link: '/en/references/papers' },
          { text: 'Related Resources', link: '/en/references/resources' }
        ]
      },
      {
        text: 'Kernel Atlas',
        items: [
          { text: 'GEMM Kernels', link: '/en/api/gemm' },
          { text: 'Attention Kernels', link: '/en/api/attention' },
          { text: 'Normalization', link: '/en/api/normalization' },
          { text: 'Convolution', link: '/en/api/convolution' },
          { text: 'Sparse Operations', link: '/en/api/sparse' },
          { text: 'Quantization', link: '/en/api/quantization' }
        ]
      },
      {
        text: 'Learn',
        items: [
          { text: 'Examples', link: '/en/examples/' },
          { text: 'FAQ', link: '/en/faq' },
          { text: 'Changelog', link: '/en/changelog' }
        ]
      }
    ]
  }
}

// Chinese sidebar
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
    '/zh/examples/': [
      {
        text: '示例',
        items: [
          { text: '概述', link: '/zh/examples/' },
          { text: 'GEMM 教程', link: '/zh/examples/gemm-tutorial' },
          { text: 'FlashAttention', link: '/zh/examples/flash-attention' },
          { text: 'Python 绑定', link: '/zh/examples/python-bindings' }
        ]
      }
    ],
    '/zh/benchmarks/': [
      {
        text: '性能基准',
        items: [
          { text: '概述', link: '/zh/benchmarks/' },
          { text: 'GEMM 性能', link: '/zh/benchmarks/gemm' },
          { text: 'Attention 性能', link: '/zh/benchmarks/attention' }
        ]
      }
    ],
    '/zh/api/': [
      {
        text: 'Kernel Atlas',
        items: [
          { text: 'GEMM 内核', link: '/zh/api/gemm' },
          { text: 'Attention 内核', link: '/zh/api/attention' },
          { text: '归一化', link: '/zh/api/normalization' },
          { text: '卷积', link: '/zh/api/convolution' },
          { text: '稀疏操作', link: '/zh/api/sparse' },
          { text: '量化', link: '/zh/api/quantization' }
        ]
      }
    ],
    '/zh/references/': [
      {
        text: '证据与参考',
        items: [
          { text: '论文引用', link: '/zh/references/papers' },
          { text: '相关资源', link: '/zh/references/resources' }
        ]
      }
    ],
    '/zh/': [
      {
        text: '起步入口',
        items: [
          { text: '展示总览', link: '/zh/' },
          { text: '快速开始', link: '/zh/getting-started' },
          { text: '架构概览', link: '/zh/architecture' }
        ]
      },
      {
        text: '技术白皮书',
        items: [
          { text: '概述', link: '/zh/whitepaper/' },
          { text: '架构设计', link: '/zh/whitepaper/architecture' },
          { text: '性能分析', link: '/zh/whitepaper/performance' },
          { text: '开发方法论', link: '/zh/whitepaper/methodology' }
        ]
      },
      {
        text: '证据',
        items: [
          { text: '性能基准', link: '/zh/benchmarks/' },
          { text: '论文引用', link: '/zh/references/papers' },
          { text: '相关资源', link: '/zh/references/resources' }
        ]
      },
      {
        text: 'Kernel Atlas',
        items: [
          { text: 'GEMM 内核', link: '/zh/api/gemm' },
          { text: 'Attention 内核', link: '/zh/api/attention' },
          { text: '归一化', link: '/zh/api/normalization' },
          { text: '卷积', link: '/zh/api/convolution' },
          { text: '稀疏操作', link: '/zh/api/sparse' },
          { text: '量化', link: '/zh/api/quantization' }
        ]
      },
      {
        text: '学习',
        items: [
          { text: '示例', link: '/zh/examples/' },
          { text: '常见问题', link: '/zh/faq' },
          { text: '更新日志', link: '/zh/changelog' }
        ]
      }
    ]
  }
}
