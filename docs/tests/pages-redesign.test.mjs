import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

const root = process.cwd()

function read(relPath) {
  return fs.readFileSync(path.join(root, relPath), 'utf8')
}

function exists(relPath) {
  return fs.existsSync(path.join(root, relPath))
}

// === Config & Theme ===

test('docs config exposes academy and evidence routes in both locales', () => {
  const config = read('docs/.vitepress/config.ts')

  assert.match(config, /\/en\/academy\//)
  assert.match(config, /\/zh\/academy\//)
  assert.match(config, /\/en\/evidence\//)
  assert.match(config, /\/zh\/evidence\//)
})

test('theme CSS uses the new token system and avoids stripe callouts', () => {
  const css = read('docs/.vitepress/theme/style.css')

  assert.match(css, /--tc-accent/)
  assert.doesNotMatch(css, /border-left:\s*[4-9]px/)
})

test('theme registers all new academic components', () => {
  const themeIndex = read('docs/.vitepress/theme/index.ts')

  assert.match(themeIndex, /TheoremBox/)
  assert.match(themeIndex, /AlgorithmBlock/)
  assert.match(themeIndex, /CitationCard/)
  assert.match(themeIndex, /TechSpec/)
  assert.match(themeIndex, /ReadingProgress/)
})

// === Homepage ===

test('homepages use the reusable showcase band component', () => {
  const enHome = read('docs/en/index.md')
  const zhHome = read('docs/zh/index.md')

  assert.match(enHome, /<ShowcaseBand/)
  assert.match(zhHome, /<ShowcaseBand/)
})

test('homepages embed GPU timeline component', () => {
  const enHome = read('docs/en/index.md')
  const zhHome = read('docs/zh/index.md')

  assert.match(enHome, /<GPUTimeline/)
  assert.match(zhHome, /<GPUTimeline/)
})

// === Whitepaper ===

test('whitepaper homepages include evolution and related-work modules', () => {
  const enWhitepaper = read('docs/en/whitepaper/index.md')
  const zhWhitepaper = read('docs/zh/whitepaper/index.md')

  assert.match(enWhitepaper, /Evolution notes/)
  assert.match(enWhitepaper, /Related open-source projects/)
  assert.match(zhWhitepaper, /演进思考/)
  assert.match(zhWhitepaper, /相关开源项目/)
})

test('whitepaper performance page uses academic components', () => {
  const enPerf = read('docs/en/whitepaper/performance.md')

  assert.match(enPerf, /<TheoremBox/)
  assert.match(enPerf, /\$\$/)
})

test('whitepaper architecture pages are monolingual', () => {
  const enArch = read('docs/en/whitepaper/architecture.md')
  const zhArch = read('docs/zh/whitepaper/architecture.md')

  assert.doesNotMatch(enArch, /[\u4e00-\u9fff]/)
  assert.match(zhArch, /[\u4e00-\u9fff]/)
})

// === References ===

test('reference papers page uses citation cards', () => {
  const enPapers = read('docs/en/references/papers.md')

  assert.match(enPapers, /<CitationCard/)
})

test('reference pages expose reading strategy and comparison framing', () => {
  const enPapers = read('docs/en/references/papers.md')
  const zhPapers = read('docs/zh/references/papers.md')
  const enResources = read('docs/en/references/resources.md')
  const zhResources = read('docs/zh/references/resources.md')

  assert.match(enPapers, /How to read these citations/)
  assert.match(zhPapers, /如何使用这些引用/)
  assert.match(enResources, /What to borrow, what to resist/)
  assert.match(zhResources, /哪些值得借鉴，哪些应该克制/)
})

// === Mermaid & Assets ===

test('mermaid sources no longer hard-code the old green palette', () => {
  const mermaidPages = [
    'docs/en/architecture.md',
    'docs/zh/architecture.md',
    'docs/en/whitepaper/architecture.md',
    'docs/zh/whitepaper/architecture.md',
    'docs/en/references/resources.md',
    'docs/zh/references/resources.md'
  ].map(read)

  for (const content of mermaidPages) {
    assert.doesNotMatch(content, /76B900|5A9100|2E7D32|F4F7F1/)
  }
})

test('evidence pages use theme-aware SVG assets for diagrams', () => {
  const enEvidence = read('docs/en/evidence/index.md')
  const zhEvidence = read('docs/zh/evidence/index.md')

  assert.match(enEvidence, /<ThemeImage/)
  assert.match(zhEvidence, /<ThemeImage/)
  assert.equal(exists('docs/public/images/diagrams/architecture-dark.svg'), true)
  assert.equal(exists('docs/public/images/diagrams/gemm-optimization-path-dark.svg'), true)
  assert.equal(exists('docs/public/images/diagrams/performance-benchmarks-dark.svg'), true)
})

// === Brand & Assets ===

test('public-facing copy and brand assets reflect the rebuilt site language', () => {
  const enHome = read('docs/en/index.md')
  const lightLogo = read('docs/public/images/logo.svg')
  const darkLogo = read('docs/public/images/logo-dark.svg')

  assert.match(enHome, /Academy|academy/)
  assert.match(enHome, /Evidence|evidence/)
  assert.doesNotMatch(lightLogo, /76B900|8ED000/)
  assert.doesNotMatch(darkLogo, /76B900|8ED000/)
})

test('all theme-aware SVG diagram pairs exist', () => {
  const pairs = [
    ['docs/public/images/diagrams/architecture.svg', 'docs/public/images/diagrams/architecture-dark.svg'],
    ['docs/public/images/diagrams/gemm-optimization-path.svg', 'docs/public/images/diagrams/gemm-optimization-path-dark.svg'],
    ['docs/public/images/diagrams/performance-benchmarks.svg', 'docs/public/images/diagrams/performance-benchmarks-dark.svg']
  ]

  for (const [light, dark] of pairs) {
    assert.equal(exists(light), true, `missing ${light}`)
    assert.equal(exists(dark), true, `missing ${dark}`)
  }
})

// === Components ===

test('theme image component resolves asset URLs through the VitePress base path', () => {
  const themeImage = read('docs/.vitepress/theme/components/ThemeImage.vue')

  assert.match(themeImage, /withBase\(/)
})

test('root docs redirect uses base-aware locale paths', () => {
  const rootIndex = read('docs/index.md')

  assert.match(rootIndex, /withBase\('/)
})

test('llms plugin uses an origin-only domain without duplicating the site base', () => {
  const config = read('docs/.vitepress/config.ts')

  assert.match(config, /domain:\s*'https:\/\/aicl-lab\.github\.io'/)
})

test('localized evidence strips can render Chinese labels', () => {
  const component = read('docs/.vitepress/theme/components/EvidenceStrip.vue')
  const zhEvidence = read('docs/zh/evidence/index.md')

  assert.match(component, /methodLabel/)
  assert.match(component, /sourceLabel/)
  assert.match(zhEvidence, /method-label="方法"/)
  assert.match(zhEvidence, /source-label="来源"/)
})

test('new academic vue components are present and self-contained', () => {
  const components = [
    'docs/.vitepress/theme/components/TheoremBox.vue',
    'docs/.vitepress/theme/components/AlgorithmBlock.vue',
    'docs/.vitepress/theme/components/CitationCard.vue',
    'docs/.vitepress/theme/components/TechSpec.vue',
    'docs/.vitepress/theme/components/ReadingProgress.vue'
  ]

  for (const c of components) {
    assert.ok(exists(c), `missing component: ${c}`)
    const content = read(c)
    assert.match(content, /<template>/)
    assert.match(content, /<script/)
    assert.match(content, /<style/)
  }
})

test('benchmark chart component uses theme-aware text colors', () => {
  const chart = read('docs/.vitepress/theme/components/BenchmarkChart.vue')

  assert.doesNotMatch(chart, /#000000/)
  assert.doesNotMatch(chart, /#000[^f]/)
})

// === Academy & Evidence Content ===

test('academy pages use optimization path component', () => {
  const enAcademy = read('docs/en/academy/index.md')
  const zhAcademy = read('docs/zh/academy/index.md')

  assert.match(enAcademy, /<OptimizationPath/)
  assert.match(zhAcademy, /<OptimizationPath/)
})

test('evidence pages include benchmark chart component', () => {
  const enEvidence = read('docs/en/evidence/index.md')

  assert.match(enEvidence, /<BenchmarkChart/)
})

test('config loads KaTeX and academic fonts', () => {
  const config = read('docs/.vitepress/config.ts')

  assert.match(config, /katex/)
  assert.match(config, /Fraunces/)
  assert.match(config, /JetBrains/)
})
