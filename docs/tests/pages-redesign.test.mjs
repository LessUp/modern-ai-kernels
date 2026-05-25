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

test('docs config exposes academy and evidence routes in both locales', () => {
  const config = read('docs/.vitepress/config.ts')

  assert.match(config, /\/en\/academy\//)
  assert.match(config, /\/zh\/academy\//)
  assert.match(config, /\/en\/evidence\//)
  assert.match(config, /\/zh\/evidence\//)
})

test('theme CSS uses the new token system and avoids stripe callouts', () => {
  const css = read('docs/.vitepress/theme/style.css')

  assert.match(css, /--tc-accent:/)
  assert.doesNotMatch(css, /border-left:\s*[23-9]px/)
})

test('homepages use the reusable showcase band component', () => {
  const enHome = read('docs/en/index.md')
  const zhHome = read('docs/zh/index.md')

  assert.match(enHome, /<ShowcaseBand/)
  assert.match(zhHome, /<ShowcaseBand/)
})

test('whitepaper homepages include evolution and related-work modules', () => {
  const enWhitepaper = read('docs/en/whitepaper/index.md')
  const zhWhitepaper = read('docs/zh/whitepaper/index.md')

  assert.match(enWhitepaper, /Evolution notes/)
  assert.match(enWhitepaper, /Related open-source projects/)
  assert.match(zhWhitepaper, /演进思考/)
  assert.match(zhWhitepaper, /相关开源项目/)
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

test('public-facing copy and brand assets reflect the rebuilt site language', () => {
  const readme = read('README.md')
  const openspecDesign = read('openspec/changes/elevate-project-showcase/design.md')
  const lightLogo = read('docs/public/images/logo.svg')
  const darkLogo = read('docs/public/images/logo-dark.svg')

  assert.match(readme, /Academy|academy/)
  assert.match(readme, /Evidence|evidence/)
  assert.match(openspecDesign, /academy/i)
  assert.doesNotMatch(lightLogo, /76B900|8ED000/)
  assert.doesNotMatch(darkLogo, /76B900|8ED000/)
})

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
