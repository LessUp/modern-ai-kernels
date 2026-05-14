/**
 * Sync CHANGELOG.md from root to docs
 * Transforms changelog entries for documentation format
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)
const root = join(__dirname, '..')

// Read root CHANGELOG.md
const changelogPath = join(root, '../CHANGELOG.md')
if (!existsSync(changelogPath)) {
  console.log('CHANGELOG.md not found in root, skipping sync')
  process.exit(0)
}

const changelog = readFileSync(changelogPath, 'utf-8')

// Transform changelog for docs format
function transformChangelog(content, lang) {
  // Add frontmatter
  const frontmatter = lang === 'zh'
    ? `---
title: 更新日志
---

# 更新日志

`
    : `---
title: Changelog
---

# Changelog

`

  // Process the content
  let processed = content

  // Convert markdown links to VitePress format if needed
  // Keep the rest of the content as-is since it's already markdown

  return frontmatter + processed
}

// Write to docs/en/changelog.md
const enDir = join(root, 'en')
const zhDir = join(root, 'zh')

// Ensure directories exist
if (!existsSync(enDir)) mkdirSync(enDir, { recursive: true })
if (!existsSync(zhDir)) mkdirSync(zhDir, { recursive: true })

// Write English version
writeFileSync(join(enDir, 'changelog.md'), transformChangelog(changelog, 'en'))
console.log('✓ Synced changelog to docs/en/changelog.md')

// Write Chinese version (same content for now, could add translation later)
writeFileSync(join(zhDir, 'changelog.md'), transformChangelog(changelog, 'zh'))
console.log('✓ Synced changelog to docs/zh/changelog.md')