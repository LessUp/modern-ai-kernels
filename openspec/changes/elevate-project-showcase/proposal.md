# Change Proposal: elevate-project-showcase

## Why

TensorCraft-HPC already has real technical substance, but its outward-facing surfaces do not yet
present that substance with enough clarity or consistency.

Today the repository shows several kinds of drift that weaken trust:

- GitHub Pages behaves more like a generic documentation bundle than a deliberate technical
  whitepaper / architecture showcase.
- README, Pages, edit links, release copy, and clone examples still point at mixed repository
  identities (`LessUp` and `AICL-Lab`).
- the documentation specification still describes Jekyll-era behavior even though the site is built
  with VitePress.
- multiple SVG diagrams are hard-coded for dark backgrounds and become difficult to read in light
  mode.

This change establishes a spec-backed path to rebuild the project surface so the repository can be
presented as a coherent, high-signal open-source artifact for interviews, technical readers, and
the wider GitHub community.

## What Changes

- reposition GitHub Pages as the primary technical showcase, not just a documentation index
- define a stronger information architecture for whitepaper, architecture, evidence, and reference
  content
- align repository branding, links, metadata, and public copy across README, Pages, workflows, and
  examples
- replace outdated GitHub Pages / documentation spec language with the current VitePress-based model
- require light/dark-safe visual assets and theme-aware diagrams

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `core`: documentation structure and published-docs positioning
- `architecture`: canonical repository layout for the VitePress documentation system
- `polish`: documentation completeness and GitHub Pages implementation requirements

## Impact

- `openspec/specs/{core,architecture,polish}/spec.md`
- `docs/.vitepress/**`
- `docs/{index.md,en/**,zh/**,public/**}`
- `README.md`, `README.zh-CN.md`
- `.github/workflows/pages.yml` and related public-facing workflow copy

---

## Checklist

- [x] Specs reviewed and understood
- [x] Implementation approach discussed
- [x] Tests planned
- [x] Documentation updated
