# Tasks: elevate-project-showcase

## Overview

Rebuild TensorCraft-HPC's outward-facing project surface so GitHub Pages acts as a technical
whitepaper / architecture showcase, while README, workflows, visual assets, and accepted specs stay
aligned with that direction.

## 1. Preparation

- [x] 1.1 Read relevant specs in `openspec/specs/`
- [x] 1.2 Audit the current docs stack, public links, workflows, and visual assets
- [x] 1.3 Review the `kimi-cli` docs structure for reusable VitePress patterns

## 2. Implementation

- [x] 2.1 Create the `elevate-project-showcase` OpenSpec change set
- [x] 2.2 Redesign docs information architecture and homepage narrative
- [x] 2.3 Rebuild the VitePress theme and showcase components
- [x] 2.4 Repair public branding, links, metadata, and deployment-path consistency
- [x] 2.5 Replace light/dark-incompatible diagrams with theme-safe assets or components
- [x] 2.6 Expand whitepaper, benchmark-evidence, and references content

## 3. Testing

- [x] 3.1 Run `npm ci` in `docs/` if needed for a clean docs toolchain
- [x] 3.2 Run `npm run build` in `docs/`
- [x] 3.3 Run `cmake --preset cpu-smoke`
- [x] 3.4 Run `cmake --build --preset cpu-smoke --parallel 2`
- [x] 3.5 Run `python3 -m build --wheel`

## 4. Documentation

- [x] 4.1 Align `README.md` with the new Pages positioning
- [x] 4.2 Align `README.zh-CN.md` with the new bilingual showcase surface
- [x] 4.3 Update changelog / release-facing copy if the public positioning changes materially

## 5. Finalization

- [x] 5.1 Review repository-wide public links and branding one final time
- [x] 5.2 Run a merge-blocking review before integration because this is a structural and workflow-facing change
- [ ] 5.3 Archive the completed change after accepted specs are merged

---

## Validation Notes

- `cd docs && npm ci`
- `cd docs && npm run build`
- `cmake --preset cpu-smoke`
- `cmake --build --preset cpu-smoke --parallel 2`
- `python3 -m build --wheel`

## Progress

| Section | Total | Completed | Status |
|---------|-------|-----------|--------|
| Preparation | 3 | 3 | ✅ |
| Implementation | 6 | 6 | ✅ |
| Testing | 5 | 5 | ✅ |
| Documentation | 3 | 3 | ✅ |
| Finalization | 3 | 2 | ⏳ |
| **Total** | **20** | **19** | ⏳ |
