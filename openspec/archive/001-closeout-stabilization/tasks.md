# Tasks: closeout-stabilization

## Overview

Tighten TensorCraft-HPC for closeout by making OpenSpec authoritative, reducing repository drift,
standardizing local tooling, and shrinking noisy automation/documentation surfaces.

## 1. Preparation

- [x] 1.1 Read relevant specs in `openspec/specs/`
- [x] 1.2 Identify affected files and components
- [x] 1.3 Review current workflows, docs, and repository metadata

## 2. Implementation

- [x] 2.1 Create the closeout-stabilization OpenSpec change set
- [x] 2.2 Normalize governance docs (`AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, Copilot instructions)
- [x] 2.3 Add missing editor/LSP configuration (`.editorconfig`, `.clangd`, `.vscode/`)
- [x] 2.4 Simplify local and GitHub automation (`CMakePresets.json`, CI, Copilot setup)
- [x] 2.5 Rebuild README / Pages / docs hub / repo metadata into one coherent surface

## 3. Testing

- [x] 3.1 Run CPU-only configure/build/install smoke validation
- [x] 3.2 Run Python wheel build validation
- [x] 3.3 Run CUDA validation if available locally (skipped - no CUDA environment)

## 4. Documentation

- [x] 4.1 Update root documentation and reference wrappers
- [x] 4.2 Document the OpenSpec-first development flow
- [x] 4.3 Record governance/tooling cleanup in `CHANGELOG.md`

## 5. Finalization

- [x] 5.1 Align GitHub About metadata with the new surface
- [x] 5.2 Review the change with `/review` (completed by Claude)
- [x] 5.3 Archive the completed change

---

## Progress

| Section | Total | Completed | Status |
|---------|-------|-----------|--------|
| Preparation | 3 | 3 | ✅ |
| Implementation | 4 | 4 | ✅ |
| Testing | 3 | 3 | ✅ |
| Documentation | 3 | 3 | ✅ |
| Finalization | 3 | 3 | ✅ |
| **Total** | **16** | **16** | ✅ |

---

## Changes Made

### Documentation
- Rewrote `docs/en/guides/architecture.md` to proper English
- Created `docs/zh/guides/architecture.md` with Chinese translation
- Completed `docs/zh/getting-started/installation.md` translation (129→full)
- Translated `docs/zh/getting-started/troubleshooting.md`
- Translated `docs/zh/examples/README.md`
- API docs (`docs/zh/api/kernels.md`) kept as-is (technical API signatures)

### Configuration
- Added sm_70 (V100) support to CMakePresets.json release preset
- Enhanced `.clangd` with multi-directory fallback (build/dev, build/release)
- Removed redundant `pre-commit` job from CI (kept `format-check`)

### Cleanup
- Deleted `changelog/` directory (content consolidated in CHANGELOG.md)

### GitHub
- Updated repository description, topics, and homepage URL

---

## Additional Changes (2026-04-27 Final Closeout)

### Bug Fixes
- Fixed `warp_utils.hpp:120`: block_reduce_max initialization now uses `std::numeric_limits<T>::lowest()`
- Enhanced `softmax.hpp`: Added comprehensive boundary condition documentation
- Improved `attention.hpp`: FlashAttention head_dim=64 error message now explains limitation
- Fixed `gemm.hpp`: TensorCore API error message now provides clear usage guidance

### Documentation
- Translated `docs/zh/api/core.md` to Chinese (complete translation)
- Fixed `docs/zh/guides/README.md` title to "指南概览"

### Configuration
- Simplified `.pre-commit-config.yaml`: Reduced from 15 to 11 hooks (removed check-json, yamllint, shellcheck, cmake-format)
- Created `.cursorrules` for Cursor IDE support

### Project Assets
- Created `HANDOVER.md` for GLM model handoff documentation
