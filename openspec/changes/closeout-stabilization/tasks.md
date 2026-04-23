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
- [ ] 2.2 Normalize governance docs (`AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, Copilot instructions)
- [ ] 2.3 Add missing editor/LSP configuration (`.editorconfig`, `.clangd`, `.vscode/`)
- [ ] 2.4 Simplify local and GitHub automation (`CMakePresets.json`, CI, Copilot setup)
- [ ] 2.5 Rebuild README / Pages / docs hub / repo metadata into one coherent surface

## 3. Testing

- [ ] 3.1 Run CPU-only configure/build/install smoke validation
- [ ] 3.2 Run Python wheel build validation
- [ ] 3.3 Run CUDA validation if available locally

## 4. Documentation

- [ ] 4.1 Update root documentation and reference wrappers
- [ ] 4.2 Document the OpenSpec-first development flow
- [ ] 4.3 Record governance/tooling cleanup in `CHANGELOG.md`

## 5. Finalization

- [ ] 5.1 Align GitHub About metadata with the new surface
- [ ] 5.2 Review the change with `/review`
- [ ] 5.3 Archive the completed change

---

## Progress

| Section | Total | Completed | Status |
|---------|-------|-----------|--------|
| Preparation | 3 | 3 | ✅ |
| Implementation | 4 | 1 | ⏳ |
| Testing | 3 | 0 | ⏳ |
| Documentation | 3 | 0 | ⏳ |
| Finalization | 3 | 0 | ⏳ |
| **Total** | **16** | **4** | ⏳ |
