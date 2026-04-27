# Change Proposal: closeout-stabilization

## Why

The repository already contains a strong kernel/code foundation, but its project surface has drifted:
OpenSpec and legacy SDD language coexist, governance docs are inconsistent, editor/LSP setup is
missing, GitHub automation has avoidable noise, and the documentation surface is broader and less
trustworthy than it needs to be for a project that is approaching maintenance closeout.

This change consolidates the repository around a single active workflow, a smaller and clearer
documentation surface, and a leaner automation/tooling baseline.

## What Changes

- make OpenSpec the only active development workflow
- add and align project-specific AI collaboration surfaces
- add missing editor/LSP/repo configuration expected by the specs
- simplify GitHub automation and Copilot cloud-agent bootstrap
- tighten the repository surface so README, Pages, workflows, and About metadata tell the same story

## Capabilities

### New Capabilities

- `closeout-governance`: closeout-oriented workflow, review discipline, and AI instruction surfaces

### Modified Capabilities

- `architecture`: canonical repository structure and active workflow surfaces
- `polish`: development environment, CI behavior, GitHub Pages positioning, and repo metadata hygiene

## Impact

- `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`
- `README*`, `docs/`, `CHANGELOG.md`
- `.editorconfig`, `.clangd`, `.vscode/`
- `.github/workflows/*`, issue/PR templates, repo metadata
- `CMakePresets.json` and related local developer ergonomics

---

## Checklist

- [x] Specs reviewed and understood
- [x] Implementation approach discussed
- [x] Tests planned
- [x] Documentation updated
