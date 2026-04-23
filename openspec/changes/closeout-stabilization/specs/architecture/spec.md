# Architecture Delta: closeout-stabilization

## MODIFIED Requirements

### Requirement: Directory Structure (ARCH-002)

The canonical top-level repository layout SHALL make the active workflow and collaboration surfaces
explicit:

```text
modern-ai-kernels/
├── AGENTS.md
├── CLAUDE.md
├── .github/copilot-instructions.md
├── .editorconfig
├── .clangd
├── .vscode/
├── openspec/
├── specs/                      # historical archive only
├── include/tensorcraft/
├── src/python_ops/
├── tests/
├── benchmarks/
├── docs/
└── changelog/                  # process archive, not release log
```

#### Scenario: Active versus archived specification surfaces

- **WHEN** contributors or agents choose a workflow source
- **THEN** `openspec/specs/` and `openspec/changes/` SHALL be treated as active workflow surfaces
- **AND** `specs/` SHALL be treated as historical archive only

## ADDED Requirements

### Requirement: AI Collaboration Surfaces (ARCH-008)

**User Story:** As a maintainer, I want each coding assistant to have a concise project-specific
instruction surface, so that tool behavior stays aligned without duplicating bulky guidance.

#### Scenario: Repository-wide baseline

- **WHEN** an assistant needs repository-wide rules
- **THEN** `AGENTS.md` SHALL provide the shared baseline

#### Scenario: Tool-specific overlays

- **WHEN** Claude or Copilot need tool-specific instructions
- **THEN** `CLAUDE.md` and `.github/copilot-instructions.md` SHALL exist
- **AND** they SHALL stay consistent with `AGENTS.md`
