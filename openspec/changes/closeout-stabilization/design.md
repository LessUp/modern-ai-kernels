# Design: closeout-stabilization

## Context

TensorCraft-HPC no longer needs broad platform expansion as much as it needs coherence. The largest
risks are no longer missing features in isolation, but repository drift:

- active workflow language differs between files
- required developer-experience assets from the specs are missing
- GitHub automation is noisier than its value
- root docs, Pages, and About metadata are not yet intentionally aligned

## Goals / Non-Goals

### Goals

- Make OpenSpec the only active workflow surface.
- Reduce governance/documentation duplication.
- Add a practical LSP/editor baseline for C++/CUDA work.
- Keep GitHub automation minimal but strict.
- Support Copilot cloud agent with only the setup it actually needs.

### Non-Goals

- Introduce large new kernel features.
- Add heavyweight external infrastructure that increases maintenance burden.
- Preserve every historical process artifact in its current prominence.

## Decisions

### Decision 1: OpenSpec is the sole active process

**Context**: Mixed OpenSpec/SDD language creates ambiguity for humans and agents.

**Options Considered**:
1. Keep both workflows documented side by side.
2. Treat legacy specs as archive only and make OpenSpec authoritative.

**Decision**: Option 2.

**Rationale**: The project needs one active source of truth, not parallel workflows.

### Decision 2: AI instructions are split by audience but share one message

**Context**: Repo-wide, Claude-specific, and Copilot-specific instructions all exist for different
tools, but they must not drift.

**Options Considered**:
1. Keep one very large universal instruction file.
2. Keep `AGENTS.md`, `CLAUDE.md`, and `.github/copilot-instructions.md` concise and aligned.

**Decision**: Option 2.

**Rationale**: Each tool gets the surface it expects without forcing unrelated detail into every
context window.

### Decision 3: Standardize local navigation on clangd

**Context**: The repository is C++/CUDA-heavy and lacks a checked-in LSP baseline.

**Options Considered**:
1. Leave editor/LSP setup undocumented.
2. Standardize on `clangd` + `build/dev/compile_commands.json`.

**Decision**: Option 2.

**Rationale**: It works across editors and agents with the least repository-specific complexity.

### Decision 4: Keep automation strict, not chatty

**Context**: A success-shaped workflow with ignored failures or low-value summary jobs increases
noise without improving trust.

**Options Considered**:
1. Keep permissive workflow behavior and explanatory note jobs.
2. Remove low-value jobs and fail on meaningful checks.

**Decision**: Option 2.

**Rationale**: Closeout-mode repositories benefit from smaller, more trustworthy automation.

### Decision 5: Copilot cloud-agent setup stays minimal

**Context**: Cloud-agent setup can easily become another maintenance surface.

**Options Considered**:
1. Preinstall a broad toolchain "just in case."
2. Install only the tools needed for CPU-side validation and repo hygiene.

**Decision**: Option 2.

**Rationale**: It supports deterministic setup without pulling the project into workflow bloat.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Some historical documents lose prominence | Keep them in archive-oriented locations and clarify their role |
| Contributors used to old workflow language may need to relearn | Make root governance docs short and explicit |
| `clangd` baseline may not cover every local CUDA layout | Pair it with generated compile commands and avoid hardcoded machine-specific paths |

## Implementation Notes

- Prefer replacing sprawling governance docs with shorter, higher-signal versions.
- Prefer root canonical docs plus lightweight documentation wrappers.
- Apply GitHub metadata changes only after README/Pages positioning is aligned.
