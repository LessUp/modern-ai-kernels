# OpenSpec Workflow Guide

This directory contains the active OpenSpec specification and change management system for TensorCraft-HPC.

## Quick Navigation

| Directory | Purpose |
|-----------|---------|
| **`specs/`** | Accepted baseline specifications (source of truth) |
| **`changes/`** | Active change proposals and implementation tasks |
| **`archive/`** | Completed changes (historical reference) |
| **`explorations/`** | Research documents and pre-proposal investigations |
| **`templates/`** | Document templates for proposals, specs, and tasks |
| **`config.yaml`** | OpenSpec configuration and metadata |

## The OpenSpec Workflow

TensorCraft-HPC uses OpenSpec as the single source of truth for system behavior and changes.

### For Making Changes

1. **Understand the baseline** — Read relevant specs in `specs/`
2. **Create a proposal** — Use templates in `templates/` to draft a change proposal in `changes/<name>/`
3. **Design and refine** — Document design decisions and deltas in the change directory
4. **Implement** — Execute implementation tasks from the proposal
5. **Validate** — Ensure changes pass tests and align with spec
6. **Archive** — Move completed change to `archive/`

### Canonical Specs

The baseline specifications are organized as:

- **`specs/core/`** — Core library concepts and architecture (REQ-001 to REQ-009)
- **`specs/polish/`** — Best practices, ergonomics, and quality standards (REQ-010 to REQ-017)
- **`specs/api/`** — C++ API contracts and public interfaces
- **`specs/data-structures/`** — Memory layouts, tensor formats, and type contracts
- **`specs/architecture/`** — Design decisions, rationale, and constraints

### Active Changes

Current active changes are in `changes/`. Each change directory contains:

- `proposal.md` — What is being changed and why
- `design.md` — Design decisions and tradeoffs
- `tasks.md` — Implementation checklist
- `specs/` — Delta specs or new specifications introduced by this change

### Historical Reference

Completed changes are archived in `archive/` for historical reference and to avoid clutter in the active `changes/` directory.

## For AI Agents and Developers

Before making any change:

1. **Read the relevant spec first** — Don't skip this step
2. **Check for conflicting changes** — Scan `changes/` to see if another change overlaps your work
3. **Create a spec-backed proposal** — All changes must be traceable to `specs/` or justified as deltas
4. **Validate with tests** — Changes must pass existing tests and include tests for new behavior
5. **Use `/review` before merge** — For structural, workflow, or policy changes

See `AGENTS.md` at the repository root for the complete workflow and principles.

---

**Project status**: In stabilization / closeout mode. Priority is coherence and maintainability, not feature expansion.
