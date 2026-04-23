# Change Proposal: Initial State

> **Archived**: 2026-04-23
> **Status**: ✅ Complete

---

## Why

This proposal represents the initial state of TensorCraft-HPC after migration from SDD to OpenSpec. All requirements were already implemented before the migration, so this is a snapshot of the completed baseline.

## What Changes

No changes. This is a historical record of the initial state at migration time.

## Capabilities

### Implemented Capabilities

All capabilities were already implemented:

**Core Requirements (REQ-001 to REQ-009):**
- REQ-001: Project Build System
- REQ-002: Core Utility Library
- REQ-003: GEMM Matrix Multiplication
- REQ-004: LLM Key Operators
- REQ-005: Normalization Operators
- REQ-006: Python Bindings
- REQ-007: Testing and Continuous Integration
- REQ-008: Documentation
- REQ-009: Community Governance

**Polish Requirements (REQ-010 to REQ-017):**
- REQ-010: Code Quality Tool Configuration
- REQ-011: GitHub Community Documentation
- REQ-012: GitHub Templates
- REQ-013: CI/CD Pipeline
- REQ-014: Release Process
- REQ-015: Project Documentation Completeness
- REQ-016: GitHub Pages
- REQ-017: Development Environment Configuration

## Impact

This is a historical record. No impact on current code.

---

## Migration Notes

### Original SDD Structure

The original specs were in `specs/` directory:

```
specs/
├── api/cxx-api.md              # API specifications
├── db/data-structures.md       # Data structure specs
├── product/
│   ├── tensorcraft-hpc.md      # Core PRD (REQ-001 to REQ-009)
│   └── project-polish.md       # Polish PRD (REQ-010 to REQ-017)
├── rfc/
│   ├── 0001-core-architecture.md
│   └── 0002-project-polish.md
└── testing/
    ├── tensorcraft-hpc-impl.md (52 tasks, 12 phases)
    └── project-polish-impl.md (29 tasks, 8 phases)
```

### OpenSpec Migration

The specs were migrated to OpenSpec format in `openspec/specs/`:

```
openspec/specs/
├── core/spec.md            # REQ-001 to REQ-009 (scenario format)
├── polish/spec.md          # REQ-010 to REQ-017 (scenario format)
├── api/spec.md             # API contracts
├── data-structures/spec.md # Memory layouts
└── architecture/spec.md    # Design decisions
```

### Format Conversion

Original PRD format:
```markdown
### REQ-001: Project Build System
**Acceptance Criteria:**
| ID | Criterion |
|----|-----------|
| REQ-001-AC1 | The Build_System SHALL use CMake 3.20+ |
```

Converted to OpenSpec scenario format:
```markdown
### Requirement: Project Build System (REQ-001)
#### Scenario: CMake Configuration
- **WHEN** a developer configures the project
- **THEN** the Build_System SHALL use CMake 3.20+ with CMakePresets.json
```
