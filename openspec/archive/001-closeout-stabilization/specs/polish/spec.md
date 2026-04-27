# Polish Delta: closeout-stabilization

## MODIFIED Requirements

### Requirement: CI/CD Pipeline (REQ-013)

#### Scenario: Meaningful workflow outcomes

- **WHEN** GitHub workflows run
- **THEN** they SHALL fail on meaningful quality violations
- **AND** they SHALL NOT hide failing checks behind success-shaped defaults unless a job is
  intentionally informational

#### Scenario: Copilot cloud-agent setup

- **WHEN** Copilot cloud agent is used on the repository
- **THEN** the project MAY provide `.github/workflows/copilot-setup-steps.yml`
- **AND** that workflow SHALL install only the minimum tools needed for deterministic repo work

### Requirement: GitHub Pages (REQ-016)

#### Scenario: Landing page positioning

- **WHEN** a user visits the documentation site
- **THEN** the landing page SHALL communicate project value, maturity, and navigation clearly
- **AND** it SHALL NOT simply mirror the repository README

### Requirement: Development Environment Configuration (REQ-017)

#### Scenario: Editor and LSP baseline

- **WHEN** contributors configure a local editor
- **THEN** the project SHALL provide `.editorconfig`, `.clangd`, and `.vscode/` recommendations
- **AND** the recommended C++/CUDA LSP baseline SHALL be `clangd` with generated compile commands

## ADDED Requirements

### Requirement: Closeout Workflow Governance (REQ-018)

**User Story:** As a maintainer, I want a lightweight but strict closeout workflow, so that the
repository can converge cleanly without long-lived drift.

#### Scenario: Structural change entry point

- **WHEN** a change affects structure, workflows, governance, or major documentation surfaces
- **THEN** the work SHALL begin from an active OpenSpec change in `openspec/changes/`

#### Scenario: Review discipline

- **WHEN** a structural or workflow change is ready
- **THEN** maintainers SHOULD run `/review` before merge
- **AND** branches SHOULD be merged promptly to avoid divergence

### Requirement: Repository Metadata Alignment (REQ-019)

**User Story:** As a user discovering the project on GitHub, I want the About box, README, and
Pages site to agree, so that I can quickly understand what the repository actually is.

#### Scenario: GitHub About consistency

- **WHEN** repository metadata is updated
- **THEN** the description, homepage, and topics SHALL align with the current README and Pages
  positioning
