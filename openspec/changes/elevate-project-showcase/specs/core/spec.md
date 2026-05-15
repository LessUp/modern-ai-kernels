# Core Specification Delta: elevate-project-showcase

## MODIFIED Requirements

### Requirement: Documentation (REQ-008)

**User Story:** As a learner, evaluator, or contributor, I want documentation that introduces the
project clearly and then leads me into deeper technical material, so that I can quickly understand
what makes TensorCraft-HPC valuable.

#### Scenario: Documentation directory
- **WHEN** exploring documentation
- **THEN** the documentation SHALL remain organized in `docs/`

#### Scenario: Mirrored bilingual routes
- **WHEN** navigating published documentation
- **THEN** the site SHALL provide mirrored English and Simplified Chinese route trees under
  `docs/en/` and `docs/zh/`

#### Scenario: Documentation structure
- **WHEN** navigating the docs
- **THEN** the documentation SHALL include sections for `whitepaper/`, `guides/`, `api/`,
  `examples/`, `benchmarks/`, and `references/`

#### Scenario: Showcase-first entry
- **WHEN** a reader lands on the published site
- **THEN** the homepage SHALL act as a project showcase that highlights project positioning,
  architecture, evidence, and learning paths before deep reference content

#### Scenario: Documentation deployment
- **WHEN** documentation is published
- **THEN** the documentation SHALL be deployed via GitHub Pages
