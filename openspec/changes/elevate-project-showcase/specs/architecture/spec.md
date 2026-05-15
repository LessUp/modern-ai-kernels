# Architecture Specification Delta: elevate-project-showcase

## MODIFIED Requirements

### Requirement: Directory Structure (ARCH-002)

**User Story:** As a developer, I want the repository layout to reflect the actual documentation
system and public project surface, so that I can find and maintain the relevant files without
guesswork.

#### Scenario: Project layout
- **WHEN** navigating the project
- **THEN** the following directory structure SHALL be used:

```
modern-ai-kernels/
├── openspec/                 # Active OpenSpec workflow
├── include/tensorcraft/      # Header-only kernel library
├── src/python_ops/           # Python bindings
├── tests/                    # Validation
├── benchmarks/               # Performance benchmarks
├── docs/                     # VitePress documentation site
│   ├── .vitepress/           # Site config, theme, and components
│   ├── public/               # Static assets
│   ├── en/                   # English docs routes
│   └── zh/                   # Simplified Chinese docs routes
└── .github/                  # Workflows, templates, repo automation
```

#### Scenario: Documentation system boundaries
- **WHEN** changing the public documentation surface
- **THEN** site structure, theme code, and static assets SHALL live under `docs/` instead of being
  split across legacy documentation systems
