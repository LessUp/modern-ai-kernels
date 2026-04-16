# RFC 0002: Project Polish — Open Source Best Practices

> **RFC Number**: 0002
> **Title**: Project Polish — Open Source Best Practices
> **Status**: ✅ Accepted — Implemented
> **Type**: Process
> **Created**: 2024-01-01
> **Last Updated**: 2026-04-17

---

## Summary

This RFC describes how to elevate TensorCraft-HPC into a professional open-source project following community best practices.

---

## Motivation

While TensorCraft-HPC has solid core functionality, it lacks the standard infrastructure expected of professional open-source projects:

1. Code quality tool configuration
2. CI/CD pipelines
3. Community governance documentation
4. Standardized contribution process

This RFC defines the approach to address these gaps.

---

## Design Principles

1. **Minimal Intrusion**: No modification to existing functional code; only adding configuration and documentation
2. **Industry Standards**: Adopt widely recognized open-source project conventions
3. **Automation-First**: Maximize quality assurance through tool automation
4. **Easy Maintenance**: Keep configuration simple for future updates

---

## Architecture

### Project Structure After Polish

```
TensorCraft-HPC/
├── .github/                    # GitHub configuration
│   ├── ISSUE_TEMPLATE/         # Issue templates
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── workflows/              # CI/CD pipelines
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   └── pages.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS
├── .vscode/                    # VS Code configuration
│   ├── settings.json
│   └── extensions.json
├── docs/                       # Documentation
│   ├── README.md               # Documentation navigation
│   ├── en/                     # English documentation
│   ├── zh/                     # Chinese documentation
│   ├── getting-started/
│   ├── guides/
│   ├── api/
│   ├── examples/
│   └── reference/
├── specs/                      # Specification documents
│   ├── product/
│   ├── rfc/
│   ├── api/
│   ├── db/
│   └── testing/
├── .clang-format               # Code formatting config
├── .clang-tidy                 # Static analysis config
├── .editorconfig               # Editor config
├── .gitignore                  # Git ignore rules
├── .gitattributes              # Git attributes
├── .pre-commit-config.yaml     # Pre-commit hooks
├── AGENTS.md                   # AI agent instructions
├── CHANGELOG.md                # Version history
├── CODE_OF_CONDUCT.md          # Community standards
├── CONTRIBUTING.md             # Contribution guide
├── LICENSE                     # MIT License
├── README.md                   # Project overview (English)
├── README.zh-CN.md             # Project overview (Chinese)
├── SECURITY.md                 # Security policy
└── _config.yml                 # Jekyll configuration
```

---

## Components

### 1. Code Quality Tool Configuration

| File | Purpose |
|------|---------|
| `.clang-format` | Based on Google style, adjusted for CUDA projects |
| `.clang-tidy` | Static analysis configuration for C++17/CUDA |
| `.editorconfig` | Unified cross-editor configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks for automated checks |

### 2. GitHub Community Documentation

| File | Purpose |
|------|---------|
| `CODE_OF_CONDUCT.md` | Contributor Covenant v2.1 standard |
| `SECURITY.md` | Security vulnerability reporting process |
| `CONTRIBUTING.md` | Contributing guide |
| `CHANGELOG.md` | Keep a Changelog format |

### 3. GitHub Templates

| File | Purpose |
|------|---------|
| `.github/ISSUE_TEMPLATE/bug_report.yml` | Bug report template |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | Feature request template |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR checklist |
| `.github/CODEOWNERS` | Code review owners |

### 4. CI/CD Pipeline

| Workflow | Purpose |
|----------|---------|
| `.github/workflows/ci.yml` | Format check, CPU build, Python packaging |
| `.github/workflows/release.yml` | Automated version release |
| `.github/workflows/pages.yml` | GitHub Pages deployment |

### 5. GitHub Pages Configuration

| File | Purpose |
|------|---------|
| `_config.yml` | Jekyll configuration |
| `index.md` | Documentation landing page |
| `docs/en/` | English documentation |
| `docs/zh/` | Chinese documentation |

---

## Implementation Details

### CI/CD Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│                         CI Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: Format Check                                          │
│  └── clang-format --dry-run --Werror                           │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: CPU Build (cpu-smoke)                                 │
│  └── cmake --preset cpu-smoke && cmake --build --preset cpu-smoke│
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: Python Package                                        │
│  └── pip install -e . && python -c "import tensorcraft_ops"    │
└─────────────────────────────────────────────────────────────────┘
```

### Release Process

1. Tag commit with semantic version (e.g., `v2.0.0`)
2. GitHub Actions builds release artifacts
3. Release notes generated from CHANGELOG.md
4. GitHub Release created automatically

---

## Verification Checklist

- [x] `.clang-format` can be correctly parsed by clang-format
- [x] `.clang-tidy` can be correctly parsed by clang-tidy
- [x] `.pre-commit-config.yaml` can be correctly parsed by pre-commit
- [x] GitHub Actions workflow syntax is valid
- [x] All Markdown files are correctly formatted
- [x] Documentation navigation links are valid
- [x] GitHub Pages can be deployed successfully

---

## Impact

### Benefits

1. **Professional appearance**: Project looks mature and well-maintained
2. **Easier contributions**: Clear guidelines and templates
3. **Automated quality**: CI catches issues early
4. **Better documentation**: Comprehensive, bilingual docs

### Risks

- None identified (no functional code changes)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-04-17 | Converted to new SDD format |
| 1.0.0 | 2024-01-01 | Initial implementation |

---

## References

- [Product Specification: Project Polish](../product/project-polish.md)
- [Implementation Plan](../testing/project-polish-impl.md)
- [RFC 0001: Core Architecture](0001-core-architecture.md)
