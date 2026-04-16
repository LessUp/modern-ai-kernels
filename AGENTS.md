# AGENTS.md — AI Agent Workflow Instructions

> **Last Updated**: 2026-04-17
> **Version**: 2.0.0

---

## Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development** paradigm. All code implementations take the specification documents under `/specs` as the **Single Source of Truth**.

---

## Project Overview

**TensorCraft-HPC** is a modern C++/CUDA AI high-performance computing kernel library for learning, validating, and implementing core algorithms including GEMM, Attention, Convolution, Normalization, Sparse Operators, and Quantization.

### Project Slogan

**Demystifying High-Performance AI Kernels with Modern C++ & CUDA**

### Tech Stack

| Component | Version |
|-----------|---------|
| C++ Standard | C++17/20/23 |
| CMake | 3.20+ |
| CUDA | 12.8 (targeted), 11.x - 13.1 compatible |
| Python | 3.8+ (for bindings) |

---

## Directory Context

| Directory | Purpose |
|-----------|---------|
| `/specs/product/` | Product feature definitions and acceptance criteria (PRDs) |
| `/specs/rfc/` | Technical design documents and architecture decisions |
| `/specs/api/` | API specification definitions |
| `/specs/db/` | Data structure and memory layout specifications |
| `/specs/testing/` | BDD test case specifications and implementation plans |
| `/include/tensorcraft/` | Header-only kernel library |
| `/src/python_ops/` | Python bindings (pybind11) |
| `/tests/` | Unit tests (GoogleTest) |
| `/benchmarks/` | Performance benchmarks |
| `/docs/` | User-facing documentation |
| `/docs/en/` | English documentation |
| `/docs/zh/` | 简体中文文档 |
| `/examples/` | Example code |
| `/changelog/` | Development changelog |

---

## AI Agent Workflow Instructions

When you (the AI agent) are asked to develop a new feature, modify existing functionality, or fix a bug, **you MUST strictly follow this workflow. Do NOT skip any steps**:

### Step 1: Review Specs

- First, read the relevant documents under `/specs/` — product specs, RFCs, and API definitions.
- If the user's instruction conflicts with existing specs, **stop coding immediately** and point out the conflict. Ask the user whether the spec needs to be updated first.

### Step 2: Spec-First Update

- If this is a new feature, or if existing interfaces/database structures need to change, **you MUST first propose modifying or creating the corresponding spec documents** (e.g., `specs/product/`, `specs/rfc/`, or `specs/api/`).
- Wait for user confirmation of the spec changes before entering the code writing phase.

### Step 3: Implementation

- When writing code, **100% comply with spec definitions** (including variable naming, API paths, data types, status codes, etc.).
- **Do NOT add features not defined in the spec** (No Gold-Plating).
- All generated code should be traceable back to specific spec requirements.

### Step 4: Test Against Spec

- Write unit tests and integration tests based on the acceptance criteria in `/specs/`.
- Ensure test cases cover all boundary conditions described in the specs.
- Use GoogleTest framework for C++ tests.
- Use property-based testing for general correctness properties.

---

## Code Generation Rules

### API Changes

- Any API change exposed externally **MUST** synchronize with the corresponding spec files.
- If uncertain about technical details, consult `specs/rfc/` for architecture conventions. **Do NOT fabricate design patterns.**

### Code Style

| Aspect | Convention |
|--------|------------|
| C++ Standard | C++17 as base |
| Style Guide | Google C++ Style Guide (with CUDA exceptions) |
| Indentation | 4 spaces |
| Class Names | `PascalCase` |
| Function Names | `snake_case` |
| Variable Names | `snake_case` |
| Constants | `kConstantName` or `CONSTANT_NAME` |
| Template Parameters | `PascalCase` |

### CUDA Conventions

| Aspect | Convention |
|--------|------------|
| Kernel Functions | `__global__` prefix |
| Device Functions | `__device__ __forceinline__` |
| Pointer Hints | Use `__restrict__` |
| Launch Bounds | Explicitly specify `__launch_bounds__` |

### Documentation

- All public APIs need documentation comments
- Use Doxygen style comments
- Complex algorithms need explanation

---

## Build System

### CMake Presets

| Preset | Purpose |
|--------|---------|
| `dev` | Recommended CUDA development preset |
| `python-dev` | Lighter build for Python bindings |
| `release` | Full release build with benchmarks |
| `debug` | Debug-oriented CUDA build |
| `cpu-smoke` | CPU-only configure/install validation |

### Build Commands

```bash
# Configure
cmake --preset dev

# Build
cmake --build --preset dev --parallel 2

# Test
ctest --preset dev --output-on-failure

# Install Python bindings
python -m pip install -e .
```

---

## Testing Requirements

### Unit Tests

- All new features need tests
- Use GoogleTest framework
- Test files go in `tests/` directory
- Property-based tests validate general correctness properties
- Unit tests verify specific examples and edge cases

### Running Tests

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
```

---

## Pull Request Checklist

Before submitting a PR, please confirm:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New features have corresponding tests
- [ ] Documentation is updated
- [ ] No compiler warnings
- [ ] No significant performance regression
- [ ] Changes are traceable to spec requirements

---

## Documentation Standards

### README

- README defaults to English
- Link to Chinese version (`README.zh-CN.md`)

### Bilingual Documentation

Follow the project's bilingual documentation structure:
- English: `/docs/en/`
- Chinese: `/docs/zh/`

---

## GPU Architecture Support

| Architecture | SM | Tensor Core | TMA | WGMMA |
|--------------|-----|-------------|-----|-------|
| Volta | 70 | ✅ | ❌ | ❌ |
| Turing | 75 | ✅ | ❌ | ❌ |
| Ampere | 80 | ✅ | ❌ | ❌ |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ |
| Hopper | 90 | ✅ | ✅ | ✅ |
| Blackwell | 100 | ✅ | ✅ | ✅ |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `AGENTS.md` | AI agent workflow instructions (this file) |
| `README.md` | Project overview (English) |
| `README.zh-CN.md` | Project overview (Chinese) |
| `CHANGELOG.md` | Version history |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CODE_OF_CONDUCT.md` | Community standards |
| `SECURITY.md` | Security policy |
| `CMakeLists.txt` | Build configuration |
| `CMakePresets.json` | Build presets |

---

## Contact & Resources

- **GitHub Repository**: https://github.com/LessUp/modern-ai-kernels
- **Online Documentation**: https://lessup.github.io/modern-ai-kernels/
- **Issue Tracker**: https://github.com/LessUp/modern-ai-kernels/issues

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-04-17 | Complete SDD workflow restructure |
| 1.0.0 | 2024-01-01 | Initial version |
