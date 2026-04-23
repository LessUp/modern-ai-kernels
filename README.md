# TensorCraft-HPC

<div align="center">

**Demystifying High-Performance AI Kernels with Modern C++ & CUDA**

[English](README.md) | [简体中文](README.zh-CN.md) | [Docs](https://lessup.github.io/modern-ai-kernels/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://github.com/LessUp/modern-ai-kernels/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/modern-ai-kernels/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17%2B-00599C?logo=c%2B%2B&logoColor=white)

</div>

TensorCraft-HPC is a **header-only C++/CUDA kernel library** for learning, validating, and packaging
modern AI operators. The repository keeps implementations readable, exposes progressive optimization
paths where that helps learning, and ships a lightweight Python binding surface for smoke-level
integration.

**Project posture:** this repository is being tightened toward a stable closeout state. The priority
is a coherent codebase, trustworthy docs, and maintainable automation.

## Why this repository exists

- readable C++/CUDA kernels with a clear module layout
- progressive operator implementations for study and comparison
- OpenSpec-driven repository workflow
- CPU-only smoke validation plus optional local CUDA validation
- bilingual documentation and GitHub Pages

## Capability snapshot

| Area | Scope |
|------|-------|
| Core utilities | CUDA checks, feature detection, type traits, warp helpers |
| Memory | `Tensor`, aligned vectors, memory pool |
| Kernels | GEMM, attention, normalization, convolution, sparse, fusion |
| Python | `tensorcraft_ops` bindings for smoke/integration workflows |
| Validation | CPU smoke build/install, Python wheel build, optional local CUDA tests |

## Quick start

### CPU-only smoke validation

```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke --parallel 2
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
python3 -m build --wheel
```

### CUDA-enabled local validation

```bash
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)
ctest --preset dev --output-on-failure
```

## Documentation

- **Documentation hub**: <https://lessup.github.io/modern-ai-kernels/>
- **Getting started**: `docs/en/getting-started/`
- **Architecture and optimization guides**: `docs/en/guides/`
- **API surface**: `docs/en/api/`
- **Chinese docs**: `docs/zh/`

## OpenSpec workflow

This repository uses **OpenSpec** as the active development workflow.

1. Review the accepted specs in `openspec/specs/`.
2. For behavioral, structural, workflow, or major documentation changes, create or update a change
   under `openspec/changes/`.
3. Implement against that change and keep docs/configs aligned.
4. Run validation before merge.
5. Use `/review` before merging structural or workflow changes.

`specs/` remains in the repository as a historical archive only.

## Repository layout

```text
modern-ai-kernels/
├── AGENTS.md                      # Repo-wide AI workflow rules
├── CLAUDE.md                      # Claude-specific guidance
├── .github/copilot-instructions.md
├── openspec/                      # Active spec workflow
├── specs/                         # Legacy archive
├── include/tensorcraft/           # Header-only C++/CUDA library
├── src/python_ops/                # Python bindings
├── tests/                         # Validation
├── benchmarks/                    # Benchmark binaries
└── docs/                          # GitHub Pages + documentation
```

## Tooling baseline

- **Build system**: CMake presets
- **Formatting / hooks**: `.clang-format`, `.clang-tidy`, `pre-commit`
- **LSP**: `clangd` with `build/dev/compile_commands.json`
- **GitHub automation**: CI, Pages, release workflow, Copilot setup steps

## License

Released under the [MIT License](LICENSE).
