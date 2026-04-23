# Contributing to TensorCraft-HPC

TensorCraft-HPC is maintained as a **stabilizing reference project**. Contributions are welcome,
but the bar is intentionally biased toward correctness, clarity, and maintainability.

## What Fits This Repository

Good contributions usually do one of these:

- fix correctness, build, packaging, or documentation issues
- tighten the project surface so it is easier to understand and maintain
- improve tests, validation, or workflow reliability
- align implementation with OpenSpec requirements

Changes that add broad new scope should start with a strong OpenSpec rationale.

## Required Workflow

1. Review the relevant files in `openspec/specs/`.
2. For behavior, architecture, workflow, or documentation-surface changes, create or update an
   active change under `openspec/changes/`.
3. Keep the implementation tied to that change.
4. Run the minimum meaningful validations before opening a PR.
5. Use `/review` before merge for structural or governance changes.

`specs/` remains in the repository as historical context only.

## Local Setup

### Baseline validation

```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke --parallel 2
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
python3 -m build --wheel
```

### CUDA-enabled validation

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
```

### Hooks

```bash
python3 -m pip install pre-commit
pre-commit install
```

## Editor and LSP

The recommended baseline is:

- `clangd` for C++/CUDA navigation
- `cmake --preset dev` to generate `build/dev/compile_commands.json`
- the checked-in `.vscode/` recommendations if you use VS Code

## Pull Requests

Every PR should:

- reference the active OpenSpec change when relevant
- explain what was removed, consolidated, or fixed
- list the validations that were run
- keep docs, workflows, and repo metadata aligned

Merge quickly after review. Avoid long-lived branch drift.
