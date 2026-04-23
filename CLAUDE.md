# CLAUDE.md

## Operating mode

Treat TensorCraft-HPC as a **closeout/stabilization** repository:

- prefer pruning and consolidation over expansion
- prefer clear specs and trustworthy automation over feature count
- prefer one coherent change over a stack of drifting partial edits

## Before editing

1. Read the active OpenSpec change in `openspec/changes/` if one exists.
2. Read the relevant accepted spec in `openspec/specs/`.
3. Check whether a root governance document already covers the topic before adding a new doc.

## Implementation rules

- `specs/` is historical reference only.
- For structural or workflow changes, update OpenSpec first.
- Keep root docs authoritative; keep `docs/en/reference/` and `docs/zh/reference/` lightweight.
- Prefer `python3 -m ...` in local-facing commands.
- Prefer `clangd` + `compile_commands.json` for LSP guidance.
- For GitHub-facing changes, align README, Pages, and About metadata before editing with `gh`.

## Validation

Run these first unless the task is docs-only:

```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke --parallel 2
python3 -m build --wheel
```

If CUDA is available, also run:

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
```

## Review

Use `/review` for governance, workflow, or architecture changes before merge.
