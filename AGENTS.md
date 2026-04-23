# AGENTS.md

## Purpose

TensorCraft-HPC is in **stabilization / closeout mode**. The priority is to leave the repository
coherent, trustworthy, and cheap to maintain, not to expand scope with speculative features.

## Canonical Sources

Use this precedence order whenever instructions conflict:

1. `openspec/changes/<change-name>/` for the active change
2. `openspec/specs/` for the accepted baseline
3. The implementation under `include/`, `src/`, `tests/`, and `docs/`

`specs/` is a **legacy archive only**. It can inform history, but it is not the active source of
truth for new work.

## Default Workflow

1. Read the relevant OpenSpec files first.
2. If the task changes behavior, structure, workflow, or public documentation, create or update an
   OpenSpec change in `openspec/changes/`.
3. Prefer consolidation, deletion, and simplification over adding new moving parts.
4. Keep changes small in concept, even if they touch several files.
5. Validate with the lightest meaningful checks first:
   - `cmake --preset cpu-smoke`
   - `cmake --build --preset cpu-smoke`
   - `cmake --install build/cpu-smoke --prefix <tmp-dir>`
   - `python3 -m build --wheel`
6. Run CUDA-dependent build/tests only when CUDA is available locally:
   - `cmake --preset dev`
   - `cmake --build --preset dev --parallel <n>`
   - `ctest --preset dev --output-on-failure`
7. Use `/review` before merge on structural, workflow, or policy changes.

## Repository Rules

- Keep `openspec/specs/` and implementation aligned.
- Keep root governance files concise and project-specific.
- Keep GitHub Actions meaningful; avoid green workflows that hide failures.
- Keep GitHub Pages focused on project value and navigation, not README duplication.
- Keep reference docs thin when a root document already exists.
- Keep AI assistant surfaces aligned:
  - `AGENTS.md` = repo-wide baseline
  - `CLAUDE.md` = Claude-specific operating notes
  - `.github/copilot-instructions.md` = Copilot-specific instructions

## Repository Map

- `openspec/` — active spec workflow
- `specs/` — legacy archive
- `include/tensorcraft/` — header-only library
- `src/python_ops/` — Python bindings
- `tests/` — GTest and Python smoke tests
- `benchmarks/` — benchmark binaries
- `docs/` — GitHub Pages site and documentation hub
- `.github/` — workflows, templates, repo automation

## Closeout Guardrails

- Do not introduce new subsystems unless they are required to remove ambiguity or fix a defect.
- Do not keep duplicate docs with the same purpose.
- Do not add heavyweight automation that creates long-term maintenance cost.
- Prefer `clangd` + `compile_commands.json` as the C++/CUDA LSP baseline.
- Prefer one well-scoped branch/PR at a time over parallel branch drift.

## Review Checklist

Before considering a change done, confirm:

- OpenSpec change/proposal exists if the change is structural or behavioral
- README, Pages, and GitHub metadata tell the same story
- CI/workflow changes fail loudly on real problems
- Docs and instructions removed more ambiguity than they added
