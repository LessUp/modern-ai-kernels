# Copilot instructions for TensorCraft-HPC

- Use `openspec/changes/` for active work and `openspec/specs/` for accepted requirements.
- Treat `specs/` as historical archive, not the active source of truth.
- This repository is in stabilization/closeout mode: prefer cleanup, consistency, and bug fixing over
  new surface area.
- Keep docs lean and project-specific. Prefer updating root docs over duplicating content.
- Keep README, GitHub Pages, and repository About metadata aligned.
- Prefer `clangd` and `build/dev/compile_commands.json` for C++/CUDA navigation.
- Use existing validations before inventing new ones:
  - `cmake --preset cpu-smoke`
  - `cmake --build --preset cpu-smoke`
  - `python3 -m build --wheel`
- Use `/research` for broad repository analysis and `/remote` for GitHub-side actions when they make
  the workflow clearer.
- For structural or governance changes, use `/review` before merge.
