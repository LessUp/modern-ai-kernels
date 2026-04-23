# OpenSpec: Archive Completed Change

Archive a completed change proposal.

## Usage

```
/opsx:archive <change-name>
```

## Arguments

- `<change-name>`: The name of a completed change proposal

## Workflow

### Step 1: Verify Completion

Check that all tasks in `tasks.md` are marked complete:
```markdown
- [x] 1.1 Task description  # All should be [x]
```

If incomplete tasks exist, warn user and ask to proceed or cancel.

### Step 2: Verify Delta Specs Applied

Check if delta specs exist in `openspec/changes/<change-name>/specs/`:
- If yes, verify they've been merged into source specs
- If not merged, ask user to run `/opsx:sync` first

### Step 3: Move to Archive

1. Create archive directory: `openspec/archive/YYYY-MM-DD-<change-name>/`
2. Move all files from `openspec/changes/<change-name>/` to archive
3. Add archive metadata:
   - Completion timestamp
   - Summary of changes
   - Links to affected specs

### Step 4: Clean Up

Remove the change directory from `openspec/changes/`.

## Archive Format

```
openspec/archive/2026-04-23-add-fp8-gemm/
├── proposal.md
├── tasks.md
├── design.md (if exists)
├── specs/ (delta specs, kept for reference)
└── archive-meta.json
```

## archive-meta.json

```json
{
  "name": "add-fp8-gemm",
  "completed": "2026-04-23T10:30:00Z",
  "tasks_completed": 12,
  "requirements_added": ["REQ-018", "REQ-019"],
  "requirements_modified": [],
  "specs_affected": ["core", "api"]
}
```
