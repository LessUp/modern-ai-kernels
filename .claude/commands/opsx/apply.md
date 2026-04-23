# OpenSpec: Apply Change Proposal

Implement the tasks defined in a change proposal.

## Usage

```
/opsx:apply <change-name>
```

## Arguments

- `<change-name>`: The name of an existing change proposal in `openspec/changes/`

## Workflow

### Step 1: Read Proposal Context

Read the following files:
- `openspec/changes/<change-name>/proposal.md`
- `openspec/changes/<change-name>/tasks.md`
- `openspec/changes/<change-name>/design.md` (if exists)
- Delta specs in `openspec/changes/<change-name>/specs/`
- Related source specs in `openspec/specs/`

### Step 2: Execute Tasks

Work through tasks in `tasks.md` in order:
1. Mark task as in-progress
2. Implement the task
3. Mark task as completed (`- [ ]` → `- [x]`)
4. Reference requirement IDs in code comments

### Step 3: Update Specs

If delta specs exist:
1. Apply ADDED requirements to source specs
2. Apply MODIFIED requirements to source specs
3. Remove REMOVED requirements from source specs
4. Apply RENAMED requirements to source specs

### Step 4: Report Progress

Show:
- Completed tasks count
- Remaining tasks count
- Any blocked tasks

## Task Format

Tasks in `tasks.md` use checkboxes:

```markdown
## 1. Preparation

- [ ] 1.1 Read relevant specs
- [x] 1.2 Identify affected files  # Completed
- [ ] 1.3 Review patterns
```

## Completion

When all tasks are complete, prompt user to run:
```
/opsx:archive <change-name>
```
