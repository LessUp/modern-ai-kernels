# OpenSpec: Create Change Proposal

Create a new change proposal in the OpenSpec workflow.

## Usage

```
/opsx:propose <change-name>
```

## Arguments

- `<change-name>`: A kebab-case name for the change (e.g., `add-fp8-gemm`, `optimize-flash-attention`)

## Workflow

### Step 1: Create Proposal Directory

Create the following structure:

```
openspec/changes/<change-name>/
├── proposal.md
├── tasks.md
└── specs/
```

### Step 2: Generate proposal.md

Use the template from `openspec/templates/proposal.md`:

```markdown
# Change Proposal: <change-name>

## Why

<!-- Explain the motivation for this change -->

## What Changes

<!-- Describe what will change -->

## Capabilities

### New Capabilities
<!-- List new capabilities -->

### Modified Capabilities
<!-- List modified capabilities -->

## Impact

<!-- Affected code, APIs, dependencies -->
```

### Step 3: Generate tasks.md

Use the template from `openspec/templates/tasks.md`.

### Step 4: Notify User

Report the created proposal location and next steps.

## Examples

```
/opsx:propose add-fp8-gemm
/opsx:propose optimize-flash-attention
/opsx:propose add-python-flash-attention
```

## Next Steps

After creating a proposal:
1. Fill in the proposal details
2. Create delta specs if modifying existing requirements
3. Run `/opsx:apply <change-name>` to implement
