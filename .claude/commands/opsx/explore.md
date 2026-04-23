# OpenSpec: Research Before Committing

Create an exploration document for research purposes. Explorations are read-only and do not affect specs.

## Usage

```
/opsx:explore <topic>
```

## Arguments

- `<topic>`: The research topic (kebab-case, e.g., `tensor-core-performance`, `memory-bandwidth-analysis`)

## Workflow

### Step 1: Create Exploration Document

Create `openspec/explorations/<topic>.md` with initial structure:

```markdown
# Exploration: <topic>

> **Created**: YYYY-MM-DD
> **Status**: In Progress

## Question

<!-- What are we investigating? -->

## Context

<!-- Background information -->

## Findings

<!-- Research findings -->

### Finding 1: <title>

<!-- Details -->

## Decisions

<!-- Any decisions made based on findings -->

## References

<!-- Links to sources, papers, etc. -->

## Next Steps

<!-- What to do next -->
```

### Step 2: Document Research

The exploration document can contain:
- Research findings
- Performance measurements
- Literature review
- Code analysis
- Decision rationale

## Key Points

- Explorations do **NOT** affect specs
- Explorations are for research only
- Use explorations to inform proposals
- Explorations can be referenced from proposals

## Example Flow

```
# Research phase
/opsx:explore tensor-core-performance

# After research, create proposal
/opsx:propose optimize-gemm-tensor-core

# Reference exploration in proposal
# proposal.md:
# > See exploration: [tensor-core-performance](../explorations/tensor-core-performance.md)
```

## Cleanup

Explorations can be:
- Kept for reference
- Moved to archive with related change
- Deleted if no longer relevant
