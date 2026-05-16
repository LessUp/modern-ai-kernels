# Design: elevate-project-showcase

## Context

TensorCraft-HPC is in closeout / stabilization mode, so the goal is not to broaden scope for its
own sake. The goal is to make the existing project easier to trust, easier to understand, and more
compelling to evaluate.

The current documentation stack is already close to the desired implementation direction:

- the site already uses VitePress, Mermaid, and GitHub Pages deployment
- the content already includes whitepaper, architecture, examples, benchmarks, and references
- bilingual routes already exist

The problem is not missing infrastructure. The problem is that the current site and repository
surface still feel transitional:

- identity and links drift between `LessUp` and `AICL-Lab`
- the homepage does not clearly establish the project as a technical whitepaper / architecture site
- the docs information architecture does not create a strong reader journey
- several diagrams and assets are not safe across light and dark themes
- accepted specs still describe Jekyll-era behavior and outdated directory assumptions

## Current gap matrix

| Area | Current state | Target state | Primary files |
|------|---------------|--------------|---------------|
| Repository identity | Public links, badges, edit links, and examples still mix `LessUp` and `AICL-Lab` | One canonical repository owner, docs domain, and edit-link target across every public surface | `README*.md`, `docs/.vitepress/config.ts`, `.github/workflows/pages.yml`, docs content pages |
| Homepage narrative | Current homepage reads like a default docs landing page | Homepage frames the project as a technical whitepaper / architecture showcase with clear first-read paths | `docs/en/index.md`, `docs/zh/index.md`, `docs/.vitepress/theme/**` |
| Information architecture | Whitepaper, architecture, benchmarks, references, and guides exist but do not form a strong reader journey | Navigation and landing pages create a deliberate flow from whitepaper to academy to evidence to deep reference | `docs/.vitepress/config.ts`, `docs/en/**`, `docs/zh/**` |
| Visual system | Existing theme has custom styling, but hierarchy, spacing, and showcase components are still thin | A reusable token system, stronger section composition, and clearer content emphasis | `docs/.vitepress/theme/style.css`, `docs/.vitepress/theme/components/**` |
| Theme compatibility | Several SVG assets are hard-coded for dark backgrounds and degrade in light mode | Visual assets remain legible in both themes through paired assets or theme-aware rendering | `docs/public/images/logo*.svg`, `docs/public/images/diagrams/*.svg` |
| Spec alignment | Accepted specs still describe Jekyll and outdated directory assumptions | Accepted specs describe the current VitePress architecture and showcase-first docs model | `openspec/specs/{core,polish,architecture}/spec.md` |
| Public-surface coherence | README, Pages, workflow copy, and release-facing text do not tell one consistent story | Every public entry point reinforces the same positioning and learning value | `README*.md`, `docs/**`, `.github/workflows/*.yml` |

## Goals / Non-Goals

### Goals

- Make GitHub Pages the primary showcase for project positioning, technical depth, and learning
  value.
- Define a stable information architecture for whitepaper, academy, benchmark evidence, and
  reference content.
- Align repository identity, links, and metadata across all public entry points.
- Update OpenSpec so accepted requirements match the actual VitePress-based documentation system.
- Require a visual system that remains legible across light and dark themes.

### Non-Goals

- Introduce large new kernel subsystems unrelated to repository presentation and coherence.
- Rebuild the documentation stack on a different framework.
- Preserve outdated structure purely for backward compatibility.
- Create duplicate docs that repeat the same purpose in multiple places.

## Decisions

### Decision 1: GitHub Pages becomes the project's primary public showcase

**Context**: Most readers judge the project before they read the source code, so the site needs to
communicate why the project matters, what it teaches, and why it is trustworthy.

**Options Considered**:
1. Keep Pages as a traditional docs landing page.
2. Reposition Pages as a technical whitepaper / architecture showcase with docs behind it.

**Decision**: Option 2.

**Rationale**: It matches the user's stated goal, strengthens first impressions, and better exposes
the educational and engineering value already present in the repository.

### Decision 2: Reuse the current VitePress stack and borrow `kimi-cli` structural patterns

**Context**: The current site already uses the same core tooling family as the reference project.

**Options Considered**:
1. Switch frameworks again to chase a more ambitious visual stack.
2. Keep VitePress and redesign information architecture, theme tokens, and page composition.

**Decision**: Option 2.

**Rationale**: The project already has the right foundation. The highest-leverage change is better
structure, stronger content hierarchy, and higher-quality visuals — not another framework
migration.

### Decision 3: English and Chinese docs keep mirrored routes and the same information architecture

**Context**: The site already serves both English and Chinese readers, and the showcase should feel
equally intentional in both locales.

**Options Considered**:
1. Let each locale drift independently.
2. Keep one mirrored route structure and shared content model across both locales.

**Decision**: Option 2.

**Rationale**: A mirrored IA reduces maintenance ambiguity and keeps the site trustworthy no matter
which locale a reader enters from.

### Decision 4: Diagrams must be theme-aware, not hard-coded to dark backgrounds

**Context**: Several SVG assets currently assume dark mode and become weak or unreadable in light
mode.

**Options Considered**:
1. Keep static dark-themed diagrams and accept degraded light-mode behavior.
2. Require theme-aware diagrams via paired assets, CSS-variable-driven SVG, or component rendering.

**Decision**: Option 2.

**Rationale**: Cross-theme legibility is part of the product quality of the site, not a cosmetic
nice-to-have.

### Decision 5: Public repository surfaces must tell one story

**Context**: Mixed repository owners, domains, edit links, and example commands make the project
feel less maintained than it actually is.

**Options Considered**:
1. Fix obvious links opportunistically.
2. Treat README, GitHub Pages, workflows, metadata, and examples as one unified public surface.

**Decision**: Option 2.

**Rationale**: The project should feel coherent from every public entry point, especially for
interview and community evaluation.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| The rewrite touches many docs files at once | Stage the work through OpenSpec, IA first, then visuals, then content expansion |
| Aggressive homepage changes could create translation drift | Keep mirrored route structure and update both locale entry pages together |
| Richer visuals can add maintenance burden | Prefer reusable components and a constrained token system over bespoke one-off styling |
| Brand normalization can miss hidden references | Add an explicit audit pass for repository URLs, badges, edit links, and workflow copy |

## Implementation Notes

- Start with the OpenSpec change and accepted-spec deltas before large docs edits.
- Treat `docs/.vitepress/config.ts`, `docs/.vitepress/theme/`, `docs/en/index.md`,
  `docs/zh/index.md`, and the new academy / evidence landing pages as the first implementation front.
- Keep the resulting design biased toward trust, clarity, and evidence instead of marketing copy.
- Validate documentation changes with the existing docs build and repository smoke validations.
