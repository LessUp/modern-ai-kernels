---
name: TensorCraft-HPC
description: Technical whitepaper and architecture showcase for modern AI kernels
colors:
  obsidian-ink: "#0f141a"
  graphite-panel: "#171e26"
  steel-line: "#2c3643"
  oxide-copper: "#c96e32"
  oxide-copper-soft: "#e8b38f"
  signal-mint: "#5db7a4"
  paper-warm: "#f3efe8"
  paper-panel: "#e7e0d7"
  slate-text: "#1d2733"
  slate-muted: "#506071"
typography:
  display:
    fontFamily: "\"Inter Tight\", \"Segoe UI\", sans-serif"
    fontSize: "clamp(2.8rem, 6vw, 5.4rem)"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "\"Inter Tight\", \"Segoe UI\", sans-serif"
    fontSize: "clamp(1.8rem, 3vw, 2.75rem)"
    fontWeight: 650
    lineHeight: 1.05
    letterSpacing: "-0.03em"
  title:
    fontFamily: "\"Inter Tight\", \"Segoe UI\", sans-serif"
    fontSize: "1.25rem"
    fontWeight: 600
    lineHeight: 1.2
  body:
    fontFamily: "\"Inter\", \"Segoe UI\", sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.75
  label:
    fontFamily: "\"JetBrains Mono\", \"SFMono-Regular\", monospace"
    fontSize: "0.76rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "0.08em"
rounded:
  sm: "10px"
  md: "18px"
  lg: "28px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "18px"
  lg: "28px"
  xl: "44px"
components:
  button-primary:
    backgroundColor: "{colors.oxide-copper}"
    textColor: "{colors.obsidian-ink}"
    rounded: "{rounded.sm}"
    padding: "12px 18px"
  button-ghost:
    backgroundColor: "{colors.graphite-panel}"
    textColor: "{colors.paper-warm}"
    rounded: "{rounded.sm}"
    padding: "12px 18px"
  insight-panel:
    backgroundColor: "{colors.graphite-panel}"
    textColor: "{colors.paper-warm}"
    rounded: "{rounded.md}"
    padding: "20px 22px"
  paper-panel:
    backgroundColor: "{colors.paper-panel}"
    textColor: "{colors.slate-text}"
    rounded: "{rounded.md}"
    padding: "20px 22px"
---

# Design System: TensorCraft-HPC

## 1. Overview

**Creative North Star: "The annotated lab instrument"**

TensorCraft-HPC should feel like an engineering artifact that has been carefully annotated for public review. The site is not a generic docs shell and not a venture-backed landing page. It should communicate that the reader is entering a serious technical environment where performance claims, architectural boundaries, and research lineage are visible.

The visual tone is dark-led and high-contrast, with warm metallic accents against graphite neutrals. Light mode keeps the same hierarchy and contrast pattern through warm paper surfaces rather than sterile white. The system rejects startup gradients, decorative glass, ornamental serif-editorial styling, and timid academic green. It should read as precise, deliberate, and slightly obsessive.

**Key Characteristics:**

- Technical confidence without visual clutter
- Diagram-centric storytelling
- Strong typographic hierarchy with restrained motion
- Theme-safe surfaces and SVG assets
- Bilingual layouts with mirrored structure

## 2. Colors

The palette pairs machine-room neutrals with an oxide-copper accent and a cool signal tone for evidence and charts.

### Primary

- **Oxide Copper** (`#c96e32`): Primary calls to action, active states, diagram emphasis, and anchor links that need deliberate attention.

### Secondary

- **Signal Mint** (`#5db7a4`): Performance evidence, secondary highlights, chart series, and positive technical signals.

### Neutral

- **Obsidian Ink** (`#0f141a`): Main dark background and hero surfaces.
- **Graphite Panel** (`#171e26`): Elevated panels, nav surfaces, code framing, and dark cards.
- **Steel Line** (`#2c3643`): Borders, dividers, axis lines, and diagram strokes.
- **Paper Warm** (`#f3efe8`): Main light background with less glare than pure white.
- **Paper Panel** (`#e7e0d7`): Elevated light surfaces and diagram frames.
- **Slate Text** (`#1d2733`): Primary light-mode text.
- **Slate Muted** (`#506071`): Secondary light-mode copy and metadata.

### Named Rules

**The Instrument Rule.** Accent color is used to direct attention, not to paint whole sections. Large surfaces rely on neutrals; copper and mint mark the important evidence.

## 3. Typography

**Display Font:** `Inter Tight, Segoe UI, sans-serif`
**Body Font:** `Inter, Segoe UI, sans-serif`
**Label/Mono Font:** `JetBrains Mono, SFMono-Regular, monospace`

**Character:** Headlines should feel condensed, exact, and engineered. Body text should stay neutral and highly readable. Mono is reserved for commands, metadata, and operator labels rather than used as a costume for the whole site.

### Hierarchy

- **Display** (700, `clamp(2.8rem, 6vw, 5.4rem)`, 0.95): Home hero, section openers, and large proof statements.
- **Headline** (650, `clamp(1.8rem, 3vw, 2.75rem)`, 1.05): Section titles and landing page bands.
- **Title** (600, `1.25rem`, 1.2): Card titles, diagram captions, and module headers.
- **Body** (400, `1rem`, 1.75): Long-form whitepaper content capped around 68ch.
- **Label** (600, `0.76rem`, 1.4, `0.08em`): Metrics, section markers, benchmark chips, and metadata.

### Named Rules

**The Proof Stack Rule.** Headings state the claim, labels frame the evidence, body copy explains the reasoning. Never repeat the same message in all three layers.

## 4. Elevation

Depth should come primarily from tonal separation, border discipline, and shadow restraint. In dark mode, panels lift through graphite-on-obsidian contrast with a soft edge glow. In light mode, panels rely more on warm paper layers and crisp border contrast.

### Shadow Vocabulary

- **Ambient Lift** (`0 16px 40px rgba(7, 11, 18, 0.18)`): Hero bands and showcase panels.
- **Focus Lift** (`0 0 0 1px rgba(201, 110, 50, 0.35), 0 14px 28px rgba(7, 11, 18, 0.16)`): Interactive hover and focus states on key tiles.

### Named Rules

**The Flat Until Needed Rule.** Surfaces stay mostly flat at rest. Elevation appears only to guide reading order or interaction.

## 5. Components

### Buttons

- **Shape:** Compact rounded rectangle (`10px`)
- **Primary:** Oxide Copper background with dark text, medium weight, compact horizontal padding
- **Hover / Focus:** Slight lift, stronger outline, no glow blur
- **Secondary / Ghost:** Graphite or transparent surface with visible border

### Chips

- **Style:** Mono labels on subtle neutral fills with thin border contrast
- **State:** Active chips switch to copper or mint emphasis without removing text contrast

### Cards / Containers

- **Corner Style:** `18px`
- **Background:** Tonal panels, never pure black or white
- **Shadow Strategy:** Ambient Lift only on strategic modules
- **Border:** Thin, high-contrast line for structure
- **Internal Padding:** `20px` to `28px`

### Inputs / Fields

- **Style:** Neutral background, visible border, label-first composition
- **Focus:** Copper outline and border shift, no saturated glow
- **Error / Disabled:** Error tone changes border and helper text, not just background

### Navigation

- **Style:** Compact, editorially aligned left, with stronger active-state contrast and theme-safe separators.

### Signature Component

- **Architecture band:** A long-form section module combining eyebrow label, claim heading, explanatory copy, evidence list, and diagram anchor in one asymmetric layout.

## 6. Do's and Don'ts

### Do:

- **Do** keep long-form content at readable measure, around 65 to 75 characters.
- **Do** pair every performance or architecture claim with a visible evidence or reference path.
- **Do** make diagrams and SVGs inherit theme-safe tokens or provide paired assets.
- **Do** use asymmetry, section bands, and typographic contrast to create rhythm instead of repetitive card grids.

### Don't:

- **Don't** ship gradient text, glassmorphism, neon AI tropes, or startup-style marketing copy.
- **Don't** use left-border callout stripes or citation blocks that violate the site's structural language.
- **Don't** rely on monochrome green academic styling or generic documentation templates.
- **Don't** let light mode become a second-class theme for charts, icons, or logos.
