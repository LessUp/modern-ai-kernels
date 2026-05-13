# TensorCraft-HPC Documentation

This directory contains the VitePress-based GitHub Pages site for TensorCraft-HPC.

## Structure

```
docs/
├── .vitepress/           # VitePress configuration
│   ├── config.ts         # Main config (i18n, plugins, theme)
│   ├── theme/            # Custom NVIDIA-style theme
│   │   ├── index.ts      # Theme entry
│   │   ├── style.css     # NVIDIA green dark theme
│   │   └── components/   # Vue components
│   └── dist/             # Build output (gitignored)
├── en/                   # English documentation
│   ├── index.md          # Landing page
│   ├── getting-started.md
│   ├── architecture.md
│   ├── api/              # API reference
│   ├── guides/           # User guides
│   └── references/       # Papers, resources
├── zh/                   # Chinese documentation (mirrors en/)
├── public/               # Static assets (images)
├── index.md              # Root redirect page
└── package.json          # npm dependencies
```

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Deployment

GitHub Actions workflow in `.github/workflows/pages.yml` handles:
- Node.js 20 setup
- VitePress build
- GitHub Pages deployment

## Features

- **NVIDIA-style dark theme** with green brand color (#76B900)
- **Mermaid diagrams** for architecture visualization
- **Local search** built-in
- **LLM-ready docs** (`llms.txt`, `llms-full.txt`)
- **Bilingual** (English + Chinese)

## Adding Documentation

1. Create a new `.md` file in the appropriate language directory
2. Add frontmatter if needed:

   ```yaml
   ---
   title: Your Page Title
   ---
   ```

3. Write your content in Markdown
4. Use Mermaid for diagrams:

   ```mermaid
   flowchart LR
       A --> B
   ```

5. Use custom blocks for tips:

   ```markdown
   ::: tip
   Helpful tip here
   :::

   ::: warning
   Warning message
   :::
   ```