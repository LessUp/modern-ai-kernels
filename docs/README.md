# TensorCraft-HPC Documentation

This directory contains the Jekyll-based GitHub Pages site for TensorCraft-HPC.

The site is intentionally split into:

- a landing page that explains the repository quickly
- language-specific documentation hubs under `en/` and `zh/`
- thin `reference/` wrappers that point back to canonical root governance documents

## Structure

```
docs/
├── _config.yml              # Jekyll configuration
├── _layouts/                # Custom Jekyll layouts
│   ├── landing.html         # Landing page layout
│   └── docs.html            # Documentation page layout
├── _includes/               # Reusable components
├── assets/
│   └── css/
│       ├── landing.scss     # Landing page styles
│       └── docs.scss        # Documentation styles
├── assets/js/
│   ├── landing.js           # Landing page interactions
│   └── docs.js              # Documentation interactions
├── en/                      # English documentation
├── zh/                      # Chinese documentation
├── 404.html                 # Custom 404 page
└── index.html               # Landing page
```

## Local Development

```bash
cd docs
bundle install
bundle exec jekyll serve --livereload --incremental
# Open http://localhost:4000
```

## Adding Documentation

1. Create a new `.md` file in the appropriate language directory
2. Add frontmatter:

   ```yaml
   ---
   title: Your Page Title
   lang: en  # or zh
   ---
   ```

3. Write your content in Markdown
4. Prefer linking to a canonical root document instead of duplicating long governance content
5. The docs layout will automatically add the sidebar navigation and TOC

## Style Guide

- Use clear, concise language
- Include code examples where relevant
- Follow existing document structures
- Add alerts for important notes:
  - `{: .note }` - Informational
  - `{: .tip }` - Helpful tip
  - `{: .warning }` - Warning
  - `{: .danger }` - Critical warning
  - `{: .important }` - Important note
