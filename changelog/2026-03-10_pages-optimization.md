# GitHub Pages 彻底优化 (2026-03-10)

## 变更内容

### _config.yml
- 添加完整 `exclude` 列表，排除源码/构建/IDE 等无关目录，加速 Jekyll 构建

### pages.yml
- 精确化 `paths` 触发过滤器，避免无关 `.md` 变更触发部署
- 扩展 `sparse-checkout` 显式列出所有文档文件

### index.md
- 重写为专业 GitHub Pages 落地页
- 添加 CI/License/CUDA/C++ 徽章
- 添加架构总览图、算子矩阵（含优化技术）、核心特性表
- 添加快速开始（含 C++ 代码示例）、GPU 架构支持表
- 添加结构化文档导航表

### README.md
- 添加 CI 徽章
- 修复错误仓库地址 `your-username/tensorcraft-hpc` → `LessUp/modern-ai-kernels`

### README.zh-CN.md
- 全面扩充：添加 CI/Docs/全套徽章
- 添加项目愿景、扩充算子表（含优化技术列）、技术特性
- 添加详细快速开始（环境要求、CMake Presets、集成方式）
- 添加完整使用示例、项目结构、GPU 架构表、文档链接

### CHANGELOG.md
- 修复底部比较链接 `username/tensorcraft-hpc` → `LessUp/modern-ai-kernels`
- 添加 v2.0.0 版本链接

### .gitignore
- 添加 `_site/`、`.jekyll-cache/`、`.sass-cache/` (Jekyll)
- 添加 `.pytest_cache/`、`.ruff_cache/`、`.cache/`

### docs/INSTALL.md
- 修复 4 处错误仓库地址 `username/tensorcraft-hpc` → `LessUp/modern-ai-kernels`

### docs/TROUBLESHOOTING.md
- 修复 Getting Help 中错误的 Issues 链接
