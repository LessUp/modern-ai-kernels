# 2026-04-16 文档全面优化重构

## 变更背景

对 TensorCraft-HPC 项目进行全面的文档优化和重构，包括：
- 完善 changelog 文档
- 优化 .kiro 规格文档
- 增强 .github 配置
- 修复和增强 GitHub Pages
- 优化 Workflows

## 主要变更

### 1. CHANGELOG 重构

- 重写 `docs/reference/changelog.md`，整合所有版本历史
- 添加版本历史汇总表
- 完善版本比较链接
- 补充 v1.0.1 和 v1.1.0 版本记录

### 2. .kiro 规格文档更新

**tensorcraft-hpc/**
- 更新 `design.md`：简化架构描述，添加版本历史表
- 更新 `requirements.md`：添加实现状态标记
- 更新 `tasks.md`：精简任务列表，添加版本历史

**project-polish/**
- 更新 `design.md`：简化组件描述，添加验证清单
- 更新 `requirements.md`：添加所有需求的完成状态
- 更新 `tasks.md`：更新任务完成状态

### 3. .github 配置增强

- 更新 `bug_report.yml`：添加更多 CUDA 版本选项，添加 CMake preset 字段
- 更新 `feature_request.yml`：优化模板格式
- 更新 `PULL_REQUEST_TEMPLATE.md`：添加本地验证命令
- 更新 `CODEOWNERS`：修正为实际用户名
- 新增 `FUNDING.yml`：添加 GitHub 赞助配置

### 4. GitHub Pages 增强

- 更新 `_config.yml`：
  - 添加导航配置
  - 添加更多 Jekyll 排除项
  - 添加 include 列表
- 更新 `index.md`：
  - 完善文档导航
  - 添加快速开始命令
  - 添加 GPU 架构支持表
  - 添加阅读路径指南

### 5. Workflows 优化

**ci.yml**
- 更新 clang-format-action 到 v4.14.0
- 添加 CMake setup step
- 添加 $GITHUB_STEP_SUMMARY 输出
- 优化 CI 覆盖说明

**release.yml**
- 更新 softprops/action-gh-release 到 v2
- 改进 changelog 提取逻辑
- 添加快速开始命令到 release body
- 使用 CUDA 12.8.0 容器

**pages.yml**
- 添加 job 名称
- 简化配置

## 验证结果

- ✅ 所有文档文件存在
- ✅ 文档导航链接完整
- ✅ GitHub 配置语法正确
- ✅ Workflows 可以正常运行
- ✅ .kiro 规格文档已同步更新

## 文件变更清单

| 操作 | 文件 |
|------|------|
| 修改 | docs/reference/changelog.md |
| 修改 | .kiro/specs/tensorcraft-hpc/design.md |
| 修改 | .kiro/specs/tensorcraft-hpc/requirements.md |
| 修改 | .kiro/specs/tensorcraft-hpc/tasks.md |
| 修改 | .kiro/specs/project-polish/design.md |
| 修改 | .kiro/specs/project-polish/requirements.md |
| 修改 | .kiro/specs/project-polish/tasks.md |
| 修改 | .github/ISSUE_TEMPLATE/bug_report.yml |
| 修改 | .github/ISSUE_TEMPLATE/feature_request.yml |
| 修改 | .github/PULL_REQUEST_TEMPLATE.md |
| 修改 | .github/CODEOWNERS |
| 新增 | .github/FUNDING.yml |
| 修改 | _config.yml |
| 修改 | index.md |
| 修改 | .github/workflows/ci.yml |
| 修改 | .github/workflows/release.yml |
| 修改 | .github/workflows/pages.yml |
