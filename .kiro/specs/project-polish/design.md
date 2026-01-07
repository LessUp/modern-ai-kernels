# Design Document: Project Polish

## Overview

本设计文档描述如何将 TensorCraft-HPC 项目完善为符合开源最佳实践的专业级项目。设计遵循以下原则：

1. **最小侵入**: 不修改现有功能代码，只添加配置和文档
2. **行业标准**: 采用广泛认可的开源项目规范
3. **自动化优先**: 尽可能通过工具自动化质量保障
4. **易于维护**: 配置简洁，便于后续更新

## Architecture

```
TensorCraft-HPC/
├── .github/                    # GitHub 配置
│   ├── ISSUE_TEMPLATE/         # Issue 模板
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── workflows/              # CI/CD 流水线
│   │   ├── ci.yml
│   │   └── release.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS
│   └── FUNDING.yml
├── .vscode/                    # VS Code 配置
│   ├── settings.json
│   ├── extensions.json
│   └── launch.json
├── docs/                       # 文档
│   ├── INSTALL.md
│   └── TROUBLESHOOTING.md
├── examples/                   # 示例代码
│   ├── CMakeLists.txt
│   ├── basic_gemm.cu
│   └── attention_example.cu
├── .clang-format               # 代码格式化配置
├── .clang-tidy                 # 静态分析配置
├── .editorconfig               # 编辑器配置
├── .gitignore                  # Git 忽略规则
├── .gitattributes              # Git 属性配置
├── .pre-commit-config.yaml     # Pre-commit hooks
├── CODE_OF_CONDUCT.md          # 行为准则
├── SECURITY.md                 # 安全策略
├── CHANGELOG.md                # 变更日志
└── ... (existing files)
```

## Components and Interfaces

### 1. 代码质量工具配置

#### 1.1 .clang-format

基于 Google 风格，针对 CUDA 项目进行调整：

```yaml
# .clang-format
---
Language: Cpp
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 100
AccessModifierOffset: -4
AlignAfterOpenBracket: Align
AlignConsecutiveAssignments: false
AlignConsecutiveDeclarations: false
AlignOperands: true
AlignTrailingComments: true
AllowAllParametersOfDeclarationOnNextLine: true
AllowShortBlocksOnASingleLine: false
AllowShortCaseLabelsOnASingleLine: false
AllowShortFunctionsOnASingleLine: Inline
AllowShortIfStatementsOnASingleLine: false
AllowShortLoopsOnASingleLine: false
BinPackArguments: true
BinPackParameters: true
BreakBeforeBraces: Attach
BreakConstructorInitializers: BeforeColon
IncludeBlocks: Regroup
IncludeCategories:
  - Regex: '^<cuda'
    Priority: 1
  - Regex: '^<'
    Priority: 2
  - Regex: '^"tensorcraft/'
    Priority: 3
  - Regex: '.*'
    Priority: 4
IndentCaseLabels: true
NamespaceIndentation: None
PointerAlignment: Left
SortIncludes: true
SpaceAfterCStyleCast: false
SpaceAfterTemplateKeyword: true
SpaceBeforeAssignmentOperators: true
SpaceBeforeParens: ControlStatements
SpaceInEmptyParentheses: false
SpacesInAngles: false
SpacesInContainerLiterals: true
SpacesInParentheses: false
SpacesInSquareBrackets: false
Standard: c++17
TabWidth: 4
UseTab: Never
```

#### 1.2 .clang-tidy

针对 C++17/CUDA 的静态分析配置：

```yaml
# .clang-tidy
---
Checks: >
  -*,
  bugprone-*,
  -bugprone-easily-swappable-parameters,
  clang-analyzer-*,
  cppcoreguidelines-*,
  -cppcoreguidelines-avoid-magic-numbers,
  -cppcoreguidelines-pro-bounds-pointer-arithmetic,
  -cppcoreguidelines-pro-type-reinterpret-cast,
  modernize-*,
  -modernize-use-trailing-return-type,
  performance-*,
  readability-*,
  -readability-magic-numbers,
  -readability-identifier-length

WarningsAsErrors: ''

HeaderFilterRegex: 'include/tensorcraft/.*'

CheckOptions:
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: lower_case
  - key: readability-identifier-naming.VariableCase
    value: lower_case
  - key: readability-identifier-naming.ConstantCase
    value: UPPER_CASE
  - key: readability-identifier-naming.TemplateParameterCase
    value: CamelCase
  - key: cppcoreguidelines-special-member-functions.AllowSoleDefaultDtor
    value: true
```

#### 1.3 .editorconfig

跨编辑器的统一配置：

```ini
# .editorconfig
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true

[*.{cpp,hpp,cu,cuh,h}]
indent_style = space
indent_size = 4

[*.{cmake,txt}]
indent_style = space
indent_size = 4

[*.{json,yml,yaml}]
indent_style = space
indent_size = 2

[*.md]
trim_trailing_whitespace = false

[Makefile]
indent_style = tab
```

### 2. GitHub 社区文档

#### 2.1 CODE_OF_CONDUCT.md

采用 Contributor Covenant v2.1 标准。

#### 2.2 SECURITY.md

安全漏洞报告流程：
- 私密报告方式
- 响应时间承诺
- 漏洞披露政策

#### 2.3 Issue Templates

Bug Report 模板字段：
- 问题描述
- 复现步骤
- 期望行为
- 环境信息（OS、CUDA 版本、GPU 型号）
- 日志输出

Feature Request 模板字段：
- 功能描述
- 使用场景
- 可能的实现方案

#### 2.4 PR Template

检查清单：
- [ ] 代码遵循项目风格规范
- [ ] 所有测试通过
- [ ] 新功能有对应测试
- [ ] 文档已更新
- [ ] CHANGELOG 已更新

### 3. CI/CD 流水线

#### 3.1 CI Workflow (ci.yml)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  format-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check formatting
        uses: jidicula/clang-format-action@v4
        with:
          clang-format-version: '17'
          check-path: 'include src tests'

  build-cuda11:
    runs-on: ubuntu-latest
    container: nvidia/cuda:11.8.0-devel-ubuntu22.04
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: apt-get update && apt-get install -y cmake ninja-build
      - name: Configure
        run: cmake --preset release
      - name: Build
        run: cmake --build build/release

  build-cuda12:
    runs-on: ubuntu-latest
    container: nvidia/cuda:12.2.0-devel-ubuntu22.04
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: apt-get update && apt-get install -y cmake ninja-build
      - name: Configure
        run: cmake --preset release
      - name: Build
        run: cmake --build build/release
```

#### 3.2 Release Workflow (release.yml)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          generate_release_notes: true
```

### 4. 版本管理

#### 4.1 版本号规范

遵循语义化版本 2.0.0：
- MAJOR: 不兼容的 API 变更
- MINOR: 向后兼容的功能新增
- PATCH: 向后兼容的问题修复

#### 4.2 CHANGELOG 格式

遵循 Keep a Changelog 规范：

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- ...

### Changed
- ...

### Fixed
- ...

## [1.0.0] - 2024-XX-XX

### Added
- Initial release with core kernel implementations
- GEMM optimizations (Naive, Tiled, Double Buffer, Tensor Core)
- FlashAttention-style attention kernel
- Normalization kernels (LayerNorm, RMSNorm, BatchNorm)
- Convolution kernels (Naive, Im2Col, Depthwise)
- Sparse matrix operations (CSR, CSC, SpMV, SpMM)
- Python bindings via pybind11
- Comprehensive documentation
```

### 5. 开发环境配置

#### 5.1 VS Code 配置

settings.json:
- C++ 扩展配置
- CUDA 语法高亮
- clang-format 集成
- CMake 工具配置

extensions.json:
- ms-vscode.cpptools
- ms-vscode.cmake-tools
- nvidia.nsight-vscode-edition
- xaver.clang-format

#### 5.2 .gitignore

覆盖：
- 构建目录 (build/, cmake-build-*)
- IDE 文件 (.idea/, *.swp)
- 编译产物 (*.o, *.so, *.a)
- Python 缓存 (__pycache__, *.pyc)
- 系统文件 (.DS_Store, Thumbs.db)

#### 5.3 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/pre-commit/mirrors-clang-format
    rev: v17.0.6
    hooks:
      - id: clang-format
        types_or: [c++, cuda]
```

## Data Models

本设计不涉及数据模型变更，仅添加配置文件和文档。



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

本设计主要涉及配置文件和文档的创建，大多数验收标准是文件存在性检查（example 类型），不适合属性测试。以下是可验证的属性：

### Property 1: Configuration File Completeness

*For any* required configuration file (.clang-format, .clang-tidy, .editorconfig, .gitignore, .gitattributes, .pre-commit-config.yaml), the file SHALL exist in the project root directory.

**Validates: Requirements 1.1, 1.2, 1.3, 6.2, 6.3, 6.4**

### Property 2: GitHub Community Files Completeness

*For any* required community file (CODE_OF_CONDUCT.md, SECURITY.md, CHANGELOG.md), the file SHALL exist in the project root directory.

**Validates: Requirements 2.1, 2.2, 2.5**

### Property 3: GitHub Templates Completeness

*For any* required GitHub template (bug_report.yml, feature_request.yml, PULL_REQUEST_TEMPLATE.md), the file SHALL exist in the .github/ directory structure.

**Validates: Requirements 2.3, 2.4**

### Property 4: CI Workflow Configuration

*For any* CI workflow file, it SHALL contain triggers for push and pull_request events on the main branch.

**Validates: Requirements 3.1**

### Property 5: Documentation Completeness

*For any* required documentation file (INSTALL.md, TROUBLESHOOTING.md), the file SHALL exist in the docs/ directory.

**Validates: Requirements 5.1, 5.2**

## Error Handling

本设计不涉及运行时错误处理，所有内容为静态配置文件。

配置文件语法错误将在以下阶段被检测：
- YAML 文件：CI 流水线启动时
- clang-format/clang-tidy：工具执行时
- CMake：配置阶段

## Testing Strategy

### 验证方法

由于本设计主要涉及配置文件和文档，测试策略以手动验证和 CI 集成测试为主：

1. **文件存在性检查**: 通过 CI 脚本验证所有必需文件存在
2. **配置语法检查**: 通过工具执行验证配置文件语法正确
3. **CI 流水线测试**: 通过实际运行 CI 验证流水线配置正确

### 验证清单

- [ ] .clang-format 可被 clang-format 正确解析
- [ ] .clang-tidy 可被 clang-tidy 正确解析
- [ ] .pre-commit-config.yaml 可被 pre-commit 正确解析
- [ ] GitHub Actions workflow 语法正确
- [ ] 所有 Markdown 文件格式正确

