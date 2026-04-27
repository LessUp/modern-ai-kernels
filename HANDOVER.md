# TensorCraft-HPC 项目交接文档

## 项目状态

**当前阶段**: 稳定化/收尾完成

TensorCraft-HPC 是一个 CUDA 高性能计算库，提供 GEMM、FlashAttention、Softmax 等 GPU 内核的现代实现。项目已完成全面重构和规范化，达到可归档的稳定状态。

---

## 已完成事项

### Phase 1: OpenSpec 审查与 Bug 修复 ✅

| 项目 | 状态 | 说明 |
|------|------|------|
| closeout-stabilization 变更 | ✅ 完成 | OpenSpec 变更已归档 |
| BUG-001: warp_utils.hpp | ✅ 修复 | block_reduce_max 初始化值改为 `std::numeric_limits<T>::lowest()` |
| BUG-002: softmax.hpp | ✅ 修复 | 添加边界条件注释说明 |
| BUG-003: attention.hpp | ✅ 修复 | FlashAttention 限制错误信息更清晰 |
| BUG-004: gemm.hpp | ✅ 修复 | TensorCore API 错误信息更详细 |
| docs/zh/api/core.md | ✅ 翻译 | 完整中文翻译 |
| docs/zh/guides/README.md | ✅ 修复 | 标题翻译为"指南概览" |

### Phase 2: 工程化与 GitHub 集成 ✅

| 项目 | 状态 | 说明 |
|------|------|------|
| pre-commit hooks | ✅ 精简 | 从 15 个减少到 11 个，删除 check-json, yamllint, shellcheck, cmake-format |
| GitHub 元数据 | ✅ 对齐 | Description, topics, homepage 已更新 |

### Phase 3: AI 工具链配置 ✅

| 项目 | 状态 | 说明 |
|------|------|------|
| .cursorrules | ✅ 创建 | 与 AGENTS.md 一致的 Cursor IDE 配置 |
| AI 配置一致性 | ✅ 完成 | AGENTS.md, CLAUDE.md, copilot-instructions.md 高度一致 |

---

## 项目结构

```
TensorCraft-HPC/
├── openspec/           # 活跃规范工作流 (真相源)
│   ├── specs/          # 已接受的基准规范
│   ├── changes/        # 活跃变更 (当前为空)
│   └── archive/        # 已完成变更归档
├── include/tensorcraft/  # 仅头文件库
│   ├── core/           # 核心工具 (cuda_check, features, type_traits, warp_utils)
│   ├── memory/         # 内存管理 (memory_pool, tensor, aligned_vector)
│   └── kernels/        # 内核实现 (gemm, softmax, attention, conv2d, etc.)
├── src/python_ops/     # Python 绑定
├── tests/              # GTest 和 Python 测试
├── benchmarks/         # 性能基准
├── docs/               # GitHub Pages 站点 (中英双语)
├── specs/              # 遗留归档 (历史参考)
└── .github/            # CI/CD 工作流
```

---

## 已知限制

### 代码限制

| 限制 | 位置 | 说明 |
|------|------|------|
| FlashAttention head_dim | `attention.hpp` | 仅支持 head_dim=64，其他值需要额外模板实例化 |
| TensorCore GEMM | `gemm.hpp` | 仅支持 half 精度，需直接调用 `launch_gemm_wmma()` |
| Softmax 边界 | `softmax.hpp` | 当所有输入为 -FLT_MAX 时输出全 0（安全行为） |

### 架构限制

- CUDA 12.0+ 推荐，支持 SM 70-90 (V100 到 H100)
- Python 3.8-3.12
- C++17/20/23 自动检测

---

## 验证命令

### CPU 验证 (无 CUDA)
```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke --parallel 2
python3 -m build --wheel
```

### CUDA 验证 (如有 GPU)
```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
```

### 代码质量
```bash
pre-commit run --all-files
```

---

## 工作流指南

### OpenSpec 驱动开发

1. **研究阶段**: 使用 `/explore` 创建研究文档
2. **提案阶段**: 使用 `/propose` 创建变更提案
3. **实施阶段**: 使用 `/apply` 实现任务
4. **归档阶段**: 使用 `/archive` 归档完成变更

### 规范优先级

1. `openspec/changes/<change-name>/` — 活跃变更
2. `openspec/specs/` — 已接受基准
3. 实现代码 (`include/`, `src/`, `tests/`, `docs/`)

---

## 后续建议

### 可选改进 (非必需)

| 项目 | 优先级 | 说明 |
|------|--------|------|
| FlashAttention head_dim 泛化 | 低 | 支持 32, 128 等其他维度 |
| CUTLASS 集成 | 低 | 替代手写 WMMA 代码 |
| 性能回归测试 | 中 | 添加基准测试框架 |
| Python 绑定扩展 | 低 | 暴露更多激活函数 |

### 维护注意事项

- 保持 OpenSpec 为真相源
- 避免添加新功能，优先简化
- CI/CD 保持精简有效
- 文档保持中英双语对称

---

## 联系与资源

- **GitHub**: https://github.com/shane030316/TensorCraft-HPC
- **GitHub Pages**: https://shane030316.github.io/TensorCraft-HPC/
- **OpenSpec 工作流**: 参见 `openspec/README.md`

---

*本文档由 Claude (Sonnet 4) 生成，用于 GLM 模型接手收尾任务。*
