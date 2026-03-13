# 2026-03-13 文档与 Pages 信息架构规范化

## 变更背景

- 按批次 A 的统一要求，继续收敛仓库入口与文档入口职责。
- 此前 `README.md`、`README.zh-CN.md` 与 `index.md` 都承担了大量项目说明、文档导航和示例内容，入口层级不够清晰。
- Pages workflow 仅监听 `main`，而仓库当前实际分支为 `master`，会导致文档改动无法稳定触发自动部署。

## 导航与目录调整

- 保持现有 Jekyll 文档结构不变，继续使用根目录 `index.md` 作为文档首页。
- 将 `README.md` / `README.zh-CN.md` 收敛为仓库入口，仅保留项目定位、最小构建命令与文档链接。
- 将 `docs/INSTALL.md`、`docs/architecture.md`、`docs/optimization_guide.md`、`docs/api_reference.md` 等页面明确分工为快速开始、架构设计、使用指南与参考入口。
- 继续使用 `CONTRIBUTING.md` 作为开发指南，`CHANGELOG.md` 与 `changelog/` 作为版本与归档入口。

## 首页调整

- `index.md` 改写为文档导读页，新增项目定位、适合谁、从哪里开始、推荐阅读路径与核心文档表。
- 首页不再重复完整代码示例、长篇项目结构和大段特性清单，而是把读者引导到具体文档页面继续阅读。
- 站点首页保留 CI / Pages / License 等核心徽章，方便直接判断工程状态。

## Pages / Workflow 调整

- `.github/workflows/pages.yml` 的推送触发分支从仅 `main` 扩展为 `master, main`。
- 保持现有 `actions/configure-pages`、`actions/upload-pages-artifact` 与 `actions/deploy-pages` 链路不变，仅修正分支触发与信息架构相关内容。

## 验证结果

- 已确认当前仓库分支为 `master`，workflow 现已覆盖实际使用分支。
- 已人工检查 `README`、`index.md`、`docs/INSTALL.md`、`docs/architecture.md`、`docs/optimization_guide.md`、`docs/api_reference.md`、`CONTRIBUTING.md` 与 `CHANGELOG.md` 的链接关系均对应现有文件。
- 已保留此前 Pages 优化中形成的文档目录与工作流结构，只做入口职责收敛。
- 本次未运行本地 Jekyll 构建；后续可在具备 Ruby / Jekyll 环境时补充静态构建验证。

## 后续待办

- 如后续继续扩充文档内容，可进一步把 `docs/` 收敛到更明确的 `guide/`、`reference/`、`archive/` 层级。
- 可在后续单独整理 `CHANGELOG.md` 与 `changelog/` 的职责边界，减少版本记录与过程记录的重复。
