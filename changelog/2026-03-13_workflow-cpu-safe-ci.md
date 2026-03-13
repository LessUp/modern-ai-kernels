# Workflow CPU-safe CI 调整

日期：2026-03-13

## 变更内容

- 将主线 CI 从停用中的 CUDA 构建矩阵，收敛为 `Format Check` 与 `Configure CPU Build`
- 在无 CUDA 环境下执行 CMake configure，并验证 `cmake --install` 流程，确保头文件库的 CPU-safe 打包路径可用
- 保留 `workflow_dispatch` 以便手工复核，同时避免 GitHub Hosted Runner 上继续维护无效 GPU job

## 背景

该仓库的核心产物以头文件库与安装配置为主，GitHub Hosted Runner 不适合承担真实 CUDA 构建测试。本次调整将主线 CI 聚焦到格式与无 CUDA 配置能力，减少无意义失败。
