# dev-guide — 内部开发指南

> 本目录是**开发过程内部文档**，非赛题交付物。记录规范、进展与规划，便于协作与追溯。

| 文件 | 用途 |
|---|---|
| [`log-naming-convention.md`](log-naming-convention.md) | ★ 实验日志命名规范（强制） |
| [`environment-setup.md`](environment-setup.md) | 环境搭建步骤（uv / 克隆 / 安装命令） |
| [`install-status.md`](install-status.md) | 当前安装状态 + 已知问题 + 待办 |
| [`roadmap.md`](roadmap.md) | MVP → M1..M11 进展与优先级 |

## 工作约定

1. **所有 uv 命令在仓库根目录执行**，隔离环境固定在 `./.venv`（Python 3.11）。
2. **所有实验日志**按 [`log-naming-convention.md`](log-naming-convention.md) 落盘到 `../logs/`。
3. **before/after 对照**：同硬件 / 同模型 / 同 prompt / 同 seed，各跑 3 次取中位数。
4. **任务成功率红线**：优化不得使成功率下降超过 2 个百分点。
5. **commit**：每个优化模块独立分支 / 独立提交，附 before/after 数字。
