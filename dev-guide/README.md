# dev-guide — 内部开发指南

> 本目录是**开发过程内部文档**，非赛题交付物。记录规范、进展与规划，便于协作与追溯。

| 文件 | 用途 |
|---|---|
| [`log-naming-convention.md`](log-naming-convention.md) | ★ 实验日志命名规范（强制） |
| [`environment-setup.md`](environment-setup.md) | 环境搭建步骤（uv / 克隆 / 安装命令） |
| [`install-status.md`](install-status.md) | 当前安装状态 + 已知问题 + 待办 |
| [`roadmap.md`](roadmap.md) | MVP → M1..M11 进展与优先级 |
| [`roles/`](roles/) | ★ 三人并行协作模型（Model C）+ 各角色 MVP/扩展任务清单 |

## 三人协作分工

开发采用 **Model C（2 特性开发 + 1 测量/质量中枢）**，详见 [`roles/README.md`](roles/README.md)。三份角色任务清单：

- [`roles/p1-engine-backend.md`](roles/p1-engine-backend.md) — 引擎/后端（M1/M4/M5/M8…）
- [`roles/p2-agent-app.md`](roles/p2-agent-app.md) — Agent/应用（M2/M6…）
- [`roles/p3-measure-quality.md`](roles/p3-measure-quality.md) — 测量骨干 + 质量门（M3/M10…）+ 全模块 before/after

> 关键：P3 **day 1 先交付契约 + stub**，P1/P2 对契约并行开发，不被互相阻塞。

## 工作约定

1. **所有 uv 命令在仓库根目录执行**，隔离环境固定在 `./.venv`（Python 3.11）。
2. **所有实验日志**按 [`log-naming-convention.md`](log-naming-convention.md) 落盘到 `../logs/`。
3. **before/after 对照**：同硬件 / 同模型 / 同 prompt / 同 seed，各跑 3 次取中位数。
4. **任务成功率红线**：优化不得使成功率下降超过 2 个百分点。
5. **commit**：每个优化模块独立分支 / 独立提交，附 before/after 数字。
