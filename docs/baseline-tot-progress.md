# 多路径决策 Baseline — 进展记录

> 分支：`feat/baseline-tot` | 日期：2026-07-22

## 目标

搭建多路径决策场景的 baseline——不改 KV、不做优化，纯跑 ToT 实验，
为后续 F7 分支 KV 共享提供对照基准。

## 已完成

- **Princeton Tree of Thoughts** 代码导入 `third_party/tree-of-thought-llm/`（不改）
- **Game of 24** benchmark 数据集（1362 题，csv）
- 实验脚本 `scripts/tot_game24_bench.py`：
  - 环境变量指向本地引擎（`OPENAI_API_BASE=http://localhost:8000/v1`）
  - BFS 搜索（propose + value + greedy selection）
  - 20 题，采集准确率、墙钟、token 用量
- `.venv-tot` 隔离环境（openai 0.27 + numpy 1.26 + pandas + sympy）
- 共享 `.venv` 不受影响

## 踩坑记录

- Cogitator：JSON schema 模式 Qwen2.5-7B 不支持 → 弃用
- Princeton ToT 用旧 openai API（`ChatCompletion.create`）→ 需 openai < 1.0
- numpy/pandas 版本冲突 → `numpy==1.26.4`
- GitHub 被墙 → 手动拷入

## 待做

- 跑通 20 题 baseline
- 对比独立前缀 vs 共享前缀的 KV 命中率
- 接入我们的指标采集（mem_peak、kv_cache_hit_rate）
