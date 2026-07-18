# logs — 实验日志

本目录存放所有实验 / 安装日志。**原始日志不入 git**（见根 `.gitignore`），仅保留本说明。

## 命名规范

**必须遵守** [`../dev-guide/log-naming-convention.md`](../dev-guide/log-naming-convention.md)。
速览：

```
logs/<YYYYMMDD-HHMMSS>_<engine>_<model>_<config>_run<N>/   # 单次 run
logs/_summaries/<YYYYMMDD>_<study-name>_comparison.md       # 对照汇总
logs/_installs/<YYYYMMDD-HHMMSS>_uv-install/install.log     # 安装日志
```

## 子目录

- `_installs/` — 环境/依赖安装日志
- `_summaries/` — 多 run 聚合的对照报告（baseline vs prefix-cache vs optimized 等）
- `<run-dir>/` — 单次实验 run（含 metrics.json / engine.log / agent.log / mem_timeseries.csv 等）
