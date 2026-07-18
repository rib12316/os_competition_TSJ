# 实验日志命名规范（强制）

> 目标：所有实验日志**全 ASCII、可 glob、可排序、可复现**。命名一旦定下，后续所有 run 必须遵守。

## 1. 单次 run 目录

```
logs/<YYYYMMDD-HHMMSS>_<engine>_<model>_<config>_run<N>/
```

| 字段 | 规则 | 取值词汇表 |
|---|---|---|
| `YYYYMMDD-HHMMSS` | 本地时，24h，run 开始时刻 | 如 `20260718-143022` |
| `engine` | 推理引擎，小写 | `vllm` \| `sglang` \| `vllm-ascend` |
| `model` | 模型短名，小写连字符 | `qwen25-7b` \| `minicpm3-4b` \| `qwen3-0.6b` … |
| `config` | 配置/优化档，小写连字符 | `baseline` \| `prefix-cache` \| `fp8-kv` \| `lmcache` \| `m3-session` \| `m5-branch` \| `m6-lazyload` … |
| `run<N>` | 同配置第 N 次重复 | `run1` \| `run2` \| `run3`（取中位数，建议 3 次） |

**分隔规则**：字段内部用连字符 `-`；字段之间用下划线 `_`。**禁止空格、中文、大写混用**。

**示例：**
```
logs/20260718-143022_vllm_qwen25-7b_baseline_run1/
logs/20260718-144510_vllm-ascend_qwen25-7b_prefix-cache_run2/
logs/20260718-150033_sglang_qwen25-7b_fp8-kv_run3/
logs/20260718-160120_vllm_minicpm3-4b_lmcache_run1/
```

## 2. run 目录内固定文件名

| 文件 | 内容 | 备注 |
|---|---|---|
| `metrics.json` | 6 大指标结构化结果 | 见下方 schema |
| `engine.log` | 引擎服务端 stdout+stderr | vllm / sglang / vllm-ascend |
| `agent.log` | agent loop trace | Qwen-Agent 多轮 |
| `mem_timeseries.csv` | 显存按时间采样 | 表头 `timestamp,used_mb` |
| `vllm_metrics.json` | `/metrics` 端点原始抓取 | KV 命中率等 |
| `config.yaml` | 本次完整配置快照 | 可复现 |
| `git_commit.txt` | 仓库 commit hash | `git rev-parse HEAD` |
| `env.txt` | 关键版本快照 | engine/torch/torch_npu/CANN/NPU id |

### `metrics.json` schema（建议）

```json
{
  "run_id": "20260718-143022_vllm_qwen25-7b_baseline_run1",
  "engine": "vllm",
  "model": "qwen25-7b",
  "config": "baseline",
  "e2e_latency_p50_ms": 0.0,
  "e2e_latency_p95_ms": 0.0,
  "qps": 0.0,
  "mem_peak_mb": 0,
  "kv_cache_hit_rate": 0.0,
  "task_success_rate": 0.0,
  "ttft_ms": 0.0,
  "seed": 42,
  "started_at": "2026-07-18T14:30:22"
}
```

## 3. 对照汇总

```
logs/_summaries/<YYYYMMDD>_<study-name>_comparison.md
```

- `study-name`：小写连字符，描述本次对照主题，如 `mvp-three-tier`、`m1-fp8`、`m5-dual-engine`、`multi-model`。
- 内容：把同 study 下各 run 的 `metrics.json` 聚合成对照表（before/after），附结论。
- **示例**：`logs/_summaries/20260718_mvp-three-tier_comparison.md`

## 4. 安装日志

```
logs/_installs/<YYYYMMDD-HHMMSS>_<task>/install.log
```

- `task` ∈ {`uv-install`, `cann-upgrade`, `vllm-ascend-build`, …}
- 完整捕获命令 stdout+stderr；末尾附 `uv pip freeze` / 版本信息。

## 5. 排序与特殊目录

- `_` 前缀目录（`_installs/`、`_summaries/`）在 `ls` / glob 中排序置顶，与 run 目录区分。
- 同一 study 的多次 run 仅 `run<N>` 与时间戳不同，便于 `ls logs/*_<config>_run*` 批量聚合。

## 6. 检查清单（每次 run 前自检）

- [ ] 目录名严格匹配 `<ts>_<engine>_<model>_<config>_run<N>`，无空格/中文
- [ ] `config.yaml` + `git_commit.txt` + `env.txt` 已写入（可复现）
- [ ] `metrics.json` 字段齐全
- [ ] 同配置至少 run1/run2/run3 三次（取中位数）
