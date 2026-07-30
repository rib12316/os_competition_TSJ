# F2 LLMLingua2 tau-bench 115-task 串行测试

> 日期：2026-07-29
> 范围：tau-bench `retail/test`，`task_id=0..114`
> 执行：复用已归档 task 0..9，本轮串行补跑 task 10..114

## 1. 实验配置

- Agent：本地 vLLM `Qwen2.5-7B-Instruct`；USER：`mimo-v2.5-pro`
- `seed=42`、`max_steps=20`、`concurrency=1`、`runs=1`
- LLMLingua2：`trigger_tokens=3000`、`rate=0.60`、`recompress_delta_tokens=2000`、`keep_hot=6`
- 结构保护：`tool_aware=true`、`assistant_rate=0.60`、`tool_result_rate=0.60`、`hot_tool_trigger_tokens=1000`
- 静态优化：`retail_compact` 与工具描述去重开启

## 2. 115-task 汇总

| 指标 | 结果 |
|---|---:|
| 成功率 | 22/115（19.13%） |
| 总步骤 / 模型调用 | 1883 / 1883 |
| e2e p50 | 81.8 s |
| e2e p95 | 146.6 s |
| e2e mean | 87.6 s |
| task 级 TTFT 中位数 | 55.6 ms |
| 有效执行口径 QPS | 0.01019 task/s |
| HBM peak（见口径说明） | 56,813 MB |

成功 task：`1, 6, 7, 10, 12, 13, 16, 22, 24, 25, 38, 43, 44, 50, 52, 59, 60, 61, 68, 79, 88, 114`。

115 个正式结果均为 `error=null`，没有 transport/runtime 错误或结果级重试。执行器在 task 46 与 47 之间中断一次；task 47 的未完成 start 未产生正式结果，恢复后从检查点重新执行。QPS 使用旧 10-task 实际墙钟加 105 个有效 attempt 的起止时间计算，排除了中断等待时间；不要使用续跑 `summary.json` 内由恢复段 runner 自动生成的 `0.01603`。

逐任务 success、steps、latency、TTFT、attempt/error 见 `merged-0-114/task_results.csv`、`task_results.json` 和 `task_results.md`。

## 3. Prompt token 与压缩

| 同轨迹配对口径 | token |
|---|---:|
| 变换前 Prompt | 11,607,341 |
| 变换后 Prompt（canonical tokenizer） | 9,357,860 |
| vLLM 实际上报 Prompt | 9,354,692 |
| 配对节省 | 2,249,481（19.38%） |

- 固定静态前缀每次节省 1,187 token，1,883 次共 2,235,121，占全部配对节省 99.36%。
- 超出固定静态前缀的额外配对节省为 14,360 token：compress 调用 5,331、reuse 调用 7,173、skip 调用 1,856。其中包含动态冷历史、热工具正文和计量取整，不能全部归为冷历史收益。
- canonical 变换后计数与 vLLM usage 相差 3,168 token（-3,168），严格节省比例始终使用同一 tokenizer 的变换前后配对值。

## 4. F2 事件与压缩器耗时

| 指标 | 结果 |
|---|---:|
| compress / reuse / skip | 10 / 13 / 1860 |
| 触发动态冷历史的 session | 10/115 |
| 触发 task | 5, 29, 52, 62, 64, 65, 95, 100, 108, 112 |
| 触发时可压正文 | 31,623 token |
| 完整冷区：压缩前 -> 压缩后 | 57,974 -> 52,946 |
| 冷历史压缩器总耗时 | 78.87 s |
| 冷历史单次 mean / median / p95 | 7.89 / 8.54 / 9.91 s |
| 冷历史单次 min / max | 5.42 / 10.23 s |
| 热工具处理事件 / 事件侧节省 | 93 / 2,196 token |
| 热工具非零计时样本 / 总耗时 | 31 / 38.04 s |
| 冷历史 + 热工具显式压缩耗时 | 116.91 s |

3000 阈值按 tool-aware 隔离后的可压自然语言正文计量，最终只有 10/115 个 session 触发动态冷历史。10 次触发时完整冷区合计减少 5,028 token；后续 13 次 reuse 继续复用缓存结果。

## 5. 基础设施指标口径

- task 10..114 的 vLLM 前后 counter delta 恰好包含 1,713 次请求，KV prefix-cache hit rate 为 93.89%；归档 task 0..9 的快照值为 93.07%。两段来自不同 vLLM counter window，不伪造统一 115-task KV 命中率。
- HBM peak 的 56,813 MB 是旧 0..9 与恢复后采样段的最大观测；中断前 task 10..46 的时序文件未完整落盘。vLLM 预分配主导该数值，不能作为 F2 token 节省的直接证据。
- task 0..9 与 10..114 配置相同，但不是同一连续进程窗口。成功率、延迟、Prompt/F2 日志可按 task 合并；进程级 KV/HBM 指标只分段报告。

## 6. 归档

- 完整汇总：`docs/f2-results/llmlingua2-115-20260729/merged-0-114/`
- 新跑原始结果：`docs/f2-results/llmlingua2-115-20260729/continuation-10-114/`
- 复用来源：`docs/f2-results/llmlingua2-10-20260729/f2/`
- 配置：`docs/f2-results/llmlingua2-115-20260729/f2-llmlingua2-115.yaml`
