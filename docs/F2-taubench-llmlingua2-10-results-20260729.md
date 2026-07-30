# F2 LLMLingua2 tau-bench 10-task 串行测试

> 日期：2026-07-29
> 范围：tau-bench `retail/test`，`task_id=0..9`
> 对照：复用同日 LongLLMLingua 实验的固定 baseline；LLMLingua2 组单并发运行 1 次

## 1. 实验配置

- Agent 模型：本地 vLLM `Qwen2.5-7B-Instruct`，地址 `http://127.0.0.1:8000/v1`
- USER 模型：`mimo-v2.5-pro`
- `seed=42`，`max_steps=20`
- 用户可见参数与 LongLLMLingua 组相同：`trigger_tokens=3000`、`rate=0.60`、`recompress_delta_tokens=2000`
- F2 方法：`llmlingua2`，multilingual BERT CPU 子进程压缩器，worker 数量 1
- Lingua2 实际产品配置：`tool_aware=true`、`assistant_rate=0.60`、`tool_result_rate=0.60`、`hot_tool_trigger_tokens=1000`
- 静态优化：`retail_compact` 系统提示词精简与工具描述去重均开启
- 10 个任务全部首轮完成，无网络或运行时错误

“同等参数”指前端暴露的阈值、保留率与复压增量相同。Lingua2 会额外启用工具结构保护，因此 3000 token 阈值统计的是可压自然语言正文，而 LongLLMLingua 统计整段冷历史；两者的实际动态压缩工作量并不相同。

## 2. 逐任务结果

| task | baseline 成功 | Lingua2 成功 | baseline 步数 | Lingua2 步数 | baseline 延迟 | Lingua2 延迟 |
|---:|:---:|:---:|---:|---:|---:|---:|
| 0 | 是 | 否 | 10 | 12 | 85.0 s | 73.6 s |
| 1 | 否 | 是 | 18 | 10 | 151.3 s | 93.3 s |
| 2 | 否 | 否 | 20 | 20 | 128.7 s | 72.5 s |
| 3 | 否 | 否 | 20 | 20 | 143.4 s | 105.1 s |
| 4 | 否 | 否 | 20 | 17 | 175.3 s | 131.2 s |
| 5 | 否 | 否 | 20 | 17 | 128.0 s | 128.7 s |
| 6 | 否 | 是 | 15 | 19 | 112.6 s | 89.4 s |
| 7 | 否 | 是 | 12 | 17 | 112.0 s | 81.6 s |
| 8 | 否 | 否 | 20 | 20 | 133.9 s | 100.5 s |
| 9 | 否 | 否 | 18 | 18 | 146.8 s | 132.0 s |

baseline 成功任务为 task 0；Lingua2 成功任务为 task 1、6、7。两次 MIMO Agent 对话是独立生成轨迹，成功任务和步数差异不能直接归因于压缩方法。

## 3. 三方汇总

| 指标 | baseline | LongLLMLingua | LLMLingua2 |
|---|---:|---:|---:|
| 任务成功率 | 1/10（10%） | 1/10（10%） | 3/10（30%） |
| e2e p50 | 131.3 s | 148.3 s | 96.9 s |
| e2e p95 | 164.5 s | 266.4 s | 131.6 s |
| TTFT | 68.6 ms | 63.3 ms | 66.0 ms |
| QPS | 0.00691 | 0.00587 | 0.00890 |
| KV cache hit rate | 95.29% | 93.51% | 93.07% |
| HBM 峰值 | 56,810 MB | 56,814 MB | 56,813 MB |
| 模型调用数 | 173 | 171 | 170 |
| 同轨迹配对 Prompt 节省 | 0 | 25.02% | 17.25% |
| compress / reuse / skip | 0 / 0 / 0 | 10 / 36 / 125 | 1 / 3 / 166 |
| 压缩器累计耗时 | 0 | 201.95 s | 5.42 s |

相对固定 baseline，Lingua2 本轮观测到 p50 `-26.2%`、p95 `-20.0%`、QPS `+28.8%`、TTFT `-3.8%`。相对 LongLLMLingua，Lingua2 p50 `-34.7%`、p95 `-50.6%`，但两组动态压缩次数为 1 次和 10 次，不能将全部差异解释为压缩器速度。

HBM 峰值继续由 vLLM 预分配 KV pool 主导，不能反映实际上下文或 KV token 的减少。

## 4. Lingua2 压缩明细

Lingua2 自身轨迹包含 170 次模型调用：

| 同轨迹配对口径 | token |
|---|---:|
| 变换前 Prompt | 1,179,503 |
| 变换后 Prompt | 976,015 |
| 节省 | 203,488（17.25%） |

- 每次调用均有 1,187 token 的固定前缀节省，共 201,790 token，占总节省约 99.17%。
- 超出固定前缀的额外节省为 1,698 token，来自动态冷历史、热工具正文处理及 token 计量取整。
- 15 次请求处理了超过热区门限的工具正文，事件侧累计节省 171 token，无 BERT 压缩耗时。
- 动态冷历史仅在 task 5 的 step 14 触发 1 次，随后复用 3 次。
- 该次可压正文为 3,031 token；完整冷区为 5,390 token，压缩为 5,013 token，压缩比约 `1.1x`，耗时 5.42 秒。

除 task 5 外，各 session 的最大可压正文均低于 3000 token，即使其完整冷区已经达到 3303 到 6483 token，也会因工具参数、调用 ID、关键 JSON 字段等结构内容被隔离保护而保持 `skip`。

## 5. 结论

1. 在这组单次 10-task 观测中，Lingua2 的通过率、p50、p95 和 QPS 均优于固定 baseline 与 LongLLMLingua，是当前更适合前端默认档的结果。
2. Lingua2 的压缩器开销明显较低：本轮唯一一次 BERT 动态压缩耗时 5.42 秒；LongLLMLingua 10 次累计 201.95 秒、平均 20.2 秒。不过 Lingua2 只有 1 次样本，尚不足以给出稳定倍速结论。
3. Lingua2 的配对 Prompt 节省为 17.25%，低于 LongLLMLingua 的 25.02%；主要原因是结构保护后动态压缩仅覆盖 1/10 个 session，而非压缩功能失效。
4. 本轮主要验证了静态提示词优化、工具结构保护和低触发率下的 Lingua2 运行代价，尚未充分验证长冷历史场景中的动态压缩收益。
5. 若下一轮目标是公平比较两种动态压缩器，应按“可压正文 token”校准 Lingua2 阈值，或使用更长的固定对话轨迹；直接沿用 3000 总量阈值会导致两种方法压缩工作量严重不等。

## 6. 原始产物

- 项目内归档：`docs/f2-results/llmlingua2-10-20260729/`
- 临时完整目录：`/tmp/f2-llmlingua2-eval-20260729/`
- baseline run：`20260729-110956_vllm_qwen25-7b_baseline-10_run1`
- Lingua2 run：`20260729-124248_vllm_qwen25-7b_f2-llmlingua2-10_run1`
- 已归档文件：两组 YAML 配置、`comparison.md`、`comparison.json`、两组 `summary.json`、逐任务结果、Prompt token 日志与 F2 事件日志
