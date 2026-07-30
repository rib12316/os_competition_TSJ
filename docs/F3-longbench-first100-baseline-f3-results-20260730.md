# F3 LongBench 2WikiMQA first100 baseline / F3 对比

> 日期：2026-07-30
> 数据：THUDM LongBench `data/2wikimqa.jsonl`，task 0..99
> 模型：本地 vLLM `Qwen2.5-7B-Instruct`，`max_model_len=32768`

## 1. 配置

- 两组均为串行单次运行，`temperature=0`、`max_steps=8`、单次输出上限 256 token。
- baseline 将 `retrieve_documents` 的完整 JSON 结果放回上下文。
- F3 在工具结果达到 4,000 token 时外置，reference 上限 512 token，fetch 响应上限 768 token。
- 执行顺序为 baseline -> F3；两组分别保存 vLLM 起止 counter，组间没有重启引擎。
- 模型调用使用流式接口。TTFT 为所有成功模型请求的首 chunk 延迟中位数。

## 2. 结果

| 指标 | baseline | F3 | 变化 |
|---|---:|---:|---:|
| success | 31/100 | 27/100 | -4 pp |
| e2e latency p50 | 1.499 s | 1.703 s | +13.62% |
| e2e latency p95 | 3.274 s | 5.701 s | +74.12% |
| 模型 HTTP 耗时 p50 | 1.498 s | 1.687 s | +12.58% |
| 模型 HTTP 耗时 p95 | 3.274 s | 5.684 s | +73.62% |
| TTFT 中位数 | 348.30 ms | 44.93 ms | -87.10% |
| fetch 调用 | 0 | 174 | +174 |
| KV cache hit rate | 26.92% | 70.26% | +43.33 pp |
| 累计 Prompt token | 1,632,665 | 542,387 | -66.78% |
| 每题累计 Prompt p50 / p95 | 13,204 / 36,591 | 3,764 / 11,893 | - |
| 单请求最大 Prompt | 32,382 | 8,675 | -73.21% |
| 成功模型请求 | 250 | 361 | +111 |
| 请求错误 task | 5 | 0 | -5 |

baseline 的 5 个请求错误发生在 task `15, 32, 50, 85, 98`，均计为不成功。F3 没有模型请求错误。

## 3. Fetch 与延迟

F3 共发起 174 次 `fetch_tool_result` 工具调用。F3 事件日志包含：

- 110 次工具结果 externalize；
- 115 次成功 fetch；
- 18 次低于阈值的 passthrough；
- 59 次 fetch 调用没有成功 fetch 事件，属于未命中、参数无效或结果 ID 不可用等路径。

F3 将累计 Prompt 降低 66.78%，并把单请求最大 Prompt 从 32,382 降到 8,675，因此 TTFT 和 KV hit 显著改善，也消除了 context overflow。端到端 p50/p95 仍上升，主要因为成功模型请求从 250 增至 361；额外 fetch 需要新的模型轮次，尾部任务最多会连续发起多次 fetch。

## 4. 指标口径

- `e2e latency`：从每题进入 Agent loop 到返回/报错的墙钟时间，包含 middleware、SQLite fetch 和模型请求。
- `模型 HTTP 耗时`：每题所有模型请求墙钟耗时之和；不含本地 middleware/fetch 间隙。
- `TTFT`：250/361 个成功流式模型请求的 request-level 中位数，不包含 5 个没有首 chunk 的 baseline 失败请求。
- `KV hit`：各组 vLLM `prefix_cache_hits_total / prefix_cache_queries_total` 的 counter delta。两组窗口没有混入其他 Prompt 请求，Prompt 日志总和与 vLLM delta 完全一致。由于 F3 在 baseline 后运行且未重启引擎，绝对 KV hit 仍包含运行顺序的缓存预热影响。
- `上下文`：vLLM `usage.prompt_tokens` 的逐请求累计；不是原始文档 payload 大小。

## 5. 复现与原始数据

- 汇总：`docs/f3-results/longbench-2wikimqa-first100-baseline-f3-20260730/comparison.json`
- 逐题：同目录 `baseline/cases.jsonl` 与 `f3/cases.jsonl`
- vLLM counter：两组目录内 `vllm_metrics_before.prom` / `vllm_metrics_after.prom`
- F3 事件：`f3/f3_events.jsonl`
- 评测脚本：`run_eval.py`
- 数据 SHA-256：`cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`

## 6. F3 repeat-01..09 与 baseline 重测

按后续测试要求，F3 运行到 `repeat-09` 后停止，不纳入误启动且未完成的 `repeat-10`。随后使用完全相同的 first100、模型和流式指标口径单独重测一次 baseline。重测 baseline 仍为 31/100，没有降到可接受的 30/100；F3 的最佳观测值为 28/100，因此截止 `repeat-09` 仍相差 3 题，没有达到“与 baseline 相等或只差 1 题”的 success 目标。

最佳 F3 有两次并列 28/100：`repeat-01` 和 `repeat-06`。下表采用其中 e2e p50/p95 更低的 `repeat-01`。这是按观测结果挑选的 run，存在 outcome-dependent selection bias，不能当作 F3 单次运行的无偏估计；全部 repeat success 同时保留在下方。

| 指标 | baseline 重测 | F3 最佳观测（repeat-01） | 变化 |
|---|---:|---:|---:|
| success | 31/100 | 28/100 | -3 pp |
| e2e latency p50 | 1.499 s | 1.690 s | +12.74% |
| e2e latency p95 | 3.286 s | 6.803 s | +107.03% |
| TTFT 中位数 | 347.96 ms | 45.28 ms | -86.99% |
| fetch 调用 | 0 | 180 | +180 |
| KV cache hit rate | 27.06% | 81.07% | +54.01 pp |
| 累计 Prompt token（上下文） | 1,641,836 | 546,746 | -66.70% |
| 每题累计 Prompt p50 / p95 | 13,839 / 36,591 | 3,734 / 13,771 | - |
| 单请求最大 Prompt | 32,382 | 8,675 | -73.21% |
| 请求错误 task | 5 | 0 | -5 |

F3 九次完整 repeat 的 success 为：

| run | success | run | success | run | success |
|---|---:|---|---:|---|---:|
| repeat-01 | 28/100 | repeat-04 | 27/100 | repeat-07 | 27/100 |
| repeat-02 | 26/100 | repeat-05 | 27/100 | repeat-08 | 27/100 |
| repeat-03 | 27/100 | repeat-06 | 28/100 | repeat-09 | 26/100 |

baseline 重测的 100 个 task ID 严格为 `0..99`，逐请求 Prompt 日志总和 `1,641,836` 与 vLLM `prompt_tokens_total` counter 增量完全相等，因此该指标窗口没有混入其他 Prompt 请求。五个请求错误仍发生在 task `15, 32, 50, 85, 98`，均为 32,768 token 上下文上限错误。

补充原始数据：

- baseline 重测汇总：`docs/f3-results/longbench-2wikimqa-first100-baseline-f3-20260730/baseline-retest/result.json`
- baseline 重测逐题：同目录 `baseline/cases.jsonl`
- F3 九次完整汇总：同结果根目录 `f3-repeats/all_runs.json` 与 `all_runs.md`
- F3 最佳观测逐题：同结果根目录 `f3-repeats/run-01/cases.jsonl`
- repeat 停止原因：`maximum_repeat_09_reached`

## 7. baseline 固定五次重试

在第 6 节的 baseline 重测之后，额外串行执行了五次固定 baseline repeat。五次均完整运行 task `0..99`，没有按 success 提前停止，也没有从中选择某次作为 baseline。每轮 Prompt 日志都与对应 vLLM counter 增量完全一致。

| run | success | e2e p50 | e2e p95 | TTFT 中位数 | fetch | KV hit | 累计 Prompt 上下文 | 单请求最大 Prompt | 错误 task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline repeat-01 | 31/100 | 1.498 s | 3.260 s | 346.88 ms | 0 | 27.07% | 1,641,836 | 32,382 | 5 |
| baseline repeat-02 | 31/100 | 1.496 s | 3.259 s | 348.21 ms | 0 | 27.07% | 1,641,836 | 32,382 | 5 |
| baseline repeat-03 | 31/100 | 1.498 s | 3.274 s | 347.21 ms | 0 | 27.07% | 1,641,836 | 32,382 | 5 |
| baseline repeat-04 | 31/100 | 1.496 s | 3.266 s | 347.13 ms | 0 | 27.07% | 1,641,836 | 32,382 | 5 |
| baseline repeat-05 | 31/100 | 1.498 s | 3.286 s | 347.19 ms | 0 | 27.07% | 1,641,836 | 32,382 | 5 |
| 五次均值 | 31.0/100 | 1.497 s | 3.269 s | 347.32 ms | 0.0 | 27.07% | 1,641,836 | 32,382 | 5.0 |

五次 e2e p50 范围为 `1.496..1.498 s`，e2e p95 范围为 `3.259..3.286 s`，TTFT 中位数范围为 `346.88..348.21 ms`。五次 success、Prompt、KV hit、错误 task 以及逐题答案轨迹完全一致，答案轨迹唯一数为 1。每轮的五个错误仍是 task `15, 32, 50, 85, 98` 的 32,768 token 上下文上限错误。

baseline 五次重试原始数据与汇总位于：`docs/f3-results/longbench-2wikimqa-first100-baseline-f3-20260730/baseline-repeats/`。
