# Tool-aware LLMLingua-2 小测（8 tasks）

- 日期：2026-07-23
- 实现：tool-aware serializer + LLMLingua-2 BERT small + body hash cache
- F2 原始目录：`/tmp/f2-toolaware8-v2`
- 持久产物：`f2_events_toolaware8.jsonl`、`prompt_tokens_toolaware8.jsonl`、
  `metrics_toolaware8.json`
- baseline：此前严格 `middleware.active=[]` 的 `/tmp/f2-true8-usage`
- 两档：`tau-0..7`、concurrency=4、max_steps=20、mimo user simulator

## 汇总

| 指标 | strict baseline | tool-aware BERT | 变化 |
|---|---:|---:|---:|
| success | 1/8 | 2/8 | +1 task（样本过小） |
| 模型调用次数 | 134 | 134 | 0 |
| 真实 prompt token | 884,845 | 918,230 | +3.77%（轨迹内容不同） |
| 平均 token/调用 | 6,603 | 6,852 | +3.77% |
| latency p50 | 78.45s | 98.64s | +25.74% |
| latency p95 | 120.24s | 129.81s | +7.96% |
| TTFT | 77.12ms | 82.57ms | +7.06% |
| KV hit | 0.94984 | 0.94768 | -0.00216 |

两档虽然调用总数相同，但各任务步数与消息内容不同，真实 token 总量不能作为同轨迹因果
对照。tool-aware 本轮 success 2/8，至少没有出现明显的协议/硬字段破坏；精度仍须 full115。

## 压缩器行为

- 8 次 cold compress，24 次 reuse，102 次 skip；
- BERT 压缩耗时均值 7.01s，范围 2.85-13.62s；
- 首次 cold 合计 `17,828 -> 16,545`，约 -7.2%；
- 0 次 hot tool result：retail 本轮没有单个结果超过估算 1,000 token 阈值；
- 所有 134 次响应均返回真实 `usage.prompt_tokens`，无 null。

安全保护导致压缩率低于旧纯文本 GPT-2，这是预期取舍：user 原文、tool name/call ID/
arguments、ID/status/金额等 JSON 字段都不交给 BERT。该实现对大 JSON、搜索和 RAG 工具
结果的潜在收益高于 retail，本轮没有覆盖到 hot-result 主场。

## 与旧 GPT-2 8 条小测

| 指标 | 旧 GPT-2 F2 | tool-aware BERT |
|---|---:|---:|
| success | 0/8 | 2/8 |
| 压缩均值 | 19.15s | 7.01s |
| latency p50 | 145.92s | 98.64s |
| latency p95 | 176.97s | 129.81s |

不同运行轨迹仍有噪声，但压缩器直接耗时下降 63.4%，证明模型切换方向有效。

## 下一步

1. 跑 full115，验证 success 下降不超过 2pp；
2. 保存 canonical messages 做固定轨迹 Qwen tokenizer replay；
3. 增加大 JSON/搜索结果 workload，验证 hot tool-result 路径；
4. 若 retail 需要更高压缩率，单独 ablation 冷 user `rate=0.85`，不能直接牺牲保护。
