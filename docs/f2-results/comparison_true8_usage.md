# F2 真实 prompt_tokens 小样本对照（8 + 8）

- 日期：2026-07-23
- 任务：tau-bench retail `tau-0..tau-7`
- 两档均为 1 run、`max_steps=20`、concurrency=4、mimo user simulator
- baseline：严格 `middleware.active=[]`
- F2：LongLLMLingua/gpt2，rate=0.65，trigger=2000，keep_hot=6，worker pool=4
- token：vLLM 响应 `usage.prompt_tokens`；两档均无 `null`
- 原始运行目录：`/tmp/f2-true8-usage`

## 总结果

| 指标 | baseline | F2 | 变化 |
|---|---:|---:|---:|
| success | 1/8 (0.125) | 0/8 | -12.5pp（样本过小） |
| 模型调用次数 | 134 | 145 | +8.21% |
| 真实 prompt token 总量 | 884,845 | 1,013,338 | +14.52% |
| 每次调用平均 token | 6,603 | 6,989 | +5.83% |
| latency p50 | 78.45s | 145.92s | +86.00% |
| latency p95 | 120.24s | 176.97s | +47.18% |
| TTFT | 77.12ms | 75.82ms | -1.69% |
| KV cache hit | 0.94984 | 0.94937 | 基本不变 |
| mem peak | 58,046MB | 58,046MB | 不变（预分配池） |

直接总量变差不能解释为“压缩增加 token”。两档虽然任务 ID 相同，但 mimo/模型生成的
轨迹不同：F2 多运行 11 步，并且多个任务达到 20 步上限。独立生成 benchmark 不能提供
严格的同轨迹 token 因果对照。

## F2 行为

| 动作 | 次数 | 对应真实 prompt token |
|---|---:|---:|
| compress | 7 | 51,297 |
| reuse | 39 | 347,632 |
| skip | 99 | 614,409 |

- 7/8 个任务触发压缩；共 46/145（31.7%）次调用使用压缩段或复用。
- 7 次压缩总耗时 134.04s，平均 19.15s，范围 11.80-26.10s。
- 按 F2 自己的 message chars/4 同口径反推，压缩/复用累计减少约 52,134 个 message
  估算 token。该值不包含工具 schema，不能当作引擎真实节省量。

在 baseline 仍有相同步号的 6 个首次压缩点中，5 个 F2 prompt 更短；这 6 步合计：

| 对照 | token |
|---|---:|
| baseline 相同步号 | 50,035 |
| F2 首次压缩步 | 43,091 |
| 变化 | -6,944（-13.88%） |

其中 `tau-0` 已在此前产生不同轨迹，压缩步反而比 baseline 相同步号长。因此该表只能证明
压缩命中步骤通常变短，不能替代固定轨迹 replay。

## 为什么整体收益小

按 Qwen2.5 tokenizer 的静态前缀测量：每次请求约有 3,047 token 工具 schema 和 1,188
token system policy，当前 F2 都不处理。仅这两部分在 F2 的 145 次调用中约为 614,075
token，占全部真实输入约 60.6%。

冷历史压缩只覆盖 31.7% 的调用、只改变其中一部分 message；与此同时，每个触发任务在
关键路径上支付约 19s CPU 压缩时间。因此本轮 TTFT 没明显变化，而 E2E 延迟显著增加。

## 决策

1. 不根据 8 条 success 的 1/8 -> 0/8 判断精度退化；全量 115 的 -1pp 更可信。
2. 当前 gpt2 LongLLMLingua 不适合作为低延迟方案；需换 LLMLingua2/BERT 或离线压缩。
3. 下一步优先做固定 system policy 精简 + 工具描述去重，目标每步减少 900-1,200 真
   token，运行时开销为 0，并保持所有请求前缀一致以保护 prefix cache。
4. 再做 token 因果评估时，应保存同一份 canonical messages 并用 Qwen tokenizer 同时计算
   原始和变换后的完整 chat template（含 tools），不能用两次独立生成轨迹的总量相减。
