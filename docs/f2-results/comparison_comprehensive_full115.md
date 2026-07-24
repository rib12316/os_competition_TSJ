# 全面 Prompt 优化 full115

- 日期：2026-07-24
- 任务：tau-bench retail `tau-0..114`
- 方案：retail compact system + tool description 去重 + tool-aware LLMLingua-2
- 配置：hot tool result 1,000 token；compressible cold body 8,000 token；并发 4
- 代码提交：`49f6724`
- 原始目录：`/tmp/f2-comprehensive-full115`
- 持久产物：`prompt_tokens_comprehensive_full115.jsonl`、
  `f2_events_comprehensive_full115.jsonl`、`metrics_comprehensive_full115.json`

## 运行完整性

- 115/115 个 session 完成；共 1,780 次模型调用；prompt meter 与 F2 event 均为 1,780 行。
- 每个 session 5--20 次调用，中位数 17；51/115 个 session 达到 20 次上限。
- vLLM 最终记录 1,780 次成功响应；实际输入合计 8,798,998 token。

## 同轨迹完整 Prompt 收益

| 指标 | 数值 |
|---|---:|
| 模型调用 | 1,780 |
| canonical prompt | 10,933,355 token |
| transformed prompt | 8,820,495 token |
| 累计节省 | 2,112,860 token |
| 整体节省 | **19.32%** |
| 有收益调用 | 1,780/1,780 |
| 每步节省 | 1,187 token |

`transformed` 是本地 Qwen2.5 完整 chat template 的严格配对计量；vLLM usage 为
8,798,998 token，比本地计量少 21,497（0.24%）。两边 canonical/transformed 使用同一个
本地 tokenizer，故 19.32% 是本轮可归因的同轨迹降幅。

## 动态压缩触发

| 事件 | 数值 |
|---|---:|
| static system/tools 优化 | 1,780/1,780 次 |
| hot tool BERT | 0 次 |
| cold BERT | 0 次 |
| 最大 cold body | 6,067 token |
| 最大 compressible cold body | 4,097 token |

全部事件均为 `skip`：345 次历史短于 hot window，1,435 次低于 cold 8k 门槛。没有单个
hot tool result 达到 1k 门槛。因此本轮 2,112,860 token 收益全部来自确定性的 system/tools
精简，LLMLingua-2 动态上下文贡献为 0，同时也没有 BERT 推理延迟。

## 性能与成功率

| 指标 | 历史 strict baseline | 本轮全面优化 | 差异 |
|---|---:|---:|---:|
| success | 26/115（22.61%） | 23/115（20.00%） | -3 tasks（-2.61pp） |
| latency p50 | 99.51s | 98.02s | -1.50% |
| latency p95 | 156.07s | 161.64s | +3.57% |
| QPS | 0.03693 | 0.03804 | +3.00% |
| TTFT | 72.84ms | 100.08ms | +27.24ms |
| KV prefix cache hit | 95.81% | 95.05% | -0.76pp |

这里的 baseline 与本轮是两次独立生成，不是配对轨迹。历史另一次相同规模 baseline 为
23/115，本轮与它相同；两次 baseline 自身相差 3 tasks。故当前 full115 没有显示大幅退化，
但相对最近的 26/115 baseline 名义下降 2.61pp，超过 2pp 红线 0.61pp，不能据此宣称
“已确认 ≤2pp”。需要同环境 baseline 重跑或多次 A/B 才能区分 compact policy 的真实影响
与 mimo/user-sim 的运行波动。

## 结论

最终 8k 策略在 retail full115 上稳定兑现了全 prompt 约 19.3% 的零模型成本收益，且消除了
旧方案的 BERT 延迟。动态 LLMLingua 路径保留给更长上下文，但本 workload 未验证其实际收益。
success 结果处在历史 baseline 波动范围内，不过严格的 2pp 红线仍未被这一次独立运行坐实。
