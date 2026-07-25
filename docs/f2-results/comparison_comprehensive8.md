# 全面 Prompt + 上下文优化配对小测（8 tasks）

- 日期：2026-07-23
- 方案：retail system compact + tool description 去重 + tool-aware LLMLingua-2
- 测试配置门槛：compressible cold 4,000 token；最终配置根据成本收益调到 8,000
- 原始目录：`/tmp/f2-comprehensive8-v3`
- 持久产物：`prompt_tokens_comprehensive8.jsonl`、`f2_events_comprehensive8.jsonl`、
  `metrics_comprehensive8.json`

## 同轨迹完整 Prompt 收益

| 指标 | 数值 |
|---|---:|
| 模型调用 | 139 |
| canonical prompt | 972,762 token |
| transformed prompt | 802,653 token |
| 累计节省 | 170,109 token |
| 整体节省 | 17.49% |
| 有收益调用 | 139/139 |
| 每步最小节省 | 1,219 token |
| 每步最大节省 | 1,887 token |
| 每步平均节省 | 1,224 token |

该次运行的固定前缀每步至少节省 1,219 token：system compact 约 759，工具描述去重约 460。
工具优化保持 16 个工具、name/type/required/enum/参数结构不变；15 个重复 description
替换为 4 条共享 conventions，并移除 7 条已由 system policy 覆盖的 confirmation 重复句。

提交前语义审计补回 profile 构成、50 类产品/variant、修改支付后仍为 pending 三项规则；
最终代码离线实测首轮 `4,272 -> 3,085`，固定节省调整为 1,187 token（27.79%）。

## 性能与成功率

| 指标 | strict baseline 8 | 全面优化 8 |
|---|---:|---:|
| success | 1/8 | 1/8 |
| latency p50 | 78.45s | 80.36s |
| latency p95 | 120.24s | 139.85s |
| TTFT | 77.12ms | 80.37ms |
| mem peak | 58,046MB | 58,046MB |

两次独立轨迹不能严格比较 latency，但 p50 差约 +2.4%，已远优于旧 GPT-2 F2 的
145.92s。success 小样本相同；完整红线仍需 full115。

## Cold 成本决策

4k 配置只在 `tau-1` 最后一步触发一次 BERT：

- compressible cold 4,191；
- 耗时 15.89s；
- 在固定 1,219 token 之外只多省 668 token；
- 没有后续步骤复用。

成本收益不合理，因此最终配置将 cold 门槛提高到 8,000。叠加上述语义审计后，按同轨迹
最终固定前缀预计节省约 164,993 token（16.96%），同时避免 15.89s 关键路径开销。超过
1,000 token 的单个 hot tool result 仍可提前走 BERT，长 RAG/累计 body 超 8k 才走 cold。

## 与 cold-only 对照

此前 tool-aware cold-only 配对收益只有 4,264 / 726,409（0.59%），且仅 20/111 次调用
生效。全面方案达到约 17.5%，所有调用生效。提高 cold 压缩率不是主要方向；固定
system/tools 去重贡献了绝大多数收益。
