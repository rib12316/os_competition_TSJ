# 同轨迹完整 Prompt 配对计量（8 tasks）

- 日期：2026-07-23
- 方法：同一次模型调用内，用本地 Qwen2.5 tokenizer + 完整 chat template + 同一份
  `tools=` 分别计算 canonical 和 transformed messages
- vLLM `usage.prompt_tokens` 用于校验本地 transformed 计数
- 原始目录：`/tmp/f2-paired8`
- 持久产物：`prompt_tokens_paired8.jsonl`、`f2_events_paired8.jsonl`、
  `metrics_paired8.json`

## 严格配对收益

| 指标 | 数值 |
|---|---:|
| 模型调用 | 111 |
| canonical 完整 prompt | 726,409 token |
| transformed 完整 prompt | 722,145 token |
| 同轨迹累计节省 | 4,264 token |
| 整体节省比例 | 0.59% |
| 有节省的调用 | 20 |
| 无变化调用 | 91 |
| 负收益调用 | 0 |
| 单步最大节省 | 260 token |

本地 transformed 计数为 722,145，vLLM usage 合计 721,422；单步 drift 在 -16 到 0，
总漂移约 0.10%。这是 chat template 实现细节的小差异，不影响同一 tokenizer 下的
`original - transformed` 配对差值。

计量本身平均 16.77ms/步，p95 26.23ms；tokenizer 首次加载已移动到 benchmark 每任务
计时开始之前。

## F2 命中

- 5 次 cold compress，15 次 reuse，91 次 skip；
- 5 次 cold 压缩估算 `11,243 -> 10,382`；
- 0 次 hot tool result，retail 本轮没有结果超过阈值；
- success 2/8，E2E p50 108.93s，压缩耗时均值 6.75s。

## 结论

当前 tool-aware BERT 仍主要优化 cold history。由于只在 20/111 次请求生效，完整 prompt
收益仅 0.59%。即使把 cold 压缩率提高一倍，整体也很难超过约 1.2%，并会增加信息损失。

对照目标“system prompt、工具描述去重与精简”，下一步必须优化每一步都存在的固定前缀，
而不是继续单独提高 cold 压缩强度。优先级：

1. system 与 tool description 的跨区域重复规则去重；
2. 重复参数描述提取为一次性 conventions + 短引用；
3. 保持 tool name/type/required/enum 不变；
4. 使用配对计量直接验证每步固定收益，再跑 full115 success。
