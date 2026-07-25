# F2 静态 Prompt / 工具描述压缩调研

- 日期：2026-07-23
- 模型与 tokenizer：本地 `Qwen2.5-7B-Instruct`
- 场景：tau-bench retail，完整 policy wiki + 16 个工具 schema
- 统计方式：`tokenizer.apply_chat_template(..., tools=..., add_generation_prompt=True)`

## 1. 结论

可行，而且静态 prompt 是当前更大的优化对象。但不应把 system prompt 和工具 schema
直接交给 LongLLMLingua 在线压缩。推荐采用：

1. system policy 离线、人工校验的确定性精简；
2. 工具 schema 删除重复措辞，但保留工具选择、参数约束和 confirmation 语义；
3. 如继续优化，按 session 使用稳定的工具组，不要每步改变工具集合；
4. 与现有 F2 冷历史压缩叠加，分别做 ablation。

这样没有额外压缩模型延迟，输出前缀稳定，也更利于 vLLM prefix cache。

## 2. Token 构成

以一条短 user 请求为例，首轮完整 prompt 为 4,271 token：

| 部分 | token | 占首轮 prompt |
|---|---:|---:|
| system retail policy | 1,188 | 27.8% |
| 16 个工具 schema | 3,047 | 71.3% |
| user + chat template 其余开销 | 36 | 0.9% |

因此原事件日志用 message 文本 `chars/4` 得到约 1,430 token，会漏掉单独通过
`tools=` 传入的约 3,047 token。响应 `usage.prompt_tokens` 才是正确总量。

## 3. System Policy

原始 `retail/wiki.md` 文本为 1,204 token。将规则改写为紧凑、无重复的结构化 policy，
同时保留认证、单用户、确认、状态、退款、退换货等约束后，样稿为 445 token：

- 节省 759 token（63%）；
- 完整首轮 prompt 从 4,271 降到 3,512；
- 这是离线静态变换，运行时成本为 0。

风险：system policy 直接决定任务成功率。不能只按语言冗余自动删除；每条业务约束必须
建立“原规则 -> 紧凑规则”的可审计映射，并跑 full115 success ablation。

## 4. 工具 Schema

工具描述里存在大量重复：

- `order_id` 的 `#W...` 说明重复 7 次；
- payment method 说明重复 4 次；
- 两个 address 工具的 6 个字段说明重复；
- “解释修改并获得显式确认”重复 7 次，且 system policy 已规定一次。

精确重复描述的多余副本约 435 token。可以把共同约定集中放入 system policy，再把字段
描述缩为必要约束。不同裁剪强度的上限测量：

| 方案 | 完整首轮 token | 节省 | 风险 |
|---|---:|---:|---|
| 原始 16 工具 | 4,271 | - | - |
| 删除所有参数 description | 3,123 | 1,148 | 中高，参数格式/约束可能丢失 |
| 删除所有 function + 参数 description | 2,465 | 1,806 | 高，工具选择和参数正确率会下降 |

“删除所有 description”只能作为压缩收益上限，不能作为上线方案。安全版本应保留：

- function 的用途与调用前置条件；
- enum、required、类型以及易混淆 ID 的格式；
- 同产品换 item、gift card 余额、一次性调用等关键约束。

## 5. 工具分组 / 按需暴露

工具 schema 是最大的单项。只暴露稳定子集时收益明显：

| 工具集合 | 首轮 token | 相对完整 16 工具 |
|---|---:|---:|
| 认证/用户/思考/转人工 5 工具 | 1,883 | -2,388（-56%） |
| 上述 + 查订单/取消订单 7 工具 | 2,228 | -2,043（-48%） |

但当前工具通过 OpenAI 请求的 `tools=` 独立传入，现有 middleware 只能变换 messages，
不能直接裁剪工具。若实现，需要增加 `transform_tools` 或统一的请求变换钩子。

工具集合应在 session 开始时按意图选择并保持稳定，而不是每一步变化。原因是当前
`kv_cache_hit_rate` 约 0.96；每步改变 system/tools 前缀会制造多个 prefix 版本，可能
用 token 节省换来 prefix-cache 命中下降和额外 KV 占用。

## 6. 推荐实施顺序

### P0：真实计量（本次已实现）

- 非流式：读取 `response.usage.prompt_tokens`；
- 流式：请求 `stream_options.include_usage=true`，读取最终 usage chunk；
- `sent_tokens` 只记录真实返回值；服务端不返回时为 `null/unavailable`；
- 原 `chars/4` 数值保留为 `estimated_sent_tokens`，只用于诊断。

### P1：固定静态精简（优先）

- 人工压缩 retail system policy，并建立逐条规则映射；
- 去掉工具描述中与 system 重复的 confirmation 等措辞；
- 合并重复 ID/address 约定，但保留 schema 类型、required、enum；
- 所有请求使用同一份压缩结果，保持 prefix cache 友好。

预期静态前缀可先减少约 900-1,200 token，且运行时开销为 0。

### P2：稳定工具组（收益大、风险也大）

- 增加工具变换钩子；
- 只按 session 选择一次工具组，组内始终包含认证、查询、think、转人工和目标动作工具；
- 无法分类时回退完整 16 工具；
- 记录 `tools_tokens`、工具组 ID 和 fallback 次数。

### P3：与冷历史压缩组合

静态 prompt 精简负责每一步都存在的固定前缀，LongLLMLingua/LLMLingua2 只负责超过
阈值后的冷历史。两者优化对象不同，可以叠加。

## 7. 验证要求

至少做四档 full115：baseline、静态精简、冷历史 F2、静态精简 + F2。每档记录：

- 引擎真实 `prompt_tokens` 总量和每步 p50/p95；
- success rate（相对 baseline 下降不超过 2pp）；
- latency p50/p95；
- prefix-cache hit rate；
- 工具选择错误、参数校验错误、遗漏 confirmation 的次数。

独立运行会产生不同轨迹，token 总量不能直接视为严格配对结果。最好额外保存 baseline
请求轨迹并做离线 tokenizer replay，用同一轨迹比较静态 prompt 的纯 token 收益。
