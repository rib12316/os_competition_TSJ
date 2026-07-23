# F2 Tool-aware LLMLingua 调研与改造方案

- 日期：2026-07-23
- 上游：<https://github.com/microsoft/LLMLingua>
- 目标：让 F2 覆盖 agent 对话与 API/tool 结果，同时保持 OpenAI tool-call 协议和业务硬字段

## 1. 当前实现到底压了什么

当前 cold 是除 system 和最近 `keep_hot` 条消息之外的所有旧消息，因此包含：

- 旧 user/assistant 文本；
- 旧 `role=tool` 的结果 content；
- 旧 assistant tool call 的工具名。

但 `_msg_to_text` 把结构化 `tool_calls` 降成 `[called: name]`，丢掉 arguments、call ID 和
role 边界。冷 tool result 的 content 虽然进入 LLMLingua，但 JSON 被当作普通文本压缩。

每步通过 OpenAI `tools=` 传入的工具 schema 不属于 messages/cold，当前 F2 完全不处理。

## 2. 官方能力与边界

官方不仅提供普通 `compress_prompt`，还提供：

- `structured_compress_prompt`：分段设置 `compress=False` 或独立 rate；
- `compress_json`：每个 JSON key 配置 rate、compress、value_type、pair_remove；
- `force_tokens`：LLMLingua-2 可强制保留关键字符；
- LongLLMLingua question-aware/context-level 过滤；
- LLMLingua-2：task-agnostic BERT encoder，官方称比 LLMLingua 快 3-6 倍。

官方原则同时指出 instruction/question 对压缩敏感，context/API result 相对不敏感；chat
模式比 completion 模式更容易受到 token-level 压缩影响。因此不能对所有字段使用同一 rate。

LLMLingua 不原生理解 OpenAI `tool_calls` 协议。结构安全必须由我们的 serializer 和
reconstructor 保证。

## 3. 推荐架构

### A. System 与 tools schema：结构层，不做在线 LLMLingua

- system policy、工具名、JSON Schema key/type/required/enum 保持确定性；
- 只允许人工/规则去重或 session 级稳定工具组；
- 不让 token compressor 改写可执行 schema。

理由：这些属于高敏感 instruction，且是固定 prefix；在线压缩会增加延迟并可能破坏
prefix cache。

### B. Hot tool result：保留协议，只压 content

对超过阈值的最新工具结果：

```text
assistant.tool_calls：原样保留 name、arguments、id
tool：原样保留 role、tool_call_id、name，只替换 content
```

如果 content 是 JSON，先 parse，再按字段策略调用 `compress_json`：

- 永不压：ID、status、amount/price、quantity、timestamp、enum、error、布尔值；
- 默认保留：address、payment method、用户明确要求的字段；
- 可压：长 description、产品说明、历史备注、大列表中的低相关文本；
- `pair_remove=False` 用于关键字段，防止整个键值对消失。

这样 OpenAI tool pairing 始终合法，并且大结果不必等到变冷后才获益。

### C. Cold 完整 turn：结构化语义记忆

按完整 turn/tool group 序列化，不能再用当前无 role 的纯文本拼接：

```text
[USER] ...
[ASSISTANT_TOOL_CALL]
name=<不压>
arguments=<不压或字段级 JSON 压缩>
[TOOL_RESULT name=...]
critical_fields=<不压>
narrative_fields=<按 0.5-0.7 压缩>
[ASSISTANT] ...
```

通过 `structured_compress_prompt` 给 role marker、工具名、参数、关键 JSON 片段设置
`compress=False`，只对自然语言和低敏感结果设置 rate。question 使用最近 user 请求，
`reorder_context=original`，保持 agent 时序。

冷 group 最终仍可折叠成一条 `[compressed history]` system message；因为它已经离开 hot
协议区，不再需要保留可执行 `tool_call_id`，但语义信息必须完整。

### D. 压缩模型与缓存

- 优先 LLMLingua-2 BERT small，替代当前 GPT-2 LongLLMLingua；
- 每个 tool result/完整 turn 只压一次，按内容 hash 缓存；
- 新 turn 到来时增量追加，不重新压整个 cold；
- 只有问题变化明显或累计新增量超过阈值时，才做 question-aware 二次整理。

这比当前“首次跨 2k 后整段重压”更细粒度，可避免一次 12-26s 的关键路径停顿。

## 4. 不建议的方案

- 把完整 OpenAI messages/tools JSON 序列化后统一 `rate=0.5`：可能破坏 key、ID 和参数；
- 在线压缩 system/tool schema：高敏感、固定前缀、收益会被压缩成本抵消；
- 删除冷 tool call arguments：会丢失用户目标、已执行动作和关键实体；
- 每步重新压全部历史：CPU 成本不可接受；
- 为减少工具 token 每步改变 tools 集合：会破坏 prefix cache，应单独实验。

## 5. 实施顺序

### P0：修正 serializer 与计量

1. 新增 `ToolAwareHistorySerializer`，按完整 tool group 输出结构化片段；
2. arguments 和关键字段必须原样；
3. 日志按 role 记录原始/压缩后的真实 Qwen token；
4. 增加 JSON 可解析、ID/金额/status 不丢、tool pairing 合法测试。

### P1：LLMLingua-2 冷历史

1. 切 `llmlingua2-bert-base-multilingual-cased-meetingbank`；
2. role marker/工具名/arguments 使用 `compress=False`；
3. user/assistant 文本先用 rate=0.75，tool narrative 用 rate=0.6；
4. 保持 `keep_hot=6`，先只替换当前 cold 路径。

### P2：Hot 大工具结果

1. 仅对超过 800-1,000 真 token 的 tool content 启用；
2. JSON 使用字段策略，非 JSON 使用 structured text；
3. 保留 canonical 原文，只变换发引擎副本；
4. 按 hash 缓存，下一步直接复用。

### P3：增量 turn 缓存

以完整 user -> assistant/tool -> tool result group 为缓存单元，替代整段 cold 重压。

## 6. 验证矩阵

| 档位 | 目的 |
|---|---|
| baseline | 严格 `active=[]` |
| current F2 | 现有纯文本 cold serializer |
| tool-aware + GPT-2 | 隔离 serializer 的质量影响 |
| tool-aware + LLMLingua-2 | 验证速度与质量 |
| 上述 + hot result | 验证更早获得 token 收益 |

先做固定轨迹 tokenizer replay，再做 full115：

- success 下降不超过 2pp；
- 关键字段保持率 100%；
- tool-call JSON/配对错误为 0；
- 真实 prompt token 总量、每 role token、压缩耗时 p50/p95；
- E2E p50/p95 与 prefix-cache hit。

## 7. 预期

该方案不会让 LLMLingua 处理所有 4k 固定 prefix；那不是它最安全的用途。它能把能力从
“普通冷文本”扩展到“工具结果 + 完整工具轨迹”，同时保护结构字段。真正的收益取决于
tool result 在任务中的长度：对于短 retail JSON，收益仍有限；对搜索/RAG/大 JSON 工具，
收益会显著高于当前 cold-only 实现。
