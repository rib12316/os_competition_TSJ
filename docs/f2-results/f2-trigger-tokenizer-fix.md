# F2 触发门 tokenizer 计量修正

- 日期：2026-07-24
- tokenizer：本地 `Qwen2.5-7B-Instruct`
- 范围：cold 8k、hot tool 1k、recompress 4k、压缩前后收益判断
- 原始摘要：`f2_trigger_tokenizer_result.json`

## 修正内容

原实现用固定 `sum(chars) // 4` 决定是否压缩。该值既不是模型 token，也会因语言和 JSON
结构产生系统性偏差。现在正式 F2 配置从 `engine.model` 自动取得 tokenizer，并与完整 Prompt
meter 共享进程级缓存；事件用 `token_count_source` 明确记录来源。

未配置模型的轻量 middleware 调用使用 Unicode-aware heuristic，避免引入强制 transformers
依赖；正式 benchmark 不走该回退路径。

## 复核结果

| 场景 | 旧 chars/4 | Qwen 精确触发值 | 8k 决策 |
|---|---:|---:|---|
| 5k tool-aware | 7,620 | 5,136 | skip |
| 9k tool-aware | 12,858 | 8,667 | compress |
| 28.9k generic | 30,681 | 20,815 | compress |

精确值按 middleware 真正送入计数器的正文片段，以 `\n\n` 边界序列化后计算。它与“逐 body
分别求和”会有少量边界 token 差异，例如 generic 的逐 body 值为 20,772。

触发修正不改变已有压缩输出：5k/9k 完整 Prompt 仍分别下降 20.83%/21.88%，generic 仍为
`28,860 -> 20,936`（-27.46%）。

## 性能与验证

精确计数按序列化文本 SHA-256 做 4,096 项有界缓存。28.9k 相同历史的无模型推理探针：

- 首次 transform：120.6ms；
- 相同 session reuse：5.9ms。

回归结果：223 passed；Ruff 通过。cold、hot tool 和 recompress 三类门槛均有独立单测。
