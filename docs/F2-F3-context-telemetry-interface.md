# F2/F3 上下文 telemetry 接口

本接口属于方法层，不依赖 Gradio。前端可以实时轮询内存 buffer，也可以读取相同 schema 的
JSONL。schema 当前版本为 `1`。

## 1. 接入

```python
from agent_mem.context_telemetry import ContextEventBuffer
from agent_mem.middleware import middlewares_from_config
from agent_mem.bench.runners.qwen_agent import QwenAgentRunner

events = ContextEventBuffer()
stack = middlewares_from_config(cfg)
runner = QwenAgentRunner(
    engine_url=engine_url,
    model=cfg.engine.model,
    middlewares=stack.middlewares,
    context_event_sink=events,
)
```

前端增量读取：

```python
new_events = events.events(after_sequence=last_sequence)
last_sequence = new_events[-1]["sequence"] if new_events else last_sequence
```

前端直接取合并状态：

```python
snapshot = events.snapshot(session_id)
f2 = snapshot["f2"]
f3 = snapshot["f3"]
prompt = snapshot["prompt"]
```

需要同时落盘时：

```python
from agent_mem.context_telemetry import (
    CompositeContextEventSink,
    JsonlContextEventSink,
)

sink = CompositeContextEventSink(
    events,
    JsonlContextEventSink(run_dir / "context-events.jsonl"),
)
```

`TauBenchAgent`、`run_react()`、`QwenAgentRunner`、tau-bench adapter 和 LongBench adapter
均支持可选的 `context_event_sink` 参数。

## 2. 前端需要的六项数据

| 数据 | snapshot 字段 |
|---|---|
| 待压缩冷历史 | `f2.cold_before` |
| 压缩后的冷历史 | `f2.cold_after` |
| 待外置的原工具数据 | `f3.latest.original` |
| 外置后的结构摘要和短引用 | `f3.latest.externalized` |
| 完整 Prompt 变换前 token | `prompt.original_prompt_tokens` |
| 完整 Prompt 变换后 token | `prompt.transformed_prompt_tokens` |

文本字段使用统一的 `{text, chars, truncated}` 结构。默认单段 preview 最多 4,000 字符，
可通过 F2/F3 构造参数 `telemetry_preview_chars` 调整；F2 默认最多预览 30 条消息，
可通过 `telemetry_max_messages` 调整。token 数始终对应完整内容，不对应 preview。

## 3. F2 事件顺序

首次压缩：

```text
f2.history_ready
f2.compress_started
f2.compress_finished
prompt.measured
prompt.completed
f2.request_completed
```

未达到门槛：

```text
f2.history_ready
f2.skipped
prompt.measured
prompt.completed
f2.request_completed
```

复用缓存：

```text
f2.history_ready
f2.reused
prompt.measured
prompt.completed
f2.request_completed
```

`f2.compress_started` 在同步调用 LLMLingua worker 前发出，`f2.compress_finished` 在 worker
返回并构造本轮发送副本后发出。前端可据此显示等待状态。

`cold_before` 是 canonical 冷历史的有界预览；`cold_after` 是本轮发给模型的冷历史副本。
F2 不会用压缩文本覆盖 canonical history。

## 4. F3 事件顺序

大结果外置：

```text
f3.tool_result_observed
f3.externalize_started
f3.tool_result_externalized
```

小结果原样通过：

```text
f3.tool_result_observed
f3.tool_result_passthrough
```

局部取回：

```text
f3.fetch_started
f3.fetch_finished
```

失败时为 `f3.externalize_failed` 或 `f3.fetch_failed`，并提供 `reason/status/error`。

`original` 包含工具名、参数、原结果 preview、完整 token/byte 数；`externalized` 包含
`result_id`、content type、SHA-256、deterministic synopsis、reference 和 reference token。
ArtifactStore 中保存的仍是完整原文，`externalized.reference` 才是进入 Agent history 的短数据。

## 5. Prompt 口径

只要配置了 `context_event_sink`，即使没有设置 `PROMPT_TOKEN_LOG`，方法层也会尝试对同一轮
canonical/transformed 请求应用完整模型 chat template，产出：

```text
original_prompt_tokens
transformed_prompt_tokens
saved_tokens
saved_percent
meter_ms
tokenizer_source 或 meter_error
```

`prompt.completed.prompt_tokens` 是服务端 `usage.prompt_tokens`，可与
`transformed_prompt_tokens` 检查 tokenizer drift。

## 6. 并发和错误边界

- `ContextEventBuffer`、`JsonlContextEventSink` 均为线程安全，可接并发 benchmark。
- 每条事件用 `(session_id, step, tool_call_id)` 关联 Agent 轨迹。
- `events(after_sequence=...)` 用于无重复的 Timer 增量刷新。
- telemetry sink 异常不会中断 Agent 推理。
- 默认只导出有界 preview；前端不应直接读取 F3 SQLite artifact 内容。
