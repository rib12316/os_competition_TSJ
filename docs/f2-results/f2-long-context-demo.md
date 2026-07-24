# F2 长上下文 LLMLingua-2 示例

日期：2026-07-24
脚本：`agent-mem/benchmarks/f2_long_context_demo.py`
结果：`f2_long_context_demo_result.json`

## 历史 5k→2.5k 记录

旧日志中确实有两条接近这个结果的记录，但它们属于旧的非 tool-aware
LongLLMLingua/GPT-2 路径：

| 记录 | 原始 | 压缩后 | 比例 | 耗时 |
|---|---:|---:|---:|---:|
| `f2_events_mimo_trigger2000.jsonl`, tau-1 step-19 | 5,036 | 2,049 | 2.5x | 28.1s |
| `f2_events_trigger4000.jsonl`, tau-2 step-11 | 5,113 | 2,155 | 2.4x | 25.8s |

这里的 `origin_tokens`/`compressed_tokens` 是压缩器段落计量，不是完整 API Prompt，且
没有保护 tool protocol 和关键 JSON 字段。

## 可复现实验

执行：

```bash
cd /tmp/f2-wt/agent-mem
PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/f2_long_context_demo.py \
  --output /tmp/f2-long-context-result.json
```

脚本生成 tool-heavy retail 历史，包含 user 约束、assistant tool call、JSON tool result、
order/status/amount/address 等硬字段；使用当前配置的 LLMLingua-2 BERT、
`assistant_rate=0.75`、`tool_result_rate=0.60` 和 `keep_hot=6`。

## 结果

| case | 压缩触发 | 压缩段（Qwen tokenizer） | 完整 Prompt | 首次压缩 |
|---|---|---:|---:|---:|
| 5k forced | 强制触发（门槛临时设 1） | 7,750 → 5,782（-25.39%） | 8,491 → 6,722（-20.83%） | 10.94s |
| 9k production | 当前 8k 门槛触发 | 13,057 → 9,736（-25.43%） | 13,666 → 10,676（-21.88%） | 17.48s |

另有一个只包含重复叙述正文的 direct LLMLingua-2 对照：

`5,000 → 2,720`，节省 **45.60%**，耗时 2.76s。它接近早期的 5k→2.5k 现象，说明
压缩器本身仍能达到这个力度；差异来自当前 tool-aware 外层保护，而不是 BERT 失效。

5k 生产门槛检查结果为 `skip / below_trigger`：当前触发器用 chars/4 估算的
`compressible_cold_tokens=7,620`，低于 8,000。9k case 估算值为 12,858，因此真实触发
一次 cold 压缩。

## 安全性与复用

两个 middleware case 均通过：

- 所有 user 原文、tool call ID、function arguments、order ID 均保留；
- `status=pending`、金额、价格、shipping address 均保留；
- hot tool tail 没有孤立 `tool_call_id`；
- 同一 session 第二次变换 action=`reuse`，输出完全一致，耗时约 1--3ms。

## 结论

早期“5k→2.5k”是真实的，但它测的是可自由删减的普通正文。当前策略对真实 agent 轨迹
先保护业务语义和协议结构，再压剩余正文，所以整体压缩约 25%，完整 Prompt 约 21%。这
是有意的安全-收益折中。当前 retail 的 8k 门槛仍会跳过约 5k 级历史；面对更长历史（本
例 9k）会触发 LLMLingua-2，且复用缓存几乎没有额外成本。
