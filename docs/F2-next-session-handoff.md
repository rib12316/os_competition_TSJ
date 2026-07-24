# F2 下一会话交接（截至 2026-07-24）

> 新会话请先读本文件。详细历史数据再查 `F2-work-log.md`、`F2-ablation-results.md` 和
> `docs/f2-results/`，不要从旧的“62% 上下文下降”直接推导完整 Prompt 收益。

## 1. 一句话现状

F2 已形成可冻结的安全基线：

```text
固定前缀：retail 人工 compact 或任意 policy 的 MIMO 离线 artifact
工具 schema：完全重复 description 提取为共享 conventions
动态历史：tool-aware LLMLingua-2，hot 1k / cold 8k / 增量 4k
安全保护：user、arguments、call ID、业务硬字段和 hot tail
```

retail full115 完整 Prompt 严格配对下降 19.32%；非 retail 28.9k synthetic trace 下降
27.46%。通用代码 checkpoint 是 `1fb6e51`，计量澄清是 `9ad0100`；本文件对应的最新提交
以 `git log -1` 为准。本轮没有在通用化改动后重跑 full115。

## 2. 分支与 worktree

- 分支：`feat/f2-prompt-compress`，未 push。
- worktree：`/tmp/f2-wt`，机器重启可能丢。
- 重建：

```bash
git -C /data/os_competition_TSJ worktree add /tmp/f2-wt feat/f2-prompt-compress
```

- 主代码目录：`/tmp/f2-wt/agent-mem`。
- 主环境：openEuler 24.03、Qwen2.5-7B-Instruct、vLLM-Ascend、Ascend 910B2C。

## 3. 当前实现

### 3.1 固定 System/Tools

`transform_request(messages, tools)` 同时处理 system 和工具 schema：

- `system_prompt_mode=retail_compact`：继续使用人工审计的 `COMPACT_RETAIL_POLICY`，行为与
  通用化前一致。
- `system_prompt_mode=compiled`：从 policy artifact 加载 compact policy；source SHA-256、
  source units、数字/ID/状态、`must/only/never/cannot/do not` 等逐条校验。
- artifact 缺失、哈希变化或规则校验失败时默认回退原始 system；`policy_artifact_strict=true`
  才抛错。
- 工具 name/type/required/enum/层次不改；完全重复 description 替换为 `See Cn.`，原说明只在
  system conventions 中出现一次。

MIMO 只在离线编译时使用，运行时不调用强模型。示例配置：

```yaml
optimize_static_prompt: true
system_prompt_mode: compiled
policy_artifact_path: agent-mem/configs/policies/knowledge_incident.json
policy_artifact_strict: false
```

### 3.2 动态 Tool-aware LLMLingua-2

当前配置：

```yaml
method: llmlingua2
trigger_tokens: 8000
recompress_delta_tokens: 4000
keep_hot: 6
assistant_rate: 0.75
tool_result_rate: 0.60
hot_tool_trigger_tokens: 1000
worker_pool_size: 4
```

保护规则：

- user 消息完整保留；
- assistant tool call 的 call ID/name/arguments 完整保留；
- tool result 的 ID/status/state/金额/数量/时间/address/payment/error/name/refund 等保留；
- 通用化新增 severity/priority/owner/assignee/organization/tenant/account/permission/role，
  以及 `_at/_time/_date/_timestamp` 后缀；
- 最近 hot tail 保留；若单个 hot tool result 估算超过 1k，只压 narrative body；
- cold 边界 snap 到完整 assistant-tool 组，压缩后不会留下孤立 `tool_call_id`。

缓存有两层：session 级压缩段复用，以及 `rate + SHA-256(body)` 正文缓存。首次超过 8k 才
压缩；新增可压 body 小于 4k 时直接复用，新增部分原样附加。

### 3.3 Worker

主 venv 是 transformers 5.x，llmlingua 0.2.2 需要 transformers 4.x，因此使用：

```text
主 Agent -> JSONL stdin/stdout -> .venv-compress -> LLMLingua-2 BERT/CPU
```

worker 模型懒加载、常驻；池大小 4，每 worker 自动约 32 CPU 线程。不要把主 venv 降级。

## 4. 关键实验结论

### 4.1 最终 retail full115

| 指标 | 结果 |
|---|---:|
| 任务/模型调用 | 115 / 1,780 |
| canonical/transformed | 10,933,355 / 8,820,495 |
| 完整 Prompt 降幅 | 2,112,860（19.32%） |
| success | 23/115 |
| p50/p95 | 98.02s / 161.64s |
| hot/cold BERT | 0 / 0 |

本轮每次固定省 1,187 token，全部来自 system/tools；最大可压 cold body 4,097，未过 8k。
Success 与两次历史 baseline 23/115、26/115 比分别是 0pp、-2.61pp，不能严格宣称通过 2pp。

### 4.2 旧“5k→2.5k”与当前安全策略

- 旧 LongLLMLingua/GPT-2 记录：`5,036 -> 2,049`、`5,113 -> 2,155`，但不是完整 Prompt，
  也没有当前的 tool-aware 保护。
- 当前 LLMLingua-2 纯 narrative：`5,000 -> 2,720`（-45.60%，2.76s）。
- 当前 5k 级 tool-aware：完整 Prompt `8,491 -> 6,722`（-20.83%），硬字段保留。
- 当前 9k 级 tool-aware：完整 Prompt `13,666 -> 10,676`（-21.88%）。

### 4.3 非 retail 数万 token

MIMO 离线编译 knowledge/incident policy，运行 45 轮、181 消息、8 工具：

| 配置 | Prompt token | 降幅 |
|---|---:|---:|
| 原始 | 28,860 | - |
| compiled policy + tool dedup | 28,712 | 0.51% |
| 静态 + LLMLingua-2 | 20,936 | 27.46% |

- Qwen 精确可压 body：20,772 token。
- middleware `chars/4` 触发估算：30,681，高估 47.7%；不要把它当真实 token。
- 首次 BERT：36.27s；相同 session reuse：5.5ms。
- 43 个 cold incident ID、所有 call ID/arguments、severity/status/owner/time 均保留；hot tail
  完全一致。

## 5. 压缩率解释与当前决策

当前不存在一个控制完整 Prompt 的统一 rate：

- `assistant_rate=0.75`：只作用 assistant narrative；
- `tool_result_rate=0.60`：只作用 tool narrative；
- `rate=0.65`：仅非 tool-aware 回退；
- user/arguments/硬字段/hot tail：100% 保留。

因此纯正文可下降约 45%，而受保护内容合并后的完整 Prompt 下降约 20%--27%。决策是冻结
0.75/0.60 安全档；不要为了更大数字直接调低。后续把 `(0.70,0.55)`、`(0.65,0.50)` 作为
独立质量 ablation。

## 6. 本次聊天做了什么（精简历程）

1. 澄清旧文档 `20219 -> 7772（-62%）` 是跨轨迹峰值冷区表示，不是完整 Prompt；提交
   `49f6724`。
2. 运行最终综合 full115，得到完整 Prompt -19.32%、动态 BERT 0 次；报告提交 `8864000`。
3. 找到早期 5k→约2k 历史记录，并新建真实 LLMLingua-2 长上下文 demo；提交 `d9420c9`。
4. 明确 system policy 是固定 Agent 业务规则，不是每次变化的 user 输入；user 当前和历史均保护。
5. 新增通用 policy artifact、MIMO 离线 compiler、哈希/逐条/literal/modal 校验与 fail-closed
   fallback；现有 retail 路径不改；提交 `1fb6e51`。
6. 构造非 retail 28.9k Prompt，证明动态 LLMLingua 触发后完整 Prompt -27.46%，并据审计新增
   severity/owner/created_at 等通用硬字段保护。
7. 发现 `chars/4` 将 20,772 精确 body 高估为 30,681，修正文档；提交 `9ad0100`。
8. 决定冻结安全压缩率，后续再做门槛计量、rate 和端到端质量/延迟改进。

## 7. 复现命令

编译一个新 policy（需要 `MIMO_KEY`）：

```bash
cd /tmp/f2-wt/agent-mem
PYTHONPATH=src /data/os_competition_TSJ/.venv/bin/python benchmarks/compile_policy.py \
  --input benchmarks/generic_policy_source.md \
  --output configs/policies/knowledge_incident.json \
  --policy-id knowledge-incident
```

运行非 tau 的数万 token benchmark：

```bash
cd /tmp/f2-wt/agent-mem
PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/generic_context_benchmark.py \
  --output /tmp/generic-context-result.json
```

回归测试：

```bash
PYTHONPATH=/tmp/f2-wt/agent-mem/src \
  /data/os_competition_TSJ/.venv/bin/pytest -q /tmp/f2-wt/agent-mem/tests
/data/os_competition_TSJ/.venv/bin/ruff check /tmp/f2-wt/agent-mem/src/agent_mem
```

最后一次结果：215 passed，Ruff 通过。

## 8. 关键代码与文档

代码：

- `agent-mem/src/agent_mem/middleware/compress.py`：hot/cold/tool-aware/缓存/worker 接线。
- `agent-mem/src/agent_mem/middleware/static_prompt.py`：retail compact、compiled artifact、工具去重。
- `agent-mem/src/agent_mem/middleware/policy.py`：通用 artifact 和 fail-closed 校验。
- `agent-mem/benchmarks/compile_policy.py`：MIMO 离线编译器。
- `agent-mem/benchmarks/generic_context_benchmark.py`：28.9k synthetic benchmark。

文档：

- `docs/F2-work-log.md`：完整时间线和所有实验摘要。
- `docs/F2-ablation-results.md`：旧 GPT-2/LLMLingua、full115 ablation 和口径澄清。
- `docs/f2-results/comparison_comprehensive_full115.md`：最终 retail full115。
- `docs/f2-results/f2-long-context-demo.md`：5k/9k LLMLingua-2 示例。
- `docs/f2-results/generic-policy-long-context.md`：通用 policy 和 28.9k 结果。
- `docs/F2-static-prompt-compression-research.md`：system/tools token 构成和风险分析。
- `docs/F2-tool-aware-llmlingua-plan.md`：tool-aware 设计依据。

## 9. 下一步优先级

1. 修正触发门计量：用 Qwen tokenizer、校准估算或语言自适应估算替换固定 `chars/4`。
2. 在非 tau 的长上下文任务上做端到端模型正确率和 prefill/latency 对照；当前 synthetic 只证明
   token 与结构保护。
3. 对 0.75/0.60、0.70/0.55、0.65/0.50 做质量 ablation，不直接改默认值。
4. 降低 20k body 首次 BERT 36s 成本，评估分块、线程、worker 调度或更小模型。
5. 用实际 KV block/可承载并发证明显存收益；vLLM 预分配 `mem_peak` 看不出来。
6. 如需严格证明 success <=2pp，做同环境 baseline/F2 多 run；不要用两次独立轨迹总 token
   归因。

## 10. 结论边界

可以说：F2 token 压缩有效、工具结构保护有效、通用 policy 接入机制已实现，并在两个业务域
获得证据。不能说：任意模型/任务都无损、已证明显存下降、已严格通过 success 2pp 或首次
压缩一定降低端到端延迟。
