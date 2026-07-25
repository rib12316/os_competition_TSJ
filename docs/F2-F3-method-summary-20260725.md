# F2/F3 方法、用法、实现与评测总结

- 日期：2026-07-25
- F2 分支：`feat/f2-prompt-compress`，checkpoint `1e64bc8`
- F3 分支：`feat/f3-tool-data-lazyload`，checkpoint `62a6488`
- 运行环境：openEuler 24.03、Ascend 910B2C、Qwen2.5-7B-Instruct、vLLM-Ascend
- 文档目的：统一说明 F2/F3 解决的问题、使用方法、运行框架、实现细节、适用边界、开源归属和现有证据

## 1. 总览

赛题要求针对 Agent 推理的长生命周期、多轮对话和工具调用特点，降低上下文及 KV Cache
占用，同时尽量保持任务成功率。F2 和 F3 都位于 Agent 层的上下文中间件，不修改模型权重，
也不依赖特定 NPU 内核：

- **F2：Prompt / 上下文压缩。** 优化每轮都会发送的 system policy、工具 schema 和逐步增长
  的历史消息。固定前缀使用确定性精简；动态历史只把允许有损的自然语言正文交给
  LLMLingua-2，工具协议、用户要求和业务硬字段由项目代码物理隔离并原样保护。
- **F3：工具调用数据 lazy-load。** 大型工具结果不再完整进入 Agent 历史，而是先保存到
  session-scoped ArtifactStore；历史只保留结构摘要和不透明 `result_id`。模型通过稳定的本地
  `fetch_tool_result` 工具按 JSON Pointer、字段条件、行或字符范围取回有界数据。
- **F2+F3：分层组合。** F3 先处理单次超大结果，F2 再处理固定前缀、中等工具结果和累积冷历史。

两者优化对象不同：F2 主要解决“固定内容每轮重复”和“许多中小消息累计变长”；F3 主要解决
“一次工具调用返回大量数据”。总上下文很长并不必然触发 F3，单次工具结果很大也不应优先交给
有损文本压缩。

```text
业务工具结果 ---------------------> F3: 大结果外置、按需取回
                                         |
canonical messages/tools ----------> F2: 固定前缀和动态历史变换
                                         |
                                         v
                               本地 vLLM-Ascend / Qwen
```

## 2. 用户视角：如何使用

### 2.1 前置条件

Agent 模型由本地 vLLM-Ascend 提供 OpenAI-compatible API。OpenAI Python SDK 只是本地 HTTP
客户端，并不表示 Agent 推理调用了 OpenAI 托管模型。以下命令假设服务位于
`http://127.0.0.1:8000/v1`，served model 名为 `Qwen2.5-7B-Instruct`。

F2 的动态压缩还需要独立 Python 环境。原因是项目主环境的 vLLM 使用 transformers 5.x，
而 `llmlingua==0.2.2` 依赖 transformers 4.x 的旧式 `past_key_values`：

```bash
uv venv /data/os_competition_TSJ/.venv-compress
uv pip install --python /data/os_competition_TSJ/.venv-compress/bin/python \
  "transformers==4.43.4" llmlingua \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

默认模型为
`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`。模型运行在 CPU，主 Qwen
继续运行在 NPU；不要为安装 LLMLingua 降级主环境的 transformers。

### 2.2 三种运行档位

只启用 F2：

```yaml
middleware:
  active: [compress]
```

对应完整配置：`agent-mem/configs/f2-compress.yaml`。

只启用 F3：

```yaml
middleware:
  active: [lazyload]
```

对应完整配置：`agent-mem/configs/f3-lazyload.yaml`。

组合启用：

```yaml
middleware:
  active: [lazyload, compress]
```

对应完整配置：`agent-mem/configs/f2-f3-combined.yaml`。顺序不能反转：F3 必须先保存原始大型
工具结果，F2 才能在后续请求中处理短引用和其他历史。

### 2.3 运行 tau-bench

从 `agent-mem` 目录执行：

```bash
PYTHONPATH=src MIMO_KEY="$MIMO_KEY" \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/runner.py \
  --config configs/f2-f3-combined.yaml \
  --runner qwen-agent \
  --engine-url http://127.0.0.1:8000/v1 \
  --runs 1 --max-tasks 5 --max-steps 20 --concurrency 1
```

`f2-compress.yaml` 和 `f2-f3-combined.yaml` 默认用 MIMO 充当 tau-bench 的 USER simulator；
被测 Agent 始终是本地 Qwen。`MIMO_KEY` 只从环境读取，不写入仓库。若不配置外部 user-sim，
runner 也支持让 user-sim 走本地服务，但轨迹长度和难度会发生变化，不能与 MIMO 结果直接混比。

### 2.4 F2 常用配置

最终安全基线是：

```yaml
compress:
  method: llmlingua2
  trigger_tokens: 8000
  recompress_delta_tokens: 4000
  keep_hot: 6
  tool_aware: true
  assistant_rate: 0.75
  tool_result_rate: 0.60
  hot_tool_trigger_tokens: 1000
  optimize_static_prompt: true
  system_prompt_mode: retail_compact
  deduplicate_tool_descriptions: true
  backend: subprocess
  worker_venv: /data/os_competition_TSJ/.venv-compress/bin/python
  worker_pool_size: 4
  worker_threads: 0
```

其中 `assistant_rate` 和 `tool_result_rate` 只控制可压缩正文的目标保留率；user、工具参数、
call ID、关键 JSON 字段和 hot tail 均不是按该 rate 统一压缩。配置中的 `rate=0.65` 只用于
`tool_aware=false` 的兼容路径，不能把它解释成完整 Prompt 保留 65%。

`system_prompt_mode` 有三种：

| 模式 | 用途 |
|---|---|
| `none` | 不改 system policy |
| `retail_compact` | 使用经过人工审计的 tau-bench retail 紧凑 policy |
| `compiled` | 加载离线编译并通过确定性校验的通用 policy artifact |

通用 policy 只在离线阶段使用 MIMO 编译，运行时不请求 MIMO：

```bash
PYTHONPATH=src /data/os_competition_TSJ/.venv/bin/python \
  benchmarks/compile_policy.py \
  --input benchmarks/generic_policy_source.md \
  --output configs/policies/knowledge_incident.json \
  --policy-id knowledge-incident
```

运行时配置为：

```yaml
system_prompt_mode: compiled
policy_artifact_path: agent-mem/configs/policies/knowledge_incident.json
policy_artifact_strict: false
```

source hash 或逐条规则校验失败时，默认继续使用原始 system prompt；strict 模式才中止运行。

### 2.5 F3 常用配置

```yaml
lazyload:
  store: sqlite
  store_path: ""
  ttl_seconds: 3600
  externalize_trigger_tokens: 4000
  max_reference_tokens: 512
  fetch_max_tokens: 768
  fetch_default_lines: 20
  fetch_max_lines: 100
  max_parse_bytes: 5000000
  on_store_error: passthrough
  exempt_tools: [fetch_tool_result]
  tool_overrides: {}
```

- `store_path=""` 时，每个进程使用 `/tmp/agent-mem-f3-<pid>.sqlite3`。
- `tool_overrides` 可为某一业务工具设置不同的外置阈值。
- `on_store_error=passthrough` 以任务成功率优先；还可选择 `head_tail` 或 `raise`。
- 只有引用确实比原文小、且不超过 512 token 时才替换原结果。
- `fetch_tool_result` 从第一轮起稳定加入 tools，避免工具前缀在外置首次发生后才变化。

模型通常会先看到如下引用，再自行构造 fetch 参数：

```json
{
  "_agent_mem": "external_tool_result",
  "result_id": "tr_<opaque-id>",
  "status": "stored",
  "tool_name": "retrieve_documents",
  "content_type": "application/json",
  "token_count": 9000,
  "summary": {
    "kind": "json",
    "arrays": [{"pointer": "/documents", "length": 10}]
  },
  "fetch": {"name": "fetch_tool_result", "arguments": {"result_id": "tr_<opaque-id>"}}
}
```

字段搜索示例：

```json
{
  "result_id": "tr_<opaque-id>",
  "json_pointer": "/documents",
  "match_field": "title",
  "match_value": "Target title",
  "match_mode": "iexact",
  "max_matches": 3
}
```

当前 selector 仅支持单个直接字段的 `exact`、`iexact`、`contains`、`icontains`，不支持
任意 JSONPath、正则、表达式执行、多字段 AND/OR、数值范围、排序或聚合。

### 2.6 观测与日志

可同时开启三类日志：

```bash
export F2_EVENT_LOG=/tmp/f2-events.jsonl
export F3_EVENT_LOG=/tmp/f3-events.jsonl
export PROMPT_TOKEN_LOG=/tmp/prompt-tokens.jsonl
```

- `F2_EVENT_LOG`：记录 `skip/compress/reuse`、冷热 token、压缩耗时、静态精简和 hot tool 事件。
- `F3_EVENT_LOG`：记录 passthrough、externalize、fetch、token 节省及本地处理耗时。
- `PROMPT_TOKEN_LOG`：对同一轮 canonical/transformed 请求应用完整 Qwen chat template，并记录
  服务端 `usage.prompt_tokens`。F3 会只在计量副本中恢复原始 artifact，实际短历史不受影响。

正式配置会从 `engine.model` 自动注入 tokenizer；`PROMPT_TOKENIZER_PATH` 可覆盖本地路径。
若没有 tokenizer，轻量调用才回退 Unicode-aware heuristic。历史文档中的固定 `chars/4` 只能作
旧实验参考，不能作为当前触发门或完整 Prompt 指标。

## 3. 公共功能框架

### 3.1 中间件挂载点

项目自建了有序 `MiddlewareStack`，主要钩子如下：

| 钩子 | 时机 | F2 | F3 |
|---|---|---|---|
| `prepare` | benchmark 计时前 | 预热 tokenizer | 预热 tokenizer |
| `transform_request` | 每次模型请求前 | 变换 messages 和 tools 副本 | 注入 fetch schema |
| `intercept_tool_result` | 工具结果回灌前 | 默认不改 canonical 结果 | 大结果存储并改成引用 |
| `handle_internal_tool_call` | 模型发出工具调用后 | 不处理 | 本地执行 fetch，不进入业务环境 |
| `measurement_baseline` | 严格配对计量时 | identity | 只在计量副本中恢复原文 |
| `after_model_call` | 模型返回 usage 后 | 落真实 token 日志 | 当前无需处理 |

F2 的所有压缩都只作用于“发给模型的副本”，canonical 对话历史仍保留完整内容。F3 的目标正是
阻止大结果进入 canonical 历史，因此 canonical 中保存引用，但完整原文仍存在 ArtifactStore。

### 3.2 一轮 Agent 请求

```text
canonical messages + business tools
    |
    | 1. measurement_baseline（仅开启严格 meter 时）
    | 2. lazyload.transform_request：加入稳定 fetch schema
    | 3. compress.transform_request：静态前缀 + 动态历史
    v
local Qwen chat.completions
    |
    +-- 普通回答 ------------------------------> 结束或回复用户
    |
    +-- 业务工具调用 --> tau env.step / execute_tool
    |                       |
    |                       +--> lazyload.intercept_tool_result
    |                               小结果：原样回灌
    |                               大结果：store + reference
    |
    +-- fetch_tool_result --> MiddlewareStack 内部处理
                              不调用 tau-bench env.step
                              返回 <= 768-token 工具结果
```

F3 内部工具绕过业务环境很重要：它只读取已经产生的工具结果，不能再次修改订单或重复执行
真实 API。相应逻辑已同时接入通用 ReAct loop 和 tau-bench Agent。

## 4. F2 是怎么实现的

### 4.1 固定 system policy 优化

tau-bench retail 首轮完整 Prompt 中，system policy 约 1,188 token，16 个工具 schema 约
3,047 token，两者几乎占满首轮输入。把高敏感 instruction 在线交给 token compressor 风险较高，
还会每轮增加 CPU 延迟，因此最终采用离线、确定性的方式：

1. retail 使用逐条审计的 `COMPACT_RETAIL_POLICY`，保留认证、单用户、显式确认、订单状态、
   退款、退换货和一次性调用等行为约束。
2. 通用业务可离线生成 source-unit artifact。每个 source unit 必须按原顺序存在，并验证
   source SHA-256、数字、ID、引号字面量、状态词及 `must/only/never/cannot/do not` 等 modal。
3. artifact 失效时保留原文，避免未经验证的压缩 policy 进入运行时。

### 4.2 工具 schema 去重

工具的 name、type、层次、required、enum 和参数结构保持不变。项目代码遍历所有 description：

- system 已覆盖显式确认规则时，删除工具描述中完全重复的确认句；
- 只对完全相同且确有 token 收益的 description 建立 `C1/C2/...` convention；
- 原位置替换成 `See Cn.`，完整说明只在 system 的 shared conventions 中出现一次。

该变换不会动态裁剪工具集合，因此每轮 system/tools 前缀稳定，仍有利于 vLLM prefix cache。

### 4.3 动态历史的冷热分层

去掉 system 后，最近 `keep_hot=6` 条为 hot tail，其余为 cold history。切分边界会向前移动，
直到 assistant tool call 与其 `role=tool` 结果构成完整组，避免 OpenAI tool-call 协议出现孤立
tool message。

处理规则：

- 历史不足 hot window：只检查是否有超大 hot tool result；否则直接跳过。
- cold 可压正文不足 8,000 Qwen token：不启动 BERT。
- 首次超过 8,000：压缩整段 cold，并记录已冻结消息数。
- 之后新增可压 cold 不足 4,000：复用旧压缩段，新冷消息原样附加。
- 新增量达到 4,000：重新构造 cold，但逐 body 缓存可复用已处理内容。

### 4.4 Tool-aware 结构保护

LLMLingua 不原生理解 OpenAI `tool_calls`。项目侧先把一条历史拆成 protected prefix 与
compressible body，只有 body 会送进 BERT：

```text
[USER]
完整用户消息                                      100% 保留

[ASSISTANT_TOOL_CALL]
call_id=...
name=...
arguments=...                                    100% 保留
assistant narrative                              rate=0.75

[TOOL_RESULT]
call_id=...
name=...
critical_fields={...}                            100% 保留
description / note / long text                   rate=0.60
```

关键字段按 JSON key 识别，包括 ID、status/state、amount/price/total/balance、quantity/count、
time/date、email/address/payment、reason/confirm/error/name、refund、severity/priority、
owner/assignee、organization/tenant/account、permission/role，以及 `_id/_ids/_at/_time/_date/
_timestamp` 后缀。数值、布尔值和 null 也会进入保护快照。

这是一种面向 Agent 协议的保守保护，不等于通用语义无损：未被关键字段规则覆盖的长文本仍可能
被压缩掉，业务接入时应审计字段命名，并通过质量 ablation 决定是否扩充保护表。

### 4.5 Hot 大工具结果

单个 hot tool result 达到 1,000 token 时，无需等它变冷：F2 保留原 assistant tool call 和
tool message 元数据，只替换发送副本里的 content。关键字段与 narrative 分离后，只压正文；
若替换结果不比原文短，则保留原文。

### 4.6 LLMLingua-2 worker 与缓存

```text
CompressMiddleware
    -> JSONL stdin/stdout RPC
    -> 常驻 .venv-compress worker pool
    -> llmlingua.PromptCompressor
    -> multilingual LLMLingua-2 BERT / CPU
```

worker 模型懒加载并跨 session 复用。`worker_pool_size=4` 支持任务级并发；
`worker_threads=0` 自动设置为 `CPU 核数 / pool size`，避免多个 PyTorch worker 各自占满全部核。

缓存包括：

- session 级冻结 cold 段和增量复用；
- `rate + SHA-256(body)` 压缩结果缓存；
- 4,096 项、按序列化文本 SHA-256 索引的精确 token-count cache；
- 进程级 tokenizer cache。

正文为空、worker 返回异常形状、压缩后更长或结果为空时，项目代码保留原正文，不用负收益或
无效输出替换它。

## 5. F3 是怎么实现的

### 5.1 精确门槛与原文外置

工具返回后，F3 用与引擎一致的 tokenizer 计算原文：

1. exempt tool 或 per-tool threshold `<=0`：原样通过；
2. 小于阈值：原样进入历史；
3. 大于等于 4,000 token：生成确定性 synopsis，并先把完整原文写入 store；
4. 生成带 `result_id` 的引用；只有引用小于原文且不超过 512 token 时才正式替换；
5. 存储失败按 `passthrough/head_tail/raise` 策略处理。

这不是截断：成功外置时，原始数据按 UTF-8 文本完整保存在 store 中，后续 fetch 可以继续读取。

### 5.2 ArtifactStore

项目定义了统一 `ArtifactStore` protocol，并实现两种后端：

- `MemoryArtifactStore`：线程安全、易失，主要用于单测和 dry-run；
- `SQLiteArtifactStore`：WAL、`synchronous=NORMAL`、busy timeout、TTL 清理和
  `(session_id, result_id)` 联合主键。

每个 artifact 包含随机 `tr_...` ID、session ID、工具名、content type、完整 content、
SHA-256、byte/token count 和创建时间。读取必须同时匹配当前 session；跨 session 使用同一个
result ID 也只会得到 `not_found`。fetch 前还会重新校验 SHA-256。

当前 SQLite 后端适合单进程本地 Agent；多进程或多节点部署需替换为共享 store，但应保持同样的
session ACL、完整性和 TTL 接口。

### 5.3 确定性摘要

F3 不调用额外 LLM 生成摘要，而是使用 Python 标准库解析：

| 类型 | 摘要内容 |
|---|---|
| JSON | 根 shape、前 12 个 key、最多 4 个 array pointer、长度、item keys、最多 12 个 title |
| HTML | title、最多 8 个 h1/h2/h3、行数 |
| CSV/TSV | 列名和行数 |
| text | 行数、受限 opening/closing excerpt |
| 超过 5 MB | 不做完整解析，只给 oversized/text 摘要和 opening |

摘要的作用是让模型知道“数据是什么形状、应从哪里取”，不是替代原文回答所有问题。

### 5.4 有界取回接口

`fetch_tool_result` 支持四类访问：

1. RFC 6901 JSON Pointer，例如 `/items/0`；
2. JSON array 单字段搜索，可显式指定 array pointer，也可在唯一候选数组时自动推断；
3. 按 1-based 行号分页，最多 100 行；
4. 按字符 offset 继续读取，单次最多检查 20,000 字符。

字段搜索最多返回 10 条，并给出稳定 `match_pointers`、扫描数、匹配总数和返回数。超大匹配记录
不会切出无效的半个 JSON object，而是逐条减少结果或返回结构有效的字段 preview。所有 fetch
响应连同 metadata 一起受 768-token 硬上限约束，并提供 `next_start_line` 或
`next_start_char` 供继续读取。

安全边界包括：无任意代码执行、无正则或 JSONPath 表达式、字段搜索 5 MB parse bound、
session-scoped 访问和完整性检查。工具描述明确把取回数据标为 untrusted；F3 本身不对原始工具
内容做 prompt-injection 语义净化，模型仍需按 system policy 把工具内容视为不可信数据。

### 5.5 为什么增加了结构化字段搜索

首版 F3 只有行/字符分页。18,658-token 的 360 条 minified JSON 位于一行中，Qwen 虽然会调用
fetch，却总是拿到开头记录，head/middle/tail 定位测试为 0/3。这说明“存储和触发正常”不等于
“模型能够找到目标”。

加入受限字段 selector 后，Qwen 根据用户给出的 `case_id` 自行生成 `match_field/match_value`，
每题一次 fetch，baseline、F3、F2+F3 均达到 3/3。这个修复是项目自己的接口设计与实现，
不是 DeerFlow 代码移植。

## 6. F2+F3 的组合契约

工具结果刚返回时的默认分层为：

| 单次结果大小 | 立即处理 |
|---|---|
| `< 1,000` token | inline 原样保留 |
| `1,000-3,999` token | F2 可对 hot result 的 narrative 做结构保护压缩 |
| `>= 4,000` token | F3 先外置，F2 只看到短引用 |

随着会话继续，旧的 inline 小结果、assistant narrative 和 F3 引用摘要仍可能进入 cold 区；只有
累计可压正文达到 8,000 token 才触发 F2 cold compression。fetch 响应不超过 768 token，
因此不会立即触发 F2 的 1,000-token hot gate；`fetch_tool_result` 同时被 F3 exempt，避免
“fetch 结果再次外置”的循环。

在 synthetic 长 JSON 轨迹上，F2-only 需要 6.64 秒压缩大型 body；F2+F3 只处理引用，变换为
3.24 ms。组合的价值不仅是叠加 token 缩减，也包括让不适合有损压缩的超大结构化数据优先走
可恢复的外置路径。

## 7. 起作用的场景与不适用场景

### 7.1 F2 适用场景

- system policy 和工具 schema 很长，而且每轮重复；
- 多轮 Agent 累积了大量旧 assistant 解释、检索摘要、日志或工具 narrative；
- 单次工具结果处于中等规模，不值得建立外部 artifact，但正文有明显冗余；
- 任务需要保留工具 ID、参数、状态、金额、时间等结构字段，同时可以压缩自然语言说明；
- 同一 session 后续多步能复用已压缩 cold 段，摊薄首次 CPU BERT 成本。

F2 不适合：短会话、几乎没有自然语言冗余、所有旧文本都必须逐字保留，或对首次请求延迟极敏感
且无法预热/异步处理的场景。对新的业务 schema 直接沿用 retail 关键字段表也不够，应先审计。

### 7.2 F3 适用场景

- 一次工具返回数千商品、历史订单、航班、库存、物流事件、网页或检索文档；
- 数据整体很大，但当前问题只需要一两条记录或一个局部字段；
- 数据有稳定 JSON ID/title，或文本天然可按行/字符分页；
- baseline 接近模型 context limit，重复携带工具结果会溢出；
- 更重视 Prompt/KV 占用、并发容量和“能够完成”，而不是最少模型轮次。

F3 不适合：20 次各 500-token 的小结果、长 system/tool schema，或必须同时遍历全部数据做全局
count/min/max/sum、多条件筛选和排序的任务。当前单字段 selector 可能迫使模型多次 fetch，
节省 Prompt 的同时增加串行 decode 和调度延迟。

### 7.3 更长 tau-bench 中的预期

如果新的 tau-bench 工具单次返回 5k-20k token 商品/订单/航班列表，F3 机制仍然有效，而且
会比当前 retail 更有价值。若只是总轮数增加、每次结果仍很小，主要由 F2 cold history 负责。

合理预期是 Prompt token、KV token 和 context overflow 减少；单会话端到端延迟不保证下降。
若 baseline 已超过 32k 而 F3 能完成任务，F3 的首要价值会从“是否更快”转为“是否可运行”。

## 8. 我们实际测了什么

### 8.1 tau-bench retail

tau-bench 是环境交互 benchmark：USER simulator 与 Agent 多轮对话，Agent 调真实 retail 工具
修改环境，最终由环境 reward 判断任务成功。当前配置可用 MIMO 模拟 USER，本地 Qwen 作为被测
Agent。

标准 retail 工具结果并不大。五任务 F2+F3 探针共观察 27 个结果，最大仅 1,416 Qwen token，
所以 F3 外置 0 次。该实验只证明组合栈能与真实 tau-bench/MIMO 流程兼容，并在不合适时安全
no-op，不能证明 F3 在 retail 上带来收益。

### 8.2 2Wiki Agent 化 benchmark

当前 2Wiki 测试不是 LongBench 原始的“一次性长上下文问答”，而是改造成多轮工具检索 Agent：

```text
2Wiki question + 约 10 篇候选文档 + gold answer
    -> context 转成 JSON documents
    -> Agent 调 retrieve_documents，得到全部候选文档
    -> baseline 把原文放入历史；F3 把原文外置
    -> F3 Agent 根据标题调用 fetch_tool_result 完成一到两跳取证
    -> 最终短答案标准化后与 gold answer 匹配
```

它没有 MIMO 用户，也不会修改订单环境。它主要测长工具结果下的检索、上下文溢出和回答质量。

### 8.3 p50 延迟口径

2Wiki 报告中的 `median_wall_ms` 对每题执行以下计算：

1. 记录该题所有本地 Qwen `chat.completions.create` HTTP 请求耗时；
2. 把同一题的多次模型请求耗时相加；
3. 对 100 个任务的和取中位数。

它不是完整端到端耗时，不包含 SQLite、fetch 本地处理、F2 middleware、LLMLingua CPU 推理和
Python Agent 调度。F3 本地操作为毫秒级，但动态 F2 真正触发时该指标会低估完整开销。

## 9. 当前实验结果

### 9.1 F2

| 工作负载 | 结果 | 正确解读 |
|---|---:|---|
| retail full115 严格配对完整 Prompt | `10,933,355 -> 8,820,495`，-19.32% | 1,780 次调用都固定省 1,187；动态 BERT 0 次，收益来自静态 system/tools |
| retail full115 success | 23/115 | 历史 baseline 为 23/115 和 26/115，单次独立轨迹未严格证明下降 `<=2pp` |
| generic 28.9k synthetic trace | `28,860 -> 20,936`，-27.46% | 动态 LLMLingua-2 触发；证明 token 与结构保护，不是端到端任务正确率 |
| 5k forced / 9k production tool-aware demo | 完整 Prompt -20.83% / -21.88% | 5k 为临时把门槛设为 1 的能力展示，生产 8k 门槛会 skip；9k 真实触发，关键字段与 hot tail 保留 |
| 纯 5k narrative | `5,000 -> 2,720`，-45.60% | 只代表可压正文，不代表含工具协议的完整 Prompt |

generic 轨迹首次 BERT 压缩耗时 36.27 秒；相同 session 复用约 5.9 ms。当前证据支持“长上下文
token 缩减、协议保护和通用 policy 接入有效”，不支持“所有任务无损”或“首次压缩必然降低
端到端延迟”。

### 9.2 F3 deterministic 与受控检索

| 指标 | 结果 |
|---|---:|
| synthetic 完整 Prompt | `21,350 -> 1,499`，-92.98% |
| vLLM server Prompt | `21,371 -> 1,520`，-92.89% |
| SQLite externalize p95 | 8.53 ms |
| bounded fetch p95 | 5.08 ms |
| exact JSON search p95 | 0.15 ms |
| reference 最大值 | 222 token，门限 512 |
| fetch 最大值 | 768 token |
| 360-record head/middle/tail | baseline、F3、F2+F3 均为 3/3 |
| 受控累计 Prompt | `57,570 -> 8,293`，-85.59% |

vLLM 的 1-output-token prefill 探针中，baseline/F3 p50 为 1,822.29/81.23 ms。该探针主要测
prefill，不能外推为正常多轮 Agent 的 decode 或端到端加速。

### 9.3 2Wiki first 100

| 配置 | 正确数 | 累计 Prompt | 相对 baseline | p50 模型请求总耗时 | 模型调用/题 |
|---|---:|---:|---:|---:|---:|
| baseline | 31/100 | 1,632,665 | - | 1,269.00 ms | 240 / 2.40 |
| F3 | 31/100 | 554,657 | -66.03% | 1,726.44 ms | 368 / 3.68 |
| F2+F3 | 27/100 | 504,313 | -69.11% | 1,660.37 ms | 355 / 3.55 |

平均每次模型调用 Prompt 分别约为 6,803、1,507 和 1,421 token。F3 显著缩短单次 Prompt，
但模型需要额外决定 fetch、生成工具参数、读取结果并继续推理，调用次数增加，所有步骤又必须串行。
在当前 NPU 上，省下的 prefill 小于额外轮次的 HTTP、调度和 decode 成本，所以 F3 p50 增加
36.05%，F2+F3 增加 30.84%。

baseline 有 5 题因重复 inline retrieval 超过 32,768 context 而请求失败，F3/F2+F3 为 0。
在 95 个 baseline/F3 都没有 request error 的任务上，正确数为 31/95 和 30/95，双方独有成功
14/13，exact McNemar `p=1.0`。因此本次没有观察到严重或显著的 F3 总体质量下降，但样本仅一轮、
模型单一，不能宣称任意任务质量无损。F2+F3 的 27/100 也仍是需要复测的风险点。

### 9.4 tau-bench first 5

| 指标 | baseline | F2+F3 |
|---|---:|---:|
| success | 1/5 | 2/5 |
| e2e p50 | 148.48 s | 67.31 s |
| server Prompt | 686,290 | 267,463 |
| model calls | 89 | 48 |

两档的 MIMO 对话走了不同轨迹，因此该表是运行观察，不是 F2/F3 的因果 A/B。F2+F3 在自身
轨迹中的严格 paired meter 为 `302,142 -> 267,806`，-11.36%；F3 未触发。

## 10. 哪些是自研，哪些直接使用开源，哪些只借鉴

### 10.1 直接使用的开源项目或模型

| 上游 | 使用方式 | 不应归为自研的部分 |
|---|---|---|
| Microsoft LLMLingua / LongLLMLingua / LLMLingua-2 | 隔离 worker 中直接调用 `llmlingua.PromptCompressor` | token 选择/压缩算法及其模型 |
| LLMLingua-2 multilingual BERT | F2 CPU narrative compressor | BERT 模型权重与上游推理实现 |
| Qwen2.5-7B-Instruct | 本地 Agent 推理与 tokenizer | 基础模型能力 |
| vLLM / vLLM-Ascend | 本地 OpenAI-compatible serving、usage 与 NPU 运行时 | 推理引擎和 Ascend 适配 |
| OpenAI Python SDK | 访问本地 vLLM endpoint | HTTP 客户端协议实现；不是 OpenAI 托管推理 |
| tau-bench | retail/airline 环境、工具、任务和 reward | benchmark 环境与数据 |
| LongBench / 2WikiMultihopQA | F3 多跳质量 probe 数据 | 原始问题、文档和 gold answer |

LLMLingua 和 DeerFlow 的上游许可/致谢应按现有 attribution 记录保留在对外发布材料中。

### 10.2 借鉴设计、没有复制代码

F3 主要借鉴 ByteDance DeerFlow `ToolOutputBudgetMiddleware` 的思想：per-tool output budget、
在历史增长前持久化、compact synopsis、bounded retrieval、fallback 和相应测试原则。

项目没有 import、vendor 或复制 DeerFlow、LangChain、LangGraph 的实现；仓库中也没有这些运行时
依赖。早期方案还参考过 reference-ID/retriever 通用模式，但最终代码以独立中间件接口实现。

### 10.3 项目自行实现

F2 自研部分：

- Agent 中间件挂载与 request/messages/tools 联合变换；
- hot/cold 切分、完整 assistant-tool group 边界和 canonical 副本语义；
- tool-aware serializer、user/arguments/call ID/关键字段物理保护；
- assistant/tool body 分 rate 调度、hot result 路径和负收益回退；
- Qwen tokenizer 精确 gate、增量 recompression、三类缓存；
- JSONL worker RPC、worker pool、初始化锁和 CPU 线程控制；
- retail compact policy、通用 compiled artifact、hash/literal/modal 校验和原文回退；
- 工具 description 去重、完整 Prompt meter、tau-bench 接入、测试和 benchmark。

F3 自研部分：

- `LazyLoadMiddleware`、稳定 `fetch_tool_result` schema 和 ReAct/tau-bench 内部工具接入；
- Memory/SQLite ArtifactStore、TTL、WAL、随机 ID、session ACL 和 SHA-256 校验；
- JSON/HTML/CSV/TSV/text 确定性 synopsis；
- RFC 6901 pointer、行/字符分页、单字段 bounded search 和 token-capped response；
- match pointer、整记录截断/preview、store error 策略、F2+F3 顺序与防循环规则；
- paired measurement restoration、单测、deterministic/quality/LongBench benchmark 和报告。

另外，项目当前 Agent loop 是自写 ReAct 和 tau-bench adapter；虽然仓库存在 Qwen-Agent 上游，
这里并未使用 Qwen-Agent `Assistant` 来实现 F2/F3 流程。Python 的 SQLite、JSON、CSV 和
`HTMLParser` 属于标准库，不是外部项目代码。

MIMO 是外部 API 服务，只用于 tau-bench USER simulation 和通用 policy 的离线编译；它不参与
本地 Agent 推理，也不是 F2 动态压缩或 F3 synopsis/fetch 的运行时依赖。

## 11. 正确的结论与当前限制

可以据现有证据陈述：

- F2 已形成固定前缀确定性优化 + tool-aware 动态压缩的完整框架；
- F2 能在 retail 完整 Prompt 上稳定减少固定 token，并能在数万 token 通用轨迹中触发动态压缩；
- F3 能把大型工具结果移出 Prompt/KV，并以 session-scoped、有界接口恢复局部数据；
- F3 在受控 ID 检索保持 3/3，在 100 条 2Wiki 中累计 Prompt -66.03%、总体正确数与 baseline 相同；
- F3 避免了该轮 5 次 32k context overflow；F2+F3 能运行真实 tau-bench/MIMO 流程。

目前不能据此宣称：

- F2/F3 对任意模型、任务和业务 policy 都质量无损；
- retail full115 已严格证明 F2 success 下降不超过 2pp；
- Prompt token 下降已经等价证明进程级 HBM peak 下降；vLLM 预分配 KV pool 会掩盖这一指标；
- F3 一定降低单会话延迟；当前多跳 workload 的额外 fetch 轮次反而增加 p50；
- 2Wiki probe 等同 LongBench 官方一次性长上下文分数；它是我们改造的 Agent benchmark；
- 本地 SQLite 已满足分布式生产部署。

后续最有价值的工作是：

1. 为 F3 加入受限多字段 AND/OR、数值范围、投影、排序/top-k 和 count/min/max/sum，使一次
   fetch 能完成常见结构化查询，减少模型轮次；
2. 重复 2Wiki 100 条或跑完整 200 条，并复测 F2+F3 的 4 个答案点差；
3. 在真正返回 5k-20k 工具结果的 tau-bench-like workload 上做同轨迹质量和端到端实验；
4. 用实际 KV block、可承载并发 session 数和 context overflow 率衡量内存收益；
5. 对 F2 `(0.75,0.60)`、`(0.70,0.55)`、`(0.65,0.50)` 做配对质量 ablation，并优化首次
   BERT 36 秒的分块、线程和异步调度。

## 12. 代码与复现索引

核心代码：

- `agent-mem/src/agent_mem/middleware/base.py`：公共中间件契约与有序 stack；
- `agent-mem/src/agent_mem/middleware/compress.py`：F2 动态压缩、缓存和 worker 接线；
- `agent-mem/src/agent_mem/middleware/static_prompt.py`：system/tools 确定性优化；
- `agent-mem/src/agent_mem/middleware/policy.py`：通用 policy artifact 校验；
- `agent-mem/src/agent_mem/token_counting.py`：共享 tokenizer 和精确计量；
- `agent-mem/src/agent_mem/middleware/lazyload.py`：F3 外置与 bounded fetch/search；
- `agent-mem/src/agent_mem/middleware/artifact_store.py`：Memory/SQLite store；
- `agent-mem/src/agent_mem/middleware/tool_synopsis.py`：确定性结构摘要；
- `agent-mem/src/agent_mem/agent/react.py`、`tau_bench_agent.py`：通用和 tau-bench Agent 接入。

F2 benchmark：

```bash
PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/f2_long_context_demo.py

PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/generic_context_benchmark.py \
  --output /tmp/generic-context-result.json
```

F3 benchmark：

```bash
PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/f3_long_tool_benchmark.py \
  --model Qwen2.5-7B-Instruct --served-model Qwen2.5-7B-Instruct \
  --results 3 --result-tokens 5000 --perf-runs 20 \
  --engine-url http://127.0.0.1:8000/v1 --vllm-rounds 3

PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/f3_retrieval_quality_benchmark.py \
  --records 360 --variants baseline f3 f2_f3

PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/f3_longbench_quality_benchmark.py \
  --data-zip /tmp/longbench-data.zip --start 0 --limit 100 \
  --variants baseline f3 f2_f3
```

回归：

```bash
PYTHONPATH=src /data/os_competition_TSJ/.venv/bin/python -m pytest -q tests
/data/os_competition_TSJ/.venv/bin/ruff check src/agent_mem tests benchmarks
```

在 F3 checkpoint `62a6488` 上复核结果为 **245 passed**，Ruff 全部通过。

原始证据与扩展说明：

- `docs/F2-next-session-handoff.md`
- `docs/F2-ablation-results.md`
- `docs/f2-results/comparison_comprehensive_full115.md`
- `docs/f2-results/generic-policy-long-context.md`
- `docs/F3-tool-data-lazyload-report.md`
- `docs/F2-F3-implementation-attribution.md`
- `docs/F2-F3-extended-evaluation-20260725.md`
- `docs/f3-results/f3_tool_data_result.json`
- `docs/f3-results/f3_retrieval_quality_result.json`
- `docs/f3-results/f3_longbench_2wikimqa_first100_summary.json`
- `docs/f3-results/f3_longbench_2wikimqa_first100_probe.json`

## 13. 最终总结

F2 和 F3 不是两个互斥的“压缩算法”，而是 Agent 上下文生命周期中的两层治理：F3 对单次大型
工具数据采用可恢复外置，F2 对仍需进入模型的固定前缀和累积 narrative 做保守精简。二者共同
目标是减少无须在每轮 Prompt/KV 中常驻的数据，同时把可执行协议、用户意图和关键业务字段放在
优化边界之外。

当前实现已完成从配置、Agent 挂载、存储/压缩、精确计量、错误回退到 benchmark 的闭环。
最可靠的现有收益是 Prompt/KV token 和 context overflow 控制；质量与延迟则高度依赖任务是否
适合局部检索、压缩是否真正触发以及额外模型轮次。后续应把重点从继续追求单个压缩率数字，转向
更强的本地结构化查询、更严格的配对质量实验和实际并发/KV 容量测量。
