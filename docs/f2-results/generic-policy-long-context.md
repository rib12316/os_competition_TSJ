# 通用 Policy 编译 + 数万 token 上下文实验

日期：2026-07-24
编译模型：`mimo-v2.5-pro`（离线，只用于生成 policy artifact）
动态压缩：当前 tool-aware LLMLingua-2 BERT
原始结果：`f2_generic_context_result.json`

## 实现内容

新增通用 policy artifact 管线：

```text
system source -> MIMO offline compiler -> source-unit artifact
             -> SHA-256/literal/modal validation -> runtime compact policy
```

运行时不会调用 MIMO。artifact 必须满足：

- source SHA-256 与当前 system 内容一致；
- 每个 source unit 都有同序的 compact rule；
- compact policy 是 source units 的确定性渲染；
- 数字、引号字面量、ID、状态和强约束词保留；
- 校验失败默认回退原始 system，strict 模式才抛错。

Retail 的 `retail_compact` 路径未改变。通用路径通过：

```yaml
system_prompt_mode: compiled
policy_artifact_path: agent-mem/configs/policies/knowledge_incident.json
```

实现文件：`agent-mem/src/agent_mem/middleware/policy.py`、
`agent-mem/benchmarks/compile_policy.py`。

## MIMO 编译结果

示例 policy 是非 retail 的 knowledge/incident support Agent，包含认证、组织隔离、
不可信检索结果、confirmation、P0/P1/P2、15 分钟响应、凭据保护等规则。

| 指标 | 数值 |
|---|---:|
| source | 862 chars |
| compiled artifact | 818 chars |
| policy units | 11 |
| 编译结果 | validation passed |
| tool description dedup | 24 replacements / 3 conventions |

该 policy 本身已经比较紧凑，所以静态 policy 编译只节省少量 token；这不是编译器失效，
而是 source 没有大量重复 prose。运行时用 hash 不匹配会安全回退原文。

## 通用长上下文 benchmark

脚本：`agent-mem/benchmarks/generic_context_benchmark.py`
生成：45 轮 incident/search/tool 交互，8 个工具，181 条消息，Qwen tokenizer 计量。

| 配置 | Prompt token | 节省 |
|---|---:|---:|
| 原始 system + tools + history | 28,860 | - |
| 只 compiled policy + tool dedup | 28,712 | 148（0.51%） |
| 静态 + LLMLingua-2 | **20,936** | **7,924（27.46%）** |

动态事件显示：

- 可压 cold body 逐段 Qwen 计数为 20,772 token；按 middleware 实际 `\n\n` 序列化边界
  精确计数为 20,815，超过当前 8k 门槛；旧 `chars/4` 值 30,681 仅作历史对照；
- cold BERT 触发 1 次，压缩耗时 36.27s；
- worker 预热耗时 4.63s；
- 相同 session 第二次 action=`reuse`；修正后有界 token-count cache 探针为 5.9ms；
- hot tool result 没有触发，因为本例大数据位于 cold history。
- 43 个 cold incident ID、所有 call ID/arguments、severity/status/owner/time 全部保留；
  最近 hot tail 与原始轨迹一致。

## 结论

当前方法已经不再要求 system policy 必须是 tau-bench retail：新的 policy 可由 MIMO 离线
编译，运行时按 source hash 安全加载，未知或校验失败时保留原文。数万 token 通用 Agent
轨迹中，动态 tool-aware LLMLingua-2 能提供约 27% 的完整 Prompt 降幅；本次没有运行
full115，也没有把 synthetic token 节省冒充显存或 task success 结果。
