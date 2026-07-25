# F2/F3 Implementation Attribution

- Date: 2026-07-25
- F2 branch: `feat/f2-prompt-compress` at `1e64bc8`
- F3 branch: `feat/f3-tool-data-lazyload` at `d942d52`
- Agent inference: local Qwen2.5-7B-Instruct served by vllm-ascend
- tau-bench user simulation: MIMO when enabled by the benchmark configuration

## Summary

F2 directly uses Microsoft LLMLingua as a text-compression dependency, then adds a project-owned Agent safety, scheduling, caching, static-prefix, and measurement layer.

F3 adapts ByteDance DeerFlow's tool-output-budgeting design, but does not import, vendor, or copy DeerFlow, LangChain, or LangGraph code. Its storage, synopsis, retrieval, search, middleware, Agent integration, tests, and benchmarks are project-owned implementations.

The OpenAI Python SDK is only a client for the local vLLM-compatible endpoint. It does not mean OpenAI-hosted inference is used.

## Source Classification

| Component | Classification | Usage |
|---|---|---|
| LLMLingua / LongLLMLingua / LLMLingua-2 | Direct open-source dependency | `llmlingua.PromptCompressor`, compression algorithms, and models |
| LLMLingua-2 multilingual BERT | Direct open-source model dependency | F2 narrative-body compression |
| DeerFlow ToolOutputBudgetMiddleware | Design reference only | Output budgets, externalization, synopsis, bounded retrieval concepts |
| Qwen2.5-7B-Instruct | Direct open-source model dependency | Local Agent inference |
| vLLM / vllm-ascend | Direct open-source runtime dependency | Local serving and token accounting |
| OpenAI Python SDK | Direct client-library dependency | Local vLLM-compatible HTTP protocol |
| tau-bench | Direct benchmark dependency | Environments, tools, rewards, tasks |
| LongBench / 2WikiMultihopQA | Direct benchmark-data dependency | Quality probes only |
| SQLite / JSON / CSV / HTMLParser | Python standard library | F3 persistence and deterministic parsing |

LLMLingua and DeerFlow are recorded as MIT-licensed upstream references. Their licenses must remain acknowledged in external release material.

## F2

### Direct upstream use

F2 invokes `llmlingua.PromptCompressor` from an isolated transformers-4.x process. The isolation avoids a conflict with the transformers-5.x vLLM runtime. The upstream compressor decides which narrative tokens to retain; that algorithm is not claimed as project-original.

```text
Agent middleware -> JSONL worker RPC -> llmlingua.PromptCompressor -> compressed body
```

### Project-owned work

- Agent `CompressMiddleware` integration and request transformation.
- Cold-history/hot-tail split and complete assistant-tool group boundaries.
- User message, tool-call ID/name/arguments, and critical-field protection.
- Tool-aware serialization with separate assistant and tool-body rates.
- Qwen-tokenizer exact gates, incremental recompression, and bounded count cache.
- Session cache, SHA-256 body cache, worker pool, CPU-thread control, and error flow.
- Retail static policy compaction, generic compiled policies, source hash/literal validation, fail-closed fallback, and tool-description deduplication.
- Full-prompt metering, tau-bench integration, tests, and reports.

### Evidence and boundary

| Workload | Result |
|---|---:|
| retail full115 complete Prompt | -19.32% |
| generic 28.9k-token context | -27.46% |
| retail full115 dynamic BERT activation | 0; saving was primarily static prefix |

F2 does not have a universal no-quality-loss claim. The retail full115 comparison did not strictly establish a success-rate delta within two percentage points.

## F3

### DeerFlow reference, independent implementation

DeerFlow supplied the architectural reference only. The F3 source has no DeerFlow, LangChain, or LangGraph dependency/import. The retained concepts are per-tool budgets, persistence before history growth, compact references, bounded retrieval, and explicit fallback/testing.

### Project-owned work

- `LazyLoadMiddleware` and stable `fetch_tool_result` schema.
- Exact Qwen-token gates and per-tool overrides.
- Session-scoped Memory/SQLite ArtifactStore, TTL, WAL, random IDs, and SHA-256.
- JSON, HTML, CSV/TSV, and text synopses.
- RFC 6901 JSON Pointer, line/character pagination, and 768-token response limit.
- JSON field search: `exact`, `iexact`, `contains`, and `icontains`; bounded results, match pointers, previews, title summaries, and 5 MB parse limit.
- F2+F3 ordering, raw measurement restoration, loop exemptions, tests, and reports.

### Evidence and boundary

| Measurement | Result |
|---|---:|
| synthetic full Prompt | 21,350 -> 1,499 (-92.98%) |
| vLLM server Prompt tokens | 21,371 -> 1,520 (-92.89%) |
| SQLite externalization p95 | 8.53 ms |
| bounded fetch p95 | 5.08 ms |
| exact JSON search p95 | 0.15 ms |
| controlled 360-record head/middle/tail, F3 | 3/3 |
| controlled 360-record head/middle/tail, F2+F3 | 3/3 |
| controlled cumulative Prompt tokens, F3 | 57,570 -> 8,293 (-85.59%) |

The original F3 interface scored 0/3 on the controlled locator test because it could only paginate a minified JSON result. Project-owned bounded field search fixed that specific interface limitation.

The extended 100-example LongBench 2Wiki probe found baseline/F3/F2+F3 success of 31/100, 31/100, and 27/100. F3 and F2+F3 saved 66.03% and 69.11% cumulative Prompt tokens. On 95 request-error-free baseline/F3 pairs, success was 31/95 versus 30/95 with balanced discordant outcomes (exact McNemar p=1.0), so this run found no severe F3 quality decrease. F3 increased median latency by 36.05% because it added retrieval turns.

The five-task tau-bench retail run used MIMO as the USER simulator and local Qwen/vLLM-Ascend as the Agent. F2+F3 completed it successfully, but F3 externalized zero of 27 observed results because the largest was only 1,416 tokens. This validates safe F3 no-op behavior, not F3 benefit on retail data.

## Combined Contract

```text
< 1k token tool result      inline unchanged
1k-4k token tool result     F2 tool-aware compression when eligible
>= 4k token tool result     F3 externalization before F2 sees the body
old cold narrative >= 8k    F2 cold-history compression
```

F3 stores raw data before F2 changes the request representation. F2 remains active for static system/tools, medium results, cold narrative, and old synopsis prose. F3 fetch results are capped below F2's 1k hot-tool threshold and exempt from F3, preventing processing loops.

## Reproducible Artifacts

- `docs/F2-next-session-handoff.md`
- `docs/F2-ablation-results.md`
- `docs/F3-tool-data-lazyload-report.md`
- `docs/f3-results/f3_tool_data_result.json`
- `docs/f3-results/f3_retrieval_quality_result.json`
- `docs/f3-results/f3_longbench_2wikimqa_probe.json`
- `docs/f3-results/f3_longbench_2wikimqa_first20_probe.json`
- `docs/f3-results/f3_longbench_2wikimqa_first100_probe.json`
- `docs/f3-results/f3_longbench_2wikimqa_first100_summary.json`
- `docs/f3-results/f2_f3_taubench_mimo_first5.json`
- `docs/F2-F3-extended-evaluation-20260725.md`
