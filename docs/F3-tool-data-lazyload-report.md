# F3 Tool Result Lazy-Load: Implementation and Benchmark Report

- Date: 2026-07-24
- Branch: `feat/f3-tool-data-lazyload`
- Base: F2 checkpoint `1e64bc8`
- Upstream design reference: ByteDance DeerFlow `ToolOutputBudgetMiddleware` (MIT)
- Inference: local Qwen2.5-7B-Instruct served by vllm-ascend

## Result

F3 now externalizes oversized tool results before they enter Agent history. The
model sees a deterministic structured reference and can call the local bounded
`fetch_tool_result` tool. No OpenAI-hosted inference service is used; the Python
SDK is only the client for vLLM's compatible HTTP endpoint.

All acceptance gates frozen in `F3-deerflow-adaptation-plan.md` passed.

| Metric | Gate | Actual |
|---|---:|---:|
| Full Prompt saving, tokenizer replay | at least 70% | 95.33% |
| Reference size | at most 512 token | at most 120 |
| Fetch response | at most 768 token | at most 768 |
| SQLite externalize p95 | at most 50 ms | 8.41 ms |
| Fetch p95 | at most 20 ms | 5.09 ms |
| Below-trigger p95 | at most 5 ms | 0.08 ms |
| F2+F3 Prompt no larger than F3 | required | 996 = 996 |
| Regression | all pass | 237 passed |

## Architecture

```text
business tool result
  -> LazyLoadMiddleware exact Qwen-token gate
  -> session-scoped SQLite/Memory ArtifactStore
  -> compact JSON synopsis + opaque result_id in canonical history
  -> F2 static/tool-aware transforms
  -> local vLLM request

model fetch_tool_result call
  -> MiddlewareStack.handle_internal_tool_call
  -> session ACL + SHA-256 integrity check
  -> JSON Pointer / line / character selection
  -> hard 768-token response cap
  -> tool result enters hot tail; tau-bench env.step is not called
```

The stable fetch schema is injected from the first request whenever F3 is enabled,
so the tools prefix does not change after the first externalized result.

## F2 interaction

The combined order is `active: [lazyload, compress]`.

- Results below 1k tokens remain unchanged.
- Medium results can use F2 hot tool-aware compression.
- Results at least 4k tokens are externalized by F3.
- F2 still compresses static system/tools and accumulated cold synopsis prose.
- `result_id`, status, tool name, numeric metadata, and fetch arguments use keys
  covered by F2's critical-field protection.
- Fetch responses are capped below F2's 1k hot-tool gate and are exempt from F3,
  preventing processing loops.
- Prompt metering expands references only in a measurement copy, so baseline vs
  transformed counts include F3 savings while canonical history remains short.

On the long JSON trace, F2-only request transformation took 6.53 seconds because
LLMLingua-2 ran on the large bodies. After F3 externalization, the F2+F3 request
transform took 1.92 ms; F3 externalization itself was 8.41 ms p95 per result.

## Strict paired token result

The deterministic trace contains three tool results targeting 5,000 tokens each.
All stages use the same Qwen tokenizer and complete chat template, including tools.

| Stage | Full Prompt token | Difference from baseline |
|---|---:|---:|
| baseline | 21,350 | - |
| F2 only | 17,075 | -20.02% |
| F3 only | 996 | -95.33% |
| F2 + F3 | 996 | -95.33% |

## vLLM verification

Three cache-busted rounds alternated request order. Server-reported usage was
21,371 tokens for baseline and 1,017 for F3 in every round, a 95.24% decrease.
Wall time p50 was 1,815.32 ms vs 60.14 ms. Requests generated only one output token,
so this comparison primarily measures prefill; it does not claim decode acceleration.

## Reproduce

```bash
cd /tmp/f3-wt/agent-mem

PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/f3_long_tool_benchmark.py \
  --model Qwen2.5-7B-Instruct --served-model Qwen2.5-7B-Instruct \
  --results 3 --result-tokens 5000 --perf-runs 20 \
  --engine-url http://127.0.0.1:8000/v1 --vllm-rounds 3 \
  --output /tmp/f3-result.json

PYTHONPATH=src /data/os_competition_TSJ/.venv/bin/python -m pytest -q tests
/data/os_competition_TSJ/.venv/bin/ruff check src/agent_mem tests benchmarks
```

## Boundaries

- The token and vLLM results use a deterministic synthetic long-tool trace, not
  tau-bench task success evaluation.
- Retail full115 has no hot tool result above 1k tokens, so it is not a meaningful
  F3 benefit workload; it remains useful for no-trigger regression.
- vLLM preallocates its KV pool, so these results do not by themselves prove lower
  process-level HBM peak. They prove fewer prompt/KV tokens and lower paired prefill.
- SQLite is local-process storage. Distributed Agent workers would need a shared
  store backend while preserving the same session-scoped interface.

Raw result: `docs/f3-results/f3_tool_data_result.json`.
