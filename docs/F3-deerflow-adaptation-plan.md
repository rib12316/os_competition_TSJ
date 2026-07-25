# F3 Tool Result Lazy-Load: DeerFlow Adaptation and Acceptance Plan

- Date: 2026-07-24
- Branch: `feat/f3-tool-data-lazyload`
- Base: F2 checkpoint `1e64bc8`
- Inference path: local Qwen served by `vllm-ascend`; the Python client only uses
  vLLM's OpenAI-compatible HTTP protocol.

## Scope

F3 externalizes oversized tool results before they enter canonical agent history.
The model receives a compact, structured reference and may retrieve a bounded slice
through the stable `fetch_tool_result` tool. F3 runs in the Agent middleware layer;
it does not replace or modify vLLM.

The implementation adapts these DeerFlow mechanisms without importing DeerFlow,
LangChain, or LangGraph:

- per-tool output budgets and exemptions;
- full-result persistence with a compact deterministic synopsis;
- bounded on-demand retrieval;
- safe fallback and comprehensive edge-case tests.

## F2 + F3 Contract

The combined middleware order is `lazyload` then `compress`:

1. F3 injects one stable fetch schema and externalizes large results.
2. F2 optimizes the resulting stable system/tools prefix.
3. F2 may compress old synopsis prose, but `result_id`, status, tool name, numeric
   metadata, and fetch arguments must remain byte-identical.
4. Fetch responses are capped below F2's 1,000-token hot-tool threshold and are
   exempt from F3 externalization, preventing recompression/externalization loops.

Recommended size tiers:

| Result size | Action |
|---|---|
| below 1,000 tokens | inline, unchanged |
| 1,000-4,000 tokens | F2 hot tool-aware compression when F2 is enabled |
| at least 4,000 tokens | F3 externalization |
| accumulated cold narrative at least 8,000 tokens | F2 cold-history compression |

## Frozen Acceptance Gates

These gates are defined before implementation tests are run.

### Correctness and safety

- Stored content is restored byte-for-byte; SHA-256 matches in 100% of cases.
- Cross-session fetches and unknown IDs return a generic not-found result.
- Every fetch response is at most 768 Qwen tokens, including metadata.
- F3 never leaves an orphan tool result or changes the original tool-call ID.
- After forced F2 cold compression, every F3 `result_id` and fetch argument remains
  byte-identical.
- `fetch_tool_result` is always present while F3 is enabled and is never
  re-externalized.
- With F3 disabled, existing F2 behavior and test results are unchanged.

### Token effectiveness

- On a deterministic long-tool workload, full Prompt tokens after F3 decrease by
  at least 70% versus the same messages with inline raw results. This comparison
  includes the added fetch-tool schema.
- `F2 + F3` must not use more Prompt tokens than F3 alone on the long workload.
- The compact reference itself is at most 512 Qwen tokens.

### Local overhead

- SQLite externalization p95 is at most 50 ms for the benchmark payloads.
- Bounded fetch p95 is at most 20 ms.
- A below-threshold result adds at most 5 ms p95 middleware overhead.

### vLLM protocol verification

- The current local vLLM endpoint accepts the stable fetch schema and structured
  tool-result reference without a 4xx response.
- Server-reported `usage.prompt_tokens` is lower for the F3 request than for the
  paired inline request. Wall-clock latency is reported, but one small run is not
  treated as a statistically strong latency claim.

## Required Comparisons

The benchmark reports four stages using the same deterministic trace and tokenizer:

1. baseline: raw inline tool results;
2. F2 only;
3. F3 only;
4. F2 + F3.

F3's measurement baseline expands stored references only in a measurement copy;
the short canonical history sent to vLLM remains unchanged.

