# F3 Tool Result Lazy-Load: Implementation and Quality Report

- Date: 2026-07-24
- Branch: `feat/f3-tool-data-lazyload`
- Base: F2 checkpoint `1e64bc8`
- Inference: local Qwen2.5-7B-Instruct served by vllm-ascend

## Result

F3 externalizes oversized tool results before they enter Agent history. The model
sees a deterministic structured reference and can call the local bounded
`fetch_tool_result` tool. The implementation now supports bounded JSON field search,
which was required by end-to-end head/middle/tail quality tests.

No OpenAI-hosted inference service is used. The Python SDK is only the client for
vLLM's compatible local HTTP endpoint.

| Metric | Gate | Actual |
|---|---:|---:|
| Full Prompt saving, tokenizer replay | at least 70% | 92.98% |
| Reference size | at most 512 token | at most 222 |
| Fetch/search response | at most 768 token | at most 768 / 25 |
| SQLite externalize p95 | at most 50 ms | 8.53 ms |
| Fetch p95 | at most 20 ms | 5.08 ms |
| Exact JSON search p95 | at most 20 ms | 0.15 ms |
| Below-trigger p95 | at most 5 ms | 0.08 ms |
| Controlled lookup correctness | all head/middle/tail | 3/3 |
| Controlled F2+F3 correctness | all head/middle/tail | 3/3 |
| Regression | all pass | 245 passed |

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
  -> JSON Pointer / line / character / bounded field search
  -> hard 768-token response cap
  -> tool result enters hot tail; tau-bench env.step is not called
```

The stable fetch schema is injected from the first request whenever F3 is enabled,
so the tools prefix does not change after the first externalized result.

## Structured search added after quality testing

The first real Qwen retrieval test exposed a concrete failure. With a minified
18,658-token JSON array, baseline answered head/middle/tail records correctly, but
the original F3 implementation scored 0/3. Qwen called `fetch_tool_result` in all
three cases, received only the beginning of the one-line JSON, and returned
`CASE-0000` instead of the requested record. Storage and fetch activation worked;
the retrieval interface could not locate a record by ID.

`fetch_tool_result` therefore gained a restricted selector:

- `json_pointer` selects an array, or may be omitted when exactly one searchable
  top-level array exists;
- `match_field` selects one direct object field;
- `match_value` is a string value;
- `match_mode` is one of `exact`, `iexact`, `contains`, or `icontains`;
- `max_matches` is capped at 10;
- JSON search is rejected above the configured 5 MB parse bound;
- search results include stable RFC 6901 pointers to matched records;
- oversized matching records return structurally valid bounded previews rather than
  partial JSON objects;
- JSON synopses advertise array pointers, item keys, and at most 12 document titles.

There is no regex, expression evaluator, JSONPath engine, or arbitrary query code.
All search responses retain the existing session ACL, integrity check, and 768-token
hard limit.

## Controlled end-to-end quality

The benchmark uses a single 360-record JSON result containing 18,658 Qwen tokens.
The requested records are at indexes 3, 180, and 356. The local Qwen model must call
the business tool, recognize the externalized reference, construct the selector,
fetch the record, and return its exact verification code. The benchmark never
injects the correct JSON pointer or answer.

| Variant | Correct | Cumulative Prompt token | Saving vs baseline | Fetch calls |
|---|---:|---:|---:|---:|
| baseline | 3/3 | 57,570 | - | 0 |
| F3 | 3/3 | 8,293 | 85.59% | 3 |
| F2 + F3 | 3/3 | 8,295 | 85.59% | 3 |

Every F3 task used exactly one search fetch. The raw tool arguments show that Qwen
generated the requested `case_id` selector itself. These are controlled exact-ID
lookups, not evidence that arbitrary long-tool reasoning is quality-neutral.

## Exploratory LongBench probe

Twenty consecutive unmodified `2wikimqa` examples were parsed into JSON document
arrays. The reference exposed document titles but not bodies, answers, or supporting
facts. Qwen was instructed to complete two evidence hops. All variants used the same
local Qwen/vLLM-Ascend service with a 32,768-token context limit.

| Variant | Correct | Cumulative Prompt token | Saving vs baseline | Fetch calls |
|---|---:|---:|---:|---:|
| baseline | 6/20 | 267,911 | - | 0 |
| F3 | 4/20 | 97,753 | 63.51% | 27 |
| F2 + F3 | 4/20 | 83,486 | 68.84% | 23 |

One baseline task overflowed the 32,768-token context after two inline retrievals.
Across the remaining 19 directly comparable tasks, baseline was 6/19 and F3/F2+F3
were 4/19. F3 often found a first-hop document but returned the intermediate entity
instead of fetching the second document. This is a quality-risk signal for local
Qwen-7B multi-hop retrieval, not a quality-neutrality claim. See the extended report
for paired task differences and the MIMO tau-bench compatibility probe.

## F2 interaction

The combined order remains `active: [lazyload, compress]`.

- Results below 1k tokens remain unchanged.
- Medium results can use F2 hot tool-aware compression.
- Results at least 4k tokens are externalized by F3.
- F2 still compresses static system/tools and accumulated cold synopsis prose.
- Fetch/search responses are capped below F2's 1k hot-tool gate and are exempt from F3.
- `result_id` and all assistant tool-call arguments remain protected by F2.

On the synthetic long JSON trace, F2-only transformation took 6.64 seconds because
LLMLingua-2 processed the large bodies. F2+F3 transformed the short references in
3.24 ms; F3 externalization itself was 8.53 ms p95 per result.

## Strict paired token result

All stages use the same deterministic trace, Qwen tokenizer, complete chat template,
and tool schema.

| Stage | Full Prompt token | Difference from baseline |
|---|---:|---:|
| baseline | 21,350 | - |
| F2 only | 17,075 | -20.02% |
| F3 only | 1,499 | -92.98% |
| F2 + F3 | 1,499 | -92.98% |

The previous 996-token result predated structured search and title-aware synopses.
The larger schema/reference cost 503 tokens on this trace but kept the saving above
the 70% gate and enabled the controlled lookup tasks to recover the correct data.

## vLLM verification

Three cache-busted rounds alternated request order. Server-reported usage was 21,371
tokens for baseline and 1,520 for F3 in every round, a 92.89% decrease. Wall time p50
was 1,822.29 ms vs 81.23 ms. Requests generated only one output token, so this
comparison primarily measures prefill; it does not claim decode acceleration.

## Reproduce

```bash
cd /tmp/f3-wt/agent-mem

PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python benchmarks/f3_long_tool_benchmark.py \
  --model Qwen2.5-7B-Instruct --served-model Qwen2.5-7B-Instruct \
  --results 3 --result-tokens 5000 --perf-runs 20 \
  --engine-url http://127.0.0.1:8000/v1 --vllm-rounds 3 \
  --output ../docs/f3-results/f3_tool_data_result.json

PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python \
  benchmarks/f3_retrieval_quality_benchmark.py \
  --records 360 --variants baseline f3 f2_f3 \
  --output ../docs/f3-results/f3_retrieval_quality_result.json

PYTHONPATH=src HF_HOME=/data/huggingface_home \
  /data/os_competition_TSJ/.venv/bin/python \
  benchmarks/f3_longbench_quality_benchmark.py \
  --data-zip /tmp/longbench-data.zip --start 0 --limit 3 \
  --variants baseline f3 f2_f3 \
  --output ../docs/f3-results/f3_longbench_2wikimqa_probe.json

PYTHONPATH=src /data/os_competition_TSJ/.venv/bin/python -m pytest -q tests
/data/os_competition_TSJ/.venv/bin/ruff check src/agent_mem tests benchmarks
```

## Boundaries

- Controlled ID lookup now preserves correctness while reducing cumulative Prompt
  tokens, but arbitrary semantic retrieval and multi-hop quality are not proven.
- The 20-example LongBench probe shows a Qwen-7B multi-hop quality risk; it is still
  too small and model-specific for a general F3 success-rate claim.
- The MIMO tau-bench retail probe validates F2+F3 compatibility, but F3 did not
  trigger because its largest observed tool result was 1,416 tokens.
- Retail full115 has no hot tool result above 1k tokens and remains a no-trigger
  regression workload rather than an F3 benefit workload.
- vLLM preallocates its KV pool, so these results prove fewer prompt/KV tokens and
  lower paired prefill, not lower process-level HBM peak.
- SQLite is local-process storage. Distributed workers need a shared backend while
  preserving the session-scoped interface.

Raw results:

- `docs/f3-results/f3_tool_data_result.json`
- `docs/f3-results/f3_retrieval_quality_before_search.json`
- `docs/f3-results/f3_retrieval_quality_result.json`
- `docs/f3-results/f3_longbench_2wikimqa_probe.json`
- `docs/f3-results/f3_longbench_2wikimqa_first20_probe.json`
- `docs/f3-results/f2_f3_taubench_mimo_first5.json`
- `docs/F2-F3-extended-evaluation-20260725.md`
