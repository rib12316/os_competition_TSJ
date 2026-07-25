# F2/F3 Extended Evaluation

- Date: 2026-07-25
- Agent inference: local Qwen2.5-7B-Instruct served by the same vLLM-Ascend instance
- Server settings: `max_model_len=32768`, `gpu_memory_utilization=0.92`, Hermes tool parser
- Scope: LongBench 2Wiki first 20 examples and tau-bench retail first 5 tasks

## Purpose

This evaluation separates two questions:

1. Does externalizing long structured tool data reduce tokens while preserving answer quality on a larger multi-hop sample?
2. Can the F2+F3 middleware stack execute real tau-bench retail tasks with the required MIMO user simulator?

It is not a statistical benchmark. Each variant has one run, and MIMO user trajectories are non-deterministic.

## LongBench 2Wiki First 20

The first 20 consecutive examples from `data/2wikimqa.jsonl` were run with the same local Qwen Agent. Each task exposes its documents as a JSON tool result. Baseline receives inline documents; F3 and F2+F3 receive the F3 reference and may use bounded title/field fetches. The Agent has at most 8 tool-calling steps and 256 output tokens per request.

| Variant | Correct | Cumulative Prompt tokens | Saving vs baseline | Median wall time |
|---|---:|---:|---:|---:|
| baseline | 6/20 | 267,911 | - | 1,108.76 ms |
| F3 | 4/20 | 97,753 | 63.51% | 1,554.98 ms |
| F2 + F3 | 4/20 | 83,486 | 68.84% | 1,595.39 ms |

One baseline task (example 15) made two large inline retrieval calls and exceeded the 32,768-token server limit on its second request. F3 and F2+F3 completed that request sequence but gave the wrong final answer. Excluding that context-overflow baseline failure leaves 19 directly comparable tasks: baseline 6/19 versus F3 4/19 and F2+F3 4/19, a -10.53 percentage-point result for both optimized variants.

The paired changes are transparent in the raw artifact:

- Example 3: baseline wrong; F3 and F2+F3 correct.
- Examples 8, 10, and 19: baseline correct; F3 and F2+F3 wrong.

Therefore this sample shows a real quality-risk signal. The current F3 title/field retrieval interface reduces Prompt tokens substantially, but local Qwen-7B still frequently stops after the first evidence hop or selects the wrong document. F2 did not add another observed success drop beyond F3 in this sample.

The higher wall time is expected for F3 because each lookup may add model tool turns. The token saving is not a claim of lower end-to-end latency for multi-hop retrieval workloads.

## tau-bench Retail with MIMO USER

The first five `retail/test` tasks (indices 0-4) were run once per variant, with the required MIMO user simulator. The Agent model was always local Qwen through the same vLLM-Ascend server. The `baseline` metrics file labels its engine as `vllm` because that is the preset metadata; it did not run on a different server or hardware backend.

| Metric | baseline | F2 + F3 |
|---|---:|---:|
| task success rate | 1/5 (20%) | 2/5 (40%) |
| e2e p50 | 148.48 s | 67.31 s |
| e2e p95 | 187.81 s | 116.86 s |
| QPS | 0.00615 | 0.01167 |
| TTFT median | 94.47 ms | 106.85 ms |
| server Prompt tokens across run | 686,290 | 267,463 |
| model calls | 89 | 48 |

The two MIMO conversations took different paths, so the aggregate token and latency differences are observations, not a causal A/B attribution. F2+F3's internal paired meter recorded 302,142 original Prompt tokens and 267,806 transformed Prompt tokens in its own trajectory, saving 34,336 tokens (11.36%).

### Middleware telemetry

| Middleware observation | Result |
|---|---:|
| F2+F3 request transforms | 48 |
| static retail system/tool compactions | 48 |
| F2 hot tool-body compressions | 8 |
| F2 cold-history compressions | 0 |
| F3 observed tool results | 27 |
| F3 externalizations | 0 |
| largest observed tool result | 1,416 Qwen tokens |

F3 did not trigger because all observed retail tool results were below its 4,000-token externalization threshold. Thus this tau-bench slice validates that F2+F3 can run against the real environment with MIMO USER and that F3 safely remains a no-op when inappropriate. It does not demonstrate F3 large-tool-result benefit; the LongBench and controlled JSON workloads provide that evidence.

## Conclusions

- F2+F3 is operationally compatible with real tau-bench retail/MIMO interactions.
- Standard retail tau-bench is primarily an F2/static-prefix and small-result workload; it is not representative of F3's target regime.
- On 20 real 2Wiki examples, F3/F2+F3 reduced cumulative Prompt tokens by 63.51%/68.84% but showed a quality decrease relative to inline baseline. Do not claim quality neutrality for multi-hop document QA with Qwen2.5-7B.
- The next quality experiment should use a larger multi-hop sample with an Agent/model that has a stronger inline baseline, then compare success under matched retrieval traces or multiple MIMO/user seeds.

## Raw Results

- `docs/f3-results/f3_longbench_2wikimqa_first20_probe.json`
- `docs/f3-results/f2_f3_taubench_mimo_first5.json`
