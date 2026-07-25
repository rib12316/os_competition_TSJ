# F2/F3 Extended Evaluation

- Date: 2026-07-25
- Agent inference: local Qwen2.5-7B-Instruct served by the same vLLM-Ascend instance
- Server settings: `max_model_len=32768`, `gpu_memory_utilization=0.92`, Hermes tool parser
- Scope: LongBench 2Wiki first 100 examples and tau-bench retail first 5 tasks

## Purpose

This evaluation separates two questions:

1. Does externalizing long structured tool data reduce tokens while preserving answer quality on a larger multi-hop sample?
2. Can the F2+F3 middleware stack execute real tau-bench retail tasks with the required MIMO user simulator?

It is not a statistical benchmark. Each variant has one run, and MIMO user trajectories are non-deterministic.

## LongBench 2Wiki First 100

The first 100 consecutive examples from `data/2wikimqa.jsonl` were run with the same local Qwen Agent. Each task exposes its documents as a JSON tool result. Baseline receives inline documents; F3 and F2+F3 receive the F3 reference and may use bounded title/field fetches. The Agent has at most 8 tool-calling steps and 256 output tokens per request.

| Variant | Correct | Cumulative Prompt tokens | Saving vs baseline | Median wall time |
|---|---:|---:|---:|---:|
| baseline | 31/100 | 1,632,665 | - | 1,269.00 ms |
| F3 | 31/100 | 554,657 | 66.03% | 1,726.44 ms |
| F2 + F3 | 27/100 | 504,313 | 69.11% | 1,660.37 ms |

Five baseline tasks made repeated inline retrieval calls and failed at the 32,768-token server limit. F3 and F2+F3 had no model-request errors because large results were externalized.

Pairwise comparison excludes request-error tasks for each pair:

| Pair | Common tasks | Correct A/B | A-only / B-only correct | Exact McNemar p |
|---|---:|---:|---:|---:|
| baseline vs F3 | 95 | 31 / 30 | 14 / 13 | 1.000 |
| baseline vs F2+F3 | 95 | 31 / 27 | 16 / 12 | 0.572 |
| F3 vs F2+F3 | 100 | 31 / 27 | 7 / 3 | 0.344 |

The first-20 subset's apparent F3 decline did not persist in the 100-example run. F3's overall success equaled baseline, and on the 95 directly comparable tasks it differed by one answer with balanced discordant pairs. There is no evidence here of a severe or statistically significant F3 success-rate decrease.

F2+F3 was four answers below F3 and baseline in the point estimate, but neither paired comparison was significant at this sample size. This benchmark disables F2's business-specific static Prompt compaction to isolate dynamic history behavior.

F3 added model turns: baseline used 240 steps, F3 368, and F2+F3 355. Median wall time increased by 36.05% for F3 and 30.84% for F2+F3. Thus F3 trades Prompt/KV tokens and context-overflow avoidance for extra Agent iterations on multi-hop retrieval workloads.

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
- On 100 real 2Wiki examples, F3 reduced cumulative Prompt tokens by 66.03% and matched baseline's overall 31% success. The paired common-task result was 30/95 versus 31/95 (`p=1.0`).
- F3 did not show a severe quality decrease at this scale, but it increased median latency by 36.05% and changed which individual tasks succeeded.
- F2+F3 reduced tokens by 69.11% and scored 27%; its 4pp point decrease was not significant but remains a follow-up risk.
- The next experiment should repeat the 100-example matrix or use the full 200 examples before making a production quality-neutrality claim.

## Raw Results

- `docs/f3-results/f3_longbench_2wikimqa_first100_probe.json`
- `docs/f3-results/f3_longbench_2wikimqa_first100_summary.json`
- `docs/f3-results/f2_f3_taubench_mimo_first5.json`
