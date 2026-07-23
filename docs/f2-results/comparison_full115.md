# Comparison: mvp-three-tier

- date: 2026-07-23
- baseline: baseline
- thresholds: mem_peak >= 30% down, e2e_latency_p50 >= 20% down, task_success_rate <= 2pp diff

## median metrics per config

| config | mem_peak_mb | e2e_latency_p50_ms | e2e_latency_p95_ms | qps | kv_cache_hit_rate | task_success_rate | ttft_ms |
|---|---|---|---|---|---|---|---|
| baseline-logged | 57954.00 | 106894.93 | 161034.79 | 0.03 | 0.96 | 0.20 | 73.20 |
| f2-compress | 57955.00 | 123774.13 | 194324.01 | 0.03 | 0.95 | 0.23 | 78.30 |

_(未找到 baseline 档 `baseline`，跳过判定。)_
