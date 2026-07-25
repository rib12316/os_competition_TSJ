# Comparison: mvp-three-tier

- date: 2026-07-22
- baseline: baseline
- thresholds: mem_peak >= 30% down, e2e_latency_p50 >= 20% down, task_success_rate <= 2pp diff

## median metrics per config

| config | mem_peak_mb | e2e_latency_p50_ms | e2e_latency_p95_ms | qps | kv_cache_hit_rate | task_success_rate | ttft_ms |
|---|---|---|---|---|---|---|---|
| baseline-logged | 57978.00 | 145621.80 | 237582.84 | 0.01 | 0.94 | 0.33 | 67.48 |
| f2-compress | 57979.00 | 109282.82 | 164064.37 | 0.01 | 0.93 | 0.17 | 64.13 |

_(未找到 baseline 档 `baseline`，跳过判定。)_
