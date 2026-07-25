# Comparison: mvp-three-tier

- date: 2026-07-23
- baseline: baseline
- thresholds: mem_peak >= 30% down, e2e_latency_p50 >= 20% down, task_success_rate <= 2pp diff

## median metrics per config

| config | mem_peak_mb | e2e_latency_p50_ms | e2e_latency_p95_ms | qps | kv_cache_hit_rate | task_success_rate | ttft_ms |
|---|---|---|---|---|---|---|---|
| baseline-logged | 57955.00 | 99513.71 | 156074.88 | 0.04 | 0.96 | 0.23 | 72.84 |
| f2-compress | 57954.00 | 109133.90 | 172402.63 | 0.03 | 0.95 | 0.22 | 76.11 |

_(未找到 baseline 档 `baseline`，跳过判定。)_
