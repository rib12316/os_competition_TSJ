# Comparison: mvp-three-tier

- date: 2026-07-22
- baseline: baseline
- thresholds: mem_peak >= 30% down, e2e_latency_p50 >= 20% down, task_success_rate <= 2pp diff

## median metrics per config

| config | mem_peak_mb | e2e_latency_p50_ms | e2e_latency_p95_ms | qps | kv_cache_hit_rate | task_success_rate | ttft_ms |
|---|---|---|---|---|---|---|---|
| baseline-logged | 57957.00 | 12125.65 | 55679.90 | 0.04 | 0.93 | 0.00 | 53.11 |
| f2-compress | 57956.00 | 33784.24 | 72831.10 | 0.03 | 0.93 | 0.00 | 53.69 |

_(未找到 baseline 档 `baseline`，跳过判定。)_
