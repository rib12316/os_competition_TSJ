"""三档对照 + comparison.md 生成测试。"""

from __future__ import annotations

import json
from pathlib import Path

from agent_mem.bench.compare import (
    compare_runs,
    compute_deltas,
    find_run_dirs,
    load_run_metrics,
    median_by_config,
    render_comparison_md,
    write_comparison_md,
)


def _metrics(config: str, *, mem: int, lat: float, success: float) -> dict:
    return {
        "run_id": f"r_{config}",
        "engine": "vllm",
        "model": "qwen25-7b",
        "config": config,
        "e2e_latency_p50_ms": lat,
        "e2e_latency_p95_ms": lat * 1.6,
        "qps": 1.0,
        "mem_peak_mb": mem,
        "kv_cache_hit_rate": 0.1,
        "task_success_rate": success,
        "ttft_ms": 100.0,
        "seed": 42,
        "started_at": "2026-07-19T00:00:00",
    }


def _make_run(root: Path, name: str, m: dict) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    return d


# ---- 加载 + 中位数 ----


def test_load_and_median(tmp_path):
    _make_run(tmp_path, "20260719-000000_vllm_qwen25-7b_baseline_run1",
              _metrics("baseline", mem=10000, lat=500, success=0.85))
    _make_run(tmp_path, "20260719-000001_vllm_qwen25-7b_baseline_run2",
              _metrics("baseline", mem=11000, lat=520, success=0.85))
    metrics = load_run_metrics(find_run_dirs(tmp_path))
    medians = median_by_config(metrics)
    assert "baseline" in medians
    # 两 run 中位数：mem 10000/11000 -> 10500
    assert medians["baseline"]["mem_peak_mb"] == 10500


def test_find_run_dirs_config_filter(tmp_path):
    _make_run(tmp_path, "x_baseline_run1", _metrics("baseline", mem=1, lat=1, success=1))
    _make_run(tmp_path, "x_prefix-cache_run1", _metrics("prefix-cache", mem=1, lat=1, success=1))
    assert len(find_run_dirs(tmp_path)) == 2
    assert len(find_run_dirs(tmp_path, config="baseline")) == 1


# ---- 判定逻辑 ----


def test_compute_deltas_pass_at_threshold():
    b = {"mem_peak_mb": 10000.0, "e2e_latency_p50_ms": 500.0, "task_success_rate": 0.85}
    t = {"mem_peak_mb": 7000.0, "e2e_latency_p50_ms": 400.0, "task_success_rate": 0.84}
    rows = {r["metric"]: r for r in compute_deltas(b, t)}
    # mem: (10000-7000)/10000 = 30% -> 恰好达标 PASS
    assert rows["mem_peak_mb"]["verdict"] == "PASS"
    # latency: (500-400)/500 = 20% -> 恰好达标 PASS
    assert rows["e2e_latency_p50_ms"]["verdict"] == "PASS"
    # success diff = (0.85-0.84)*100 = 1pp <= 2 -> PASS
    assert rows["task_success_rate"]["verdict"] == "PASS"


def test_compute_deltas_fail_on_success_regression():
    b = {"mem_peak_mb": 10000.0, "e2e_latency_p50_ms": 500.0, "task_success_rate": 0.85}
    t = {"mem_peak_mb": 5000.0, "e2e_latency_p50_ms": 300.0, "task_success_rate": 0.80}
    rows = {r["metric"]: r for r in compute_deltas(b, t)}
    assert rows["mem_peak_mb"]["verdict"] == "PASS"  # 50% down
    assert rows["task_success_rate"]["verdict"] == "FAIL"  # 5pp worse


def test_compute_deltas_na_when_baseline_zero():
    b = {"mem_peak_mb": 0, "e2e_latency_p50_ms": 0, "task_success_rate": 0.0}
    t = {"mem_peak_mb": 100, "e2e_latency_p50_ms": 10, "task_success_rate": 0.5}
    rows = {r["metric"]: r for r in compute_deltas(b, t)}
    assert rows["mem_peak_mb"]["verdict"] == "N/A"
    assert rows["e2e_latency_p50_ms"]["verdict"] == "N/A"
    # success diff = (0 - 0.5)*100 = -50pp -> target 更好 -> PASS
    assert rows["task_success_rate"]["verdict"] == "PASS"


# ---- 渲染 + 写盘 ----


def test_render_comparison_md_ascii_and_verdicts():
    medians = {
        "baseline": {"mem_peak_mb": 10000.0, "e2e_latency_p50_ms": 500.0, "task_success_rate": 0.85},
        "prefix-cache": {"mem_peak_mb": 7000.0, "e2e_latency_p50_ms": 400.0, "task_success_rate": 0.84},
        "optimized": {"mem_peak_mb": 5000.0, "e2e_latency_p50_ms": 300.0, "task_success_rate": 0.80},
    }
    md = render_comparison_md("mvp-three-tier", medians)
    assert md.isascii()  # 全 ASCII（日志规范）
    assert "[PASS]" in md
    assert "[FAIL]" in md  # optimized 成功率 FAIL
    assert "all thresholds met: NO" in md
    assert "prefix-cache vs baseline" in md


def test_write_comparison_md_filename(tmp_path):
    medians = {"baseline": {"mem_peak_mb": 10000.0, "e2e_latency_p50_ms": 500.0,
                            "task_success_rate": 0.85}}
    p = write_comparison_md(tmp_path, "mvp-three-tier", medians)
    assert p.parent.name == "_summaries"
    # 文件名形如 20260719_mvp-three-tier_comparison.md
    assert p.name.endswith("_mvp-three-tier_comparison.md")
    assert p.name[:8].isdigit()


def test_compare_runs_end_to_end(tmp_path):
    _make_run(tmp_path, "x_baseline_run1", _metrics("baseline", mem=10000, lat=500, success=0.85))
    _make_run(tmp_path, "x_prefix-cache_run1",
              _metrics("prefix-cache", mem=7000, lat=400, success=0.84))
    p = compare_runs(find_run_dirs(tmp_path), study="mvp-three-tier", log_root=tmp_path)
    assert p.exists()
    text = p.read_text()
    assert "[PASS]" in text
    assert "all thresholds met: YES" in text
