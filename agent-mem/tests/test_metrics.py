"""metrics.json schema/序列化测试（字段顺序严格对齐日志规范）。"""

from __future__ import annotations

import json

from agent_mem.metrics import METRIC_FIELDS, RunMetrics, default_metrics, to_json, write

# log-naming-convention.md 里 metrics.json schema 的字段顺序（权威基准）
_EXPECTED_FIELDS = (
    "run_id",
    "engine",
    "model",
    "config",
    "e2e_latency_p50_ms",
    "e2e_latency_p95_ms",
    "qps",
    "mem_peak_mb",
    "kv_cache_hit_rate",
    "task_success_rate",
    "ttft_ms",
    "seed",
    "started_at",
)


def test_metric_fields_order_matches_convention():
    assert METRIC_FIELDS == _EXPECTED_FIELDS


def test_default_metrics_all_zero():
    m = default_metrics(
        run_id="20260718-143022_vllm_qwen25-7b_baseline_run1",
        engine="vllm",
        model="qwen25-7b",
        config="baseline",
        seed=42,
        started_at="2026-07-18T14:30:22",
    )
    assert m.e2e_latency_p50_ms == 0.0
    assert m.mem_peak_mb == 0
    assert m.task_success_rate == 0.0


def test_to_json_has_all_fields_in_order():
    m = default_metrics(run_id="r", engine="vllm", model="qwen25-7b", config="baseline")
    obj = json.loads(to_json(m))
    assert list(obj.keys()) == list(_EXPECTED_FIELDS)


def test_to_json_roundtrip():
    m = RunMetrics(
        run_id="r",
        engine="sglang",
        model="minicpm3-4b",
        config="fp8-kv",
        e2e_latency_p50_ms=123.4,
        mem_peak_mb=8000,
        task_success_rate=0.97,
        ttft_ms=42.0,
        seed=7,
        started_at="2026-07-18T14:30:22",
    )
    obj = json.loads(to_json(m))
    again = RunMetrics(**obj)
    assert again == m


def test_to_json_is_ascii_safe_for_meta():
    # run_id 等元信息应保证 ASCII（命名规范要求）
    m = default_metrics(
        run_id="20260718-143022_vllm-ascend_qwen25-7b_prefix-cache_run1",
        engine="vllm-ascend",
        model="qwen25-7b",
        config="prefix-cache",
    )
    assert to_json(m).isascii()


def test_write_to_file(tmp_path):
    m = default_metrics(run_id="r", engine="vllm", model="qwen25-7b", config="baseline")
    p = write(m, tmp_path / "metrics.json")
    obj = json.loads(p.read_text())
    assert obj["run_id"] == "r"
    assert list(obj.keys()) == list(_EXPECTED_FIELDS)
