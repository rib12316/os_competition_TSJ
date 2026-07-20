"""vLLM metrics 抓取器测试（纯解析 + 对 stub server 联调）。"""

from __future__ import annotations

import json

import pytest

from agent_mem.bench import vllm_metrics
from agent_mem.server import stub_openai

# ---- 纯解析 ----


_FIXTURE = """\
# HELP vllm:prefix_cache_hits_total Prefix cache hits.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{model="qwen"} 80
vllm:prefix_cache_hits_total{model="sglang"} 20
# HELP vllm:prefix_cache_queries_total Prefix cache queries.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total 200
vllm:something_else 5
"""


def test_parse_strips_labels_and_comments():
    samples = vllm_metrics.parse_prometheus(_FIXTURE)
    names = [s.name for s in samples]
    assert "vllm:prefix_cache_hits_total" in names
    assert "vllm:something_else" in names
    # 注释行被忽略
    assert all(not n.startswith("#") for n in names)


def test_metric_sum_across_labels():
    # 80 + 20 = 100（两行同指标不同 label 求和）
    assert vllm_metrics.metric_sum(_FIXTURE, "vllm:prefix_cache_hits") == 100.0
    assert vllm_metrics.metric_sum(_FIXTURE, "vllm:prefix_cache_queries") == 200.0


def test_kv_cache_hit_rate():
    # 100 / 200 = 0.5
    assert vllm_metrics.kv_cache_hit_rate(_FIXTURE) == pytest.approx(0.5)


def test_kv_cache_hit_rate_zero_queries():
    assert vllm_metrics.kv_cache_hit_rate("vllm:prefix_cache_hits_total 5\n") == 0.0


def test_metrics_url_strips_v1():
    assert vllm_metrics._metrics_url("http://h:8000/v1") == "http://h:8000/metrics"
    assert vllm_metrics._metrics_url("http://h:8000") == "http://h:8000/metrics"


def test_dump_writes_summary_and_raw(tmp_path):
    p = vllm_metrics.dump(tmp_path / "vllm_metrics.json", _FIXTURE)
    obj = json.loads(p.read_text())
    assert obj["kv_cache_hit_rate"] == pytest.approx(0.5)
    assert obj["prefix_cache_queries"] == 200.0
    assert obj["raw_exposition"] == _FIXTURE


# ---- 对 stub server 联调 ----


def test_scrape_stub_metrics():
    srv, base_url = stub_openai.start_background(port=0)
    try:
        text = vllm_metrics.scrape(base_url)
        assert "vllm:prefix_cache_hits_total" in text
        assert vllm_metrics.kv_cache_hit_rate(text) == 0.0  # stub 都是 0
    finally:
        srv.shutdown()
        srv.server_close()
