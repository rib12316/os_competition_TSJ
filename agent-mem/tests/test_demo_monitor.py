"""``agent_mem.demo.monitor`` 单测（纯数据层，无 GUI / 无设备依赖）。

覆盖：历史 before/after 分组中位数、离线 engine_status、LiveMonitor 后台采样循环。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent_mem.demo import monitor
from agent_mem.demo.monitor import LiveMonitor, engine_status, load_history

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = REPO_ROOT / "logs" / "mvp-newframework"


def test_load_history_missing_dir_returns_empty(tmp_path):
    assert load_history(tmp_path / "does-not-exist") == []


def test_load_history_real_runs_grouped_by_config():
    """对真机 logs/mvp-newframework：baseline + prefix-cache 各 3 runs，按 config 分组取中位数。"""
    if not HISTORY_DIR.is_dir():
        pytest.skip(f"历史目录不存在：{HISTORY_DIR}（先跑 bench 生成）")

    hist = load_history(HISTORY_DIR)
    assert hist, "应至少解析出 1 个 config"

    by_cfg = {h.config: h for h in hist}
    # 这批数据应含 baseline 与 prefix-cache 两档
    assert "baseline" in by_cfg, f"缺 baseline，实际 configs={list(by_cfg)}"
    assert "prefix-cache" in by_cfg, f"缺 prefix-cache，实际 configs={list(by_cfg)}"

    base = by_cfg["baseline"]
    pc = by_cfg["prefix-cache"]
    assert base.n_runs == 3 and pc.n_runs == 3
    assert base.mem_peak_mb > 0
    # prefix-cache 的 KV 命中率应显著高于 baseline（真机 ~0.94 vs ~0）
    assert pc.kv_cache_hit_rate > base.kv_cache_hit_rate
    # mem_curve 至少有若干点（首条 run 的 mem_timeseries.csv）
    assert len(base.mem_curve) > 0


def test_runmetrics_from_json_ignores_unknown_fields():
    from agent_mem.demo.monitor import _runmetrics_from_json

    data = {
        "run_id": "x", "engine": "vllm", "model": "qwen", "config": "baseline",
        "mem_peak_mb": 12345, "kv_cache_hit_rate": 0.5,
        "unknown_extra_field": "ignored",
    }
    m = _runmetrics_from_json(data)
    assert m.config == "baseline"
    assert m.mem_peak_mb == 12345
    assert m.kv_cache_hit_rate == 0.5


def test_read_mem_curve_parses_and_skips_bad_rows(tmp_path):
    from agent_mem.demo.monitor import _read_mem_curve

    csv_path = tmp_path / "mem_timeseries.csv"
    csv_path.write_text("timestamp,used_mb\n0.000,100\n0.5,200\nbad,row\n1.0,300\n", encoding="utf-8")
    pts = _read_mem_curve(csv_path)
    assert pts == [(0.0, 100), (0.5, 200), (1.0, 300)]


def test_engine_status_offline_for_closed_port():
    # 8001 几乎肯定没有服务在听 → offline（不依赖任何引擎）
    assert engine_status("http://127.0.0.1:8001/v1", timeout=1.0) == "offline"


def test_live_monitor_samples_and_stops():
    """LiveMonitor 后台线程能产出快照、stop 能 join。

    用 FakeBackend（确定、快）注入显存源；base_url 指关闭端口 → kv_hit_rate=None。
    """
    from agent_mem.bench.mem_sampler import FakeBackend

    fake = FakeBackend([10000, 10100, 10200, 10300, 10400])
    mon = LiveMonitor(
        base_url="http://127.0.0.1:8001/v1", interval=0.15, device="npu", backend=fake
    )
    mon.start()
    try:
        assert mon.latest() is not None  # start() 已同步采首个
        time.sleep(0.5)  # 让后台再采几次
        snaps = mon.snapshot()
        assert len(snaps) >= 2, f"采样数不足：{len(snaps)}"
        # 注入的显存值被采到（首个=10000）；kv 因引擎离线为 None
        assert any(s.mem_mb == 10000 for s in snaps)
        assert all(s.kv_hit_rate is None for s in snaps)
    finally:
        mon.stop()


def test_root_url_strips_v1():
    assert monitor._root_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert monitor._root_url("http://127.0.0.1:8000/v1/") == "http://127.0.0.1:8000"
    assert monitor._root_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
