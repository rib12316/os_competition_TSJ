"""Runner 骨架端到端测试（DryRunRunner，不触 tau_bench）。"""

from __future__ import annotations

import json

import pytest

from agent_mem.bench.runner import (
    DryRunRunner,
    aggregate_runs,
    get_runner,
    run_concurrent_sessions,
    run_once,
    run_study,
)
from agent_mem.config import AppConfig, BenchmarkConfig, EngineConfig, MetricsConfig
from agent_mem.metrics import RunMetrics


def _cfg(runs: int = 3) -> AppConfig:
    return AppConfig(
        engine=EngineConfig(backend="vllm", model="Qwen2.5-7B-Instruct"),
        benchmark=BenchmarkConfig(runs=runs, seed=42),
        metrics=MetricsConfig(),
        config_name="baseline",
    )


def test_dry_run_runner_returns_placeholders():
    rs = DryRunRunner().run_all(_cfg())
    assert len(rs) == DryRunRunner.N_PLACEHOLDER_TASKS
    assert all(r.reward == 0.0 and not r.success for r in rs)


def test_run_once_creates_dir_and_metrics(tmp_path):
    cfg = _cfg()
    m = run_once(
        cfg, DryRunRunner(), run_n=1, run_root=tmp_path, config_text="# snap", ts="20260718-143022"
    )
    assert m.run_id == "20260718-143022_vllm_qwen25-7b_baseline_run1"
    assert m.task_success_rate == 0.0
    assert m.engine == "vllm" and m.model == "qwen25-7b" and m.config == "baseline"
    d = tmp_path / m.run_id
    assert (d / "metrics.json").exists()
    assert (d / "config.yaml").read_text(encoding="utf-8") == "# snap"
    assert (d / "git_commit.txt").exists()
    obj = json.loads((d / "metrics.json").read_text())
    assert obj["seed"] == 42


def test_run_study_three_runs_and_median(tmp_path):
    cfg = _cfg(runs=3)
    all_m, agg = run_study(cfg, DryRunRunner(), run_root=tmp_path, config_text="# snap")
    assert len(all_m) == 3
    assert len(list(tmp_path.iterdir())) == 3
    assert agg["task_success_rate"] == 0.0
    # run 目录命名 run1/run2/run3 各一
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names[0].endswith("_run1") and names[-1].endswith("_run3")


def test_aggregate_runs_takes_median():
    ms = [
        RunMetrics(
            run_id="r", engine="vllm", model="m", config="c",
            e2e_latency_p50_ms=v, task_success_rate=v / 100,
        )
        for v in (1.0, 2.0, 10.0)
    ]
    agg = aggregate_runs(ms)
    assert agg["e2e_latency_p50_ms"] == 2.0
    assert agg["task_success_rate"] == pytest.approx(0.02)


def test_aggregate_runs_empty():
    assert aggregate_runs([]) == {}


def test_get_runner_known_and_unknown():
    assert get_runner("dry-run").name() == "dry-run"
    assert get_runner("tau-bench").name() == "tau-bench"
    with pytest.raises(KeyError):
        get_runner("nope")


def test_run_concurrent_sessions_qps():
    # 3 个并发 session × 5 占位任务 = 15，qps > 0（DryRun 瞬时完成）
    merged, qps = run_concurrent_sessions(_cfg(), DryRunRunner(), n_sessions=3)
    assert len(merged) == 3 * DryRunRunner.N_PLACEHOLDER_TASKS
    assert qps > 0


# ---- 采集路径集成（device / engine_url）----


def test_run_once_scrapes_engine_metrics(tmp_path):
    """engine_url 指向 stub server → 抓 /metrics、写 vllm_metrics.json、KV=0。"""
    from agent_mem.server import stub_openai

    srv, base_url = stub_openai.start_background(port=0)
    try:
        cfg = _cfg()
        m = run_once(
            cfg, DryRunRunner(), run_n=1, run_root=tmp_path,
            config_text="# snap", ts="20260718-143022", engine_url=base_url,
        )
        d = tmp_path / m.run_id
        assert (d / "vllm_metrics.json").exists()
        assert m.kv_cache_hit_rate == 0.0  # stub 都是 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_run_once_samples_memory_with_fake_backend(tmp_path, monkeypatch):
    """device 路径：monkeypatch make_backend 返回常量 FakeBackend，峰值=该常量。"""
    from agent_mem.bench import mem_sampler

    fake = mem_sampler.FakeBackend([1234])  # 常量：无论采几次峰值都确定
    monkeypatch.setattr(mem_sampler, "make_backend", lambda device: fake)

    cfg = _cfg()
    m = run_once(
        cfg, DryRunRunner(), run_n=1, run_root=tmp_path,
        config_text="# snap", ts="20260718-143022", device="cuda",
    )
    d = tmp_path / m.run_id
    assert (d / "mem_timeseries.csv").exists()
    assert m.mem_peak_mb == 1234
