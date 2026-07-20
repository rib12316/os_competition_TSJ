"""benchmark runner：可插拔的 Runner 抽象 + 单 run / 多 run 驱动 + 聚合。

day-1 默认用 :class:`DryRunRunner`（不调任何 agent / tau_bench，返回全 0 占位结果），
让 harness 骨架能在纯 CPU、无引擎的情况下跑通并落 run 目录。
:class:`TauBenchRunner` 惰性接 τ-bench 适配器（day-1 仍只调 ``run_task_stub`` 占位，
用于验证 import 链路）。MVP 接 P2 的 agent CLI 时，新增 ``QwenAgentRunner(Runner)`` 即可，
不动本模块主体。
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from agent_mem.bench.run_dir import (
    create_run_dir,
    run_id_for,
    short_config_name,
    short_engine_name,
    short_model_name,
    timestamp_now,
    write_metrics,
)
from agent_mem.bench.stats import median, p50, p95
from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult, aggregate_success
from agent_mem.config import AppConfig
from agent_mem.metrics import RunMetrics, default_metrics

# 聚合时取中位数的数值字段
_NUMERIC_FIELDS = (
    "e2e_latency_p50_ms",
    "e2e_latency_p95_ms",
    "qps",
    "mem_peak_mb",
    "kv_cache_hit_rate",
    "task_success_rate",
    "ttft_ms",
)


class Runner(Protocol):
    """可插拔的任务执行器抽象。"""

    def run_all(self, cfg: AppConfig) -> list[TaskRunResult]:
        """跑完一个 run 的全部任务，返回每个任务的结果。"""
        ...

    def name(self) -> str:
        """runner 名（dry-run / tau-bench / qwen-agent …）。"""
        ...


class DryRunRunner:
    """默认 runner：不调任何 agent，返回固定数量全 0 占位结果。

    让 harness 骨架在无引擎、无 τ-bench 时也能端到端落 run 目录（指标全 0）。
    """

    N_PLACEHOLDER_TASKS = 5

    def run_all(self, cfg: AppConfig) -> list[TaskRunResult]:
        return [
            TaskRunResult(
                task_id=i, reward=0.0, success=False, latency_ms=0.0, n_steps=0, error=None
            )
            for i in range(self.N_PLACEHOLDER_TASKS)
        ]

    def name(self) -> str:
        return "dry-run"


class TauBenchRunner:
    """τ-bench runner：惰性加载适配器，day-1 调 ``run_task_stub`` 占位。

    用于验证 τ-bench import 链路通；MVP 时把 ``run_task_stub`` 换成真 agent 执行。
    """

    def run_all(self, cfg: AppConfig) -> list[TaskRunResult]:
        # 惰性 import（适配器顶层零 tau_bench import；这里调用才触发 litellm 加载）
        from agent_mem.bench.tasks.tau_bench_adapter import list_tasks, run_task_stub

        tasks = list_tasks(cfg.benchmark.domain, cfg.benchmark.split)
        return [
            run_task_stub(t.task_id, domain=cfg.benchmark.domain, split=cfg.benchmark.split)
            for t in tasks
        ]

    def name(self) -> str:
        return "tau-bench"


RUNNERS: dict[str, type[Runner]] = {
    "dry-run": DryRunRunner,
    "tau-bench": TauBenchRunner,
}


def get_runner(name: str) -> Runner:
    """按名取 runner 实例；未知名抛 KeyError。"""
    if name not in RUNNERS:
        raise KeyError(f"未知 runner {name!r}，可选：{list(RUNNERS)}")
    return RUNNERS[name]()


def run_once(
    cfg: AppConfig,
    runner: Runner,
    *,
    run_n: int = 1,
    run_root: str | Path,
    config_text: str | None = None,
    ts: str | None = None,
    device: str | None = None,
    engine_url: str | None = None,
) -> RunMetrics:
    """跑一次完整 run：建 run 目录 → runner 跑全部任务 → 采集指标 → 写 metrics.json。

    - 延迟 p50/p95 + QPS 始终从 results + 墙钟算（dry-run 下为 0）。
    - ``device`` 给定时启动显存采样（写 ``mem_timeseries.csv``，峰值从时序取）。
    - ``engine_url`` 给定时抓 vLLM ``/metrics``（写 ``vllm_metrics.json``，算 KV 命中率）。
    - TTFT 需在 agent 层测（C 组），当前留 0。
    """
    ts = ts or timestamp_now()
    run_id, started_at = run_id_for(cfg, run_n, ts=ts)
    run_dir = create_run_dir(run_root, cfg, run_n, config_text=config_text, ts=ts)

    # 显存采样（惰性 import，避免顶层拉 torch）
    sampler = None
    if device:
        from agent_mem.bench.mem_sampler import MemSampler, make_backend

        sampler = MemSampler(
            make_backend(device), interval=0.5, out_csv=run_dir / "mem_timeseries.csv"
        )
        sampler.start()

    t0 = time.monotonic()
    try:
        results = runner.run_all(cfg)
    finally:
        series = sampler.stop() if sampler else None
    wall = max(time.monotonic() - t0, 1e-9)

    metrics = default_metrics(
        run_id=run_id,
        engine=short_engine_name(cfg.engine.backend),
        model=short_model_name(cfg.engine.model),
        config=short_config_name(cfg.config_name),
        seed=cfg.benchmark.seed,
        started_at=started_at,
    )
    metrics.task_success_rate = aggregate_success(results)
    latencies = [r.latency_ms for r in results]
    metrics.e2e_latency_p50_ms = p50(latencies)
    metrics.e2e_latency_p95_ms = p95(latencies)
    metrics.qps = len(results) / wall
    metrics.ttft_ms = median([r.ttft_ms for r in results])
    if series is not None:
        metrics.mem_peak_mb = series.peak_mb
    if engine_url:
        from agent_mem.bench import vllm_metrics

        try:
            text = vllm_metrics.scrape(engine_url)
            vllm_metrics.dump(run_dir / "vllm_metrics.json", text)
            metrics.kv_cache_hit_rate = vllm_metrics.kv_cache_hit_rate(text)
        except Exception as e:  # noqa: BLE001 — 引擎未就绪时不阻断 run
            print(f"[runner] /metrics 抓取失败，跳过 KV 命中率：{e}", file=sys.stderr)
    write_metrics(run_dir, metrics)
    return metrics


def run_study(
    cfg: AppConfig,
    runner: Runner,
    *,
    run_root: str | Path,
    config_text: str | None = None,
    device: str | None = None,
    engine_url: str | None = None,
) -> tuple[list[RunMetrics], dict[str, float]]:
    """跑 ``cfg.benchmark.runs`` 次，返回 (每次 metrics 列表, 各字段中位数)。"""
    all_metrics = [
        run_once(
            cfg, runner, run_n=n, run_root=run_root, config_text=config_text,
            device=device, engine_url=engine_url,
        )
        for n in range(1, cfg.benchmark.runs + 1)
    ]
    return all_metrics, aggregate_runs(all_metrics)


def aggregate_runs(runs: list[RunMetrics]) -> dict[str, float]:
    """对多次 run 的每个数值字段取中位数（公平对照：3 次取中位数）。"""
    if not runs:
        return {}
    out: dict[str, float] = {}
    for field in _NUMERIC_FIELDS:
        vals = [getattr(r, field) for r in runs]
        out[field] = statistics.median(vals)
    return out


def run_concurrent_sessions(
    cfg: AppConfig,
    runner: Runner,
    *,
    n_sessions: int = 4,
) -> tuple[list[TaskRunResult], float]:
    """并发跑 ``n_sessions`` 个 agent session（各自 :meth:`Runner.run_all`），算 QPS。

    返回 ``(合并后的全部任务结果, qps)``，``qps = 总任务数 / 墙钟秒``。
    命中赛题「多 agent 并发吞吐」。骨架：DryRun 下任务瞬时完成，qps 仅验证机制；
    真 QPS 需真 agent + 引擎（C/B 组就绪后）。
    """
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n_sessions) as ex:
        futures = [ex.submit(runner.run_all, cfg) for _ in range(n_sessions)]
        sessions = [f.result() for f in futures]
    wall = max(time.monotonic() - t0, 1e-9)
    merged = [r for s in sessions for r in s]
    return merged, len(merged) / wall
