"""Benchmark 桥——前端"跑 bench"的后端。

把统一 benchmark（tau-bench via unified-tau-freq / longbench via unified-longbench）
经 ``bench.runner.run_once`` 在**后台线程**跑、落盘 ``metrics.json``，并把进度/结果
暴露给 Gradio ``gr.Timer`` 轮询。

设计要点：
- ``run_once`` 是同步的、不支持流式 → 在 daemon 线程里按 ``cfg.benchmark.runs`` 循环调用，
  每完成一次 bump ``completed_runs``（real-time 进度靠 LiveMonitor 看 /metrics，T3 显示 M/N）。
- worker 线程**只**就地改 ``BenchHandle``（``_lock`` 保护），**不**碰 Gradio API；
  T3 的 Timer 经 ``snapshot()``（拷贝）读。
- 重依赖（QwenAgentRunner / tau_bench / litellm）lazy import，保证 ``build_app`` 与无引擎测试快。
- 绝不 import ``benchmarks.runner``（那是 CLI，会拖入 argparse）。
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchHandle:
    """由 gr.State 持有、worker 线程就地改、Timer 读的 bench 状态。"""

    status: str = "idle"  # idle | queued | running | done | error
    preset: str | None = None
    total_runs: int = 0
    completed_runs: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    run_dirs: list[str] = field(default_factory=list)
    median: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        """线程安全快照（Timer 用）。"""
        with self._lock:
            return {
                "status": self.status,
                "preset": self.preset,
                "total_runs": self.total_runs,
                "completed_runs": self.completed_runs,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "run_dirs": list(self.run_dirs),
                "median": dict(self.median),
                "error": self.error,
            }


def _build_runner(cfg, *, engine_url: str, max_concurrency: int):
    """镜像 benchmarks/runner.py 的 QwenAgentRunner 构造（lazy import 重依赖）。"""
    from agent_mem.bench.runners.qwen_agent import QwenAgentRunner
    from agent_mem.middleware import middlewares_from_config

    mw = middlewares_from_config(cfg)
    us = cfg.user_sim
    user_api_key = os.environ.get(us.api_key_env) if us.api_key_env else None
    return QwenAgentRunner(
        engine_url=engine_url,
        model=cfg.engine.model,
        user_model=us.model or None,
        user_provider=us.provider,
        user_api_base=us.api_base or None,
        user_api_key=user_api_key,
        max_concurrency=max_concurrency,
        middlewares=mw.middlewares,
        dynamic=(cfg.session.strategy in ("priority-evict", "progress-evict", "combined-evict")),
        idle_timeout_s=cfg.session.idle_timeout_s,
        target_lo=cfg.session.target_lo,
        target_hi=cfg.session.target_hi,
        # hbm_pct_fn=None：LiveMonitor 已采 HBM，避免第二个 scraper
    )


def _run_study_thread(
    handle: BenchHandle,
    *,
    preset_path: str,
    engine_url: str,
    run_root: str,
    runs: int | None,
    max_concurrency: int,
    device: str,
) -> None:
    """worker：load_config → 构造 runner → run_once 循环 → aggregate。所有 IO 在 try 内。"""
    try:
        from agent_mem.bench.compare import find_run_dirs
        from agent_mem.bench.runner import aggregate_runs, run_once
        from agent_mem.config import load_config

        config_text = Path(preset_path).read_text(encoding="utf-8")
        cfg = load_config(preset_path)
        if runs is not None:
            cfg.benchmark.runs = max(1, int(runs))
        runner = _build_runner(cfg, engine_url=engine_url, max_concurrency=max_concurrency)
        handle.update(
            status="running",
            preset=Path(preset_path).stem,
            total_runs=cfg.benchmark.runs,
            completed_runs=0,
            started_at=time.time(),
        )
        for n in range(1, cfg.benchmark.runs + 1):
            run_once(
                cfg, runner, run_n=n, run_root=run_root,
                config_text=config_text, device=device, engine_url=engine_url,
            )
            # run_once 已把 metrics.json 写入 run_root 下最新 dir
            newest = find_run_dirs(run_root)
            handle.update(
                completed_runs=n,
                run_dirs=[str(newest[-1])] if newest else [],
            )
        # aggregate 用真正落盘的 metrics.json（中位数语义与 run_study 一致）
        median = aggregate_runs(_load_run_metrics_objs(run_root, cfg.benchmark.runs))
        handle.update(status="done", finished_at=time.time(), median=median)
    except Exception as e:  # noqa: BLE001
        handle.update(
            status="error",
            error=repr(e) + "\n" + traceback.format_exc(),
            finished_at=time.time(),
        )


def _load_run_metrics_objs(run_root: str, runs: int) -> list:
    """读最近 ``runs`` 个 run 的 metrics.json 成 RunMetrics（aggregate_runs 需要）。"""
    import json as _json

    from agent_mem.bench.compare import find_run_dirs
    from agent_mem.metrics import RunMetrics

    objs: list[RunMetrics] = []
    for d in find_run_dirs(run_root)[-runs:]:
        mpath = Path(d) / "metrics.json"
        if mpath.exists():
            try:
                objs.append(RunMetrics(**_json.loads(mpath.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001
                pass
    return objs


def run_bench_async(
    handle: BenchHandle,
    *,
    preset_path: str,
    engine_url: str,
    run_root: str = "logs",
    runs: int | None = None,
    max_concurrency: int = 1,
    device: str = "npu",
) -> BenchHandle:
    """起 daemon 线程跑 bench，立即返回（非阻塞）。``handle`` 由 gr.State 持有。"""
    handle.update(status="queued", error=None, completed_runs=0, run_dirs=[], median={})
    t = threading.Thread(
        target=_run_study_thread,
        kwargs=dict(
            handle=handle, preset_path=str(preset_path), engine_url=engine_url,
            run_root=run_root, runs=runs, max_concurrency=max_concurrency,
            device=device,
        ),
        daemon=True,
        name="agent-mem-bench",
    )
    t.start()
    return handle
