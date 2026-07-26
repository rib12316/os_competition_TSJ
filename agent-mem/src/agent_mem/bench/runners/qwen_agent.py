"""``QwenAgentRunner`` —— 接本地引擎真跑 τ-bench 任务的 Runner。

支持 ``max_concurrency``：>1 时用 ThreadPoolExecutor 并发跑多任务，
共享引擎 KV cache（F5 并发场景）。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_mem.bench.runner import Runner
from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult
from agent_mem.config import AppConfig
from agent_mem.scheduler.driver import ConcurrentSessionDriver


class QwenAgentRunner(Runner):
    """真跑 τ-bench 的 Runner。``max_concurrency`` > 1 时内部并发跑任务。"""

    def __init__(
        self,
        *,
        engine_url: str,
        model: str,
        user_model: str | None = None,
        user_provider: str = "openai",
        user_api_base: str | None = None,
        user_api_key: str | None = None,
        api_key: str = "stub",
        max_steps: int = 30,
        max_tasks: int | None = None,
        max_concurrency: int = 1,
        priority: int = 0,
        middlewares: list | None = None,
        dynamic: bool = False,
        idle_timeout_s: float = 30.0,
        target_lo: int = 70,
        target_hi: int = 85,
        hbm_pct_fn: Callable[[], float] | None = None,
    ):
        self.engine_url = engine_url
        self.model = model
        self.user_model = user_model
        self.user_provider = user_provider
        self.user_api_base = user_api_base
        self.user_api_key = user_api_key
        self.api_key = api_key
        self.max_steps = max_steps
        self.max_tasks = max_tasks
        self.max_concurrency = max_concurrency
        self.priority = priority
        self.middlewares = middlewares
        # F5 动态调度参数（dynamic=True 且 max_concurrency>1 时走 ConcurrentSessionDriver）
        self.dynamic = dynamic
        self.idle_timeout_s = idle_timeout_s
        self.target_lo = target_lo
        self.target_hi = target_hi
        self.hbm_pct_fn = hbm_pct_fn
        self.last_driver: ConcurrentSessionDriver | None = None  # 供 metrics/监控取 snapshot

    def name(self) -> str:
        return "qwen-agent"

    def run_all(self, cfg: AppConfig) -> list[TaskRunResult]:
        from agent_mem.bench.tasks.tau_bench_adapter import list_tasks, run_task

        tasks = list_tasks(cfg.benchmark.domain, cfg.benchmark.split)
        if self.max_tasks is not None:
            tasks = tasks[: self.max_tasks]
        # F5 动态调度路径：HBM 准入闸门 + 后台 sweep 抬 idle priority
        if self.dynamic and self.max_concurrency > 1:
            return self._run_dynamic(cfg, [t.task_id for t in tasks], run_task)

        if self.max_concurrency <= 1:
            return [
                run_task(
                    t.task_id, domain=cfg.benchmark.domain, split=cfg.benchmark.split,
                    engine_url=self.engine_url, model=self.model,
                    user_model=self.user_model, user_provider=self.user_provider,
                    user_api_base=self.user_api_base, user_api_key=self.user_api_key,
                    api_key=self.api_key, max_steps=self.max_steps,
                    priority=self.priority, middlewares=self.middlewares,
                )
                for t in tasks
            ]

        # 并发跑
        results: list[TaskRunResult] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futures = {
                ex.submit(
                    run_task, t.task_id,
                    domain=cfg.benchmark.domain, split=cfg.benchmark.split,
                    engine_url=self.engine_url, model=self.model,
                    user_model=self.user_model, user_provider=self.user_provider,
                    user_api_base=self.user_api_base, user_api_key=self.user_api_key,
                    api_key=self.api_key, max_steps=self.max_steps,
                    priority=self.priority, middlewares=self.middlewares,
                ): t.task_id
                for t in tasks
            }
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append(TaskRunResult(
                        task_id=futures[f], reward=0.0, success=False,
                        latency_ms=0.0, n_steps=0, error=repr(e),
                    ))
        return results

    def _run_dynamic(
        self, cfg: AppConfig, task_ids: list[int], run_task: Callable[..., TaskRunResult]
    ) -> list[TaskRunResult]:
        """F5 动态调度：用 :class:`ConcurrentSessionDriver` 跑，注入 priority/turn 回调。"""
        domain, split = cfg.benchmark.domain, cfg.benchmark.split

        def _runner(tid: int, on_turn, pfn) -> TaskRunResult:
            try:
                return run_task(
                    tid, domain=domain, split=split, engine_url=self.engine_url,
                    model=self.model, user_model=self.user_model,
                    user_provider=self.user_provider, user_api_base=self.user_api_base,
                    user_api_key=self.user_api_key, api_key=self.api_key,
                    max_steps=self.max_steps, priority=self.priority,
                    middlewares=self.middlewares, priority_fn=pfn, on_turn_start=on_turn,
                )
            except Exception as e:  # noqa: BLE001 — 单任务异常不杀并发 run
                return TaskRunResult(
                    task_id=tid, reward=0.0, success=False,
                    latency_ms=0.0, n_steps=0, error=repr(e),
                )

        strategy = cfg.session.strategy
        priority_mode = ("combined" if strategy == "combined-evict"
                         else "progress" if strategy == "progress-evict" else "idle")
        # Think-time profiles from config: list of [lo,hi] pairs simulating user engagement
        profiles_raw = cfg.session.options.get("think_time_profiles") or []
        think_profiles = [tuple(p) for p in profiles_raw] if profiles_raw else None
        driver = ConcurrentSessionDriver(
            max_workers=self.max_concurrency, idle_timeout_s=self.idle_timeout_s,
            target_lo=self.target_lo, target_hi=self.target_hi, hbm_pct_fn=self.hbm_pct_fn,
            priority_mode=priority_mode, think_time_profiles=think_profiles,
            max_steps=self.max_steps,
        )
        self.last_driver = driver  # 暴露给 metrics / 实时监控取 snapshot
        return driver.run(task_ids, _runner)
