"""``QwenAgentRunner`` —— 接本地引擎真跑 τ-bench 任务的 Runner。

支持 ``max_concurrency``：>1 时用 ThreadPoolExecutor 并发跑多任务，
共享引擎 KV cache（F5 并发场景）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_mem.bench.runner import Runner
from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult
from agent_mem.config import AppConfig


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

    def name(self) -> str:
        return "qwen-agent"

    def run_all(self, cfg: AppConfig) -> list[TaskRunResult]:
        from agent_mem.bench.tasks.tau_bench_adapter import list_tasks, run_task

        tasks = list_tasks(cfg.benchmark.domain, cfg.benchmark.split)
        if self.max_tasks is not None:
            tasks = tasks[: self.max_tasks]

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
