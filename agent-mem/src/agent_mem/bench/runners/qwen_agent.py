"""``QwenAgentRunner`` —— 接本地引擎真跑 τ-bench 任务的 Runner。

惰性：模块顶层只 import TauBenchAgent（轻）。真 τ-bench/openai 调用都在
:meth:`run_all` 内经 :func:`agent_mem.bench.tasks.tau_bench_adapter.run_task` 触发。
注册：因构造需参数（engine_url/model），**不**进 ``runner.RUNNERS``（那个给无参 runner），
由 CLI 在 ``--runner qwen-agent`` 时显式构造。
"""

from __future__ import annotations

from agent_mem.bench.runner import Runner
from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult
from agent_mem.config import AppConfig


class QwenAgentRunner(Runner):
    """真跑 τ-bench 的 Runner。``run_all`` 逐任务调 ``adapter.run_task``。"""

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
        # 缝D：中间件链（由 CLI 从 cfg.middleware 构造后注入；None=identity）
        self.middlewares = middlewares

    def name(self) -> str:
        return "qwen-agent"

    def run_all(self, cfg: AppConfig) -> list[TaskRunResult]:
        from agent_mem.bench.tasks.tau_bench_adapter import list_tasks, run_task

        tasks = list_tasks(cfg.benchmark.domain, cfg.benchmark.split)
        if self.max_tasks is not None:
            tasks = tasks[: self.max_tasks]
        return [
            run_task(
                t.task_id,
                domain=cfg.benchmark.domain,
                split=cfg.benchmark.split,
                engine_url=self.engine_url,
                model=self.model,
                user_model=self.user_model,
                user_provider=self.user_provider,
                user_api_base=self.user_api_base,
                user_api_key=self.user_api_key,
                api_key=self.api_key,
                max_steps=self.max_steps,
                middlewares=self.middlewares,
            )
            for t in tasks
        ]
