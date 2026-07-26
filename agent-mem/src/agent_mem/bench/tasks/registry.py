"""suite dispatch —— ``get_adapter(cfg.benchmark.suite)`` 把 runner 与具体任务解耦。

统一 benchmark 的核心 indirection：runner 不再硬 import tau_bench_adapter，而是经
``get_adapter(suite)`` 拿到一个 :class:`SuiteAdapter`，驱动 ``list_tasks(cfg)`` +
``run_task(task, ctx)``。tau / longbench 各一个 adapter；新增 suite 只需注册一个 adapter。

metrics / run_dir / run_study 层不变——它们只消费 :class:`TaskRunResult`（suite-agnostic）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult
from agent_mem.bench.tasks.types import TaskInfo
from agent_mem.config import AppConfig


@dataclass
class RunContext:
    """一次 run 内不变、但跨 task 共享的上下文（suite-agnostic 容器）。

    tau 字段（user_*、priority）longbench 忽略；longbench 字段（longbench_system_prompt）
    tau 忽略。F5 动态回调 ``priority_fn``/``on_turn_start`` 在 dynamic 路径由 driver 注入。
    """

    cfg: AppConfig
    engine_url: str
    model: str
    api_key: str = "stub"
    max_steps: int = 30
    middlewares: list[Any] | None = None
    # tau-bench user-simulator
    user_model: str | None = None
    user_provider: str = "openai"
    user_api_base: str | None = None
    user_api_key: str | None = None
    priority: int = 0
    # F5 动态调度回调（None → tau_bench_agent 退回静态 priority）
    priority_fn: Callable[[], int] | None = None
    on_turn_start: Callable[[int], None] | None = None
    # longbench system prompt 覆盖（None → 用 2WikiMQA 默认）
    longbench_system_prompt: str | None = None


class SuiteAdapter:
    """一个 benchmark suite 的任务源 + 执行器。"""

    suite: str = ""

    def list_tasks(self, cfg: AppConfig) -> list[TaskInfo]:
        raise NotImplementedError

    def run_task(self, task: TaskInfo, ctx: RunContext) -> TaskRunResult:
        raise NotImplementedError


class _TauAdapter(SuiteAdapter):
    """tau-bench：包一层现有 tau_bench_adapter（保持其 task_id 签名不变）。"""

    suite = "tau-bench"

    def list_tasks(self, cfg: AppConfig) -> list[TaskInfo]:
        from agent_mem.bench.tasks.tau_bench_adapter import list_tasks as _lt

        return _lt(cfg.benchmark.domain, cfg.benchmark.split)

    def run_task(self, task: TaskInfo, ctx: RunContext) -> TaskRunResult:
        from agent_mem.bench.tasks.tau_bench_adapter import run_task as _rt

        return _rt(
            task.task_id,
            domain=task.domain,
            split=task.split,
            engine_url=ctx.engine_url,
            model=ctx.model,
            user_model=ctx.user_model,
            user_provider=ctx.user_provider,
            user_api_base=ctx.user_api_base,
            user_api_key=ctx.user_api_key,
            api_key=ctx.api_key,
            max_steps=ctx.max_steps,
            priority=ctx.priority,
            middlewares=ctx.middlewares,
            priority_fn=ctx.priority_fn,
            on_turn_start=ctx.on_turn_start,
        )


class _LongbenchAdapter(SuiteAdapter):
    """longbench：长上下文 QA 经 F2/F3 middleware。"""

    suite = "longbench"

    def list_tasks(self, cfg: AppConfig) -> list[TaskInfo]:
        from agent_mem.bench.tasks.longbench_adapter import list_tasks as _lt

        return _lt(cfg)

    def run_task(self, task: TaskInfo, ctx: RunContext) -> TaskRunResult:
        from agent_mem.bench.tasks.longbench_adapter import run_task as _rt

        return _rt(
            task,
            engine_url=ctx.engine_url,
            model=ctx.model,
            api_key=ctx.api_key,
            middlewares=ctx.middlewares,
            max_steps=ctx.max_steps,
            system_prompt=ctx.longbench_system_prompt,
        )


_ADAPTERS: dict[str, SuiteAdapter] = {
    "tau-bench": _TauAdapter(),
    "longbench": _LongbenchAdapter(),
}


def get_adapter(suite: str) -> SuiteAdapter:
    """按 ``cfg.benchmark.suite`` 取 adapter；未知名抛 :class:`KeyError`。"""
    if suite not in _ADAPTERS:
        raise KeyError(
            f"未知 suite {suite!r}，已注册：{sorted(_ADAPTERS)}。"
            f"（新增 suite 在此注册一个 SuiteAdapter）"
        )
    return _ADAPTERS[suite]


def registered_suites() -> list[str]:
    return sorted(_ADAPTERS)
