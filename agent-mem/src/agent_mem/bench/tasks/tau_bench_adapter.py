"""τ-bench 任务适配器。

**惰性 import**：``import tau_bench`` 会拖入 litellm（~3.6s）并污染 pytest 收集，
因此本模块顶层**零** ``tau_bench.*`` import，全部放在函数体内。模块本身只定义
轻量 dataclass + 纯函数，import 本模块是廉价的；只有真正调用 :func:`list_tasks`
等才会加载 τ-bench。

任务枚举走 ``tau_bench.envs.get_env(domain, ...).tasks``；reward 判定对齐官方
``is_successful := 0.999999 <= reward <= 1.000001``。

day-1 骨架：:func:`run_task_stub` 不调 agent，直接返回占位结果（reward=0）；
真执行（``get_env(...).reset`` → agent.solve → reward）留 MVP。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_mem.bench.stats import median

SUPPORTED_ENVS = ("retail", "airline")
SUPPORTED_SPLITS = ("train", "test", "dev")


@dataclass(frozen=True)
class TaskInfo:
    """与 ``tau_bench.types.Task`` 解耦的轻量 DTO（避免顶层 import tau_bench）。"""

    task_id: int
    user_id: str
    instruction: str
    action_names: tuple[str, ...]  # [a.name for a in task.actions]
    n_outputs: int
    domain: str  # retail | airline
    split: str  # test | train | dev


@dataclass(frozen=True)
class TaskRunResult:
    """单任务 run 结果（骨架期 reward/latency_ms/ttft_ms 填 0）。"""

    task_id: int
    reward: float  # 0.0 骨架；真值 ∈ {0.0, 1.0}
    success: bool  # 0.999999 <= reward <= 1.000001
    latency_ms: float
    n_steps: int
    error: str | None
    ttft_ms: float = 0.0  # 该任务各步首 token 时间的中位数（骨架 0）


def is_successful(reward: float) -> bool:
    """对齐 ``tau_bench.run.display_metrics`` 的成功判定。"""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def aggregate_success(results: list[TaskRunResult]) -> float:
    """task_success_rate = mean(success)。空列表返回 0.0。"""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def _check_domain_split(domain: str, split: str) -> None:
    if domain not in SUPPORTED_ENVS:
        raise ValueError(f"domain={domain!r} 不在支持列表 {SUPPORTED_ENVS}")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"split={split!r} 不在支持列表 {SUPPORTED_SPLITS}")


def list_tasks(domain: str = "retail", split: str = "test") -> list[TaskInfo]:
    """惰性 import tau_bench，枚举任务。重依赖（litellm）只在本函数首次调用时加载。

    用 ``UserStrategy.HUMAN``（不需要 LLM 的 user 模拟器），仅用于枚举；
    MVP 真跑 agent 时换 ``LLM`` 策略。
    """
    _check_domain_split(domain, split)
    from tau_bench.envs import get_env  # 惰性
    from tau_bench.envs.user import UserStrategy  # 惰性

    env = get_env(
        domain,
        user_strategy=UserStrategy.HUMAN,
        user_model="stub",  # HUMAN 策略不使用模型，占位即可
        task_split=split,
        task_index=0,
    )
    return [
        TaskInfo(
            task_id=i,
            user_id=t.user_id,
            instruction=t.instruction,
            action_names=tuple(a.name for a in t.actions),
            n_outputs=len(t.outputs),
            domain=domain,
            split=split,
        )
        for i, t in enumerate(env.tasks)
    ]


def n_tasks(domain: str = "retail", split: str = "test") -> int:
    """任务总数（惰性）。"""
    return len(list_tasks(domain, split))


def load_task(domain: str, split: str, task_id: int) -> TaskInfo:
    """加载单个任务的 DTO（骨架：从 list_tasks 取；MVP 可直 get_env(task_index)）。"""
    tasks = list_tasks(domain, split)
    if not 0 <= task_id < len(tasks):
        raise IndexError(f"task_id={task_id} 越界（共 {len(tasks)} 个任务）")
    return tasks[task_id]


def run_task_stub(task_id: int, *, domain: str = "retail", split: str = "test") -> TaskRunResult:
    """骨架：不调 agent，直接返回占位结果（reward=0, latency=0, success=False）。

    MVP 阶段替换为：``get_env(..., task_index=task_id)`` → ``agent.solve(env)`` →
    由 reward 算 success/latency。
    """
    _check_domain_split(domain, split)  # 早失败，参数错误立刻暴露
    return TaskRunResult(
        task_id=task_id,
        reward=0.0,
        success=False,
        latency_ms=0.0,
        n_steps=0,
        error=None,
    )


def _resolve_user_sim(
    *,
    engine_url: str,
    model: str,
    user_model: str | None,
    user_provider: str,
    user_api_base: str | None,
    user_api_key: str | None,
    api_key: str,
) -> tuple[str, str, dict[str, str]]:
    """解析 user-simulator 的 (user_model, user_provider, 环境变量覆盖)。

    三种情形：
    - **默认本地**（无外部参数）：user-sim 也走本地引擎（``engine_url``），完全开源可复现。
    - **外部 OpenAI 兼容**（``user_api_base`` 给定，如 GPT-4o 官方 / MiniMax / DeepSeek-API）：
      指向 ``user_api_base`` + ``user_api_key``，provider=openai。
    - **非 openai 系 provider**（``user_provider`` != openai，如 anthropic）：不动 ``OPENAI_*``，
      由用户自行 ``export`` 该 provider 的环境变量。

    返回的 env dict 由 :func:`run_task` 写入 ``os.environ``，仅供 litellm user-sim 读取；
    **agent 侧用显式 OpenAI client 直连本地引擎，不受此影响。**
    """
    if user_provider != "openai":
        return (user_model or model), user_provider, {}
    if user_api_base:
        env = {"OPENAI_API_BASE": user_api_base}
        if user_api_key is not None:
            env["OPENAI_API_KEY"] = user_api_key
        return (user_model or "gpt-4o"), user_provider, env
    # 默认：user-sim 走本地优化引擎（合规 + 可复现）
    return (user_model or model), user_provider, {
        "OPENAI_API_BASE": engine_url,
        "OPENAI_API_KEY": api_key,
    }


def run_task(
    task_id: int,
    *,
    domain: str = "retail",
    split: str = "test",
    engine_url: str,
    model: str,
    user_model: str | None = None,
    user_provider: str = "openai",
    user_api_base: str | None = None,
    user_api_key: str | None = None,
    api_key: str = "stub",
    max_steps: int = 30,
    priority: int = 0,
    middlewares: list | None = None,
) -> TaskRunResult:
    """真路径：用本地引擎跑 agent，返回带真 reward/latency/ttft 的结果。

    **agent 侧**：openai SDK 直连本地 ``engine_url``（被测/被优化的对象）。
    **user-simulator 侧**：走 litellm，端点由 :func:`_resolve_user_sim` 决定
    （默认本地引擎；可通过 ``user_api_base`` 切外部 OpenAI 兼容 API）。

    ``priority``：vLLM 调度优先级（0=最高，数值越大越先被踢，见 F5）。
    ``middlewares``：缝D 中间件链（F2/F3），透传给 :class:`TauBenchAgent`。
    """
    import os
    import time

    _check_domain_split(domain, split)
    from openai import OpenAI
    from tau_bench.envs import get_env  # 惰性
    from tau_bench.envs.user import UserStrategy  # 惰性

    from agent_mem.agent.tau_bench_agent import TauBenchAgent

    um, up, env_overrides = _resolve_user_sim(
        engine_url=engine_url, model=model, user_model=user_model,
        user_provider=user_provider, user_api_base=user_api_base,
        user_api_key=user_api_key, api_key=api_key,
    )
    os.environ.update(env_overrides)  # 仅供 litellm user-sim 读取

    env = get_env(
        domain,
        user_strategy=UserStrategy.LLM,
        user_model=um,
        task_split=split,
        user_provider=up,
        task_index=task_id,
    )
    client = OpenAI(base_url=engine_url, api_key=api_key)  # agent 侧：始终本地引擎
    agent = TauBenchAgent(client, model, priority=priority, middlewares=middlewares)

    t0 = time.monotonic()
    try:
        out = agent.solve(env, task_index=task_id, max_num_steps=max_steps)
        latency_ms = (time.monotonic() - t0) * 1000
        ttft_ms = median(out.ttft_ms_list) if out.ttft_ms_list else 0.0
        return TaskRunResult(
            task_id=task_id,
            reward=out.reward,
            success=is_successful(out.reward),
            latency_ms=latency_ms,
            n_steps=out.n_steps,
            error=None,
            ttft_ms=ttft_ms,
        )
    except Exception as e:  # noqa: BLE001 — 单任务失败不杀整个 run
        latency_ms = (time.monotonic() - t0) * 1000
        return TaskRunResult(
            task_id=task_id, reward=0.0, success=False, latency_ms=latency_ms,
            n_steps=0, error=repr(e),
        )
