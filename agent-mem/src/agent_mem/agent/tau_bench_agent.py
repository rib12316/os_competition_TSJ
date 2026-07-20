"""τ-bench agent：自定义 ``Agent`` 子类，用 openai SDK 驱动 τ-bench env 拿真 reward。

镜像官方 ``ToolCallingAgent`` 的消息协议（``message_to_action`` + respond 分支），
但把 LLM 调用换成 openai SDK 直连我们的引擎/stub，绕开 litellm。每个 tool_call 转
``Action(name, kwargs)`` 调 ``env.step``；respond 时 env 算 reward 并 done。

惰性 import：顶层**零** ``tau_bench.*`` import（会拖入 litellm），全在 ``solve()`` 内。
对 fake env + fake/stub client 可测（fake env 测试用 ``importorskip`` 守门）。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_mem.agent.react import _as_stack, stream_chat_with_ttft
from agent_mem.middleware import Middleware, MiddlewareContext, MiddlewareStack


@dataclass
class SolveOutcome:
    """solve 结果（与 tau_bench SolveResult 解耦；runner 只读 reward/n_steps/ttft）。"""

    reward: float
    messages: list[dict] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    n_steps: int = 0
    total_cost: float | None = None
    ttft_ms_list: list[float] = field(default_factory=list)  # 每步 TTFT（首 token 时间）


def _message_to_action(next_message: dict, Action: Any, respond_name: str) -> Any:
    """镜像 tau_bench message_to_action：有 tool_calls→工具 Action，否则→respond。"""
    tcs = next_message.get("tool_calls") or []
    if tcs and tcs[0].get("function"):
        tc = tcs[0]
        try:
            kwargs = json.loads(tc["function"].get("arguments") or "{}")
        except (ValueError, TypeError):
            kwargs = {}
        return Action(name=tc["function"]["name"], kwargs=kwargs)
    return Action(name=respond_name, kwargs={"content": next_message.get("content") or ""})


class TauBenchAgent:
    """驱动 τ-bench env 的 ReAct agent（openai SDK）。

    不继承 tau_bench.agents.base.Agent（避免顶层 import tau_bench）；duck-type 兼容。
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        enable_thinking: bool = False,
        middlewares: MiddlewareStack | Sequence[Middleware] | None = None,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Qwen3 默认开 thinking（<think>…</think>），benchmark 关掉以加速 + 省 token；
        # 经 chat_template_kwargs 透传给 vLLM 的 chat template。
        self.extra_body = (
            None if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}
        )
        # 缝D：上下文中间件（F2 压缩 / F3 lazy-load）。None → 空 stack = identity。
        self.stack: MiddlewareStack = _as_stack(middlewares)

    def solve(self, env: Any, task_index: int | None = None, max_num_steps: int = 30) -> SolveOutcome:
        # 惰性 import（触发 litellm 仅在此处）
        from tau_bench.types import RESPOND_ACTION_NAME, Action

        reset = env.reset(task_index=task_index)
        messages: list[dict] = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": reset.observation},
        ]
        info: dict[str, Any] = {}
        if hasattr(reset.info, "model_dump"):
            info.update(reset.info.model_dump())
        reward = 0.0
        steps = 0
        ttft_ms_list: list[float] = []
        # 缝D：每 session 一份 context（session_id 透传给 F5/F6）
        ctx = MiddlewareContext(session_id=f"tau-{task_index}")

        for _ in range(max_num_steps):
            steps += 1
            ctx.bump_step()
            # 缝D：发引擎前变换 messages（副本），正典 messages 不动
            to_send = self.stack.transform_messages(messages, ctx)
            # 流式调用：拿到 message dict + 本步 TTFT
            next_message, ttft_s = stream_chat_with_ttft(
                self.client,
                model=self.model,
                messages=to_send,
                tools=env.tools_info,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=self.extra_body,
            )
            ttft_ms_list.append(ttft_s * 1000)
            action = _message_to_action(next_message, Action, RESPOND_ACTION_NAME)
            env_response = env.step(action)
            reward = env_response.reward
            if hasattr(env_response.info, "model_dump"):
                info.update(env_response.info.model_dump())

            if action.name != RESPOND_ACTION_NAME:
                tcs = next_message.get("tool_calls") or []
                next_message["tool_calls"] = tcs[:1]  # τ-bench 每步一个 action
                tc = tcs[0] if tcs else {"id": "x", "function": {"name": action.name}}
                messages.append(next_message)
                # 缝D：工具（env）返回值回灌前拦截（F3 把长 JSON 换成引用）
                obs = self.stack.intercept_tool_result(
                    action.name, dict(action.kwargs), env_response.observation, ctx
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": obs,
                })
            else:
                messages.append(next_message)
                messages.append({"role": "user", "content": env_response.observation})

            if env_response.done:
                break

        return SolveOutcome(
            reward=reward, messages=messages, info=info, n_steps=steps,
            ttft_ms_list=ttft_ms_list,
        )
