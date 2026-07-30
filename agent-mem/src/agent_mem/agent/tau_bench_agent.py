"""τ-bench agent：通过兼容客户端连接本地 vLLM，并驱动环境取得真实 reward。

镜像官方 ``ToolCallingAgent`` 的消息协议（``message_to_action`` + respond 分支），
但把 LLM 调用换成兼容客户端直连本地 vLLM/stub，绕开 litellm。每个 tool_call 转
``Action(name, kwargs)`` 调 ``env.step``；respond 时 env 算 reward 并 done。

惰性 import：顶层**零** ``tau_bench.*`` import（会拖入 litellm），全在 ``solve()`` 内。
对 fake env + fake/stub client 可测（fake env 测试用 ``importorskip`` 守门）。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_mem.agent.react import _as_stack, stream_chat_with_ttft
from agent_mem.agent.usage_log import (
    log_prompt_tokens,
    measure_prompt_pair,
    prepare_prompt_meter,
    prompt_meter_enabled,
)
from agent_mem.context_telemetry import ContextEventSink
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


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Parse tool arguments, repairing consecutive JSON objects from local models."""
    if isinstance(raw_arguments, dict):
        return dict(raw_arguments)
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        return {}
    text = raw_arguments.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, TypeError):
        pass

    decoder = json.JSONDecoder()
    merged: dict[str, Any] = {}
    position = 0
    parsed_any = False
    try:
        while position < len(text):
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text):
                break
            value, position = decoder.raw_decode(text, position)
            if not isinstance(value, dict):
                return {}
            merged.update(value)
            parsed_any = True
    except (ValueError, TypeError):
        return {}
    return merged if parsed_any else {}


def _message_to_action(next_message: dict, Action: Any, respond_name: str) -> Any:
    """镜像 tau_bench message_to_action：有 tool_calls→工具 Action，否则→respond。"""
    tcs = next_message.get("tool_calls") or []
    if tcs and tcs[0].get("function"):
        tc = tcs[0]
        kwargs = _parse_tool_arguments(tc["function"].get("arguments"))
        return Action(name=tc["function"]["name"], kwargs=kwargs)
    return Action(name=respond_name, kwargs={"content": next_message.get("content") or ""})


def _canonicalize_tool_call(next_message: dict, action: Any) -> dict[str, Any]:
    """Store exactly one valid tool call matching the action sent to tau-bench."""
    tool_calls = next_message.get("tool_calls") or []
    source = tool_calls[0] if tool_calls and isinstance(tool_calls[0], dict) else {}
    source_function = source.get("function") or {}
    canonical = {
        "id": source.get("id") or "call",
        "type": "function",
        "function": {
            "name": action.name or source_function.get("name"),
            "arguments": json.dumps(
                dict(action.kwargs), ensure_ascii=False, separators=(",", ":")
            ),
        },
    }
    next_message["tool_calls"] = [canonical]
    return canonical


class TauBenchAgent:
    """驱动 τ-bench env 的 ReAct agent（推理由本地 vLLM 提供）。

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
        priority: int = 0,
        priority_fn: Callable[[], int] | None = None,
        on_turn_start: Callable[[int], None] | None = None,
        middlewares: MiddlewareStack | Sequence[Middleware] | None = None,
        context_event_sink: ContextEventSink | None = None,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.priority = priority
        # F5：动态优先级（priority_fn 每轮读 session 当前 priority；None → 静态 priority）
        self._priority_fn = priority_fn
        # F5：每轮开始回调（driver 接 mgr.touch + 策略 mark_active，标记活跃、回落 priority）
        self._on_turn_start = on_turn_start
        self.context_event_sink = context_event_sink
        # extra_body：关闭 thinking + 透传 priority 给 vLLM 调度器（priority 每轮刷新）
        body: dict[str, Any] = {}
        if not enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body["priority"] = self._current_priority()
        self.extra_body = body
        # 缝D：上下文中间件（F2 压缩 / F3 lazy-load）。None → 空 stack = identity。
        self.stack: MiddlewareStack = _as_stack(middlewares)
        self.stack.prepare()
        prepare_prompt_meter(model)

    def _current_priority(self) -> int:
        """当前调度优先级：有 ``priority_fn`` 则调它（异常回退静态 priority），否则静态。"""
        if self._priority_fn is not None:
            try:
                return int(self._priority_fn())
            except Exception:
                return self.priority
        return self.priority

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
        ctx = MiddlewareContext(
            session_id=f"tau-{task_index}",
            event_sink=self.context_event_sink,
        )

        for _ in range(max_num_steps):
            steps += 1
            ctx.bump_step()
            # F5：每轮开始回调（标记 session 活跃）+ 刷新动态 priority 透传给 vLLM
            if self._on_turn_start is not None:
                self._on_turn_start(steps)
            self.extra_body["priority"] = self._current_priority()
            if prompt_meter_enabled() or ctx.telemetry_enabled:
                baseline_messages, baseline_tools = self.stack.measurement_baseline(
                    messages, env.tools_info, ctx
                )
            else:
                baseline_messages, baseline_tools = messages, env.tools_info
            # 缝D：联合变换 messages/tools（副本），正典输入不动
            to_send, to_tools = self.stack.transform_request(
                messages, env.tools_info, ctx
            )
            token_measurement = measure_prompt_pair(
                model=self.model,
                original_messages=baseline_messages,
                transformed_messages=to_send,
                original_tools=baseline_tools,
                transformed_tools=to_tools,
                extra_body=self.extra_body,
                force=ctx.telemetry_enabled,
            )
            ctx.emit("prompt.measured", token_measurement)
            # 流式调用：拿到 message dict + 本步 TTFT
            next_message, ttft_s, prompt_tokens = stream_chat_with_ttft(
                self.client,
                model=self.model,
                messages=to_send,
                tools=to_tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=self.extra_body,
            )
            log_prompt_tokens(ctx, prompt_tokens, token_measurement)
            ctx.emit("prompt.completed", {
                **token_measurement,
                "prompt_tokens": prompt_tokens,
                "source": (
                    "response.usage.prompt_tokens"
                    if prompt_tokens is not None else "unavailable"
                ),
            })
            self.stack.after_model_call(prompt_tokens, ctx)
            ttft_ms_list.append(ttft_s * 1000)
            action = _message_to_action(next_message, Action, RESPOND_ACTION_NAME)

            if action.name != RESPOND_ACTION_NAME:
                tc = _canonicalize_tool_call(next_message, action)
                messages.append(next_message)
                ctx.tool_call_id = str(tc.get("id") or "")
                ctx.tool_call_index = 0
                internal = self.stack.handle_internal_tool_call(
                    action.name, dict(action.kwargs), ctx
                )
                if internal is not None:
                    obs = self.stack.intercept_tool_result(
                        action.name, dict(action.kwargs), internal.content, ctx
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "content": obs,
                    })
                    ctx.tool_call_id = None
                    ctx.tool_call_index = None
                    continue
                env_response = env.step(action)
                reward = env_response.reward
                if hasattr(env_response.info, "model_dump"):
                    info.update(env_response.info.model_dump())
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
                ctx.tool_call_id = None
                ctx.tool_call_index = None
            else:
                env_response = env.step(action)
                reward = env_response.reward
                if hasattr(env_response.info, "model_dump"):
                    info.update(env_response.info.model_dump())
                messages.append(next_message)
                messages.append({"role": "user", "content": env_response.observation})

            if env_response.done:
                break

        return SolveOutcome(
            reward=reward, messages=messages, info=info, n_steps=steps,
            ttft_ms_list=ttft_ms_list,
        )
