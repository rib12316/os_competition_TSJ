"""ReAct 多轮 tool-calling 核心引擎（openai SDK 手写）。

循环：LLM ``chat.completions.create(tools=...)`` → 解析 ``tool_calls`` →
``execute_tool(name, args)`` 回灌 ``role=tool`` → 直到 LLM 不再调工具或达 ``max_steps``。

一个核心服务两个前端：通用 CLI agent（见 ``cli.py``）+ τ-bench agent（见
``tau_bench_agent.py``，其 ``execute_tool`` 走 ``env.step``）。对 stub server 可测。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_mem.middleware import Middleware, MiddlewareContext, MiddlewareStack

# 工具执行器签名：(name, args_dict) -> 观察文本
ExecuteTool = Callable[[str, dict[str, Any]], str]


def _as_stack(
    middlewares: MiddlewareStack | Sequence[Middleware] | None,
) -> MiddlewareStack:
    """把 list[Middleware] / None 规范成 MiddlewareStack（None → 空=identity）。"""
    if middlewares is None:
        return MiddlewareStack()
    if isinstance(middlewares, MiddlewareStack):
        return middlewares
    return MiddlewareStack(list(middlewares))


@dataclass
class ReactResult:
    final_text: str
    n_steps: int  # LLM 调用次数
    messages: list[dict] = field(default_factory=list)
    tool_calls_made: int = 0
    truncated: bool = False  # 是否因 max_steps 截断


def tool_call_to_dict(tc: Any) -> dict:
    """把 tool_call（openai 对象或 SimpleNamespace fake）转成可回灌 dict。"""
    if hasattr(tc, "model_dump"):
        return tc.model_dump(exclude_none=True)
    fn = tc.function
    fn_d = fn if isinstance(fn, dict) else {"name": fn.name, "arguments": fn.arguments}
    return {"id": tc.id, "type": getattr(tc, "type", "function"), "function": fn_d}


def assistant_message_to_dict(msg: Any) -> dict:
    """把 openai ChatCompletionMessage 转成可回灌的 dict（保留 tool_calls）。"""
    d: dict[str, Any] = {"role": "assistant", "content": msg.content}
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        d["tool_calls"] = [tool_call_to_dict(tc) for tc in tcs]
    return d


def stream_chat_with_ttft(
    client: Any,
    *,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Any], float]:
    """流式调用 + 测 TTFT（请求→首个 chunk 的秒数），返回 (可回灌 message dict, ttft_seconds)。

    在流里累积 ``delta.content`` 与 ``delta.tool_calls``（按 index 拼接 arguments 片段），
    重建出与非流式等价的 message dict，供 ReAct 循环继续用。
    """
    create_kw: dict[str, Any] = dict(
        model=model, messages=messages, tools=tools or None,
        temperature=temperature, stream=True,
    )
    if max_tokens is not None:
        create_kw["max_tokens"] = max_tokens
    if extra_body is not None:
        create_kw["extra_body"] = extra_body

    t0 = clock()
    stream = client.chat.completions.create(**create_kw)
    ttft: float | None = None
    content_parts: list[str] = []
    tc_acc: dict[int, dict[str, Any]] = {}

    for chunk in stream:
        if ttft is None:
            ttft = clock() - t0  # 首 chunk 到达
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        c = getattr(delta, "content", None)
        if c:
            content_parts.append(c)
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc, "index", 0)
            slot = tc_acc.setdefault(idx, {"id": None, "name": None, "args": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["args"] += fn.arguments

    if ttft is None:
        ttft = clock() - t0  # 流没产出任何 chunk

    msg: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts) or None}
    if tc_acc:
        msg["tool_calls"] = [
            {
                "id": s["id"] or "call",
                "type": "function",
                "function": {"name": s["name"], "arguments": s["args"] or "{}"},
            }
            for _, s in sorted(tc_acc.items())
        ]
    return msg, ttft


def run_react(
    client: Any,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    execute_tool: ExecuteTool,
    *,
    max_steps: int = 10,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    extra_body: dict[str, Any] | None = None,
    middlewares: MiddlewareStack | Sequence[Middleware] | None = None,
    session_id: str = "default",
) -> ReactResult:
    """跑 ReAct 循环，返回 :class:`ReactResult`。

    缝D 挂载点：``middlewares`` 非空时，每步发引擎前调
    :meth:`MiddlewareStack.transform_messages`（只变换发给引擎的副本，正典 ``msgs``
    不动），工具结果回灌前调 :meth:`MiddlewareStack.intercept_tool_result`（会改写
    进入正典历史的内容——lazy-load 的目的）。``None`` / 空栈 = identity，零开销。
    """
    stack = _as_stack(middlewares)
    ctx = MiddlewareContext(session_id=session_id)

    msgs = list(messages)
    n_steps = 0
    tool_calls_made = 0

    # 基础请求参数（messages 每步由中间件变换后注入）
    base_kw: dict[str, Any] = dict(
        model=model, tools=tools or None, temperature=temperature
    )
    if max_tokens is not None:
        base_kw["max_tokens"] = max_tokens
    if extra_body is not None:
        base_kw["extra_body"] = extra_body

    while n_steps < max_steps:
        n_steps += 1
        ctx.bump_step()
        # 缝D：发引擎前变换（副本），正典 msgs 不变
        to_send = stack.transform_messages(msgs, ctx)
        resp = client.chat.completions.create(**base_kw, messages=to_send)
        msg = resp.choices[0].message
        msgs.append(assistant_message_to_dict(msg))

        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            return ReactResult(
                final_text=msg.content or "",
                n_steps=n_steps,
                messages=msgs,
                tool_calls_made=tool_calls_made,
            )

        for tc in tcs:
            tool_calls_made += 1
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (ValueError, TypeError):
                args = {}
            try:
                obs = execute_tool(name, args)
            except Exception as e:  # noqa: BLE001 — 工具失败不杀 agent，把错误回灌
                obs = f"tool error: {e}"
            # 缝D：工具结果回灌前拦截（可改写进正典历史的内容）
            obs = stack.intercept_tool_result(name, args, str(obs), ctx)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": obs})

    return ReactResult(
        final_text="(max steps reached)",
        n_steps=n_steps,
        messages=msgs,
        tool_calls_made=tool_calls_made,
        truncated=True,
    )
