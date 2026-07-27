"""Interactive LongBench 2WikiMQA runner with F2/F3 telemetry for the demo UI."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_mem.context_telemetry import ContextEventSink
    from agent_mem.middleware import Middleware, MiddlewareStack


def _trunc(value: Any, limit: int = 800) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else f"{text[:limit]} ...(+{len(text) - limit} chars)"


def longbench_messages_to_chatbot(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Render canonical ReAct messages without sending full LongBench documents to the browser."""
    output: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            continue
        if role == "user":
            output.append({"role": "user", "content": _trunc(content)})
            continue
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for tool_call in tool_calls:
                    function = tool_call.get("function") or {}
                    output.append({
                        "role": "assistant",
                        "content": (
                            f"调用工具 `{function.get('name', '?')}`\n"
                            f"```json\n{_trunc(function.get('arguments'), 400)}\n```"
                        ),
                    })
            elif content:
                output.append({"role": "assistant", "content": _trunc(content)})
            continue
        if role == "tool":
            output.append({
                "role": "assistant",
                "content": (
                    f"工具结果（`{message.get('name', '')}`）\n"
                    f"```\n{_trunc(content, 500)}\n```"
                ),
            })
    return output


def run_longbench_task_streaming(
    *,
    data_zip: str,
    task_id: int,
    engine_url: str,
    model: str,
    api_key: str = "EMPTY",
    max_steps: int = 8,
    max_tokens: int = 256,
    middlewares: MiddlewareStack | Sequence[Middleware] | None = None,
    context_event_sink: ContextEventSink | None = None,
    client: Any = None,
) -> Iterator[tuple[list[dict[str, str]], str]]:
    """Run one 2WikiMQA task and yield the same trace/status shape as tau-bench."""
    from agent_mem.agent.react import _as_stack, stream_chat_with_ttft
    from agent_mem.agent.usage_log import (
        log_prompt_tokens,
        measure_prompt_pair,
        prompt_meter_enabled,
    )
    from agent_mem.bench.tasks.longbench_adapter import (
        RETRIEVE_TOOL,
        TWO_WIKI_MQA_SYSTEM_PROMPT,
        answer_correct,
        context_payload,
        load_task,
    )
    from agent_mem.middleware import MiddlewareContext

    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=engine_url, api_key=api_key)

    task = load_task(data_zip, task_id)
    example = task.payload or {}
    payload, document_count = context_payload(str(example.get("context", "")))
    answers = [str(answer) for answer in example.get("answers", [])]
    stack = _as_stack(middlewares)
    session_id = f"longbench-{task.task_id}"
    ctx = MiddlewareContext(session_id=session_id, event_sink=context_event_sink)
    tools = [RETRIEVE_TOOL]
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": TWO_WIKI_MQA_SYSTEM_PROMPT},
        {"role": "user", "content": str(example.get("input", ""))},
    ]
    started = time.monotonic()

    try:
        stack.prepare()
        yield (
            longbench_messages_to_chatbot(messages),
            f"任务 #{task.task_id}（2WikiMQA）开始：候选文档 {document_count} 篇",
        )

        for step in range(1, max(1, int(max_steps)) + 1):
            try:
                ctx.bump_step()
                if prompt_meter_enabled() or ctx.telemetry_enabled:
                    baseline_messages, baseline_tools = stack.measurement_baseline(
                        messages, tools, ctx
                    )
                else:
                    baseline_messages, baseline_tools = messages, tools
                to_send, to_tools = stack.transform_request(messages, tools, ctx)
                measurement = measure_prompt_pair(
                    model=model,
                    original_messages=baseline_messages,
                    transformed_messages=to_send,
                    original_tools=baseline_tools,
                    transformed_tools=to_tools,
                    extra_body=extra_body,
                    force=ctx.telemetry_enabled,
                )
                ctx.emit("prompt.measured", measurement)
                next_message, ttft, prompt_tokens = stream_chat_with_ttft(
                    client,
                    model=model,
                    messages=to_send,
                    tools=to_tools,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                log_prompt_tokens(ctx, prompt_tokens, measurement)
                ctx.emit("prompt.completed", {
                    **measurement,
                    "prompt_tokens": prompt_tokens,
                    "source": (
                        "response.usage.prompt_tokens"
                        if prompt_tokens is not None else "unavailable"
                    ),
                })
                stack.after_model_call(prompt_tokens, ctx)
                messages.append(next_message)

                tool_calls = next_message.get("tool_calls") or []
                if not tool_calls:
                    final_text = str(next_message.get("content") or "")
                    success = answer_correct(final_text, answers)
                    verdict = "成功" if success else "未命中"
                    elapsed_ms = (time.monotonic() - started) * 1000
                    gold = " / ".join(answers) or "(none)"
                    yield (
                        longbench_messages_to_chatbot(messages),
                        f"结束：{verdict} | 步数={step} | e2e={elapsed_ms:.0f} ms | "
                        f"TTFT={ttft * 1000:.0f} ms | 预测={_trunc(final_text, 240)} | "
                        f"gold={_trunc(gold, 240)}",
                    )
                    return

                for tool_call_index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function") or {}
                    name = str(function.get("name") or "")
                    call_id = str(tool_call.get("id") or f"call-{step}-{tool_call_index}")
                    ctx.tool_call_id = call_id
                    ctx.tool_call_index = tool_call_index
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                        if not isinstance(arguments, dict):
                            arguments = {}
                    except (TypeError, ValueError):
                        arguments = {}
                    try:
                        internal = stack.handle_internal_tool_call(name, arguments, ctx)
                        if internal is not None:
                            observation = internal.content
                        elif name == "retrieve_documents":
                            observation = payload
                        else:
                            raise ValueError(f"unexpected business tool: {name}")
                    except Exception as exc:  # noqa: BLE001 - feed tool errors back to the agent
                        observation = f"tool error: {exc}"
                    observation = stack.intercept_tool_result(
                        name, arguments, str(observation), ctx
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": observation,
                    })
                    ctx.tool_call_id = None
                    ctx.tool_call_index = None

                yield (
                    longbench_messages_to_chatbot(messages),
                    f"步骤 {step}/{max_steps}：{len(tool_calls)} 次工具调用，"
                    f"TTFT={ttft * 1000:.0f} ms",
                )
            except Exception as exc:  # noqa: BLE001 - visible per-task failure
                messages.append({
                    "role": "assistant",
                    "content": f"LongBench 第 {step} 步失败：{exc}",
                })
                yield longbench_messages_to_chatbot(messages), f"第 {step} 步失败：{exc}"
                return

        elapsed_ms = (time.monotonic() - started) * 1000
        yield (
            longbench_messages_to_chatbot(messages),
            f"达到 max_steps={max_steps}，运行 {elapsed_ms:.0f} ms 后停止",
        )
    finally:
        for middleware in stack.middlewares:
            close = getattr(middleware, "close", None)
            if close is not None:
                close()
