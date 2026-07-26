"""τ-bench 任务流式运行器（demo 用）：把真实任务跑进对话页。

**重导入惰性**：``tau_bench`` / ``litellm``（~6s）只在 :func:`run_tau_task_streaming`
内部 import，模块本身 import 廉价——避免拖慢 demo 启动 / 污染自由对话路径。

流程（镜像 :func:`agent_mem.agent.tau_bench_agent.TauBenchAgent.solve`，但改成
**生成器**逐步 yield，供 Gradio 流式刷聊天框；不修改核心 agent 文件）::

    get_env(domain, LLM user-sim → 本地引擎) → reset → 循环：
      流式调引擎(stream_chat_with_ttft) → message_to_action → env.step
      → 回灌 tool 结果 / user-sim 下一句 → yield 当前对话 → 直到 done

每个 LLM 调用都打本地引擎，故右侧监控的 KV/TTFT/延迟/吞吐曲线会在任务跑动时实时变化。
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_mem.context_telemetry import ContextEventSink
    from agent_mem.middleware import Middleware, MiddlewareStack

_TASK_CACHE: dict[tuple[str, str], list[Any]] = {}


def _trunc(s: str | None, n: int = 300) -> str:
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + f" …(+{len(s) - n}字)"


def list_tau_tasks(domain: str = "retail", split: str = "test") -> list[Any]:
    """惰性枚举任务（带缓存，避免重复 6s 加载）。返回 :class:`TaskInfo` 列表。"""
    key = (domain, split)
    if key not in _TASK_CACHE:
        from agent_mem.bench.tasks.tau_bench_adapter import list_tasks

        _TASK_CACHE[key] = list_tasks(domain, split)
    return _TASK_CACHE[key]


def tau_messages_to_chatbot(messages: list[dict]) -> list[dict]:
    """把 τ-bench messages（system/user/assistant/tool）映射成 Gradio 聊天气泡。

    - system(wiki)：跳过（太长）
    - user：用户气泡（任务指令 / user-sim 下一句）
    - assistant + tool_calls：助手气泡「🔧 调用工具 `name`」+ 参数
    - assistant 纯文本：助手气泡
    - tool 结果：助手气泡「↩️ 工具结果」+ 截断
    """
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            continue
        if role == "user":
            out.append({"role": "user", "content": _trunc(content, 800)})
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                tc = tcs[0]
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                out.append({
                    "role": "assistant",
                    "content": f"🔧 调用工具 `{fn.get('name','?')}`\n```json\n{_trunc(args, 300)}\n```",
                })
            elif content:
                out.append({"role": "assistant", "content": _trunc(content, 800)})
        elif role == "tool":
            out.append({
                "role": "assistant",
                "content": f"↩️ 工具结果（`{m.get('name','')}`）:\n```\n{_trunc(content, 300)}\n```",
            })
    return out


def _is_success(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def run_tau_task_streaming(
    *,
    domain: str,
    split: str,
    task_id: int,
    engine_url: str,
    model: str,
    api_key: str = "EMPTY",
    max_steps: int = 20,
    middlewares: MiddlewareStack | Sequence[Middleware] | None = None,
    context_event_sink: ContextEventSink | None = None,
) -> Iterator[tuple[list[dict], str]]:
    """流式跑一个 τ-bench 任务，逐步 yield ``(chatbot_history, status_text)``。

    每步一次 LLM 调用（打本地引擎），故右侧监控曲线会随之动。agent 侧用显式
    OpenAI client 直连本地引擎；user-sim 走 litellm（``OPENAI_API_BASE`` 指本地引擎）。
    """
    # 惰性重导入
    from openai import OpenAI

    from agent_mem.agent.react import _as_stack, stream_chat_with_ttft
    from agent_mem.agent.usage_log import (
        log_prompt_tokens,
        measure_prompt_pair,
        prompt_meter_enabled,
    )
    from agent_mem.agent.tau_bench_agent import _message_to_action
    from agent_mem.middleware import MiddlewareContext
    from tau_bench.envs import get_env
    from tau_bench.envs.user import UserStrategy
    from tau_bench.types import RESPOND_ACTION_NAME, Action

    stack = _as_stack(middlewares)
    ctx = MiddlewareContext(session_id=f"tau-{task_id}", event_sink=context_event_sink)
    previous_env = {
        "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    }
    os.environ["OPENAI_API_BASE"] = engine_url
    os.environ["OPENAI_API_KEY"] = api_key

    try:
        stack.prepare()
        try:
            env = get_env(
                domain,
                user_strategy=UserStrategy.LLM,
                user_model=model,
                user_provider="openai",
                task_split=split,
                task_index=task_id,
            )
        except Exception as e:  # noqa: BLE001
            yield ([{"role": "assistant", "content": f"❌ 构建 τ-bench 环境失败：{e}"}], "环境构建失败")
            return

        client = OpenAI(base_url=engine_url, api_key=api_key)
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            reset = env.reset(task_index=task_id)
        except Exception as e:  # noqa: BLE001
            yield ([{"role": "assistant", "content": f"❌ env.reset 失败：{e}"}], "reset 失败")
            return

        messages: list[dict] = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": reset.observation},
        ]
        yield tau_messages_to_chatbot(messages), f"▶ 任务 #{task_id}（{domain}/{split}）开始…"

        reward = 0.0
        step = 0
        for step in range(1, max_steps + 1):
            try:
                ctx.bump_step()
                if prompt_meter_enabled() or ctx.telemetry_enabled:
                    baseline_messages, baseline_tools = stack.measurement_baseline(
                        messages, env.tools_info, ctx
                    )
                else:
                    baseline_messages, baseline_tools = messages, env.tools_info
                to_send, to_tools = stack.transform_request(
                    messages, env.tools_info, ctx
                )
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
                next_message, _ttft, prompt_tokens = stream_chat_with_ttft(
                    client,
                    model=model,
                    messages=to_send,
                    tools=to_tools,
                    temperature=0.0,
                    max_tokens=512,
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
                action = _message_to_action(next_message, Action, RESPOND_ACTION_NAME)

                if action.name != RESPOND_ACTION_NAME:
                    tcs = (next_message.get("tool_calls") or [])[:1]
                    next_message["tool_calls"] = tcs
                    tc = tcs[0] if tcs else {
                        "id": "x",
                        "function": {"name": action.name},
                    }
                    messages.append(next_message)
                    ctx.tool_call_id = str(tc.get("id") or "")
                    ctx.tool_call_index = 0
                    internal = stack.handle_internal_tool_call(
                        action.name, dict(action.kwargs), ctx
                    )
                    if internal is not None:
                        observation = stack.intercept_tool_result(
                            action.name,
                            dict(action.kwargs),
                            internal.content,
                            ctx,
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "content": observation,
                        })
                        ctx.tool_call_id = None
                        ctx.tool_call_index = None
                        yield (
                            tau_messages_to_chatbot(messages),
                            f"步骤 {step}/{max_steps}　F3 内部工具 `{action.name}`",
                        )
                        continue

                    env_response = env.step(action)
                    reward = env_response.reward
                    observation = stack.intercept_tool_result(
                        action.name,
                        dict(action.kwargs),
                        env_response.observation,
                        ctx,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "content": observation,
                    })
                    ctx.tool_call_id = None
                    ctx.tool_call_index = None
                else:
                    env_response = env.step(action)
                    reward = env_response.reward
                    messages.append(next_message)
                    messages.append({"role": "user", "content": env_response.observation})
            except Exception as e:  # noqa: BLE001 — 单步失败给出可见错误，不崩整个 UI
                messages.append({"role": "assistant", "content": f"⚠️ 第 {step} 步出错：{e}"})
                yield tau_messages_to_chatbot(messages), f"⚠️ 第 {step} 步出错"
                return

            done = env_response.done
            yield (
                tau_messages_to_chatbot(messages),
                f"步骤 {step}/{max_steps}　reward={reward:.2f}{'　✅ done' if done else ''}",
            )
            if done:
                break

        verdict = "✅ 成功" if _is_success(reward) else "❌ 未达标"
        yield tau_messages_to_chatbot(messages), f"🏁 结束　reward={reward:.2f}　{verdict}　共 {step} 步"
    finally:
        for key, previous in previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        for middleware in stack.middlewares:
            close = getattr(middleware, "close", None)
            if close is not None:
                close()


# ---- 并发 benchmark（前端展示用）----


def _conc_rows(task_ids: list[int], results: dict[int, dict]) -> list[list]:
    """会话状态表（list-of-lists，配 gradio DataFrame headers，避免 [object Object]）。"""
    rows: list[list] = []
    for tid in task_ids:
        if tid in results:
            r = results[tid]
            status = "✅" if r["success"] else "❌"
            err = r.get("error") or ""
            err = (err[:40] + "…") if len(err) > 40 else err
            rows.append([tid, status, round(r["reward"], 2), r["n_steps"], round(r["latency_ms"]), err])
        else:
            rows.append([tid, "运行中", "", "", "", ""])
    return rows


def _conc_snapshot(
    task_ids: list[int],
    results: dict[int, dict],
    total: int,
    concurrency: int,
) -> tuple[str, list[list], dict]:
    """实时进度：(进度 Markdown, 会话状态表, per-task results)。results 供 run_conc 做 CLI 同源聚合。"""
    done = len(results)
    success = sum(1 for r in results.values() if r["success"])
    rate = (success / done * 100) if done else 0.0
    md = (
        f"### 并发 {concurrency}：完成 **{done}/{total}**　✅ {success}　❌ {done - success}\n"
        f"**成功率：{rate:.1f}%**"
    )
    return md, _conc_rows(task_ids, results), results


def _conc_final(
    task_ids: list[int],
    results: dict[int, dict],
    total: int,
    concurrency: int,
) -> tuple[str, list[list]]:
    """跑完后的醒目最终结果：(最终得分 Markdown, 完整会话表)。"""
    done = len(results)
    success = sum(1 for r in results.values() if r["success"])
    fail = done - success
    rate = (success / done * 100) if done else 0.0
    lats = [r["latency_ms"] for r in results.values() if r["latency_ms"]]
    steps_list = [r["n_steps"] for r in results.values()]
    avg_lat = sum(lats) / len(lats) if lats else 0.0
    avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0.0
    badge = "🏆" if rate >= 50 else ("✅" if rate > 0 else "⚠️")
    md = (
        f"## {badge} 最终结果\n"
        f"### 成功率 **{rate:.1f}%**（✅ {success} / ❌ {fail}，共 {done} 会话）\n"
        f"| 指标 | 值 |\n|---|---|\n"
        f"| 并发数 | {concurrency} |\n"
        f"| 总任务数 | {total} |\n"
        f"| 平均端到端延迟 | {avg_lat:.0f} ms |\n"
        f"| 平均步数 | {avg_steps:.1f} |\n"
    )
    return md, _conc_rows(task_ids, results)


def run_task_into_convo(
    tid: int,
    convo_store: dict,
    *,
    domain: str,
    split: str,
    engine_url: str,
    model: str,
    api_key: str = "EMPTY",
    max_steps: int = 10,
):
    """跑一个 τ-bench 任务，每步把当前对话（气泡）写入 ``convo_store[tid]``，返回 TaskRunResult。

    供并发 run 实时查看各会话对话。镜像 :func:`run_tau_task_streaming` 的循环，但写成
    普通函数（在线程里跑）+ 写共享 dict（而非 yield）。线程安全靠 CPython GIL（单键读写原子）。
    """
    import os
    import time

    from openai import OpenAI

    from agent_mem.agent.react import stream_chat_with_ttft
    from agent_mem.agent.tau_bench_agent import _message_to_action
    from agent_mem.bench.stats import median
    from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult, is_successful
    from tau_bench.envs import get_env
    from tau_bench.envs.user import UserStrategy
    from tau_bench.types import RESPOND_ACTION_NAME, Action

    os.environ["OPENAI_API_BASE"] = engine_url
    os.environ["OPENAI_API_KEY"] = api_key
    t0 = time.monotonic()
    try:
        env = get_env(
            domain, user_strategy=UserStrategy.LLM, user_model=model,
            user_provider="openai", task_split=split, task_index=tid,
        )
        client = OpenAI(base_url=engine_url, api_key=api_key)
        reset = env.reset(task_index=tid)
        messages: list[dict] = [
            {"role": "system", "content": env.wiki},
            {"role": "user", "content": reset.observation},
        ]
        convo_store[tid] = tau_messages_to_chatbot(messages)
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        reward = 0.0
        steps = 0
        ttfts: list[float] = []
        for step in range(1, max_steps + 1):
            steps = step
            nm, ttft, _prompt_tokens = stream_chat_with_ttft(
                client, model=model, messages=messages, tools=env.tools_info,
                temperature=0.0, max_tokens=512, extra_body=extra_body,
            )
            ttfts.append(ttft * 1000)
            action = _message_to_action(nm, Action, RESPOND_ACTION_NAME)
            er = env.step(action)
            reward = er.reward
            if action.name != RESPOND_ACTION_NAME:
                tcs = (nm.get("tool_calls") or [])[:1]
                nm["tool_calls"] = tcs
                tc = tcs[0] if tcs else {"id": "x", "function": {"name": action.name}}
                messages.append(nm)
                messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "name": tc["function"]["name"], "content": er.observation,
                })
            else:
                messages.append(nm)
                messages.append({"role": "user", "content": er.observation})
            convo_store[tid] = tau_messages_to_chatbot(messages)  # 每步更新（供 UI 实时读）
            if er.done:
                break
        return TaskRunResult(
            task_id=tid, reward=reward, success=is_successful(reward),
            latency_ms=(time.monotonic() - t0) * 1000, n_steps=steps, error=None,
            ttft_ms=median(ttfts) if ttfts else 0.0,
        )
    except Exception as e:  # noqa: BLE001
        return TaskRunResult(
            task_id=tid, reward=0.0, success=False,
            latency_ms=(time.monotonic() - t0) * 1000, n_steps=0, error=repr(e), ttft_ms=0.0,
        )


def run_concurrent_streaming(
    *,
    domain: str,
    split: str,
    task_ids: list[int],
    concurrency: int,
    engine_url: str,
    model: str,
    convo_store: dict,
    api_key: str = "EMPTY",
    max_steps: int = 10,
) -> Iterator[tuple[str, list[list], dict]]:
    """并发跑多个 τ-bench 会话，每完成一个 yield ``(进度 Markdown, 会话状态表, per-task results)``。

    底层用 :func:`agent_mem.bench.tasks.tau_bench_adapter.run_task`（每会话独立
    agent+env），ThreadPoolExecutor 控并发。每会话都打本地引擎 → 右侧监控曲线随
    并发负载实时变化（显存/KV/延迟/吞吐）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(task_ids)
    results: dict[int, dict] = {}
    yield _conc_snapshot(task_ids, results, total, concurrency)  # 初始：全部运行中

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as ex:
        futures = {
            ex.submit(
                run_task_into_convo, tid, convo_store,
                domain=domain, split=split,
                engine_url=engine_url, model=model,
                api_key=api_key, max_steps=max_steps,
            ): tid
            for tid in task_ids
        }
        for f in as_completed(futures):
            tid = futures[f]
            try:
                r = f.result()
                results[tid] = {
                    "reward": r.reward, "success": r.success, "n_steps": r.n_steps,
                    "latency_ms": r.latency_ms, "ttft_ms": r.ttft_ms, "error": r.error,
                }
            except Exception as e:  # noqa: BLE001 — 单会话失败不杀整批
                results[tid] = {
                    "reward": 0.0, "success": False, "n_steps": 0,
                    "latency_ms": 0.0, "ttft_ms": 0.0, "error": repr(e),
                }
            yield _conc_snapshot(task_ids, results, total, concurrency)
