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

from agent_mem.demo.f5_runtime import RequestAdmissionController

if TYPE_CHECKING:
    from agent_mem.context_telemetry import ContextEventSink
    from agent_mem.demo.f5_runtime import UserSimResponseCache
    from agent_mem.middleware import Middleware, MiddlewareStack

_TASK_CACHE: dict[tuple[str, str], list[Any]] = {}


class _ScriptedUserMessage:
    def __init__(self, content: str) -> None:
        self.content = content

    def model_dump(self) -> dict[str, str]:
        return {"role": "assistant", "content": self.content}


class _ScriptedUserResponse:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": _ScriptedUserMessage(content)})()]
        self._hidden_params = {"response_cost": 0.0}


def _scripted_user_completion(*args: Any, **kwargs: Any) -> _ScriptedUserResponse:
    """Return a deterministic local customer response for F5 load tests."""
    messages = kwargs.get("messages") or []
    system = next(
        (
            str(message.get("content") or "")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "system"
        ),
        "",
    )
    has_prior_user_turn = any(
        isinstance(message, dict) and message.get("role") == "assistant"
        for message in messages
    )
    if not has_prior_user_turn:
        instruction = system.partition("Instruction:")[2].partition("\nRules:")[0].strip()
        content = instruction or "Please help me complete the request."
    else:
        content = "Please continue with the request using the details already provided."
    return _ScriptedUserResponse(content)


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
    user_model: str = "mimo-v2.5-pro",
    user_provider: str = "openai",
    user_api_base: str = "https://token-plan-cn.xiaomimimo.com/v1",
    user_api_key: str | None = None,
    max_steps: int = 20,
    middlewares: MiddlewareStack | Sequence[Middleware] | None = None,
    context_event_sink: ContextEventSink | None = None,
) -> Iterator[tuple[list[dict], str]]:
    """流式跑一个 τ-bench 任务，逐步 yield ``(chatbot_history, status_text)``。

    每步一次 LLM 调用（打本地引擎），故右侧监控曲线会随之动。agent 侧用显式
    OpenAI client 直连本地引擎；user-sim 默认通过 litellm 使用 MIMO。
    """
    # 惰性重导入
    from openai import OpenAI
    from tau_bench.envs import get_env
    from tau_bench.envs.user import UserStrategy
    from tau_bench.types import RESPOND_ACTION_NAME, Action

    from agent_mem.agent.react import _as_stack, stream_chat_with_ttft
    from agent_mem.agent.tau_bench_agent import (
        _canonicalize_tool_call,
        _message_to_action,
    )
    from agent_mem.agent.usage_log import (
        log_prompt_tokens,
        measure_prompt_pair,
        prompt_meter_enabled,
    )
    from agent_mem.bench.tasks.tau_bench_adapter import _resolve_user_sim
    from agent_mem.middleware import MiddlewareContext

    stack = _as_stack(middlewares)
    ctx = MiddlewareContext(session_id=f"tau-{task_id}", event_sink=context_event_sink)
    resolved_user_model, resolved_user_provider, user_env = _resolve_user_sim(
        engine_url=engine_url,
        model=model,
        user_model=user_model,
        user_provider=user_provider,
        user_api_base=user_api_base,
        user_api_key=user_api_key,
        api_key=api_key,
    )
    previous_env = {
        "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    }
    os.environ.update(user_env)

    try:
        if user_api_base and not user_api_key:
            yield ([{
                "role": "assistant",
                "content": "❌ MIMO user-sim 未启动：环境变量 MIMO_KEY 未设置。",
            }], "MIMO_KEY 未设置")
            return
        stack.prepare()
        try:
            env = get_env(
                domain,
                user_strategy=UserStrategy.LLM,
                user_model=resolved_user_model,
                user_provider=resolved_user_provider,
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
                    tc = _canonicalize_tool_call(next_message, action)
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


def _conc_rows(
    task_ids: list[int],
    results: dict[int, dict],
    scheduler: RequestAdmissionController | None = None,
) -> list[list]:
    """会话状态表（list-of-lists，配 gradio DataFrame headers，避免 [object Object]）。"""
    scheduler_sessions = (scheduler.snapshot().get("sessions") or {}) if scheduler else {}
    rows: list[list] = []
    for tid in task_ids:
        session = scheduler_sessions.get(f"tau-{tid}", {})
        if tid in results:
            r = results[tid]
            status = "✅" if r["success"] else "❌"
            err = r.get("error") or ""
            err = (err[:40] + "…") if len(err) > 40 else err
            rows.append([
                tid,
                status,
                session.get("priority", ""),
                r["n_steps"],
                round(r["reward"], 2),
                round(r["latency_ms"]),
                err,
            ])
        else:
            rows.append([
                tid,
                session.get("state", "QUEUED"),
                session.get("priority", ""),
                session.get("step", ""),
                "",
                "",
                "",
            ])
    return rows


def _conc_snapshot(
    task_ids: list[int],
    results: dict[int, dict],
    total: int,
    concurrency: int,
    scheduler: RequestAdmissionController | None = None,
) -> tuple[str, list[list], dict]:
    """实时进度：(进度 Markdown, 会话状态表, per-task results)。results 供 run_conc 做 CLI 同源聚合。"""
    done = len(results)
    success = sum(1 for r in results.values() if r["success"])
    rate = (success / done * 100) if done else 0.0
    scheduler_text = ""
    if scheduler is not None:
        snap = scheduler.snapshot()
        scheduler_text = (
            f"　请求槽 **{snap['active_requests']}/{snap['current_limit']}**"
            f"　延迟准入 **{snap['deferred_requests']}**"
        )
    md = (
        f"### 并发 {concurrency}：完成 **{done}/{total}**　任务达成 {success}　未达成 {done - success}\n"
        f"**max_steps 内任务达成率：{rate:.1f}%**{scheduler_text}"
    )
    return md, _conc_rows(task_ids, results, scheduler), results


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
        f"### max_steps 内任务达成率 **{rate:.1f}%**"
        f"（达成 {success} / 未达成 {fail}，共 {done} 会话）\n"
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
    max_tokens: int = 512,
    user_model: str = "mimo-v2.5-pro",
    priority_strategy: str = "combined",
    think_time_profiles: list | None = None,
    request_controller: RequestAdmissionController | None = None,
    seed: int = 42,
):
    """跑一个 τ-bench 任务，每步把当前对话（气泡）写入 ``convo_store[tid]``，返回 TaskRunResult。

    供并发 run 实时查看各会话对话。镜像 :func:`run_tau_task_streaming` 的循环，但写成
    普通函数（在线程里跑）+ 写共享 dict（而非 yield）。线程安全靠 CPython GIL（单键读写原子）。
    """
    import random
    import time

    from openai import OpenAI
    from tau_bench.envs import get_env
    from tau_bench.envs.user import UserStrategy
    from tau_bench.types import RESPOND_ACTION_NAME, Action

    from agent_mem.agent.react import stream_chat_with_ttft
    from agent_mem.agent.tau_bench_agent import (
        _canonicalize_tool_call,
        _message_to_action,
    )
    from agent_mem.bench.stats import median
    from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult, is_successful

    # 不设 os.environ：agent 走 OpenAI(base_url=engine_url) 直连；
    # user-sim 走 litellm 读 os.environ（mimo），由 run_concurrent_streaming 在起线程前统一设好。
    t0 = time.monotonic()
    steps = 0
    total_prompt_tokens = 0
    try:
        env = get_env(
            domain, user_strategy=UserStrategy.LLM, user_model=user_model,
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
        ttfts: list[float] = []
        rng = random.Random(int(seed) + tid)
        think_range = think_time_profiles[tid % len(think_time_profiles)] if think_time_profiles else None
        last_request_end: float | None = None
        ewma_gap = 0.0
        session_id = f"tau-{tid}"
        for step in range(1, max_steps + 1):
            steps = step
            # think-time 用户频率仿真（step>1 时注入 inter-turn 延迟，模拟异质用户活跃度）
            if step > 1 and think_range:
                time.sleep(rng.uniform(*think_range))
            # F5 优先级策略（透传给 vLLM 调度器 via extra_body；值越大越先被抢占）
            now = time.monotonic()
            if priority_strategy == "progress":
                extra_body["priority"] = max(0, round((1 - step / max(max_steps, 1)) * 100))
            elif priority_strategy == "combined":
                if last_request_end is not None:
                    gap = max(0.0, now - last_request_end)
                    ewma_gap = gap if ewma_gap == 0.0 else 0.7 * ewma_gap + 0.3 * gap
                extra_body["priority"] = min(70, int(ewma_gap * 3.5)) + max(0, 30 - int(step * 1.2))
            # fcfs: 不设 priority（vLLM 用默认 FCFS）
            priority = int(extra_body.get("priority", 0))
            acquired = False
            try:
                if request_controller is not None:
                    request_controller.before_request(
                        session_id, step=step, priority=priority
                    )
                    acquired = True
                nm, ttft, prompt_tokens = stream_chat_with_ttft(
                    client, model=model, messages=messages, tools=env.tools_info,
                    temperature=0.0, max_tokens=max_tokens, extra_body=extra_body,
                )
            finally:
                last_request_end = time.monotonic()
                if acquired and request_controller is not None:
                    request_controller.after_request(session_id)
            total_prompt_tokens += int(prompt_tokens or 0)
            ttfts.append(ttft * 1000)
            action = _message_to_action(nm, Action, RESPOND_ACTION_NAME)
            tc = (
                _canonicalize_tool_call(nm, action)
                if action.name != RESPOND_ACTION_NAME
                else None
            )
            er = env.step(action)
            reward = er.reward
            if action.name != RESPOND_ACTION_NAME:
                assert tc is not None
                messages.append(nm)
                messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": er.observation,
                })
            else:
                messages.append(nm)
                messages.append({"role": "user", "content": er.observation})
            convo_store[tid] = tau_messages_to_chatbot(messages)  # 每步更新（供 UI 实时读）
            if er.done:
                break
        if request_controller is not None:
            request_controller.mark_done(session_id)
        return TaskRunResult(
            task_id=tid, reward=reward, success=is_successful(reward),
            latency_ms=(time.monotonic() - t0) * 1000, n_steps=steps, error=None,
            ttft_ms=median(ttfts) if ttfts else 0.0,
            prompt_tokens=total_prompt_tokens,
        )
    except Exception as e:  # noqa: BLE001
        if request_controller is not None:
            request_controller.mark_done(f"tau-{tid}", error=repr(e))
        return TaskRunResult(
            task_id=tid, reward=0.0, success=False,
            latency_ms=(time.monotonic() - t0) * 1000,
            n_steps=steps,
            error=repr(e),
            ttft_ms=0.0,
            prompt_tokens=total_prompt_tokens,
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
    max_tokens: int = 512,
    user_model: str = "mimo-v2.5-pro",
    user_api_base: str = "https://token-plan-cn.xiaomimimo.com/v1",
    user_api_key: str | None = None,
    scripted_user: bool = False,
    priority_strategy: str = "combined",
    think_time_profiles: list | None = None,
    request_controller: RequestAdmissionController | None = None,
    user_response_cache: UserSimResponseCache | None = None,
    seed: int = 42,
) -> Iterator[tuple[str, list[list], dict]]:
    """并发跑多个 τ-bench 会话，每完成一个 yield ``(进度 Markdown, 会话状态表, per-task results)``。

    每会话使用独立 agent+env；传入 ``request_controller`` 时所有 session 均启动，
    但每轮模型请求必须先取得请求槽。每会话都打本地引擎，右侧监控随负载实时变化。
    """
    import os as _os
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    import litellm
    import tau_bench.envs.user as tau_user_module

    total = len(task_ids)
    results: dict[int, dict] = {}

    # F5 极限压力使用本地固定脚本；普通路径仍可使用 MIMO/local user-sim。
    if not scripted_user and not user_api_key:
        user_api_key = _os.environ.get("MIMO_KEY")
    previous_env = {
        "OPENAI_API_BASE": _os.environ.get("OPENAI_API_BASE"),
        "OPENAI_API_KEY": _os.environ.get("OPENAI_API_KEY"),
    }
    previous_litellm = {
        "api_base": getattr(litellm, "api_base", None),
        "api_key": getattr(litellm, "api_key", None),
    }
    previous_user_completion = tau_user_module.completion
    effective_user_model = user_model
    try:
        fallback_to_local = False
        if scripted_user:
            effective_api_base = "scripted://f5"
            effective_api_key = "unused"
        else:
            fallback_to_local = not (user_api_base and user_api_key)
            if fallback_to_local:
                effective_user_model = model
                effective_api_base = engine_url
                effective_api_key = api_key
            else:
                effective_api_base = user_api_base
                effective_api_key = user_api_key

            # This LiteLLM version does not reliably resolve the OpenAI-compatible endpoint
            # from OPENAI_API_* alone; tau-bench's user simulator also reads these globals.
            _os.environ["OPENAI_API_BASE"] = str(effective_api_base)
            _os.environ["OPENAI_API_KEY"] = str(effective_api_key)
            litellm.api_base = str(effective_api_base)
            litellm.api_key = str(effective_api_key)
        if scripted_user:
            tau_user_module.completion = _scripted_user_completion
        elif user_response_cache is not None:
            def cached_user_completion(*args, **kwargs):
                return user_response_cache.complete(
                    previous_user_completion, *args, **kwargs
                )

            tau_user_module.completion = cached_user_completion
        if fallback_to_local and user_api_base:
            yield ("MIMO_KEY 未设置，user-sim 使用本地引擎。", [], {})

        yield _conc_snapshot(task_ids, results, total, concurrency, request_controller)

        worker_count = (
            len(task_ids)
            if request_controller is not None
            else max(1, int(concurrency))
        )
        with ThreadPoolExecutor(max_workers=max(1, worker_count)) as ex:
            futures = {
                ex.submit(
                    run_task_into_convo, tid, convo_store,
                    domain=domain, split=split,
                    engine_url=engine_url, model=model,
                    api_key=api_key, max_steps=max_steps, max_tokens=max_tokens,
                    user_model=effective_user_model,
                    priority_strategy=priority_strategy,
                    think_time_profiles=think_time_profiles,
                    request_controller=request_controller,
                    seed=seed,
                ): tid
                for tid in task_ids
            }
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    tid = futures[future]
                    try:
                        result = future.result()
                        results[tid] = {
                            "reward": result.reward,
                            "success": result.success,
                            "n_steps": result.n_steps,
                            "latency_ms": result.latency_ms,
                            "ttft_ms": result.ttft_ms,
                            "prompt_tokens": result.prompt_tokens,
                            "error": result.error,
                        }
                    except Exception as e:  # noqa: BLE001 — 单会话失败不杀整批
                        results[tid] = {
                            "reward": 0.0, "success": False, "n_steps": 0,
                            "latency_ms": 0.0, "ttft_ms": 0.0,
                            "prompt_tokens": 0, "error": repr(e),
                        }
                yield _conc_snapshot(
                    task_ids, results, total, concurrency, request_controller
                )
    finally:
        tau_user_module.completion = previous_user_completion
        litellm.api_base = previous_litellm["api_base"]
        litellm.api_key = previous_litellm["api_key"]
        for key, previous in previous_env.items():
            if previous is None:
                _os.environ.pop(key, None)
            else:
                _os.environ[key] = previous
