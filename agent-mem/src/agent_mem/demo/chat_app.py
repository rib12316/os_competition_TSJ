"""Gradio Blocks 演示应用：左列对话（自由对话 / τ-bench 任务）+ 右列多指标实时监控。

布局（一个浏览器窗口，两列；左列双标签，右侧监控共享）::

    ┌──── 左列：gr.Tabs ─────────────────────────────────────┐ ┌──── 右列：实时监控 ────┐
    │ 💬 自由对话：Chatbot + 输入（Qwen-Agent Assistant 流式）│ │ 引擎状态/当前值表      │
    │ 🎯 τ-bench 任务：domain/task_id → 运行真实客服任务     │ │ 6 指标实时子图：        │
    │   （agent 多轮 tool-calling，逐步流式刷对话）          │ │  HBM/KV/吞吐/TTFT/e2e/队列│
    │                                                        │ │ 历史 before/after 柱    │
    └────────────────────────────────────────────────────────┘ └────────────────────────┘

- 自由对话：``qwen_agent.agents.Assistant``，``llm.model_server`` 指向本地 vLLM。
- τ-bench：:mod:`agent_mem.demo.tau_bench_ui.run_tau_task_streaming` 流式跑真实任务。
- 监控：:class:`agent_mem.demo.monitor.LiveMonitor` 后台采 NPU HBM + vLLM ``/metrics``，
  ``gr.Timer`` 每 2s 重渲染右列。图表用 **plotly**（浏览器渲染，中文正常）。
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qwen_agent.agents import Assistant

from agent_mem.config import load_config
from agent_mem.context_telemetry import ContextEventBuffer
from agent_mem.demo import bench_runner, tau_bench_ui
from agent_mem.demo.engine_control import CONFIG_FLAGS, PENDING_CONFIGS, EngineManager
from agent_mem.demo.monitor import (
    HistoryConfig,
    LiveMonitor,
    Sample,
    WindowSeries,
    compute_window_series,
    engine_status,
    load_history,
)
from agent_mem.demo.overview import overview_html
from agent_mem.middleware import MiddlewareStack, middlewares_from_config

DEFAULT_ENGINE_URL = os.environ.get("AGENT_MEM_ENGINE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.environ.get("AGENT_MEM_MODEL", "Qwen2.5-7B-Instruct")
DEFAULT_HISTORY_DIR = os.environ.get("AGENT_MEM_HISTORY_DIR", "logs/mvp-newframework")
WINDOW_S = 10.0  # 窗口速率统计窗口（秒）
DEFAULT_RUN_ROOT = os.environ.get("AGENT_MEM_RUN_ROOT", "logs")

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
_TAU_CONTEXT_PRESETS = {
    "baseline": _CONFIGS_DIR / "baseline.yaml",
    "F2": _CONFIGS_DIR / "f2-compress.yaml",
    "F3": _CONFIGS_DIR / "f3-lazyload.yaml",
    "F2+F3": _CONFIGS_DIR / "f2-f3-combined.yaml",
}

# 统一 benchmark 场景 → preset（preset 编码 suite+middleware+session；引擎由上方按钮单独起）
BENCH_SCENARIOS: dict[str, str] = {
    "α baseline τ-bench（T0/T1 锚）": "agent-mem/configs/baseline.yaml",
    "α F5 并发回收（tau-bench·combined-evict·think-time）": "agent-mem/configs/unified-tau-freq.yaml",
    "β F2/F3 长上下文+工具（longbench·compress+lazyload）": "agent-mem/configs/unified-longbench.yaml",
    "α F1 显存（C8，需先起 +C8 引擎）": "agent-mem/configs/optimized.yaml",
    "α F4 分层（LMCache，需先起 +LMCache 引擎）": "agent-mem/configs/f4-lmcache.yaml",
}
_SCENARIO_GUIDE = (
    "### 场景 ↔ 功能 ↔ 引擎档位 对照（先起引擎，再选场景跑）\n"
    "| 演示功能 | bench 场景 | 🛠引擎按钮 | 显存上限 | 看什么指标 |\n|---|---|---|---|---|\n"
    "| **F5 动态回收** | α F5 并发回收 | **+priority(F5)** | **0.27** + **max-len 16384**(制压触发抢占) | 抢占→0 / KV命中 0.46→0.93 |\n"
    "| **F1 C8 显存** | α F1 显存 | **+C8(F1)** | 0.9 | 同 HBM token 容量 2× |\n"
    "| **F4 LMCache 分层** | α F4 分层 | **+LMCache(F4)** | 0.9 | 高并发不 OOM / p50↓ |\n"
    "| **F2 压缩 / F3 lazyload** | β F2/F3 长上下文 | baseline | 0.9 | prompt↓ / context↓（**填 data-zip**）|\n"
    "| **baseline 锚** | α baseline | baseline | 0.9 | 对照基线（T0/T1）|\n\n"
    "**流程**：① 🛠 选档位 + 设显存上限 → ▶起引擎 → ✅就绪 → ② 本页选场景 + 调参 → 🚀 → ③ 右侧实时看负载，完成后看本页结果。"
)


# ---- Qwen-Agent 消息工具 ----


def _assistant_text(response_list: list[Any]) -> str:
    """从 Assistant.run 的单次 yield（List[Message/dict]）里取最后一条 assistant 文本。

    yield 的元素可能是 Message 对象或 plain dict（取决于输入 messages 类型），
    两种都兼容：dict 走 ``.get``，Message 走 ``getattr``。
    """
    for msg in reversed(response_list):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "assistant":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _build_assistant(engine_url: str, model: str) -> Assistant:
    """构造指向本地 vLLM 的 Qwen-Agent Assistant（OpenAI 兼容，model_server 走 oai 路径）。"""
    llm_cfg = {
        "model": model,
        "model_server": engine_url,  # 以 http 开头 → 自动走 oai（OpenAI 兼容）
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
    }
    return Assistant(
        llm=llm_cfg,
        system_message="你是 agent-mem 演示助手。用中文简洁回答。",
        name="agent-mem-assistant",
    )


def _last(seq: list) -> Any:
    return seq[-1] if seq else None


def _fmt(v: float | None, unit: str = "", nd: int = 1) -> str:
    return f"{v:.{nd}f}{unit}" if v is not None else "N/A"


# ---- 图表（plotly）----


def _live_figure(series: WindowSeries, history: list[HistoryConfig]) -> go.Figure:
    """6 指标实时子图：HBM / KV命中率 / 吞吐 / TTFT / e2e延迟 / 队列。"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "NPU 显存 HBM (MB)", "KV 命中率 (%)", "吞吐 (tok/s)",
            "TTFT 首 token (ms)", "端到端延迟 (ms)", "在跑 / 等待 请求数",
        ],
    )
    t = series.t
    fig.add_trace(go.Scatter(x=t, y=series.mem, name="HBM(实时)", mode="lines",
                             line=dict(color="#2ca02c", width=2)), row=1, col=1)
    base_curve = next((h.mem_curve for h in history if h.config == "baseline" and h.mem_curve), [])
    if base_curve:
        bx, by = zip(*base_curve)
        fig.add_trace(go.Scatter(x=bx, y=by, name="baseline 参考", mode="lines",
                                 line=dict(color="#999", dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[None if v is None else v * 100 for v in series.kv_rate],
                             name="KV命中率", mode="lines", line=dict(color="#1f77b4")), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=series.throughput, name="吞吐", mode="lines",
                             line=dict(color="#ff7f0e")), row=1, col=3)
    fig.add_trace(go.Scatter(x=t, y=series.ttft, name="TTFT", mode="lines",
                             line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=series.e2e, name="e2e", mode="lines",
                             line=dict(color="#9467bd")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=series.running, name="running", mode="lines",
                             line=dict(color="#2ca02c")), row=2, col=3)
    fig.add_trace(go.Scatter(x=t, y=series.waiting, name="waiting", mode="lines",
                             line=dict(color="#d62728")), row=2, col=3)
    fig.update_layout(height=540, showlegend=False, template="plotly_white",
                      margin=dict(l=32, r=16, t=38, b=24))
    fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
    fig.update_xaxes(title_text="时间 (s)", row=2, col=2)
    fig.update_xaxes(title_text="时间 (s)", row=2, col=3)
    return fig


def _history_figure(history: list[HistoryConfig]) -> go.Figure:
    """历史 before/after 中位数对比柱（显存峰值 / KV命中率 / TTFT / e2e延迟）。"""
    if not history:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           text="无历史数据（跑 bench 后在 logs/ 生成）",
                           font=dict(size=14, color="#888"))
        fig.update_layout(height=320, template="plotly_white",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    cfgs = [h.config for h in history]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "显存峰值 (MB)", "KV 命中率 (%)", "TTFT (ms)", "端到端延迟 p50 (ms)",
    ])
    fig.add_trace(go.Bar(x=cfgs, y=[h.mem_peak_mb for h in history], name="显存峰值",
                         marker_color="#888", text=[f"{h.mem_peak_mb:.0f}" for h in history], textposition="outside"), row=1, col=1)
    fig.add_trace(go.Bar(x=cfgs, y=[h.kv_cache_hit_rate * 100 for h in history], name="KV命中率",
                         marker_color="#1f77b4", text=[f"{h.kv_cache_hit_rate*100:.1f}" for h in history], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(x=cfgs, y=[h.ttft_ms for h in history], name="TTFT",
                         marker_color="#d62728", text=[f"{h.ttft_ms:.0f}" for h in history], textposition="outside"), row=2, col=1)
    fig.add_trace(go.Bar(x=cfgs, y=[h.e2e_latency_p50_ms for h in history], name="e2e",
                         marker_color="#9467bd", text=[f"{h.e2e_latency_p50_ms:.0f}" for h in history], textposition="outside"), row=2, col=2)
    fig.update_layout(height=320, showlegend=False, template="plotly_white",
                      margin=dict(l=32, r=16, t=42, b=24))
    return fig


def _runs_figure(runs: list[dict]) -> go.Figure:
    """从 session 内累积的运行结果画对比图（6 指标 × 各运行）。"""
    if not runs:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           text="运行「📊 并发 benchmark」后，结果会自动汇总到这里做对比",
                           font=dict(size=14, color="#888"))
        fig.update_layout(height=420, template="plotly_white",
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    labels = [r.get("label", "?") for r in runs]

    def bars(field: str, row: int, col: int, scale: float = 1.0, nd: int = 0) -> None:
        vals = [(r.get(field) or 0.0) * scale for r in runs]
        fig.add_trace(go.Bar(x=labels, y=vals, text=[f"{v:.{nd}f}" for v in vals],
                             textposition="outside"), row=row, col=col)

    fig = make_subplots(rows=2, cols=3, subplot_titles=[
        "显存峰值 (MB)", "真实 KV 利用率 (%)", "KV 命中率 (%)",
        "TTFT (ms)", "e2e 延迟 p50 (ms)", "任务成功率 (%)",
    ])
    bars("hbm_peak_mb", 1, 1, 1, 0)
    bars("kv_usage_perc", 1, 2, 100, 1)
    bars("kv_hit_rate", 1, 3, 100, 1)
    bars("ttft_ms", 2, 1, 1, 0)
    bars("e2e_p50_ms", 2, 2, 1, 0)
    bars("success_rate", 2, 3, 100, 0)
    fig.update_layout(height=440, showlegend=False, template="plotly_white",
                      margin=dict(l=32, r=16, t=38, b=24))
    return fig


def _runs_table(runs: list[dict]) -> str:
    """运行结果对比表（Δ 相对首行=基线）。"""
    if not runs:
        return "_（运行「📊 并发 benchmark」后，结果自动汇总到这里。换引擎 config 再跑、改标签，逐步累加成 before/after）_"
    base = runs[0]

    def d(r: dict, field: str) -> str:
        bv, cv = base.get(field), r.get(field)
        if bv is None or cv is None or bv == 0:
            return ""
        pct = (cv - bv) / bv * 100
        return f" ({'↓' if pct < 0 else '+'}{abs(pct):.0f}%)"

    lines = [
        "| 运行 | 显存峰值 | 真实 KV 利用率 | KV 命中率 | TTFT | e2e p50 | 成功率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(runs):
        star = "  ←基线" if i == 0 else ""
        lines.append(
            f"| {r.get('label', '?')}{star} | {(r.get('hbm_peak_mb') or 0):.0f} MB{d(r, 'hbm_peak_mb')} | "
            f"{(r.get('kv_usage_perc') or 0) * 100:.1f}% | {(r.get('kv_hit_rate') or 0) * 100:.1f}% | "
            f"{(r.get('ttft_ms') or 0):.0f} ms{d(r, 'ttft_ms')} | "
            f"{(r.get('e2e_p50_ms') or 0):.0f} ms{d(r, 'e2e_p50_ms')} | "
            f"{(r.get('success_rate') or 0) * 100:.0f}% |"
        )
    return "\n".join(lines)


def _runs_summary(runs: list[dict]) -> str:
    n = len(runs)
    if n == 0:
        return ""
    return f"共 **{n}** 次运行（每次 = 一次并发 benchmark，标签 = 优化档）。Δ 相对首行基线，↓ 为改善。"


def capture_run(store: list[dict], metrics: dict | None, label: str) -> tuple[list[dict], go.Figure, str, str]:
    """把一次并发 benchmark 的最终指标（带标签）追加进 run_store，重渲染对比。"""
    store = list(store or [])
    if metrics:
        lbl = (label or "").strip() or f"运行 {len(store) + 1}"
        store.append({**metrics, "label": lbl})
    return store, _runs_figure(store), _runs_table(store), _runs_summary(store)


def clear_runs() -> tuple[list[dict], go.Figure, str, str]:
    return [], _runs_figure([]), _runs_table([]), _runs_summary([])


def _tau_context_stack(mode: str, model: str) -> MiddlewareStack:
    """Build a fresh per-task F2/F3 stack without changing the running engine."""
    if mode not in _TAU_CONTEXT_PRESETS:
        raise ValueError(f"未知上下文模式 {mode!r}")
    cfg = load_config(_TAU_CONTEXT_PRESETS[mode])
    cfg.engine.model = model
    return middlewares_from_config(cfg)


def _tau_context_view(
    buffer: ContextEventBuffer,
    *,
    session_id: str,
    mode: str,
    middleware_names: list[str],
) -> tuple[str, dict, dict, dict, dict]:
    """Render one stable Prompt/F2/F3 view from the method-layer snapshot."""
    snapshot = buffer.snapshot(session_id)
    events = buffer.events(session_id=session_id)
    prompt_events = [
        event for event in events
        if event["event"] == "prompt.completed"
        and isinstance((event.get("data") or {}).get("original_prompt_tokens"), int)
        and isinstance((event.get("data") or {}).get("transformed_prompt_tokens"), int)
    ]
    original_total = sum(
        event["data"]["original_prompt_tokens"] for event in prompt_events
    )
    transformed_total = sum(
        event["data"]["transformed_prompt_tokens"] for event in prompt_events
    )
    saved_total = original_total - transformed_total
    saved_percent = saved_total / max(1, original_total) * 100
    latest_prompt = snapshot.get("prompt") or {}
    cumulative_saved = (
        f"{saved_total} ({saved_percent:.2f}%)" if prompt_events else "—"
    )
    prompt_md = (
        f"**上下文模式** `{mode}`　**实际 middleware** "
        f"`{middleware_names or []}`　**模型调用** `{len(prompt_events)}`\n\n"
        "| Prompt token | 本轮 | 累计 |\n|---|---:|---:|\n"
        f"| canonical | {latest_prompt.get('original_prompt_tokens', '—')} | "
        f"{original_total if prompt_events else '—'} |\n"
        f"| transformed | {latest_prompt.get('transformed_prompt_tokens', '—')} | "
        f"{transformed_total if prompt_events else '—'} |\n"
        f"| saved | {latest_prompt.get('saved_tokens', '—')} | {cumulative_saved} |"
    )

    f2 = snapshot.get("f2") or {}
    f2_enabled = "compress" in middleware_names
    f2_before = f2.get("cold_before") or {
        "available": False,
        "enabled": f2_enabled,
        "phase": f2.get("phase") or "waiting",
        "note": (
            "Waiting for the first model request."
            if f2_enabled else "F2 is disabled in this mode."
        ),
    }
    f2_after = f2.get("cold_after") or {
        "available": False,
        "enabled": f2_enabled,
        "phase": f2.get("phase") or "waiting",
        "action": f2.get("action"),
        "reason": f2.get("reason"),
        "system_prompt_compacted": f2.get("system_prompt_compacted"),
        "tool_descriptions_replaced": f2.get("tool_descriptions_replaced"),
        "note": (
            "Cold history has not been dynamically compressed in this step."
            if f2_enabled else "F2 is disabled in this mode."
        ),
    }

    f3_state = snapshot.get("f3") or {}
    latest_f3 = f3_state.get("latest") or {}
    f3_enabled = "lazyload" in middleware_names
    f3_before = (
        {
            key: latest_f3.get(key)
            for key in (
                "phase",
                "operation_id",
                "tool_call_id",
                "tool_name",
                "arguments",
                "threshold_tokens",
                "original",
            )
        }
        if latest_f3 else {
            "available": False,
            "enabled": f3_enabled,
            "note": (
                "Waiting for a business tool result."
                if f3_enabled else "F3 is disabled in this mode."
            ),
        }
    )
    f3_after = latest_f3.get("externalized") or {
        "available": False,
        "enabled": f3_enabled,
        "phase": latest_f3.get("phase"),
        "action": latest_f3.get("event"),
        "reason": latest_f3.get("reason"),
        "fetches": f3_state.get("fetches") or [],
        "note": (
            "Tool result was not externalized; check phase/reason and the 4k threshold."
            if f3_enabled else "F3 is disabled in this mode."
        ),
    }
    return prompt_md, f2_before, f2_after, f3_before, f3_after


# ---- 应用工厂 ----


def build_app(
    *,
    engine_url: str,
    model: str,
    model_path: str,
    history_dir: str,
    interval: float,
    run_root: str = DEFAULT_RUN_ROOT,
) -> "gr.Blocks":  # type: ignore[name-defined]
    """构造并返回 Gradio Blocks（不 launch）。监控线程随 launch 后启动。"""
    import gradio as gr

    monitor = LiveMonitor(base_url=engine_url, interval=interval, device="npu")
    history = load_history(history_dir)
    assistant = _build_assistant(engine_url, model)
    n_runs = sum(h.n_runs for h in history)

    # 引擎管理器：前端按钮按档位起/停引擎；config 作 bench 自动标签
    engine_mgr = EngineManager(model_path=model_path, served_name=model)
    # 并发 bench 各会话的实时对话（tid → 气泡列表）；线程写、Timer 读，实时查看
    convo_store: dict[int, list[dict]] = {}
    tau_run_lock = threading.Lock()
    # ---- 自由对话 ----
    def respond(user_msg: str, chat_history: list[dict]):
        user_msg = (user_msg or "").strip()
        if not user_msg:
            yield chat_history
            return
        messages = [*chat_history, {"role": "user", "content": user_msg}]
        new_history = [
            *chat_history,
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ""},
        ]
        try:
            for response_list in assistant.run(messages=messages):
                new_history[-1] = {"role": "assistant", "content": _assistant_text(response_list)}
                yield new_history
        except Exception as e:  # noqa: BLE001 — 引擎离线/调用失败 → 给可见提示
            new_history[-1] = {
                "role": "assistant",
                "content": f"⚠️ 引擎调用失败（{engine_url}）：{e}\n请确认 vLLM 已在该地址服务。",
            }
            yield new_history

    # ---- τ-bench 任务（流式）----
    def run_tau(domain, task_id, max_steps, context_mode):
        tid = int(task_id)
        mode = str(context_mode)
        buffer = ContextEventBuffer(max_events=5000)
        if not tau_run_lock.acquire(blocking=False):
            empty_view = _tau_context_view(
                buffer,
                session_id=f"tau-{tid}",
                mode=mode,
                middleware_names=[],
            )
            yield [], "⚠️ 已有 tau-bench 任务运行，不能重复启动。", *empty_view
            return
        try:
            stack = _tau_context_stack(mode, model)
            names = stack.names
            view = _tau_context_view(
                buffer,
                session_id=f"tau-{tid}",
                mode=mode,
                middleware_names=names,
            )
            yield (
                [],
                f"⏳ 构建 τ-bench 环境（{domain} #{tid}）… "
                f"middleware={names}，首次加载 litellm ~6s",
                *view,
            )
            for hist_msgs, status in tau_bench_ui.run_tau_task_streaming(
                domain=str(domain),
                split="test",
                task_id=tid,
                engine_url=engine_url,
                model=model,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                max_steps=int(max_steps),
                middlewares=stack,
                context_event_sink=buffer,
            ):
                yield hist_msgs, status, *_tau_context_view(
                    buffer,
                    session_id=f"tau-{tid}",
                    mode=mode,
                    middleware_names=names,
                )
        except Exception as e:  # noqa: BLE001
            names = locals().get("names", [])
            yield (
                [{"role": "assistant", "content": f"❌ 运行失败：{e}"}],
                f"❌ 失败：{e}",
                *_tau_context_view(
                    buffer,
                    session_id=f"tau-{tid}",
                    mode=mode,
                    middleware_names=names,
                ),
            )
        finally:
            tau_run_lock.release()

    # ---- 并发 benchmark（流式：会话表 + 成功率；跑完出系统性能最终结果）----
    def run_conc(domain, ntasks, conc, steps):
        n, c, s = int(ntasks), int(conc), int(steps)
        task_ids = list(range(n))
        convo_store.clear()  # 清空上一轮对话
        start = monitor.latest()  # run 起点累积计数（算运行窗口性能增量）
        t0 = time.monotonic()
        yield f"⏳ 启动 {n} 个会话（并发 {c}，每会话 ≤{s} 步）… 首次加载 litellm ~6s", [], None
        last_rows: list = []
        last_results: dict = {}
        try:
            for md, rows, res in tau_bench_ui.run_concurrent_streaming(
                domain=str(domain),
                split="test",
                task_ids=task_ids,
                concurrency=c,
                convo_store=convo_store,
                engine_url=engine_url,
                model=model,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                max_steps=s,
            ):
                yield md, rows, None
                last_rows = rows
                last_results = res
        except Exception as e:  # noqa: BLE001
            yield f"❌ 并发运行失败：{e}", [], None
            return

        # 跑完：**CLI 同源聚合**（与 run_once 同口径）
        # 任务级（per-task）：成功率 / e2e p50,p95 / TTFT 中位 / qps —— 来自 run_task 的 TaskRunResult
        # 引擎级：KV 命中率 / 真实 KV 利用率（/metrics 末次累积/gauge）/ HBM 峰值均值（monitor 时序）
        import statistics
        from agent_mem.bench import vllm_metrics
        from agent_mem.bench.stats import p50, p95
        from agent_mem.demo.monitor import kv_usage_from_map, scrape_snapshot

        dt = time.monotonic() - t0
        tasks = list(last_results.values())
        total = len(tasks)
        success = sum(1 for t in tasks if t.get("success"))
        rate = (success / total * 100) if total else 0.0
        lats = [t["latency_ms"] for t in tasks if t.get("latency_ms")]
        ttfts = [t["ttft_ms"] for t in tasks if t.get("ttft_ms")]
        e2e_p50 = p50(lats)
        e2e_p95 = p95(lats)
        ttft_med = statistics.median(ttfts) if ttfts else None
        qps = total / dt if dt > 0 else 0.0
        # 引擎级：KV命中率（/metrics 累积，CLI 同源）+ 真实 KV 利用率/HBM（monitor 窗口采样）
        m = scrape_snapshot(engine_url)
        qh = m.get("vllm:prefix_cache_queries_total") if m else None
        # baseline（关前缀缓存）queries=0 → 命中率记 0%（与 CLI kv_cache_hit_rate 一致），不显示 N/A
        kv_hit = None if (not m or qh is None) else (0.0 if qh <= 0 else (m.get("vllm:prefix_cache_hits_total") or 0.0) / qh)
        snaps = monitor.snapshot()
        run_snaps = [s for s in snaps if start is None or s.t >= start.t]
        # 真实 KV 利用率：取 run 窗口内采样峰值（结束抓取在空闲时会得 0，不代表 run）
        kvu = [s.kv_usage_perc for s in run_snaps if s.kv_usage_perc is not None]
        kv_usage = max(kvu) if kvu else kv_usage_from_map(m)
        hbm_vals = [s.mem_mb for s in run_snaps if s.mem_mb is not None]
        hbm_peak = max(hbm_vals) if hbm_vals else None
        hbm_mean = (sum(hbm_vals) / len(hbm_vals)) if hbm_vals else None

        def f(v: float | None, unit: str = "", nd: int = 1) -> str:
            return f"{v:.{nd}f}{unit}" if v is not None else "N/A"

        badge = "🏆" if rate >= 50 else ("✅" if rate > 0 else "⚠️")
        final_md = (
            f"## {badge} 最终结果（与 CLI runner 同源）\n"
            f"### 任务成功率 **{rate:.1f}%**（✅ {success} / ❌ {total - success}，共 {total} 会话）\n"
            f"**系统性能（CLI 同口径：per-task 聚合 + /metrics）**\n"
            f"| 指标 | 值 |\n|---|---|\n"
            f"| e2e 延迟 p50 / p95 | {f(e2e_p50, ' ms', 0)} / {f(e2e_p95, ' ms', 0)} |\n"
            f"| TTFT 中位 | {f(ttft_med, ' ms', 1)} |\n"
            f"| 吞吐 qps | {f(qps, ' tasks/s', 2)} |\n"
            f"| 真实 KV 利用率 | {f(None if kv_usage is None else kv_usage * 100, ' %', 1)} |\n"
            f"| KV 命中率 | {f(None if kv_hit is None else kv_hit * 100, ' %', 1)} |\n"
            f"| NPU HBM 峰值 / 均值 | {f(hbm_peak, ' MB', 0)} / {f(hbm_mean, ' MB', 0)} |\n"
            f"| 并发数 / 运行时长 | {c} / {dt:.1f} s |\n"
        )
        metrics = {
            "n_tasks": total, "concurrency": c,
            "success_rate": rate / 100 if total else 0.0,
            "e2e_p50_ms": e2e_p50, "e2e_p95_ms": e2e_p95,
            "ttft_ms": ttft_med, "qps": qps,
            "kv_usage_perc": kv_usage, "kv_hit_rate": kv_hit, "hbm_peak_mb": hbm_peak,
        }
        yield final_md, last_rows, metrics

    # ---- 捕获并发结果入优化对比（标签 = 当前引擎档位，自动）----
    def capture_run_labeled(store, metrics):
        label = engine_mgr.config or f"运行 {len(store) + 1}"
        return capture_run(store, metrics, label)

    # ---- 查看某会话的实时对话（从共享 convo_store 读）----
    def render_convo(tid):
        try:
            t = int(tid)
        except Exception:  # noqa: BLE001
            t = -1
        msgs = convo_store.get(t)
        return msgs if msgs else [
            {"role": "assistant", "content": "_（该会话还没开始/无对话。运行「📊 并发 benchmark」后，此处每 2s 实时刷新选中会话的 τ-bench 对话）_"}
        ]

    # ---- 监控刷新 ----
    def refresh():
        status = engine_status(engine_url)
        samples = monitor.snapshot()
        series = compute_window_series(samples, window_s=WINDOW_S)
        latest = monitor.latest()

        kv_now = _last(series.kv_rate)
        if kv_now is None and latest and latest.kv_queries is not None:
            q = latest.kv_queries
            # baseline（关前缀缓存）queries=0 → 命中率 0%，不显示 N/A
            kv_now = 0.0 if q <= 0 else (latest.kv_hits or 0.0) / q

        def cur_mean(series_field: str, sum_attr: str, cnt_attr: str) -> float | None:
            v = _last(getattr(series, series_field))
            if v is not None or latest is None:
                return v
            s, c = getattr(latest, sum_attr), getattr(latest, cnt_attr)
            return None if (s is None or c is None or c <= 0) else s / c

        hbm = latest.mem_mb if latest else None

        if status != "online":
            status_md = (
                f"### 🔴 引擎离线：`{engine_url}`\n"
                f"模型：`{model}`　NPU 残留 HBM {_fmt(hbm, ' MB', 0)}\n"
                f"---\n"
                f"⚠️ vLLM 未在该地址服务 → **KV / TTFT / 延迟 / 吞吐 / 队列 暂不可用**。\n\n"
                f"启动 vLLM 后本面板**每 2s 自动恢复**（无需刷新页面）。"
            )
            return status_md, _live_figure(series, history), _history_figure(history)

        status_md = (
            f"### 🟢 引擎在线：`{engine_url}`\n"
            f"模型：`{model}`\n"
            f"---\n"
            f"| 当前指标 | 值 |\n|---|---|\n"
            f"| NPU HBM | {_fmt(hbm, ' MB', 0)} |\n"
            f"| KV 命中率 | {_fmt(None if kv_now is None else kv_now * 100, ' %', 1)} |\n"
            f"| TTFT | {_fmt(cur_mean('ttft', 'ttft_sum', 'ttft_count'), ' ms', 1)} |\n"
            f"| e2e 延迟 | {_fmt(cur_mean('e2e', 'e2e_sum', 'e2e_count'), ' ms', 1)} |\n"
            f"| 吞吐 | {_fmt(_last(series.throughput), ' tok/s', 1)} |\n"
            f"| 在跑/等待 | {_fmt(latest.running if latest else None, '', 0)} / "
            f"{_fmt(latest.waiting if latest else None, '', 0)} |\n"
        )
        return status_md, _live_figure(series, history), _history_figure(history)

    # ---- 统一 benchmark（后台 run_study；worker 改 bench_h，Timer 读）----
    bench_h = bench_runner.BenchHandle()

    def run_unified(scenario, runs, conc, maxtasks, datazip):
        url = engine_mgr.base_url if engine_mgr.is_alive() else None
        if not url:
            return "⚠️ 请先在 🛠 引擎控制 起一档引擎（点按钮 → ✅就绪）。"
        preset = BENCH_SCENARIOS.get(scenario)
        if not preset:
            return "❌ 未知场景。"
        bench_runner.run_bench_async(
            bench_h, preset_path=preset, engine_url=url, run_root=run_root,
            runs=int(runs), max_concurrency=int(conc), device="npu",
            data_zip=(str(datazip).strip() or None),
            max_tasks=int(maxtasks) if maxtasks else None,
        )
        return (f"▶ 已提交 **{scenario}** → 后台 run_study 落盘 `{run_root}/`。"
                f"进度每 2s 刷新于此；实时负载见右侧 📊。")

    def poll_unified():
        s = bench_h.snapshot()
        st = s["status"]
        if st == "running":
            return f"⏳ running… **{s['completed_runs']}/{s['total_runs']}** runs（preset={s['preset']}）"
        if st == "done":
            med = s["median"] or {}
            rows = "\n".join(f"| `{k}` | {v:.4g} |" for k, v in med.items()) or "| — | — |"
            latest = s["run_dirs"][-1] if s["run_dirs"] else "—"
            return (f"### ✅ 完成（{s['completed_runs']} runs）\n"
                    f"| 指标 | 中位数 |\n|---|---|\n{rows}\n\n最新 run dir：`{latest}`")
        if st == "error":
            return f"### ❌ 出错\n```\n{s['error']}```"
        return f"状态：{st}"

    # ---- 布局 ----
    with gr.Blocks(title="agent-mem 优化对比演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# agent-mem · KV/显存优化对比演示\n"
            "左：**自由对话** / **τ-bench 任务** / **📊 并发 benchmark**（Qwen-Agent agent，后端 = 本地 vLLM-Ascend）。"
            "右：**6 指标实时监控**。「⚙️ 优化对比」**实时继承**你在前端跑的并发 benchmark 结果（不读历史 logs）——"
            "换引擎 config、改标签再跑，逐步累加成 before/after。"
        )
        # 🏗 架构总览（评委展开看全：7缝/痛点→功能/三档/αβγ；操作时折叠）
        with gr.Accordion("🏗 架构总览（7 缝 / 痛点→功能 / 三档递进 / αβγ 场景）", open=False):
            gr.HTML(value=overview_html())
        run_store = gr.State([])  # 累积每次并发 benchmark 的结果（{label, ...metrics}）
        last_metrics = gr.State(None)  # 最近一次并发 benchmark 的最终指标（供捕获）
        # 引擎档位控制：点按钮直接（重）起引擎 + 开对应功能；config 自动作 bench 标签
        with gr.Accordion("🛠 引擎控制（按档位起引擎，无需手敲命令）", open=True):
            with gr.Row():
                eng_btn_baseline = gr.Button("baseline", scale=1)
                eng_btn_prefix = gr.Button("+prefix-cache", scale=1)
                eng_btn_c8 = gr.Button("+C8(F1)", scale=1)
                eng_btn_lmcache = gr.Button("+LMCache(F4)", scale=1)
                eng_btn_priority = gr.Button("+priority(F5)", scale=1)
                eng_btn_all = gr.Button("全开", scale=1)
                eng_btn_stop = gr.Button("⏹ 停止引擎", scale=1)
            with gr.Row():
                with gr.Column(scale=1):
                    eng_memutil = gr.Number(
                        value=0.9, label="显存上限（F5 制压调 0.27）",
                        minimum=0.1, maximum=0.95, step=0.01,
                    )
                with gr.Column(scale=1):
                    eng_maxmlen = gr.Number(
                        value=32768, label="max_model_len（F5 制压调 16384）",
                        minimum=2048, step=1024,
                    )
                with gr.Column(scale=2):
                    engine_status_md = gr.Markdown(
                        f"当前档位：**{engine_mgr.config or '未起'}**　引擎：`{engine_url}`"
                        "（点按钮起/换档；激活档标 🟢，换档重启 ~1-2min）"
                    )
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("💬 自由对话"):
                        chatbot = gr.Chatbot(
                            type="messages", height=560,
                            label="对话（Qwen-Agent Assistant）",
                        )
                        input_box = gr.Textbox(
                            placeholder="和 agent 对话（引擎离线时会提示）...", label="输入", scale=4,
                        )
                        with gr.Row():
                            send_btn = gr.Button("发送", variant="primary")
                            clear_btn = gr.Button("清空")
                    with gr.Tab("🎯 τ-bench 任务"):
                        gr.Markdown(
                            "选 **domain + task_id** 运行真实 τ-bench 客服任务。agent 多轮调工具解任务，"
                            "逐步流式刷对话；每步打本地引擎，**右侧指标实时变化**。"
                            "（retail 115 个任务；user-sim 也走本地引擎）"
                        )
                        with gr.Row():
                            tau_domain = gr.Dropdown(
                                ["retail", "airline"], value="retail", label="domain", scale=1
                            )
                            tau_taskid = gr.Number(value=0, minimum=0, label="task_id", scale=1)
                            tau_maxsteps = gr.Number(
                                value=20, minimum=1, maximum=40, label="max_steps", scale=1
                            )
                            tau_run = gr.Button("▶ 运行任务", variant="primary", scale=1)
                        tau_context_mode = gr.Radio(
                            choices=list(_TAU_CONTEXT_PRESETS),
                            value="F2+F3",
                            label="上下文模式（Agent middleware，不重启引擎）",
                            info="F2=Prompt 压缩，F3=工具数据 lazy-load，组合顺序固定为 [lazyload, compress]",
                        )
                        tau_chatbot = gr.Chatbot(
                            type="messages", height=460,
                            label="τ-bench agent 对话（tool-calling）",
                        )
                        tau_status = gr.Markdown()
                        with gr.Accordion("F2/F3 上下文变换（当前 tau-bench session）", open=True):
                            tau_prompt_view = gr.Markdown(
                                "等待任务开始：Prompt paired token、F2 冷历史和 F3 工具结果会在每一步刷新。"
                            )
                            with gr.Row():
                                tau_f2_before = gr.JSON(
                                    label="F2 待压缩冷历史（canonical preview）",
                                    value={"phase": "waiting"},
                                    height=300,
                                    max_height=360,
                                    scale=1,
                                )
                                tau_f2_after = gr.JSON(
                                    label="F2 压缩后冷历史（发送副本）",
                                    value={"phase": "waiting"},
                                    height=300,
                                    max_height=360,
                                    scale=1,
                                )
                            with gr.Row():
                                tau_f3_before = gr.JSON(
                                    label="F3 待结构化存储的工具数据",
                                    value={"phase": "waiting"},
                                    height=300,
                                    max_height=360,
                                    scale=1,
                                )
                                tau_f3_after = gr.JSON(
                                    label="F3 外置后的 synopsis/reference",
                                    value={"phase": "waiting"},
                                    height=300,
                                    max_height=360,
                                    scale=1,
                                )
                    with gr.Tab("📊 统一 Benchmark"):
                        gr.Markdown(_SCENARIO_GUIDE)
                        scenario_dd = gr.Dropdown(
                            choices=list(BENCH_SCENARIOS.keys()),
                            value=list(BENCH_SCENARIOS.keys())[0],
                            label="场景（= bench preset：suite + middleware(F2/F3) + session(F5) 都编码在内）",
                        )
                        with gr.Row():
                            bench_runs = gr.Slider(1, 5, value=3, step=1, label="重复次数(中位数)")
                            bench_conc = gr.Slider(1, 8, value=1, step=1, label="并发 max_concurrency")
                        with gr.Row():
                            bench_maxtasks = gr.Number(
                                value=8, minimum=1, label="max_tasks 任务数", scale=1,
                            )
                            bench_datazip = gr.Textbox(
                                value="", label="longbench data-zip 路径（β 场景必填）", scale=2,
                            )
                        bench_run_btn = gr.Button("🚀 跑统一 Benchmark", variant="primary")
                        bench_progress = gr.Markdown(
                            "选场景 + 调参 → 🚀（需先在 🛠 起匹配引擎档位）。进度/结果在此，实时负载见右侧。"
                        )
                    with gr.Tab("⚙️ 优化对比"):
                        gr.Markdown(
                            "**实时继承**前端并发 benchmark 的结果（不读历史 logs）。"
                            "每跑一次并发 benchmark（带上方标签）→ 结果自动加进来；"
                            "换引擎 config（baseline / +prefix-cache / +F1 / +F4 …）再跑、改标签，逐步累加成 before/after。"
                            " **Δ 相对首行基线，↓ 为改善**。"
                        )
                        with gr.Row():
                            compare_status = gr.Markdown("")
                            clear_compare = gr.Button("🗑 清空对比", scale=1)
                        compare_plot = gr.Plot(value=_runs_figure([]), label="各运行指标对比")
                        compare_table = gr.Markdown(_runs_table([]))
            with gr.Column(scale=2):
                status_md = gr.Markdown()
                live_plot = gr.Plot(label="实时监控（窗口=%.0fs）" % WINDOW_S)
                history_plot = gr.Plot(label="历史 before/after（中位数）")

        gr.Markdown(
            f"_历史来源：`{history_dir}`（{n_runs} runs）。"
            "实时曲线 = 当前引擎；baseline 参考线 = 历史。"
            "访问：服务绑 127.0.0.1，经 `ssh -L 7860:localhost:7860` 在笔记本浏览器打开。_"
        )

        # 事件：自由对话
        send_actions = [
            input_box.submit(respond, [input_box, chatbot], [chatbot], api_name="chat"),
            send_btn.click(respond, [input_box, chatbot], [chatbot]),
        ]
        for a in send_actions:
            a.then(lambda: "", None, [input_box])
        clear_btn.click(lambda: [], None, [chatbot])

        # 事件：τ-bench
        tau_run.click(
            run_tau,
            [tau_domain, tau_taskid, tau_maxsteps, tau_context_mode],
            [
                tau_chatbot,
                tau_status,
                tau_prompt_view,
                tau_f2_before,
                tau_f2_after,
                tau_f3_before,
                tau_f3_after,
            ],
        )

        # 事件：统一 benchmark（后台 run_study → Timer 轮询进度/结果）
        bench_run_btn.click(
            run_unified,
            [scenario_dd, bench_runs, bench_conc, bench_maxtasks, bench_datazip],
            [bench_progress], api_name="unified_bench",
        )
        bench_timer = gr.Timer(value=2.0)
        bench_timer.tick(poll_unified, None, [bench_progress])

        # 事件：引擎档位按钮（起/换引擎 → 流式状态 + 激活档标 🟢）
        _eng_cfgs = ("baseline", "prefix-cache", "c8", "lmcache", "priority", "all-engine")
        _eng_labels0 = {"baseline": "baseline（prefix关）", "prefix-cache": "+prefix-cache",
                        "c8": "+C8(F1)", "lmcache": "+LMCache(F4)",
                        "priority": "+priority(F5)", "all-engine": "全开"}

        def _eng_labels(active):
            return [f"🟢 {_eng_labels0[c]}" if c == active else _eng_labels0[c] for c in _eng_cfgs]

        _eng_outs = [engine_status_md, eng_btn_baseline, eng_btn_prefix, eng_btn_c8,
                     eng_btn_lmcache, eng_btn_priority, eng_btn_all]

        def _starter(cfg: str):
            def _h(memutil, maxmlen):
                engine_mgr.gpu_mem_util = float(memutil) if memutil else 0.9
                engine_mgr.max_model_len = int(maxmlen) if maxmlen else 32768
                for status in engine_mgr.start(cfg):
                    yield status, *_eng_labels(engine_mgr.config)
            return _h

        for _btn, _cfg in (
            (eng_btn_baseline, "baseline"), (eng_btn_prefix, "prefix-cache"),
            (eng_btn_c8, "c8"), (eng_btn_lmcache, "lmcache"),
            (eng_btn_priority, "priority"), (eng_btn_all, "all-engine"),
        ):
            _btn.click(_starter(_cfg), [eng_memutil, eng_maxmlen], _eng_outs)

        def _stop_engine():
            engine_mgr.stop()
            return "⏹ 引擎已停止", *_eng_labels(None)

        eng_btn_stop.click(_stop_engine, None, _eng_outs)

        # 事件：清空优化对比
        clear_compare.click(
            clear_runs, None, [run_store, compare_plot, compare_table, compare_status]
        )

        # 事件：会话对话查看已移除（统一 benchmark 改为后台 run_study 落盘，结果见本页 + 右侧历史）

        # 事件：监控刷新（共享，与对话/任务解耦）
        timer = gr.Timer(value=2.0)
        timer.tick(refresh, None, [status_md, live_plot, history_plot])
        demo.load(refresh, None, [status_md, live_plot, history_plot])

        demo._agent_mem_monitor = monitor  # type: ignore[attr-defined]
    return demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="agent-mem Gradio 演示：对话 + τ-bench + 多指标实时监控")
    p.add_argument("--engine-url", default=DEFAULT_ENGINE_URL, help="vLLM OpenAI base_url")
    p.add_argument("--model", default=DEFAULT_MODEL, help="--served-model-name")
    p.add_argument(
        "--model-path", default="/data/os_competition_TSJ/models/Qwen2.5-7B-Instruct",
        help="引擎加载的模型权重路径（引擎控制按钮起引擎用）",
    )
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="绑 127.0.0.1（SSH 隧道友好）")
    p.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR, help="历史 run 目录")
    p.add_argument("--interval", type=float, default=0.5, help="采样间隔（秒）")
    args = p.parse_args(argv)

    demo = build_app(
        engine_url=args.engine_url,
        model=args.model,
        model_path=args.model_path,
        history_dir=args.history_dir,
        interval=args.interval,
    )
    demo._agent_mem_monitor.start()  # type: ignore[attr-defined]
    print(
        f"[demo] http://{args.host}:{args.port}  引擎={args.engine_url}  模型={args.model}\n"
        f"[demo] 笔记本访问：ssh -L {args.port}:localhost:{args.port} <user>@<server> "
        f"→ 浏览器开 http://localhost:{args.port}",
        flush=True,
    )
    try:
        demo.launch(server_name=args.host, server_port=args.port, show_error=True, share=False)
    finally:
        demo._agent_mem_monitor.stop()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
