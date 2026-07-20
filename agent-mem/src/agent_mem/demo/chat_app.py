"""Gradio Blocks 演示应用：左列 Qwen-Agent 对话 + 右列**多指标实时监控**面板。

布局（一个浏览器窗口，两列）::

    ┌──────────────── 对话（Qwen-Agent Assistant，流式）──────────────┐ ┌──── 实时监控 ────┐
    │ Chatbot                                                          │ │ 引擎状态/当前值  │
    │ 输入框                                                           │ │ 6 指标实时子图： │
    │                                                                  │ │  HBM / KV命中率  │
    │                                                                  │ │  吞吐 / TTFT     │
    │                                                                  │ │  e2e延迟 / 队列  │
    │                                                                  │ │ 历史 before/after│
    └──────────────────────────────────────────────────────────────────┘ └──────────────────┘

- 对话后端：``qwen_agent.agents.Assistant``，``llm.model_server`` 指向本地 vLLM。
- 监控：:class:`agent_mem.demo.monitor.LiveMonitor` 后台采 NPU HBM + vLLM ``/metrics``
  （TTFT / e2e / KV / 吞吐 / 队列），``gr.Timer`` 每 2s 重渲染右列。
- 图表用 **plotly**（浏览器渲染，中文正常、可交互），无 matplotlib 字体问题。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qwen_agent.agents import Assistant

from agent_mem.demo.monitor import (
    HistoryConfig,
    LiveMonitor,
    Sample,
    WindowSeries,
    compute_window_series,
    engine_status,
    load_history,
)

DEFAULT_ENGINE_URL = os.environ.get("AGENT_MEM_ENGINE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.environ.get("AGENT_MEM_MODEL", "Qwen2.5-7B-Instruct")
DEFAULT_HISTORY_DIR = os.environ.get("AGENT_MEM_HISTORY_DIR", "logs/mvp-newframework")
WINDOW_S = 10.0  # 窗口速率统计窗口（秒）


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
    # (1,1) HBM + baseline 参考
    fig.add_trace(go.Scatter(x=t, y=series.mem, name="HBM(实时)", mode="lines",
                             line=dict(color="#2ca02c", width=2)), row=1, col=1)
    base_curve = next((h.mem_curve for h in history if h.config == "baseline" and h.mem_curve), [])
    if base_curve:
        bx, by = zip(*base_curve)
        fig.add_trace(go.Scatter(x=bx, y=by, name="baseline 参考", mode="lines",
                                 line=dict(color="#999", dash="dash", width=1.5)), row=1, col=1)
    # (1,2) KV 命中率
    fig.add_trace(go.Scatter(x=t, y=[None if v is None else v * 100 for v in series.kv_rate],
                             name="KV命中率", mode="lines", line=dict(color="#1f77b4")), row=1, col=2)
    # (1,3) 吞吐
    fig.add_trace(go.Scatter(x=t, y=series.throughput, name="吞吐", mode="lines",
                             line=dict(color="#ff7f0e")), row=1, col=3)
    # (2,1) TTFT
    fig.add_trace(go.Scatter(x=t, y=series.ttft, name="TTFT", mode="lines",
                             line=dict(color="#d62728")), row=2, col=1)
    # (2,2) e2e
    fig.add_trace(go.Scatter(x=t, y=series.e2e, name="e2e", mode="lines",
                             line=dict(color="#9467bd")), row=2, col=2)
    # (2,3) running / waiting
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


# ---- 应用工厂 ----


def build_app(
    *,
    engine_url: str,
    model: str,
    history_dir: str,
    interval: float,
) -> "gr.Blocks":  # type: ignore[name-defined]
    """构造并返回 Gradio Blocks（不 launch）。监控线程随 launch 后启动。"""
    import gradio as gr

    monitor = LiveMonitor(base_url=engine_url, interval=interval, device="npu")
    history = load_history(history_dir)
    assistant = _build_assistant(engine_url, model)
    n_runs = sum(h.n_runs for h in history)

    # ---- 对话 ----
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

    # ---- 监控刷新 ----
    def refresh():
        status = engine_status(engine_url)
        samples = monitor.snapshot()
        series = compute_window_series(samples, window_s=WINDOW_S)
        latest = monitor.latest()

        # 当前值：窗口速率优先；窗口未满则回落到累积均值 / 瞬时
        kv_now = _last(series.kv_rate)
        if kv_now is None and latest and latest.kv_queries:
            kv_now = (latest.kv_hits or 0.0) / latest.kv_queries if latest.kv_queries else None

        def cur_mean(series_field: str, sum_attr: str, cnt_attr: str) -> float | None:
            v = _last(getattr(series, series_field))
            if v is not None or latest is None:
                return v
            s, c = getattr(latest, sum_attr), getattr(latest, cnt_attr)
            return None if (s is None or c is None or c <= 0) else s / c

        hbm = latest.mem_mb if latest else None

        if status != "online":
            # 离线：明确提示，别让一排 N/A 看着像 bug
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

    # ---- 布局 ----
    with gr.Blocks(title="agent-mem 优化对比演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# agent-mem · KV/显存优化对比演示\n"
            "左：与 **Qwen-Agent** agent 对话（后端 = 本地 vLLM-Ascend）。"
            "右：**6 指标实时监控**（HBM / KV命中率 / 吞吐 / TTFT / e2e延迟 / 队列）+ 历史 **before/after**。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    type="messages", height=580,
                    label="对话（Qwen-Agent Assistant）",
                )
                input_box = gr.Textbox(
                    placeholder="和 agent 对话（引擎离线时会提示）...", label="输入", scale=4,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空")
            with gr.Column(scale=2):
                status_md = gr.Markdown()
                live_plot = gr.Plot(label="实时监控（窗口=%.0fs）" % WINDOW_S)
                history_plot = gr.Plot(label="历史 before/after（中位数）")

        gr.Markdown(
            f"_历史来源：`{history_dir}`（{n_runs} runs）。"
            "实时曲线 = 当前引擎；baseline 参考线 = 历史。"
            "访问：服务绑 127.0.0.1，经 `ssh -L 7860:localhost:7860` 在笔记本浏览器打开。_"
        )

        # 事件
        send_actions = [
            input_box.submit(respond, [input_box, chatbot], [chatbot], api_name="chat"),
            send_btn.click(respond, [input_box, chatbot], [chatbot]),
        ]
        for a in send_actions:
            a.then(lambda: "", None, [input_box])
        clear_btn.click(lambda: [], None, [chatbot])

        timer = gr.Timer(value=2.0)
        timer.tick(refresh, None, [status_md, live_plot, history_plot])
        demo.load(refresh, None, [status_md, live_plot, history_plot])

        demo._agent_mem_monitor = monitor  # type: ignore[attr-defined]
    return demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="agent-mem Gradio 演示：Qwen-Agent 对话 + 多指标实时监控")
    p.add_argument("--engine-url", default=DEFAULT_ENGINE_URL, help="vLLM OpenAI base_url")
    p.add_argument("--model", default=DEFAULT_MODEL, help="--served-model-name")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="绑 127.0.0.1（SSH 隧道友好）")
    p.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR, help="历史 run 目录")
    p.add_argument("--interval", type=float, default=0.5, help="采样间隔（秒）")
    args = p.parse_args(argv)

    demo = build_app(
        engine_url=args.engine_url,
        model=args.model,
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
