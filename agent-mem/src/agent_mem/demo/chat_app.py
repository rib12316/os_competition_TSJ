"""Gradio Blocks 演示应用：左列 Qwen-Agent 对话 + 右列实时监控面板。

布局（一个浏览器窗口，两列）::

    ┌──────────────── 对话（Qwen-Agent Assistant，流式）──────────────┐ ┌── 监控 ──┐
    │ Chatbot                                                          │ │ 引擎状态  │
    │ 输入框                                                           │ │ HBM 实时  │
    │                                                                  │ │  曲线+参考 │
    │                                                                  │ │ 当前 KV   │
    │                                                                  │ │ before/   │
    │                                                                  │ │ after 柱  │
    └──────────────────────────────────────────────────────────────────┘ └──────────┘

- 对话后端：``qwen_agent.agents.Assistant``，``llm.model_server`` 指向本地 vLLM
  （OpenAI 兼容）；引擎离线时捕获异常、给出提示。
- 监控：:class:`agent_mem.demo.monitor.LiveMonitor` 后台采 NPU HBM + vLLM ``/metrics``，
  ``gr.Timer`` 每 2s 重渲染右列；历史 before/after 来自 :func:`load_history`。
- matplotlib 用 Agg 后端（headless 服务器无需显示）。
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless 服务器：渲染到内存，不连显示
import matplotlib.pyplot as plt  # noqa: E402

from qwen_agent.agents import Assistant  # noqa: E402

from agent_mem.demo.monitor import (  # noqa: E402
    HistoryConfig,
    LiveMonitor,
    Sample,
    engine_status,
    load_history,
)

DEFAULT_ENGINE_URL = os.environ.get("AGENT_MEM_ENGINE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL = os.environ.get("AGENT_MEM_MODEL", "Qwen2.5-7B-Instruct")
DEFAULT_HISTORY_DIR = os.environ.get("AGENT_MEM_HISTORY_DIR", "logs/mvp-newframework")

# 颜色：每个 config 一个稳定颜色（baseline 红 / prefix-cache / optimized 绿 …）
_CONFIG_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


def _config_color(config: str, idx: int) -> str:
    return _CONFIG_COLORS[idx % len(_CONFIG_COLORS)]


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


# ---- 图表 ----


def _mem_figure(samples: list[Sample], history: list[HistoryConfig]) -> plt.Figure:
    """NPU HBM 显存图：baseline 历史参考线（虚线）+ 当前引擎实时曲线（实线）。"""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    # baseline 历史参考（"before"）
    for i, h in enumerate(history):
        if h.config == "baseline" and h.mem_curve:
            ts, vs = zip(*h.mem_curve)
            ax.plot(ts, vs, color="gray", linestyle="--", linewidth=1.2,
                    label=f"baseline 参考 (峰 {h.mem_peak_mb:.0f} MB)")
    # 当前引擎实时（"after" = 现在在跑的优化档）
    live = [s for s in samples if s.mem_mb is not None]
    if live:
        ts = [s.t for s in live]
        vs = [s.mem_mb for s in live]
        ax.plot(ts, vs, color="#2ca02c", linewidth=1.6,
                label=f"实时=当前引擎 (峰 {max(vs)} MB)")
    ax.set_xlabel("时间 (s)", fontsize=9)
    ax.set_ylabel("HBM 已用 (MB)", fontsize=9)
    ax.set_title("NPU 显存：实时 vs baseline 参考", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def _bars_figure(history: list[HistoryConfig]) -> plt.Figure:
    """历史 before/after 中位数对比柱（显存峰值 / e2e 延迟 / KV 命中率 / TTFT）。"""
    if not history:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.text(0.5, 0.5, "（无历史数据：跑 bench 后会在 logs/mvp-newframework 生成）",
                ha="center", va="center", fontsize=10, color="#888")
        ax.axis("off")
        return fig
    configs = [h.config for h in history]
    colors = [_config_color(c, i) for i, c in enumerate(configs)]
    spec = [
        ("显存峰值 (MB)", [h.mem_peak_mb for h in history]),
        ("e2e 延迟 p50 (ms)", [h.e2e_latency_p50_ms for h in history]),
        ("KV 命中率 (%)", [h.kv_cache_hit_rate * 100 for h in history]),
        ("TTFT (ms)", [h.ttft_ms for h in history]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.8))
    for ax, (title, vals) in zip(axes, spec):
        ax.bar(configs, vals, color=colors)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("历史 before/after 中位数（来源 logs/）", fontsize=10, y=1.02)
    fig.tight_layout()
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
    import gradio as gr  # 延迟 import：仅运行 demo 时才需要

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
        plt.close("all")  # 关上一 tick 的 figure，避免每 2s 新建导致 matplotlib 内存泄漏
        status = engine_status(engine_url)
        samples = monitor.snapshot()
        latest = monitor.latest()
        kv = latest.kv_hit_rate if latest else None
        mem = latest.mem_mb if latest else None
        badge = "🟢 online" if status == "online" else "🔴 offline"
        status_md = (
            f"### 引擎：`{engine_url}`\n"
            f"**状态**：{badge}　**模型**：`{model}`\n"
            f"---\n"
            f"**当前 HBM**：{f'{mem} MB' if mem is not None else 'N/A'}\n\n"
            f"**当前 KV 命中率**：{f'{kv*100:.1f}%' if kv is not None else 'N/A'}"
        )
        return status_md, _mem_figure(samples, history), _bars_figure(history)

    # ---- 布局 ----
    with gr.Blocks(title="agent-mem 优化对比演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# agent-mem · KV/显存优化对比演示\n"
            "左：与 **Qwen-Agent** agent 对话（后端 = 本地 vLLM-Ascend）。"
            "右：实时显存/KV 指标 + 历史 **before/after**。"
        )
        with gr.Row():
            with gr.Column(scale=3):
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
            with gr.Column(scale=2):
                status_md = gr.Markdown()
                mem_plot = gr.Plot(label="NPU HBM 显存（实时=当前引擎 / 参考=baseline）")
                bars_plot = gr.Plot(label="历史 before/after")

        gr.Markdown(
            f"_历史来源：`{history_dir}`（{n_runs} runs）。"
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
        timer.tick(refresh, None, [status_md, mem_plot, bars_plot])

        demo.load(refresh, None, [status_md, mem_plot, bars_plot])

        # 把 monitor 挂到 demo 上，launch 前 start
        demo._agent_mem_monitor = monitor  # type: ignore[attr-defined]
    return demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="agent-mem Gradio 演示：Qwen-Agent 对话 + 实时监控")
    p.add_argument("--engine-url", default=DEFAULT_ENGINE_URL, help="vLLM OpenAI base_url")
    p.add_argument("--model", default=DEFAULT_MODEL, help="--served-model-name")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="绑 127.0.0.1（SSH 隧道友好）")
    p.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR, help="历史 run 目录")
    p.add_argument("--interval", type=float, default=0.5, help="显存采样间隔（秒）")
    args = p.parse_args(argv)

    import gradio as gr

    demo = build_app(
        engine_url=args.engine_url,
        model=args.model,
        history_dir=args.history_dir,
        interval=args.interval,
    )
    # 启动监控线程
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
