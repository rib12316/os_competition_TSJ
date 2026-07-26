"""agent-mem 演示前端（6 tab）—— 架构总览 + 引擎/全流程前端执行。

Tabs：
  🏗 架构总览    静态，给评委一眼看全（overview_html）
  🎛 引擎控制    preset → 启/停（config 驱动，任何 yaml 经 start_engine）
  🚀 跑 Benchmark preset → runs/并发 → 后台 run_study → 进度+结果
  📊 实时监控    LiveMonitor 6 面板 + 历史 before/after
  📈 结果对比    从 logs/ 读各 config 的 run，中位数对照表
  💬 对话演示    Qwen-Agent 自由对话 + τ-bench 任务流式

引擎 URL 动态（在 EngineHandle 里）；monitor.base_url 随引擎启停重指。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agent_mem.demo import bench_runner, engine_control, overview
from agent_mem.demo.monitor import (
    HistoryConfig,
    LiveMonitor,
    WindowSeries,
    compute_window_series,
    engine_status,
    load_history,
)

DEFAULT_MODEL_PATH = os.environ.get("AGENT_MEM_MODEL_PATH", "models/Qwen2.5-7B-Instruct")
DEFAULT_HISTORY_DIR = os.environ.get("AGENT_MEM_HISTORY_DIR", "logs")
DEFAULT_RUN_ROOT = os.environ.get("AGENT_MEM_RUN_ROOT", "logs")
DEFAULT_DEVICE = os.environ.get("AGENT_MEM_DEVICE", "npu")
WINDOW_S = 10.0

# 只暴露有意义的 preset（过滤 baseline-logged / kv_offload / policies 之类噪音）
_MEANINGFUL_PRESETS = {
    "baseline", "prefix_cache", "optimized",
    "f1-bench-c8", "f1-int8", "f4-lmcache",
    "f5-evict", "f5-evict-dynamic", "f5-evict-dynamic-offload",
    "f5-native", "f5-priority-static", "f5-progress-only",
    "f2-compress", "f3-lazyload", "f2-f3-combined",
    "unified-tau-freq", "unified-longbench",
}


# =====================================================================
# 复用：图表 + 对话 helper（来自旧 chat_app）
# =====================================================================


def _last(seq: list) -> Any:
    return seq[-1] if seq else None


def _fmt(v: float | None, unit: str = "", nd: int = 1) -> str:
    return f"{v:.{nd}f}{unit}" if v is not None else "N/A"


def _assistant_text(response_list: list[Any]) -> str:
    for msg in reversed(response_list):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "assistant":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _build_assistant(engine_url: str, model: str):  # type: ignore[no-untyped-def]
    """构造指向本地 vLLM 的 Qwen-Agent Assistant（lazy import qwen_agent）。"""
    from qwen_agent.agents import Assistant

    llm_cfg = {
        "model": model,
        "model_server": engine_url,
        "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
    }
    return Assistant(
        llm=llm_cfg,
        system_message="你是 agent-mem 演示助手。用中文简洁回答。",
        name="agent-mem-assistant",
    )


def _live_figure(series: WindowSeries, history: list[HistoryConfig]) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "NPU 显存 HBM (MB)", "KV 命中率 (%)", "吞吐 (tok/s)",
            "TTFT 首 token (ms)", "端到端延迟 (ms)", "在跑 / 等待 请求数",
        ],
    )
    t = series.t
    fig.add_trace(go.Scatter(x=t, y=series.mem, name="HBM", mode="lines",
                             line=dict(color="#2a78d6", width=2)), row=1, col=1)
    base_curve = next((h.mem_curve for h in history if h.config == "baseline" and h.mem_curve), [])
    if base_curve:
        bx, by = zip(*base_curve)
        fig.add_trace(go.Scatter(x=bx, y=by, name="baseline 参考", mode="lines",
                                 line=dict(color="#999", dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=[None if v is None else v * 100 for v in series.kv_rate],
                             name="KV命中率", mode="lines", line=dict(color="#1baf7a")), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=series.throughput, name="吞吐", mode="lines",
                             line=dict(color="#eb6834")), row=1, col=3)
    fig.add_trace(go.Scatter(x=t, y=series.ttft, name="TTFT", mode="lines",
                             line=dict(color="#e34948")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=series.e2e, name="e2e", mode="lines",
                             line=dict(color="#4a3aa7")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=series.running, name="running", mode="lines",
                             line=dict(color="#1baf7a")), row=2, col=3)
    fig.add_trace(go.Scatter(x=t, y=series.waiting, name="waiting", mode="lines",
                             line=dict(color="#e34948")), row=2, col=3)
    fig.update_layout(height=540, showlegend=False, template="plotly_white",
                      margin=dict(l=32, r=16, t=38, b=24))
    return fig


def _history_figure(history: list[HistoryConfig]) -> go.Figure:
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
                         marker_color="#1baf7a", text=[f"{h.kv_cache_hit_rate*100:.1f}" for h in history], textposition="outside"), row=1, col=2)
    fig.add_trace(go.Bar(x=cfgs, y=[h.ttft_ms for h in history], name="TTFT",
                         marker_color="#e34948", text=[f"{h.ttft_ms:.0f}" for h in history], textposition="outside"), row=2, col=1)
    fig.add_trace(go.Bar(x=cfgs, y=[h.e2e_latency_p50_ms for h in history], name="e2e",
                         marker_color="#4a3aa7", text=[f"{h.e2e_latency_p50_ms:.0f}" for h in history], textposition="outside"), row=2, col=2)
    fig.update_layout(height=320, showlegend=False, template="plotly_white",
                      margin=dict(l=32, r=16, t=42, b=24))
    return fig


# =====================================================================
# preset 工具
# =====================================================================


def _enumerate_presets(configs_dir: str) -> list[tuple[str, str]]:
    """返回 [(stem, abs_path)]，只含 _MEANINGFUL_PRESETS，按 stem 排序。"""
    out: list[tuple[str, str]] = []
    root = Path(configs_dir)
    for p in sorted(root.glob("*.yaml")):
        if p.stem in _MEANINGFUL_PRESETS:
            out.append((p.stem, str(p)))
    return out


def _preview_config(preset_stem: str, preset_path_of: dict[str, str]) -> str:
    """load_config → 渲染关键字段的 markdown 预览（证明解析通过、显示开了哪些功能）。"""
    path = preset_path_of.get(preset_stem)
    if not path:
        return "_未选中 preset_"
    try:
        from agent_mem.config import load_config

        cfg = load_config(path)
    except Exception as e:  # noqa: BLE001
        return f"❌ 解析失败：`{e}`"
    e = cfg.engine
    kv = e.kv_transfer.connector or "（无）"
    return (
        f"### `{preset_stem}`\n"
        f"- **suite**: `{cfg.benchmark.suite}` ｜ domain: `{cfg.benchmark.domain}`\n"
        f"- **c8** (F1): `{e.c8.enabled}` ｜ **kv_transfer** (F4/F5-P2): `{kv}`\n"
        f"- **priority_scheduling** (F5): `{e.priority_scheduling}`\n"
        f"- **middleware** (F2/F3): `{cfg.middleware.active or '（无）'}`\n"
        f"- **session.strategy** (F5): `{cfg.session.strategy}`\n"
        f"- extra_args: `{e.extra_args}`"
    )


# =====================================================================
# 应用工厂
# =====================================================================


def build_app(
    *,
    configs_dir: str,
    model_path: str,
    history_dir: str,
    interval: float,
    run_root: str = DEFAULT_RUN_ROOT,
    device: str = DEFAULT_DEVICE,
) -> "gr.Blocks":  # type: ignore[name-defined]
    """构造 6-tab Blocks（不 launch）。"""
    import gradio as gr

    model_name = os.path.basename(model_path.rstrip("/")) or model_path
    presets = _enumerate_presets(configs_dir)
    preset_stems = [s for s, _ in presets]
    preset_path_of = dict(presets)
    default_stem = preset_stems[0] if preset_stems else None

    monitor = LiveMonitor(base_url=None, interval=interval, device=device)
    # 引擎/bench 句柄经闭包共享（单用户 demo；不用 gr.State——其 deepcopy 会破坏 Lock +
    # 断开 worker 线程的就地改写）。worker 线程改 bench_h，Timer 闭包读同一对象。
    engine_h = engine_control.EngineHandle()
    bench_h = bench_runner.BenchHandle()

    # ---------- 引擎控制 ----------
    def start_engine_cb(preset_stem: str):
        path = preset_path_of.get(preset_stem)
        if not path:
            yield "❌ 未选中 preset"
            return
        for msg in engine_control.start(engine_h, path, model_path=model_path):
            yield msg
        if engine_h.base_url:
            monitor.base_url = engine_h.base_url

    def stop_engine_cb():
        engine_control.stop(engine_h)
        monitor.base_url = None
        return "⏹ 引擎已停止。"

    # ---------- bench ----------
    def start_bench_cb(preset_stem, runs, conc):
        url = engine_h.base_url
        if not url:
            return "⚠️ 请先在 **🎛 引擎控制** 启动引擎。"
        path = preset_path_of.get(preset_stem)
        if not path:
            return "❌ 未选中 preset。"
        bench_runner.run_bench_async(
            bench_h, preset_path=path, engine_url=url, run_root=run_root,
            runs=int(runs), max_concurrency=int(conc), device=device,
        )
        return f"▶ 已提交 **{preset_stem}**（runs={runs}, 并发={conc}）。实时负载见 📊，进度每 2s 刷新于此。"

    def poll_bench_cb():
        s = bench_h.snapshot()
        st = s["status"]
        if st == "running":
            return f"⏳ running… **{s['completed_runs']}/{s['total_runs']}** runs（preset={s['preset']}）"
        if st == "done":
            med = s["median"] or {}
            rows = "\n".join(f"| `{k}` | {v:.4g} |" for k, v in med.items()) or "| — | — |"
            latest = s["run_dirs"][-1] if s["run_dirs"] else "—"
            return f"### ✅ 完成（{s['completed_runs']} runs）\n| 指标 | 中位数 |\n|---|---|\n{rows}\n\n最新 run dir：`{latest}`"
        if st == "error":
            return f"### ❌ 出错\n```\n{s['error']}\n```"
        return f"状态：{st}"

    # ---------- 实时监控 ----------
    def refresh():
        url = monitor.base_url
        status = engine_status(url) if url else "offline"
        samples = monitor.snapshot()
        series = compute_window_series(samples, window_s=WINDOW_S)
        latest = monitor.latest()
        history = load_history(history_dir)
        kv_now = _last(series.kv_rate)
        if kv_now is None and latest and latest.kv_queries:
            kv_now = (latest.kv_hits or 0.0) / latest.kv_queries
        hbm = latest.mem_mb if latest else None
        if status != "online":
            status_md = (
                f"### 🔴 引擎离线\n模型：`{model_name}`　NPU 残留 HBM {_fmt(hbm, ' MB', 0)}\n---\n"
                f"⚠️ vLLM 未服务 → KV/TTFT/延迟/吞吐/队列 暂不可用。启动后每 2s 自动恢复。"
            )
            return status_md, _live_figure(series, history), _history_figure(history)
        status_md = (
            f"### 🟢 引擎在线：`{url}`\n模型：`{model_name}`\n---\n"
            f"| 当前指标 | 值 |\n|---|---|\n"
            f"| NPU HBM | {_fmt(hbm, ' MB', 0)} |\n"
            f"| KV 命中率 | {_fmt(None if kv_now is None else kv_now * 100, ' %', 1)} |\n"
            f"| TTFT | {_fmt(_last(series.ttft), ' ms', 1)} |\n"
            f"| 吞吐 | {_fmt(_last(series.throughput), ' tok/s', 1)} |\n"
            f"| 在跑/等待 | {_fmt(latest.running if latest else None, '', 0)} / "
            f"{_fmt(latest.waiting if latest else None, '', 0)} |"
        )
        return status_md, _live_figure(series, history), _history_figure(history)

    # ---------- 结果对比 ----------
    def compare_refresh():
        history = load_history(run_root)
        if not history:
            return "_暂无 run（跑 bench 后此处显示各 config 中位数对照）_", _history_figure(history)
        rows = "\n".join(
            f"| `{h.config}` | {h.n_runs} | {h.mem_peak_mb:.0f} | {h.kv_cache_hit_rate*100:.1f} | "
            f"{h.ttft_ms:.0f} | {h.e2e_latency_p50_ms:.0f} | {h.task_success_rate*100:.1f} |"
            for h in history
        )
        md = (
            "### 各 config 中位数对照（来源 `"
            f"{run_root}`）\n| config | runs | 显存峰值(MB) | KV命中率(%) | TTFT(ms) | e2e p50(ms) | 成功率(%) |\n"
            "|---|---|---|---|---|---|---|\n" + rows
        )
        return md, _history_figure(history)

    # ---------- 对话 ----------
    def respond(user_msg: str, chat_history: list[dict]):
        user_msg = (user_msg or "").strip()
        url = monitor.base_url
        if not url:
            yield [*chat_history, {"role": "user", "content": user_msg},
                   {"role": "assistant", "content": "⚠️ 引擎未启动。请先到 **🎛 引擎控制** 启动。"}]
            return
        if not user_msg:
            yield chat_history
            return
        try:
            assistant = _build_assistant(url, model_name)
            messages = [*chat_history, {"role": "user", "content": user_msg}]
            new_history = [*chat_history, {"role": "user", "content": user_msg},
                           {"role": "assistant", "content": ""}]
            for response_list in assistant.run(messages=messages):
                new_history[-1] = {"role": "assistant", "content": _assistant_text(response_list)}
                yield new_history
        except Exception as e:  # noqa: BLE001
            yield [*chat_history, {"role": "user", "content": user_msg},
                   {"role": "assistant", "content": f"⚠️ 引擎调用失败（{url}）：{e}"}]

    # ============================ 布局：一屏 dashboard ============================
    # 上：架构总览(可折叠) + 引擎控制(常驻)；左：跑bench/对话(切换)；右：实时监控+结果(常驻)
    with gr.Blocks(title="agent-mem · 内存管理优化演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# agent-mem · 面向智能体的内存管理优化（赛题14）\n"
            "**一屏 dashboard**：上方控制引擎；**右侧实时监控+结果常驻**——跑 bench 时无需切页即见负载与结果。"
            " 👇 展开「🏗 架构总览」看完整架构（7 缝 / 痛点→功能 / 三档递进）。"
        )
        # 🏗 架构总览（评委展开看全；操作时折叠让出空间）
        with gr.Accordion("🏗 架构总览（7 缝 / 痛点→功能 / 5 功能 / 三档递进 / αβγ 场景）", open=False):
            gr.HTML(value=overview.overview_html())

        # 🎛 引擎控制（顶部常驻，不分页）
        with gr.Accordion("🎛 引擎控制（选 preset → 起/停引擎）", open=True):
            with gr.Row():
                preset_dd = gr.Dropdown(
                    choices=preset_stems, value=default_stem,
                    label="配置预设（configs/*.yaml）", scale=4,
                )
                start_btn = gr.Button("▶ 启动引擎", variant="primary", scale=1)
                stop_btn = gr.Button("⏹ 停止", variant="stop", scale=1)
            with gr.Row():
                with gr.Column(scale=3):
                    cfg_preview = gr.Markdown(
                        _preview_config(default_stem, preset_path_of) if default_stem else "_未选中_"
                    )
                with gr.Column(scale=2):
                    engine_status_md = gr.Markdown("引擎未启动。选 preset 后点 ▶。")

        # 主体：左（交互 Tabs）+ 右（常驻 监控+结果）
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("🚀 跑 Benchmark"):
                        bench_preset_dd = gr.Dropdown(
                            choices=preset_stems, value=default_stem, label="benchmark preset",
                        )
                        with gr.Row():
                            runs_slider = gr.Slider(1, 5, value=3, step=1, label="重复次数（中位数）")
                            conc_slider = gr.Slider(1, 8, value=1, step=1, label="并发 max_concurrency")
                        run_btn = gr.Button("🚀 跑 Benchmark", variant="primary")
                        bench_progress_md = gr.Markdown("选 preset + 调参 → 🚀（需先在上方 🎛 起引擎）。进度/结果在此。")
                    with gr.Tab("💬 对话演示"):
                        chatbot = gr.Chatbot(type="messages", height=460, label="对话（Qwen-Agent）")
                        input_box = gr.Textbox(placeholder="和 agent 对话（引擎在线时）...", label="输入", scale=4)
                        with gr.Row():
                            send_btn = gr.Button("发送", variant="primary")
                            clear_btn = gr.Button("清空")
            with gr.Column(scale=2):
                mon_status_md = gr.Markdown()
                live_plot = gr.Plot(label=f"实时监控（窗口={WINDOW_S:.0f}s）")
                history_plot = gr.Plot(label="历史 before/after（中位数）")
                compare_md = gr.Markdown()
                compare_plot = gr.Plot(label="各 config before/after")

        gr.Markdown(
            f"_服务绑 127.0.0.1，经 `ssh -L 7860:localhost:7860` 在笔记本浏览器打开。"
            f" preset 来源：`{configs_dir}` ｜ run 落盘：`{run_root}`_"
        )

        # -------- 事件 --------
        preset_dd.change(
            lambda stem: _preview_config(stem, preset_path_of), [preset_dd], [cfg_preview]
        )
        start_btn.click(start_engine_cb, [preset_dd], [engine_status_md])
        stop_btn.click(stop_engine_cb, None, [engine_status_md])

        run_btn.click(
            start_bench_cb,
            [bench_preset_dd, runs_slider, conc_slider],
            [bench_progress_md],
        )
        bench_timer = gr.Timer(value=2.0)
        bench_timer.tick(poll_bench_cb, None, [bench_progress_md])

        mon_timer = gr.Timer(value=2.0)
        mon_timer.tick(refresh, None, [mon_status_md, live_plot, history_plot])
        demo.load(refresh, None, [mon_status_md, live_plot, history_plot])

        compare_timer = gr.Timer(value=5.0)
        compare_timer.tick(compare_refresh, None, [compare_md, compare_plot])
        demo.load(compare_refresh, None, [compare_md, compare_plot])

        for a in [input_box.submit(respond, [input_box, chatbot], [chatbot]),
                  send_btn.click(respond, [input_box, chatbot], [chatbot])]:
            a.then(lambda: "", None, [input_box])
        clear_btn.click(lambda: [], None, [chatbot])

        demo._agent_mem_monitor = monitor  # type: ignore[attr-defined]
    return demo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="agent-mem Gradio 演示（6 tab）")
    p.add_argument("--configs-dir", default="agent-mem/configs", help="preset yaml 目录")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="模型权重路径")
    p.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR, help="历史 run 目录（监控）")
    p.add_argument("--run-root", default=DEFAULT_RUN_ROOT, help="bench run 落盘根")
    p.add_argument("--device", default=DEFAULT_DEVICE, help="显存采样设备")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="绑 127.0.0.1（SSH 隧道友好）")
    p.add_argument("--interval", type=float, default=0.5, help="采样间隔（秒）")
    args = p.parse_args(argv)

    demo = build_app(
        configs_dir=args.configs_dir, model_path=args.model_path,
        history_dir=args.history_dir, interval=args.interval,
        run_root=args.run_root, device=args.device,
    )
    demo._agent_mem_monitor.start()  # type: ignore[attr-defined]
    print(
        f"[demo] http://{args.host}:{args.port}  模型={args.model_path}\n"
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
