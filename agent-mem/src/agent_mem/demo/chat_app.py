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
from urllib.parse import urlparse

import pandas as _pandas  # noqa: F401 - preload before concurrent Plotly timer callbacks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qwen_agent.agents import Assistant

from agent_mem.config import load_config
from agent_mem.context_telemetry import ContextEventBuffer
from agent_mem.demo import longbench_ui, tau_bench_ui
from agent_mem.demo.context_compare import (
    render_f2_panels,
    render_f3_panels,
    waiting_panel,
)
from agent_mem.demo.engine_control import EngineManager
from agent_mem.demo.f5_runtime import (
    RequestAdmissionController,
    build_f5_tiers,
)
from agent_mem.demo.kv_optimize_ui import (
    KV_PAGE_CSS,
    SimState,
    engine_stages_for,
    render_all,
    render_quant_html,
    render_static_experiment_results_md,
    render_tier_html,
    should_reveal_static_results,
    stage_name_for,
    update_from_samples,
)
from agent_mem.demo.monitor import (
    HistoryConfig,
    LiveMonitor,
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
_CONTEXT_UI_MODES = ("baseline", "F2", "F3")
_TAU_USER_SIM_PRESET = _CONFIGS_DIR / "f2-f3-combined.yaml"
_LONGBENCH_CONTEXT_PRESET = _CONFIGS_DIR / "unified-longbench.yaml"
_CONTEXT_MODE_MIDDLEWARES = {
    "baseline": [],
    "F2": ["compress"],
    "F3": ["lazyload"],
    "F2+F3": ["lazyload", "compress"],
}
_DEFAULT_LONGBENCH_ZIP = Path(
    os.environ.get("AGENT_MEM_LONGBENCH_ZIP", "/tmp/longbench-data.zip")
).expanduser()
_F5_GPU_MEMORY_UTIL = float(os.environ.get("AGENT_MEM_F5_GPU_MEMORY_UTIL", "0.266"))
_F5_MAX_MODEL_LEN = int(os.environ.get("AGENT_MEM_F5_MAX_MODEL_LEN", "16384"))

_CONTEXT_TASK_CSS = """
.context-param-panel,
.context-chat-panel {
  border: 1px solid rgba(31, 41, 55, 0.16);
  border-radius: 8px;
  padding: 12px 14px;
  background: rgba(255, 255, 255, 0.62);
}
.context-param-title,
.context-chat-title {
  margin-bottom: 8px;
}
.context-param-title * ,
.context-chat-title * {
  margin: 0;
  font-size: 0.95rem;
}
.context-chat-panel .wrap,
.context-chat-panel .prose,
.context-chat-panel textarea,
.context-chatbot,
.context-chatbot * {
  font-size: 0.92rem;
}
.context-chatbot .message.user,
.context-chatbot .message-wrap.user .message,
.context-chatbot [data-testid="chatbot-message-user"] {
  background: #dcfce7 !important;
  border: 1px solid #86efac !important;
  color: #14532d !important;
}
.context-chatbot .message.bot,
.context-chatbot .message.assistant,
.context-chatbot .message-wrap.bot .message,
.context-chatbot .message-wrap.assistant .message,
.context-chatbot [data-testid="chatbot-message-bot"],
.context-chatbot [data-testid="chatbot-message-assistant"] {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  color: #111827 !important;
}
.live-monitor-sticky {
  position: relative;
  align-self: flex-start;
  box-sizing: border-box;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 0;
}
.live-monitor-status,
.live-monitor-plot {
  width: 100% !important;
  max-width: none !important;
  min-width: 0;
  box-sizing: border-box;
}
.live-monitor-status {
  padding: 12px 14px;
  border: 1px solid rgba(31, 41, 55, 0.16);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.62);
}
.live-monitor-status .prose,
.live-monitor-status .markdown-body {
  max-width: none !important;
}
.live-monitor-status h3 {
  margin-top: 0;
}
.live-monitor-plot,
.live-monitor-plot .plot-container,
.live-monitor-plot .js-plotly-plot,
.live-monitor-plot .plotly {
  width: 100% !important;
}
.live-monitor-plot [data-testid="block-label"] {
  position: relative !important;
  inset: auto !important;
  box-sizing: border-box;
  width: 100% !important;
  max-width: none !important;
  min-height: 32px;
  margin: 0 0 8px;
  justify-content: flex-start;
  border-radius: 6px;
}
.live-monitor-sticky.is-docked {
  position: fixed !important;
  top: 12px;
  left: var(--monitor-dock-left);
  width: var(--monitor-dock-width);
  max-height: calc(100vh - 24px);
  overflow: auto;
  z-index: 100;
  background: var(--background-fill-primary);
  box-shadow: 0 8px 24px rgba(17, 24, 39, 0.14);
  padding: 12px;
  border-radius: 8px;
}
.live-monitor-sticky.is-docked > .styler,
.live-monitor-sticky.is-docked .live-monitor-status,
.live-monitor-sticky.is-docked .live-monitor-plot {
  flex-shrink: 0 !important;
}
.live-monitor-sticky.is-docked > .styler {
  overflow: visible;
}
@media (max-width: 899px) {
  .live-monitor-sticky,
  .live-monitor-sticky.is-docked {
    position: static !important;
    width: 100%;
    max-height: none;
    overflow: visible;
    box-shadow: none;
  }
}
"""

_MONITOR_DOCK_JS = """
() => {
  if (window.__agentMemMonitorDockInstalled) return;
  window.__agentMemMonitorDockInstalled = true;

  let dockStart = null;
  let undockedHeight = 0;
  let savedPanelScrollTop = 0;
  let panelScrollInteractionUntil = 0;
  let restoringPanelScroll = false;

  const restorePanelScroll = () => {
    const panel = document.querySelector(".live-monitor-sticky");
    if (!panel || !panel.classList.contains("is-docked")) return;
    const maxScrollTop = Math.max(0, panel.scrollHeight - panel.clientHeight);
    const target = Math.min(savedPanelScrollTop, maxScrollTop);
    if (target <= 0 || Math.abs(panel.scrollTop - target) < 1) return;
    restoringPanelScroll = true;
    panel.scrollTop = target;
    restoringPanelScroll = false;
  };

  const bindPanelScroll = (panel) => {
    if (panel.dataset.monitorScrollBound === "true") return;
    panel.dataset.monitorScrollBound = "true";
    const markWheelInteraction = () => {
      panelScrollInteractionUntil = performance.now() + 500;
    };
    panel.addEventListener("wheel", markWheelInteraction, {passive: true});
    panel.addEventListener("pointerdown", () => {
      panelScrollInteractionUntil = Number.POSITIVE_INFINITY;
    });
    window.addEventListener("pointerup", () => {
      if (panelScrollInteractionUntil === Number.POSITIVE_INFINITY) {
        savedPanelScrollTop = panel.scrollTop;
        panelScrollInteractionUntil = performance.now() + 100;
      }
    });
    panel.addEventListener("scroll", () => {
      if (restoringPanelScroll) return;
      if (performance.now() <= panelScrollInteractionUntil) {
        savedPanelScrollTop = panel.scrollTop;
      } else if (
        panel.classList.contains("is-docked")
        && savedPanelScrollTop > 0
        && panel.scrollTop === 0
      ) {
        restorePanelScroll();
      }
    }, {passive: true});
  };

  const updateMonitorDock = () => {
    const panel = document.querySelector(".live-monitor-sticky");
    const column = document.querySelector(".live-monitor-column");
    if (!panel || !column) return;
    bindPanelScroll(panel);

    const wideScreen = window.matchMedia("(min-width: 900px)").matches;
    const isDocked = panel.classList.contains("is-docked");
    if (!isDocked) {
      dockStart = panel.getBoundingClientRect().top + window.scrollY;
      undockedHeight = panel.getBoundingClientRect().height;
    }

    const columnRect = column.getBoundingClientRect();
    panel.style.setProperty("--monitor-dock-left", `${columnRect.left}px`);
    panel.style.setProperty("--monitor-dock-width", `${columnRect.width}px`);
    const shouldDock = (
      wideScreen && dockStart !== null && window.scrollY + 12 >= dockStart
    );
    if (shouldDock && !isDocked) {
      column.style.minHeight = `${undockedHeight}px`;
      panel.classList.add("is-docked");
    } else if (!shouldDock && isDocked) {
      panel.classList.remove("is-docked");
      column.style.removeProperty("min-height");
      savedPanelScrollTop = 0;
    } else if (!shouldDock) {
      column.style.removeProperty("min-height");
    }
  };

  window.addEventListener("scroll", updateMonitorDock, {passive: true});
  window.addEventListener("resize", updateMonitorDock);
  document.querySelector("gradio-app")?.addEventListener(
    "domchange",
    () => window.requestAnimationFrame(() => {
      updateMonitorDock();
      restorePanelScroll();
    }),
  );
  window.requestAnimationFrame(updateMonitorDock);
}
"""

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


def _apply_f2_demo_options(
    cfg: Any,
    *,
    f2_method: str | None = None,
    f2_trigger_tokens: int | None = None,
    f2_recompress_delta_tokens: int | None = None,
    f2_retention_rate: float | None = None,
) -> None:
    """Apply frontend-only F2 controls to one freshly loaded config."""
    if "compress" in cfg.middleware.active and f2_method is not None:
        if f2_method not in {"llmlingua2", "longllmlingua"}:
            raise ValueError(f"未知 F2 压缩方法 {f2_method!r}")
        compress_options = cfg.middleware.options.setdefault("compress", {})
        compress_options["method"] = f2_method
        if f2_method == "longllmlingua":
            compress_options["tool_aware"] = False
            compress_options["model_name"] = "gpt2"
            compress_options["hot_tool_trigger_tokens"] = 0
    if "compress" in cfg.middleware.active and f2_trigger_tokens is not None:
        cfg.middleware.options.setdefault("compress", {})["trigger_tokens"] = max(
            1,
            int(f2_trigger_tokens),
        )
    if "compress" in cfg.middleware.active and f2_recompress_delta_tokens is not None:
        cfg.middleware.options.setdefault("compress", {})["recompress_delta_tokens"] = max(
            1,
            int(f2_recompress_delta_tokens),
        )
    if "compress" in cfg.middleware.active and f2_retention_rate is not None:
        retention = min(1.0, max(0.1, float(f2_retention_rate)))
        compress_options = cfg.middleware.options.setdefault("compress", {})
        compress_options["rate"] = retention
        compress_options["assistant_rate"] = retention
        compress_options["tool_result_rate"] = retention


def _tau_context_stack(
    mode: str,
    model: str,
    *,
    f2_method: str | None = None,
    f2_trigger_tokens: int | None = None,
    f2_recompress_delta_tokens: int | None = None,
    f2_retention_rate: float | None = None,
) -> MiddlewareStack:
    """Build a fresh tau-bench F2/F3 stack without changing the running engine."""
    if mode not in _TAU_CONTEXT_PRESETS:
        raise ValueError(f"未知上下文模式 {mode!r}")
    cfg = load_config(_TAU_CONTEXT_PRESETS[mode])
    cfg.engine.model = model
    _apply_f2_demo_options(
        cfg,
        f2_method=f2_method,
        f2_trigger_tokens=f2_trigger_tokens,
        f2_recompress_delta_tokens=f2_recompress_delta_tokens,
        f2_retention_rate=f2_retention_rate,
    )
    return middlewares_from_config(cfg)


def _longbench_context_stack(
    mode: str,
    model: str,
    *,
    f2_method: str | None = None,
    f2_trigger_tokens: int | None = None,
    f2_recompress_delta_tokens: int | None = None,
    f2_retention_rate: float | None = None,
) -> MiddlewareStack:
    """Build a LongBench stack without applying tau-bench's retail system policy."""
    if mode not in _CONTEXT_MODE_MIDDLEWARES:
        raise ValueError(f"未知上下文模式 {mode!r}")
    cfg = load_config(_LONGBENCH_CONTEXT_PRESET)
    cfg.engine.model = model
    cfg.middleware.active = list(_CONTEXT_MODE_MIDDLEWARES[mode])
    if "compress" in cfg.middleware.active:
        # F2-only needs a hot-result gate because a short 2Wiki trace may never build 8k cold history.
        cfg.middleware.options.setdefault("compress", {}).setdefault(
            "hot_tool_trigger_tokens", 1000
        )
    _apply_f2_demo_options(
        cfg,
        f2_method=f2_method,
        f2_trigger_tokens=f2_trigger_tokens,
        f2_recompress_delta_tokens=f2_recompress_delta_tokens,
        f2_retention_rate=f2_retention_rate,
    )
    return middlewares_from_config(cfg)


def _tau_user_sim_settings() -> dict[str, str | None]:
    """Load the frontend's default MIMO user simulator without exposing its key."""
    cfg = load_config(_TAU_USER_SIM_PRESET)
    user_sim = cfg.user_sim
    return {
        "model": user_sim.model,
        "provider": user_sim.provider,
        "api_base": user_sim.api_base,
        "api_key_env": user_sim.api_key_env,
        "api_key": os.environ.get(user_sim.api_key_env) if user_sim.api_key_env else None,
    }


def _tau_context_view(
    buffer: ContextEventBuffer,
    *,
    session_id: str,
    mode: str,
    middleware_names: list[str],
) -> tuple[str, str, str, str, str]:
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

    f2_before, f2_after = render_f2_panels(
        snapshot.get("f2") or {},
        enabled="compress" in middleware_names,
    )
    f3_before, f3_after = render_f3_panels(
        snapshot.get("f3") or {},
        enabled="lazyload" in middleware_names,
    )
    return prompt_md, f2_before, f2_after, f3_before, f3_after


def _context_workload(mode: str, baseline_workload: str = "tau-bench") -> str:
    """Route one context mode to the workload that can visibly exercise it."""
    if mode == "F2":
        return "tau-bench"
    if mode in {"F3", "F2+F3"}:
        return "longbench"
    if mode == "baseline" and baseline_workload in {"tau-bench", "longbench"}:
        return baseline_workload
    raise ValueError(f"无法路由上下文模式 {mode!r} / baseline workload {baseline_workload!r}")


def _build_context_controls(gr: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Build one routed F2/F3 mode selector and its optional F2 controls."""
    context_mode = gr.Radio(
        choices=list(_CONTEXT_UI_MODES),
        value="F2",
        label="上下文模式",
    )
    with gr.Group(visible=True) as f2_group:
        f2_threshold = gr.Number(
            value=2000,
            minimum=500,
            maximum=8000,
            step=500,
            label="压缩阈值",
        )
        f2_method = gr.Radio(
            choices=["llmlingua2", "longllmlingua"],
            value="llmlingua2",
            label="F2 压缩方法",
        )
        f2_retention = gr.Slider(
            minimum=0.2,
            maximum=0.75,
            value=0.4,
            step=0.05,
            label="F2 演示正文保留率",
        )
    return (
        context_mode,
        f2_group,
        f2_threshold,
        f2_method,
        f2_threshold,
        f2_retention,
    )


def _build_context_outputs(gr: Any, _benchmark_name: str) -> tuple[Any, Any, Any, Any, Any]:
    """Build matching Prompt/F2/F3 output panels for an interactive task tab."""
    with gr.Group():
        prompt_view = gr.Markdown()
        with gr.Row():
            f2_before = gr.HTML(
                label="F2 待压缩冷历史（canonical preview）",
                value=waiting_panel("F2 待压缩冷历史"),
                padding=False,
            )
            f2_after = gr.HTML(
                label="F2 压缩后冷历史（发送副本）",
                value=waiting_panel("F2 实际发送的冷历史副本"),
                padding=False,
            )
        with gr.Row():
            f3_before = gr.HTML(
                label="F3 待结构化存储的工具数据",
                value=waiting_panel("F3 业务工具原始结果"),
                padding=False,
            )
            f3_after = gr.HTML(
                label="F3 外置后的 synopsis/reference",
                value=waiting_panel("F3 外置后的 synopsis / reference"),
                padding=False,
            )
    return prompt_view, f2_before, f2_after, f3_before, f3_after


_F5_STATE_COLOR = {
    "RUNNING": "#22c55e", "WAITING": "#f59e0b", "THINKING": "#3b82f6",
    "DONE": "#9ca3af", "ERROR": "#ef4444", "QUEUED": "#f59e0b",
}


def _f5_timeline_figure(
    events: list[dict], sessions: dict, preempt_events: list[tuple[float, int]],
    kv_samples: list[tuple[float, float]], *, tier_label: str, t0: float,
) -> go.Figure:
    """实时 session 调度时间轴 + 抢占尖峰 + KV-pool 压力（live 动画每帧重画）。"""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.66, 0.34], vertical_spacing=0.16,
        subplot_titles=[
            f"{tier_label} · session 调度（🟢运行 🟡等待 🔵思考 ⚫完成）",
            "抢占累计（红，越高越多） / KV-pool%（蓝虚线）",
        ],
    )
    # session lanes：每 session 一条横道，从首个事件到末事件，按当前状态着色
    sess_times: dict[str, list[float]] = {}
    for ev in events or []:
        sid = ev.get("session_id") or ""
        tr = max(0.0, float(ev.get("t", t0)) - t0)
        rng = sess_times.setdefault(sid, [tr, tr])
        rng[0], rng[1] = min(rng[0], tr), max(rng[1], tr)
    sids = sorted(sess_times, key=lambda s: (len(s), s))
    for i, sid in enumerate(sids):
        start, end = sess_times[sid]
        if end <= start:
            end = start + 0.4
        sd = (sessions or {}).get(sid, {}) or {}
        st = sd.get("state", "RUNNING")
        prio = sd.get("priority", "")
        fig.add_trace(
            go.Bar(x=[end - start], y=[i], base=[start], orientation="h", width=0.68,
                   marker_color=_F5_STATE_COLOR.get(st, "#22c55e"), showlegend=False,
                   text=f"{sid} · p{prio} · {st}", textposition="inside",
                   hovertext=f"{sid}: {st}, step={sd.get('step','?')}, priority={prio}"),
            row=1, col=1,
        )
    # 抢占累计阶梯 + KV% 曲线
    if preempt_events:
        xs = [p[0] for p in preempt_events]
        ys = list(range(1, len(preempt_events) + 1))
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines+markers", name="抢占累计",
                       line=dict(color="#ef4444", width=2.2, shape="hv"),
                       marker=dict(color="#ef4444", size=9, symbol="x")), row=2, col=1)
    if kv_samples:
        fig.add_trace(
            go.Scatter(x=[k[0] for k in kv_samples], y=[k[1] for k in kv_samples],
                       mode="lines", name="KV-pool%", line=dict(color="#1f77b4", width=1.6, dash="dot")),
            row=2, col=1)
    total_preempt = int(sum(p[1] for p in preempt_events)) if preempt_events else 0
    fig.update_layout(height=440, template="plotly_white", showlegend=False,
                      margin=dict(l=40, r=16, t=48, b=26), barmode="overlay")
    fig.update_yaxes(row=1, col=1, tickvals=list(range(len(sids))), ticktext=sids, autorange="reversed")
    fig.update_xaxes(row=1, col=1, title_text="时间 (s)")
    fig.update_xaxes(row=2, col=1, title_text="时间 (s)")
    fig.update_yaxes(row=2, col=1, title_text="抢占 / KV%")
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.995, xanchor="right",
                       text=f"<b>抢占 {total_preempt}</b>", showarrow=False,
                       font=dict(color="#ef4444", size=20))
    return fig


def _f5_compare_timelines(baseline: dict | None, ours: dict | None) -> go.Figure:
    """baseline vs ours(准入控制) 抢占/KV 并排对比（两层都跑完后显示）。"""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.5, 0.5], vertical_spacing=0.20,
        subplot_titles=["baseline（FCFS）", "ours（KV-pool 准入控制）"],
    )
    for r, res in enumerate((baseline, ours), start=1):
        if not res:
            continue
        pe = res.get("preempt_events") or []
        kv = res.get("kv_samples") or []
        tot = int(sum(p[1] for p in pe)) if pe else 0
        if pe:
            fig.add_trace(
                go.Scatter(x=[p[0] for p in pe], y=list(range(1, len(pe) + 1)),
                           mode="lines+markers", name=f"{res.get('tier','?')} 抢占",
                           line=dict(color="#ef4444", width=2.2, shape="hv"),
                           marker=dict(color="#ef4444", size=8, symbol="x")), row=r, col=1)
        else:
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="text", text=["抢占 = 0 ✅"],
                                     textfont=dict(color="#22c55e", size=14), showlegend=False),
                          row=r, col=1)
        if kv:
            fig.add_trace(
                go.Scatter(x=[k[0] for k in kv], y=[k[1] for k in kv], mode="lines", name="KV-pool%",
                           line=dict(color="#1f77b4", width=1.4, dash="dot")), row=r, col=1)
        # 把抢占总数塞进该行子图标题
        fig.layout.annotations[r - 1].update(
            text=fig.layout.annotations[r - 1].text + f"　·　<b style='color:#ef4444'>抢占 {tot}</b>")
    fig.update_layout(height=460, template="plotly_white", showlegend=False,
                      margin=dict(l=40, r=16, t=54, b=26))
    fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
    fig.update_yaxes(title_text="抢占 / KV%", row=1, col=1)
    fig.update_yaxes(title_text="抢占 / KV%", row=2, col=1)
    return fig


def _f5_empty_figure(msg: str = "等待运行…") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=420,
                      margin=dict(l=20, r=16, t=30, b=20),
                      annotations=[dict(x=0.5, y=0.5, xref="paper", yref="paper",
                                        text=msg, showarrow=False, font=dict(size=15, color="#9ca3af"))],
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _f5_runner_live_figure(
    preempt_events: list[tuple[float, int]], kv_samples: list[tuple[float, float]],
    rw_samples: list[tuple[float, float, float]], eviction_tracker: dict | None,
    *, tier_label: str, t0: float,
) -> go.Figure:
    """benchmark-runner 路径实时图：在跑/等待 请求数 + 抢占累计 + KV%（含准入阈值线）。"""
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.5, 0.5], vertical_spacing=0.18,
        subplot_titles=[f"{tier_label} · 在跑(绿)/等待(红) 请求数",
                        "抢占累计（红）/ KV-pool%（蓝点）+ 准入阈值 85%（橙虚线）"],
    )
    if rw_samples:
        xs = [s[0] for s in rw_samples]
        fig.add_trace(go.Scatter(x=xs, y=[s[1] for s in rw_samples], mode="lines", name="running",
                                line=dict(color="#22c55e", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=[s[2] for s in rw_samples], mode="lines", name="waiting",
                                line=dict(color="#ef4444", width=2)), row=1, col=1)
    if preempt_events:
        fig.add_trace(go.Scatter(x=[p[0] for p in preempt_events], y=list(range(1, len(preempt_events) + 1)),
                                mode="lines+markers", name="抢占累计",
                                line=dict(color="#ef4444", width=2.2, shape="hv"),
                                marker=dict(color="#ef4444", size=9, symbol="x")), row=2, col=1)
    if kv_samples:
        fig.add_trace(go.Scatter(x=[k[0] for k in kv_samples], y=[k[1] for k in kv_samples], mode="lines",
                                name="KV-pool%", line=dict(color="#1f77b4", width=1.6, dash="dot")),
                      row=2, col=1)
    # 准入阈值线（85%）：直观显示准入控制何时介入——ours 的 KV% 应被压在 85 以下，native 冲到 100
    xspan = [s[0] for s in (rw_samples or kv_samples)] or [0, 1]
    if xspan:
        fig.add_trace(go.Scatter(x=[min(xspan), max(xspan)], y=[85, 85], mode="lines",
                                name="准入阈值85%", line=dict(color="#f59e0b", width=1.4, dash="dash")),
                      row=2, col=1)
    total_preempt = int(sum(p[1] for p in preempt_events)) if preempt_events else 0
    ev = int((eviction_tracker or {}).get("evictions", 0) or 0) if isinstance(eviction_tracker, dict) else 0
    fig.update_layout(height=440, template="plotly_white", showlegend=False,
                      margin=dict(l=40, r=16, t=48, b=26))
    fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
    fig.update_yaxes(title_text="请求数", row=1, col=1)
    fig.update_yaxes(title_text="抢占 / KV%", row=2, col=1, range=[0, 105])
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.995, xanchor="right",
                       text=f"<b>抢占 {total_preempt}</b>" + (f" · 准入拦 {ev}" if ev else ""),
                       showarrow=False, font=dict(color="#ef4444", size=18))
    return fig


def _f5_kv_pool_pct_fn(engine_url: str):
    """读 vLLM KV-pool 利用率（%）——与 benchmarks/runner.py 的 _kv_pool_pct_fn 完全同款，
    保证 demo 的准入闸门与 A 线/sweep 验证过的 reader 行为一致。抓失败返回 -1。"""
    import re

    from agent_mem.bench import vllm_metrics

    names = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")

    def _read() -> float:
        try:
            text = vllm_metrics.scrape(engine_url)
        except Exception:
            return -1.0
        for nm in names:
            vals = re.findall(rf"^{re.escape(nm)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text, re.M)
            if vals:
                try:
                    return max(0.0, float(vals[-1])) * 100.0
                except ValueError:
                    continue
        return -1.0

    return _read


# ---- 应用工厂 ----


def build_app(
    *,
    engine_url: str,
    model: str,
    model_path: str,
    history_dir: str,
    interval: float,
    run_root: str = DEFAULT_RUN_ROOT,
) -> Any:
    """构造并返回 Gradio Blocks（不 launch）。监控线程随 launch 后启动。"""
    import gradio as gr

    monitor = LiveMonitor(base_url=engine_url, interval=interval, device="npu")
    history = load_history(history_dir)
    assistant = _build_assistant(engine_url, model)
    n_runs = sum(h.n_runs for h in history)
    tau_user_sim = _tau_user_sim_settings()

    # 引擎管理器：前端按钮按档位起/停引擎；config 作 bench 自动标签
    engine_port = urlparse(engine_url).port or 8000
    engine_mgr = EngineManager(model_path=model_path, served_name=model, port=engine_port)
    # 并发 bench 各会话的实时对话（tid → 气泡列表）；线程写、Timer 读，实时查看
    convo_store: dict[int, list[dict]] = {}
    tau_run_lock = threading.Lock()
    longbench_run_lock = threading.Lock()
    f5_run_lock = threading.Lock()
    f5_cancel = threading.Event()
    f5_controller: list[RequestAdmissionController | None] = [None]
    kv_run_lock = threading.Lock()
    kv_state_lock = threading.RLock()
    kv_cancel = threading.Event()
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
    def run_tau(
        domain,
        task_id,
        max_steps,
        context_mode,
        f2_method,
        f2_trigger_tokens,
        f2_recompress_delta_tokens,
        f2_retention_rate,
    ):
        tid = int(task_id)
        mode = str(context_mode)
        demo_trigger = max(1, int(f2_trigger_tokens or 2000))
        demo_recompress_delta = max(1, int(f2_recompress_delta_tokens or 1000))
        demo_retention = min(1.0, max(0.1, float(f2_retention_rate or 0.4)))
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
            stack = _tau_context_stack(
                mode,
                model,
                f2_method=str(f2_method),
                f2_trigger_tokens=demo_trigger,
                f2_recompress_delta_tokens=demo_recompress_delta,
                f2_retention_rate=demo_retention,
            )
            names = stack.names
            user_sim = _tau_user_sim_settings()
            view = _tau_context_view(
                buffer,
                session_id=f"tau-{tid}",
                mode=mode,
                middleware_names=names,
            )
            yield (
                [],
                f"⏳ 构建 τ-bench 环境（{domain} #{tid}）… "
                f"middleware={names}，USER simulator={user_sim['model']}，"
                f"F2 method={f2_method}，"
                f"F2 demo trigger={demo_trigger} token，"
                f"recompress delta={demo_recompress_delta} token，"
                f"正文保留率={demo_retention:.2f}，首次加载 litellm ~6s",
                *view,
            )
            for hist_msgs, status in tau_bench_ui.run_tau_task_streaming(
                domain=str(domain),
                split="test",
                task_id=tid,
                engine_url=engine_url,
                model=model,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                user_model=str(user_sim["model"]),
                user_provider=str(user_sim["provider"]),
                user_api_base=str(user_sim["api_base"]),
                user_api_key=user_sim["api_key"],
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

    # ---- LongBench 2WikiMQA 任务（流式）----
    def run_longbench(
        data_zip,
        task_id,
        max_steps,
        context_mode,
        f2_method,
        f2_trigger_tokens,
        f2_recompress_delta_tokens,
        f2_retention_rate,
    ):
        tid = int(task_id)
        mode = str(context_mode)
        zip_path = str(data_zip or "").strip()
        demo_trigger = max(1, int(f2_trigger_tokens or 2000))
        demo_recompress_delta = max(1, int(f2_recompress_delta_tokens or 1000))
        demo_retention = min(1.0, max(0.1, float(f2_retention_rate or 0.4)))
        session_id = f"longbench-{tid}"
        buffer = ContextEventBuffer(max_events=5000)
        if not longbench_run_lock.acquire(blocking=False):
            empty_view = _tau_context_view(
                buffer,
                session_id=session_id,
                mode=mode,
                middleware_names=[],
            )
            yield [], "已有 LongBench 任务运行，不能重复启动。", *empty_view
            return
        try:
            if not zip_path:
                raise ValueError("请填写包含 data/2wikimqa.jsonl 的 LongBench data zip 路径")
            stack = _longbench_context_stack(
                mode,
                model,
                f2_method=str(f2_method),
                f2_trigger_tokens=demo_trigger,
                f2_recompress_delta_tokens=demo_recompress_delta,
                f2_retention_rate=demo_retention,
            )
            names = stack.names
            view = _tau_context_view(
                buffer,
                session_id=session_id,
                mode=mode,
                middleware_names=names,
            )
            yield (
                [],
                f"正在从 `{zip_path}` 加载 LongBench 2WikiMQA #{tid}… "
                f"middleware={names}，F2 method={f2_method}，"
                f"trigger={demo_trigger}，recompress delta={demo_recompress_delta}，"
                f"retention={demo_retention:.2f}",
                *view,
            )
            for hist_msgs, status in longbench_ui.run_longbench_task_streaming(
                data_zip=zip_path,
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
                    session_id=session_id,
                    mode=mode,
                    middleware_names=names,
                )
        except Exception as exc:  # noqa: BLE001 - surface path/zip/engine failures in the tab
            names = locals().get("names", [])
            yield (
                [{"role": "assistant", "content": f"LongBench 运行失败：{exc}"}],
                f"LongBench 失败：{exc}",
                *_tau_context_view(
                    buffer,
                    session_id=session_id,
                    mode=mode,
                    middleware_names=names,
                ),
            )
        finally:
            longbench_run_lock.release()

    def run_context_task(
        context_mode,
        baseline_workload,
        tau_domain,
        tau_task_id,
        tau_max_steps,
        longbench_data_zip,
        longbench_task_id,
        longbench_max_steps,
        f2_method,
        f2_trigger_tokens,
        f2_recompress_delta_tokens,
        f2_retention_rate,
    ):
        """Route one frontend action to the workload selected by context mode."""
        mode = str(context_mode)
        workload = _context_workload(mode, str(baseline_workload))
        if workload == "tau-bench":
            yield from run_tau(
                tau_domain,
                tau_task_id,
                tau_max_steps,
                mode,
                f2_method,
                f2_trigger_tokens,
                f2_recompress_delta_tokens,
                f2_retention_rate,
            )
            return
        yield from run_longbench(
            longbench_data_zip,
            longbench_task_id,
            longbench_max_steps,
            mode,
            f2_method,
            f2_trigger_tokens,
            f2_recompress_delta_tokens,
            f2_retention_rate,
        )

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
            return None if (s is None or c is None or c <= 0) else s / c * 1000.0

        hbm = latest.mem_mb if latest else None

        if status != "online":
            status_md = (
                f"### 🔴 引擎离线：`{engine_url}`\n"
                f"模型：`{model}`　NPU 残留 HBM {_fmt(hbm, ' MB', 0)}\n"
                f"---\n"
                f"⚠️ vLLM 未在该地址服务 → **KV / TTFT / 延迟 / 吞吐 / 队列 暂不可用**。\n\n"
                f"启动 vLLM 后本面板**每 2s 自动恢复**（无需刷新页面）。"
            )
            return status_md, _live_figure(series, history)

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
        return status_md, _live_figure(series, history)

    # ---- 高并发 F5：固定 workload，一键顺序运行 baseline -> vLLM 原生 -> ours ----
    def _f5_comparison_rows(results: list[dict], running_label: str | None = None) -> str:
        def cell(r: dict, k: str, fmt=None) -> str:
            v = r.get(k)
            if v is None or v == "":
                return "—"
            if fmt:
                try:
                    return fmt(v)
                except Exception:
                    return "—"
            return str(v)

        rows = [
            "| " + " | ".join([
                cell(r, "tier"), cell(r, "status"),
                cell(r, "p50_ms", lambda v: f"{v:.0f}"),
                cell(r, "preemptions", lambda v: str(int(v))),
            ]) + " |"
            for r in results
        ]
        if running_label:
            rows.append(f"| {running_label} | 运行中 |  |  |")
        header = (
            "### baseline → ours 实测对比（准入控制）\n"
            "| 层级 | 状态 | e2e p50 ms | 实际抢占 |\n"
            "|---|---|---:|---:|"
        )
        body = "\n".join(rows) if rows else "| 尚未运行 |  |  |  |"
        if len(results) == 2:
            b, o = results[0], results[1]
            try:
                dp50 = ((o["p50_ms"] / b["p50_ms"] - 1) * 100) if b.get("p50_ms") else None
                dpre = int(o.get("preemptions", 0) or 0) - int(b.get("preemptions", 0) or 0)
                p50txt = f"{dp50:+.0f}%" if dp50 is not None else "—"
                rows.append(f"| **Δ(ours−base)** |  | {p50txt} | {dpre:+d} |")
                body = "\n".join(rows)
            except Exception:
                pass
        return f"{header}\n{body}"

    def _f5_result_conclusion(results: list[dict], run_dir: Path) -> str:
        if len(results) != 2:
            return f"结果目录：`{run_dir}`"

        baseline, ours = results

        def g(r: dict, k: str, default: float = 0.0) -> float:
            v = r.get(k)
            return v if isinstance(v, (int, float)) else default

        def delta(value: float, reference: float, *, lower_is_better: bool) -> str:
            if reference == 0:
                return "N/A"
            change = (value / reference - 1.0) * 100.0
            improved = change < 0 if lower_is_better else change > 0
            return f"{change:+.1f}% ({'改善' if improved else '代价'})"

        evidence_rows = "".join(
            f"| {r.get('tier', '?')} | {g(r, 'duration_s'):.1f}s | "
            f"{g(r, 'output_tokens_per_s'):.1f} | {g(r, 'ttft_p95_ms'):.0f} ms | "
            f"{g(r, 'external_kv_hit_rate'):.1%} |\n"
            for r in results
        )
        return (
            "### 实测结论\n"
            "| ours 相对 | p50 | p95 | QPS | KV 峰值 | 抢占 |\n"
            "|---|---:|---:|---:|---:|---:|\n"
            f"| baseline | {delta(g(ours, 'p50_ms'), g(baseline, 'p50_ms'), lower_is_better=True)} | "
            f"{delta(g(ours, 'p95_ms'), g(baseline, 'p95_ms'), lower_is_better=True)} | "
            f"{delta(g(ours, 'qps'), g(baseline, 'qps'), lower_is_better=False)} | "
            f"{g(ours, 'kv_peak_pct'):.1f}% vs {g(baseline, 'kv_peak_pct'):.1f}% | "
            f"{int(g(ours, 'preemptions'))} vs {int(g(baseline, 'preemptions'))} |\n"
            "\n"
            "| 层级 | 整批完成时间 | 输出 token/s | session TTFT p95 | 外部KV命中 |\n"
            "|---|---:|---:|---:|---:|\n"
            f"{evidence_rows}"
            "\n"
            f"结果目录：`{run_dir}`"
        )

    def _f5_evidence(label: str, controller: RequestAdmissionController) -> str:
        snapshot = controller.snapshot()
        kv = snapshot["latest_kv_pct"]
        kv_text = "N/A" if kv is None else f"{kv:.1f}%"
        return (
            f"### 当前层：{label}\n"
            f"| 调度证据 | 当前值 |\n|---|---|\n"
            f"| 请求槽 active / limit | {snapshot['active_requests']} / {snapshot['current_limit']} |\n"
            f"| 本层最大 active requests | {snapshot['max_active_requests']} |\n"
            f"| 已准入请求 | {snapshot['admitted_requests']} |\n"
            f"| 被延迟请求 | {snapshot['deferred_requests']} |\n"
            f"| 当前 KV-pool | {kv_text} |\n"
            f"| 准入模式 | {'KV-pool 自适应' if snapshot['adaptive'] else '固定请求上限'} |"
        )

    def _warmup_f5_engine(base_url: str, *, require_c8_quality: bool = False) -> None:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            temperature=0.0,
            max_tokens=2,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if require_c8_quality:
            quality = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": "What is 7 times 8? Reply with only the number.",
                }],
                temperature=0.0,
                max_tokens=8,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = quality.choices[0].message.content or ""
            if content.strip() != "56":
                raise RuntimeError(f"C8 输出质量预检失败：{content[:120]!r}")

    def run_f5_tier(tier_key, n_tasks, conc, maxsteps, seed, results_state):
        import concurrent.futures
        import json

        from agent_mem.bench.runners.qwen_agent import QwenAgentRunner
        from agent_mem.bench.stats import p50, p95
        from agent_mem.demo.monitor import kv_usage_from_map, scrape_snapshot

        def _runner_evidence(label, snap, drv, preempt_events) -> str:
            tot = int(sum(p[1] for p in preempt_events)) if preempt_events else 0
            kvu = kv_usage_from_map(snap)
            running = float((snap or {}).get("vllm:num_requests_running", 0.0))
            waiting = float((snap or {}).get("vllm:num_requests_waiting", 0.0))
            et = (drv.get("eviction_tracker") or {}) if isinstance(drv, dict) else {}
            ev = int((et or {}).get("evictions", 0) or 0)
            return (f"### {label} 调度证据（A 线同款 benchmark runner）\n"
                    f"| 抢占累计 | KV-pool% | running | waiting | 驱逐(app) |\n|---|---|---|---|---|\n"
                    f"| **{tot}** | {(kvu * 100 if kvu is not None else 0):.1f} | {running:.0f} | {waiting:.0f} | {ev} |")

        if not f5_run_lock.acquire(blocking=False):
            yield ("已有 F5 运行正在进行，请先停止。", "", [], _f5_empty_figure("运行中…"),
                   "", list(results_state or []))
            return

        f5_cancel.clear()
        results: list[dict] = list(results_state or [])
        last_rows: list[list] = []
        run_dir: Path | None = None
        settings: dict = {}
        active_tier = None
        try:
            n_tasks = max(2, int(n_tasks))
            max_requests = max(1, min(int(conc), n_tasks))
            steps = max(1, int(maxsteps))

            tiers = build_f5_tiers()
            tier = next((t for t in tiers if t.key == str(tier_key)), tiers[-1])
            active_tier = tier
            # A 线同款配置：engine flags 来自 tier.features + gpu_mem 0.27；session 块来自 yaml
            cfg_name = "f5-native" if tier.key == "baseline" else "f5-evict-dynamic"
            cfg = load_config(str(_CONFIGS_DIR / f"{cfg_name}.yaml"))
            run_dir = Path(run_root) / "f5-demo" / time.strftime("%Y%m%d-%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            settings = {
                "aligned_to": "A-line (benchmarks/runner.py · QwenAgentRunner · ConcurrentSessionDriver)",
                "tier": tier.label, "config": f"{cfg_name}.yaml",
                "gpu_memory_utilization": 0.27, "max_model_len": _F5_MAX_MODEL_LEN,
                "max_tasks": n_tasks, "max_active_requests": max_requests, "max_steps": steps,
                "user_simulation": "local (与 A 线同口径)", "seed": int(seed),
            }

            engine_mgr.gpu_mem_util = 0.27   # ← A 线对齐（KV pool 1.27GiB）
            engine_mgr.max_model_len = _F5_MAX_MODEL_LEN
            for engine_message in engine_mgr.start(list(tier.features)):
                yield (f"**{tier.label}**：{engine_message}",
                       _f5_comparison_rows(results, tier.label),
                       [], _f5_empty_figure(f"{tier.label} 引擎启动中…"),
                       f"A 线同款：gpu-mem 0.27 / features={tier.features} / {cfg_name}.yaml", results)
            if not engine_mgr.is_alive():
                raise RuntimeError(f"{tier.label} 引擎启动失败：{engine_mgr.last_start_error or '?'}")

            run_url = engine_mgr.base_url
            _warmup_f5_engine(run_url, require_c8_quality=tier.c8_enabled)
            metric_start = scrape_snapshot(run_url) or {}
            started = time.monotonic()
            t0 = started
            dynamic = cfg.session.strategy in ("priority-evict", "progress-evict", "combined-evict")
            runner = QwenAgentRunner(
                engine_url=run_url, model=model, max_tasks=n_tasks, max_steps=steps,
                api_key=os.environ.get("OPENAI_API_KEY", "stub"), max_concurrency=max_requests,
                dynamic=dynamic, idle_timeout_s=cfg.session.idle_timeout_s,
                target_lo=cfg.session.target_lo, target_hi=cfg.session.target_hi,
                hbm_pct_fn=_f5_kv_pool_pct_fn(run_url),
            )
            f5_controller[0] = None  # benchmark 路径不用 demo 的 RequestAdmissionController

            preempt_events: list[tuple[float, int]] = []
            kv_samples: list[tuple[float, float]] = []
            rw_samples: list[tuple[float, float, float]] = []
            last_preempt_total = float(metric_start.get("vllm:num_preemptions_total", 0.0))
            all_results: list = []

            def _poll(snap) -> None:
                nonlocal last_preempt_total
                cur = float(snap.get("vllm:num_preemptions_total", last_preempt_total))
                d = cur - last_preempt_total
                if d > 0:
                    preempt_events.append((time.monotonic() - t0, int(d)))
                    last_preempt_total = cur
                kvu = kv_usage_from_map(snap)
                if kvu is not None:
                    kv_samples.append((time.monotonic() - t0, kvu * 100.0))
                rw_samples.append((time.monotonic() - t0,
                                   float(snap.get("vllm:num_requests_running", 0.0)),
                                   float(snap.get("vllm:num_requests_waiting", 0.0))))

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(runner.run_all, cfg)
                while not fut.done():
                    if f5_cancel.is_set():
                        break
                    snap = scrape_snapshot(run_url) or {}
                    _poll(snap)
                    drv = runner.last_driver.snapshot() if runner.last_driver else {}
                    live_fig = _f5_runner_live_figure(
                        preempt_events, kv_samples, rw_samples,
                        drv.get("eviction_tracker", {}), tier_label=tier.label, t0=t0,
                    )
                    yield (f"**{tier.label}** 运行中（A 线同款 runner）… preempt={int(sum(p[1] for p in preempt_events))}",
                           _f5_comparison_rows(results, tier.label), [], live_fig,
                           _runner_evidence(tier.label, snap, drv, preempt_events), results)
                    time.sleep(1.0)
                if f5_cancel.is_set():
                    engine_mgr.stop()
                    all_results = []
                else:
                    all_results = fut.result()

            if f5_cancel.is_set():
                yield (f"**{tier.label}** 已停止。", _f5_comparison_rows(results), [],
                       _f5_runner_live_figure(preempt_events, kv_samples, rw_samples, {},
                                              tier_label=tier.label, t0=t0),
                       "已停止；8000 引擎已回收。", results)
                return

            tasks = list(all_results)
            if not tasks:
                raise RuntimeError(f"{tier.label} 未产生任何 session 结果")
            duration = max(time.monotonic() - started, 1e-9)
            metric_end = scrape_snapshot(run_url) or {}

            def counter_delta(name: str) -> float:
                return max(0.0, float(metric_end.get(name, 0.0)) - float(metric_start.get(name, 0.0)))

            latencies = [float(t.latency_ms or 0.0) for t in tasks]
            ttfts = [float(t.ttft_ms or 0.0) for t in tasks]
            queries = counter_delta("vllm:prefix_cache_queries_total")
            hits = counter_delta("vllm:prefix_cache_hits_total")
            kv_hit = hits / queries if queries > 0 else 0.0
            preemptions = counter_delta("vllm:num_preemptions_total")
            drv_snap = runner.last_driver.snapshot() if runner.last_driver else {}
            tier_result = {
                "tier": tier.label, "status": "完成", "engine_features": list(tier.features),
                "c8_enabled": tier.c8_enabled,
                "session_ok_rate": sum(1 for t in tasks if not t.error) / len(tasks),
                "success_rate": sum(1 for t in tasks if t.success) / len(tasks),
                "p50_ms": p50(latencies), "p95_ms": p95(latencies), "ttft_p95_ms": p95(ttfts),
                "qps": len(tasks) / duration, "kv_hit_rate": kv_hit,
                "kv_peak_pct": max([k[1] for k in kv_samples] or [0.0]),
                "preemptions": int(preemptions),
                "deferred_requests": 0,
                "prompt_tokens": sum(int(getattr(t, "prompt_tokens", 0) or 0) for t in tasks),
                "offload": tier.offload_label, "duration_s": duration,
                "preempt_events": preempt_events, "kv_samples": kv_samples,
            }
            results = [r for r in results if r.get("tier") != tier.label]
            results.append(tier_result)
            (run_dir / "summary.json").write_text(
                json.dumps({"settings": settings, "tiers": results}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            last_rows = [
                [t.task_id, "✅" if t.success else "❌", "", t.n_steps,
                 round(float(t.reward or 0.0), 2), round(float(t.latency_ms or 0.0)),
                 (str(t.error)[:40] if t.error else "")]
                for t in tasks
            ]
            base_r = next((r for r in results if r.get("tier") == "baseline"), None)
            ours_r = next((r for r in results if r.get("tier") == "ours"), None)
            final_fig = (_f5_compare_timelines(base_r, ours_r) if (base_r and ours_r)
                         else _f5_runner_live_figure(preempt_events, kv_samples, rw_samples,
                                                     drv_snap.get("eviction_tracker", {}),
                                                     tier_label=tier.label, t0=t0))
            yield (f"**{tier.label}** 完成。preempt={int(preemptions)} kv_hit={kv_hit:.2f}",
                   _f5_comparison_rows(results), last_rows, final_fig,
                   _runner_evidence(tier.label, metric_end, drv_snap, preempt_events), results)
            engine_mgr.stop()
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            if active_tier is not None and not any(r.get("tier") == active_tier.label for r in results):
                results = [r for r in results if r.get("tier") != active_tier.label]
                results.append({"tier": active_tier.label, "status": "失败",
                                "engine_features": list(active_tier.features),
                                "p50_ms": 0.0, "p95_ms": 0.0, "kv_hit_rate": 0.0, "preemptions": 0,
                                "offload": active_tier.offload_label, "error": error_text})
            if run_dir is not None:
                (run_dir / "summary.json").write_text(
                    json.dumps({"settings": settings, "tiers": results, "error": error_text},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            engine_mgr.stop()
            yield (f"{active_tier.label if active_tier else 'F5'} 运行失败：{error_text}",
                   _f5_comparison_rows(results), last_rows, _f5_empty_figure("运行失败"),
                   _f5_result_conclusion(results, run_dir) if run_dir else "失败；8000 引擎已回收。", results)
        finally:
            f5_controller[0] = None
            f5_run_lock.release()

    def run_f5_baseline(total_sessions, conc, maxsteps, seed, results_state):
        yield from run_f5_tier("baseline", total_sessions, conc, maxsteps, seed, results_state)

    def run_f5_ours(total_sessions, conc, maxsteps, seed, results_state):
        yield from run_f5_tier("ours", total_sessions, conc, maxsteps, seed, results_state)

    def stop_f5():
        f5_cancel.set()
        if f5_controller[0] is not None:
            f5_controller[0].cancel()
        stopped = engine_mgr.stop()
        return (
            "F5 运行已停止，8000 端口引擎及 EngineCore 已回收。"
            if stopped else "F5 运行已停止；8000 端口未发现引擎进程。"
        )

    # ---- 布局 ----
    with gr.Blocks(
        title="agent-mem 优化对比演示",
        theme=gr.themes.Soft(),
        css=_CONTEXT_TASK_CSS + KV_PAGE_CSS,
        js=_MONITOR_DOCK_JS,
    ) as demo:
        gr.Markdown(
            "# agent-mem · KV/显存优化对比演示"
        )
        with gr.Accordion(
            "🏗 架构总览（7 缝 / 痛点→功能 / 三档递进 / αβγ 场景）",
            open=False,
            visible=False,
        ):
            gr.HTML(value=overview_html())
        with gr.Accordion("🛠 引擎控制（多选功能 → 一键启动）", open=True):
            eng_features = gr.CheckboxGroup(
                choices=[
                    ("prefix-cache（默认开）", "prefix-cache"),
                    ("C8 int8 KV (F1)", "c8"),
                    ("LMCache 分层 (F4)", "lmcache"),
                    ("priority 调度 (F5)", "priority"),
                ],
                value=["prefix-cache"], label="引擎功能（多选组合）",
            )
            with gr.Row():
                eng_start_btn = gr.Button("▶ 启动引擎", variant="primary", scale=1)
                eng_stop_btn = gr.Button("⏹ 停止", variant="stop", scale=1)
            with gr.Row():
                with gr.Column(scale=1):
                    eng_memutil = gr.Number(
                        value=0.9, label="显存上限（选 priority 自动 0.27；可改）",
                        minimum=0.1, maximum=0.95, step=0.01,
                    )
                with gr.Column(scale=1):
                    eng_maxmlen = gr.Number(
                        value=32768, label="max_model_len（选 priority 自动 16384；可改）",
                        minimum=2048, step=1024,
                    )
                with gr.Column(scale=2):
                    engine_status_md = gr.Markdown(
                        f"引擎：`{engine_url}`（未起）。勾选功能 → ▶启动 → 状态在此流式刷新。"
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
                    with gr.Tab("上下文优化任务"):
                        with gr.Group(elem_classes=["context-param-panel"]):
                            gr.Markdown("**参数调整**", elem_classes=["context-param-title"])
                            (
                                context_mode,
                                f2_options_group,
                                context_f2_trigger,
                                context_f2_method,
                                context_f2_recompress_delta,
                                context_f2_retention,
                            ) = _build_context_controls(gr)
                            baseline_workload = gr.Radio(
                                choices=[
                                    ("F2 对照 · τ-bench", "tau-bench"),
                                    ("F3 对照 · LongBench", "longbench"),
                                ],
                                value="tau-bench",
                                label="baseline 对照场景",
                                visible=False,
                            )
                        with gr.Group(visible=True) as tau_inputs_group:
                            gr.Markdown(
                                f"USER simulator：`{tau_user_sim['model']}`　"
                                f"key：`{tau_user_sim['api_key_env']}` "
                                f"{'已设置' if tau_user_sim['api_key'] else '未设置'}"
                            )
                            with gr.Row():
                                context_tau_domain = gr.Dropdown(
                                    ["retail", "airline"], value="retail", label="domain", scale=1
                                )
                                context_tau_taskid = gr.Number(
                                    value=0, minimum=0, step=1, label="τ-bench task_id", scale=1
                                )
                                context_tau_maxsteps = gr.Number(
                                    value=20, minimum=1, maximum=40, step=1,
                                    label="τ-bench max_steps", scale=1,
                                )
                        with gr.Group(visible=False) as longbench_inputs_group:
                            with gr.Row():
                                context_longbench_data_zip = gr.Textbox(
                                    value=(
                                        str(_DEFAULT_LONGBENCH_ZIP)
                                        if _DEFAULT_LONGBENCH_ZIP.is_file() else ""
                                    ),
                                    label="LongBench data zip",
                                    placeholder="包含 data/2wikimqa.jsonl 的 zip 路径",
                                    scale=3,
                                )
                                context_longbench_taskid = gr.Number(
                                    value=0, minimum=0, step=1,
                                    label="LongBench task_id", scale=1,
                                )
                                context_longbench_maxsteps = gr.Number(
                                    value=8, minimum=1, maximum=20, step=1,
                                    label="LongBench max_steps", scale=1,
                                )
                        context_run = gr.Button("▶ 运行当前场景", variant="primary")
                        with gr.Group(elem_classes=["context-chat-panel"]):
                            gr.Markdown("**上下文优化Agent对话**", elem_classes=["context-chat-title"])
                            context_chatbot = gr.Chatbot(
                                type="messages", height=460,
                                label="上下文优化 Agent 对话（自动路由）",
                                elem_classes=["context-chatbot"],
                            )
                        context_status = gr.Markdown()
                        (
                            context_prompt_view,
                            context_f2_before,
                            context_f2_after,
                            context_f3_before,
                            context_f3_after,
                        ) = _build_context_outputs(gr, "自动路由")
                    with gr.Tab("🧪 高并发·F5"):
                        f5_results = gr.State([])
                        with gr.Group(elem_classes=["context-param-panel"]):
                            gr.Markdown("**参数调整**", elem_classes=["context-param-title"])
                            with gr.Row():
                                f5_total_sessions = gr.Number(
                                    value=10, minimum=2, maximum=115, step=1,
                                    label="任务数 (max_tasks，须>conc 才能触发准入队列)", scale=1,
                                )
                                f5_conc = gr.Number(
                                    value=6, minimum=1, maximum=24, step=1,
                                    label="最大 active requests (conc)", scale=1,
                                )
                                f5_steps = gr.Number(
                                    value=25, minimum=1, maximum=30, step=1,
                                    label="每任务 max_steps (高压，确保 baseline 必抢占)", scale=1,
                                )
                                f5_seed = gr.Number(
                                    value=42, minimum=0, step=1, label="seed", scale=1,
                                )
                            gr.Markdown(
                                "对比：**baseline** = FCFS（仅 prefix-cache，并发溢出→盲目抢占重算）　|　"
                                "**ours** = **KV-pool 准入控制（背压）**：读 `kv_cache_usage_perc`，>85% 暂不放新 session、"
                                "降至 70% 再放 → **从源头防溢出、抢占归零**（+ priority 调度协同、无 offload）。两层均 APC-on。"
                                "先各点一次按钮看实时抢占差异，跑完出并排对比。"
                            )
                        with gr.Row():
                            f5_baseline_btn = gr.Button("▶ 运行 baseline（FCFS）", variant="primary", scale=1)
                            f5_ours_btn = gr.Button("▶ 运行 ours（准入控制）", variant="primary", scale=1)
                            f5_stop_btn = gr.Button("⏹ 停止", variant="stop", scale=1)
                        f5_status = gr.Markdown()
                        with gr.Group(elem_classes=["context-chat-panel"]):
                            gr.Markdown("**实时请求调度与抢占行为**", elem_classes=["context-chat-title"])
                            f5_live_plot = gr.Plot(
                                value=_f5_empty_figure("点上方按钮开始（建议先 baseline 再 ours）"),
                                label="session 调度时间轴 / 抢占尖峰 / KV 压力",
                            )
                            f5_feature_data = gr.Markdown("运行后显示抢占计数 · KV 峰值 · session 调度证据。")
                        f5_sessions = gr.DataFrame(
                            headers=[
                                "task_id", "状态", "priority", "步数", "reward", "延迟ms", "错误",
                            ],
                            label="当前层 session 状态", interactive=False,
                        )
                        f5_comparison = gr.Markdown(_f5_comparison_rows([]))
                    with gr.Tab("🧊 通用改进 · KV量化 + 分层"):
                        _kv_sim = {"s": SimState.fresh()}
                        _kv_render_cache: dict[str, tuple | None] = {"value": None}
                        with gr.Group(elem_classes=["context-param-panel"]):
                            gr.Markdown("**参数调整**", elem_classes=["context-param-title"])
                            gen_mode = gr.Radio(
                                choices=[
                                    ("baseline（fp16 · 仅 NPU）", "baseline"),
                                    ("+ KV量化 (int8)", "quant"),
                                    ("+ LMCache 三级分层", "tier"),
                                    ("量化 → 分层（依次实测）", "full"),
                                ],
                                value="full",
                                label="改进档位",
                            )
                            with gr.Row():
                                kv_domain = gr.Dropdown(
                                    choices=["retail", "airline"], value="retail",
                                    label="tau-bench domain", scale=1,
                                )
                                kv_task_id = gr.Number(
                                    value=0, minimum=0, step=1, label="task id", scale=1,
                                )
                                kv_max_steps = gr.Number(
                                    value=8, minimum=1, maximum=20, step=1,
                                    label="max steps", scale=1,
                                )
                            with gr.Row():
                                kv_start_btn = gr.Button("▶ 启动引擎并运行任务", variant="primary", scale=2)
                                kv_stop_btn = gr.Button("⏸ 停止", scale=1)
                                kv_reset_btn = gr.Button("↺ 重置", scale=1)
                            gr.Markdown(
                                "启动后先跑真实 BF16 baseline，再按所选档位启动 C8 / LMCache，"
                                "每一档都重复运行同一个 tau-bench 任务。"
                                "下方动画展示 KV 在 NPU、CPU 与 SSD 间的真实分层，以及量化后的容量提升。"
                            )
                        with gr.Group(elem_classes=["context-chat-panel"]):
                            gr.Markdown(
                                "**KV 内存与三级分层 + 量化密度（实时演示）**",
                                elem_classes=["context-chat-title"],
                            )
                            kv_tier_html = gr.HTML(render_tier_html(_kv_sim["s"], "full"))
                            kv_quant_html = gr.HTML(render_quant_html(_kv_sim["s"], "full"))
                            kv_static_results = gr.Markdown(
                                value="",
                                visible=False,
                                elem_classes=["kv-static-results"],
                            )
                        with gr.Group(elem_classes=["context-chat-panel"]):
                            gr.Markdown("**真实 tau-bench 任务轨迹**", elem_classes=["context-chat-title"])
                            kv_task_status = gr.Markdown("尚未运行")
                            kv_task_chat = gr.Chatbot(
                                type="messages", height=360, show_label=False,
                                elem_classes=["context-chatbot"],
                            )
                        kv_timer = gr.Timer(value=1.5, active=False)
            with gr.Column(scale=2, elem_classes=["live-monitor-column"]):
                with gr.Group(elem_classes=["live-monitor-sticky"]):
                    status_md = gr.Markdown(elem_classes=["live-monitor-status"])
                    live_plot = gr.Plot(
                        label=f"实时监控（窗口={WINDOW_S:.0f}s）",
                        show_label=True,
                        elem_classes=["live-monitor-plot"],
                    )

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

        # 事件：上下文模式自动路由到 τ-bench / LongBench
        def _context_mode_updates(mode, baseline_target):
            workload = _context_workload(str(mode), str(baseline_target))
            is_tau = workload == "tau-bench"
            f2_visible = str(mode) in {"F2", "F2+F3"}
            return (
                gr.update(visible=is_tau),
                gr.update(visible=not is_tau),
                gr.update(visible=str(mode) == "baseline"),
                gr.update(visible=f2_visible),
            )

        context_mode.change(
            _context_mode_updates,
            [context_mode, baseline_workload],
            [
                tau_inputs_group,
                longbench_inputs_group,
                baseline_workload,
                f2_options_group,
            ],
            queue=False,
        )
        baseline_workload.change(
            _context_mode_updates,
            [context_mode, baseline_workload],
            [
                tau_inputs_group,
                longbench_inputs_group,
                baseline_workload,
                f2_options_group,
            ],
            queue=False,
        )
        context_run.click(
            run_context_task,
            [
                context_mode,
                baseline_workload,
                context_tau_domain,
                context_tau_taskid,
                context_tau_maxsteps,
                context_longbench_data_zip,
                context_longbench_taskid,
                context_longbench_maxsteps,
                context_f2_method,
                context_f2_trigger,
                context_f2_recompress_delta,
                context_f2_retention,
            ],
            [
                context_chatbot,
                context_status,
                context_prompt_view,
                context_f2_before,
                context_f2_after,
                context_f3_before,
                context_f3_after,
            ],
            api_name="context_task",
        )

        # 事件：引擎（多选功能 → 一键启动；选 priority 自动设 0.27/16384）
        def _preset_params(features):
            return (0.27, 16384) if (features and "priority" in features) else (0.9, 32768)

        eng_features.change(
            lambda fs: _preset_params(fs), [eng_features], [eng_memutil, eng_maxmlen],
        )

        def _start_engine(features, memutil, maxmlen):
            engine_mgr.gpu_mem_util = float(memutil) if memutil else 0.9
            engine_mgr.max_model_len = int(maxmlen) if maxmlen else 32768
            yield from engine_mgr.start(list(features or []))

        eng_start_btn.click(
            _start_engine, [eng_features, eng_memutil, eng_maxmlen], [engine_status_md],
        )

        def _stop_engine():
            stopped = engine_mgr.stop()
            return (
                "⏹ 8000 端口引擎及 EngineCore 已停止。"
                if stopped else "8000 端口未发现引擎进程。"
            )

        eng_stop_btn.click(_stop_engine, None, [engine_status_md])

        # 事件：高并发 F5 一键三层对比
        f5_baseline_btn.click(
            run_f5_baseline,
            [f5_total_sessions, f5_conc, f5_steps, f5_seed, f5_results],
            [f5_status, f5_comparison, f5_sessions, f5_live_plot, f5_feature_data, f5_results],
            api_name="f5_run_baseline",
        )
        f5_ours_btn.click(
            run_f5_ours,
            [f5_total_sessions, f5_conc, f5_steps, f5_seed, f5_results],
            [f5_status, f5_comparison, f5_sessions, f5_live_plot, f5_feature_data, f5_results],
            api_name="f5_run_ours",
        )
        f5_stop_btn.click(stop_f5, None, [f5_status], queue=False, api_name="f5_stop")

        # 事件：通用改进真实演示（启动引擎 → tau-bench → /metrics 驱动动画）
        def _render_kv(s, mode):
            rendered = render_all(s, mode)
            _kv_render_cache["value"] = rendered
            return rendered

        def _kv_start(mode, domain, task_id, max_steps):
            hidden_results = gr.update(value="", visible=False)
            if not kv_run_lock.acquire(blocking=False):
                with kv_state_lock:
                    s = _kv_sim["s"]
                    s.message = "已有通用改进演示正在运行。"
                    rendered = _render_kv(s, mode)
                yield (
                    *rendered, hidden_results, [], "已有任务运行",
                    gr.Timer(active=s.running),
                )
                return
            tau_locked = False
            try:
                tau_locked = tau_run_lock.acquire(blocking=False)
                if not tau_locked:
                    raise RuntimeError("上下文优化页已有 tau-bench 任务运行，请先等待其结束")
                kv_cancel.clear()
                engine_mgr.gpu_mem_util = 0.9
                engine_mgr.max_model_len = 32768
                stages = engine_stages_for(str(mode))
                user_sim = _tau_user_sim_settings()
                tid = int(task_id)
                stage_results: dict[str, dict[str, object]] = {}
                task_chat: list[dict] = []
                stage_labels = {
                    "baseline": "原始 BF16 baseline",
                    "quant": "C8 int8",
                    "tier": "LMCache 分层",
                }

                for stage_index, features in enumerate(stages, start=1):
                    stage = stage_name_for(features)
                    label = stage_labels[stage]
                    with kv_state_lock:
                        s = SimState.fresh()
                        s.running = True
                        s.phase = "starting"
                        s.current_stage = stage
                        s.active_quant = stage == "quant"
                        s.active_tier = stage == "tier"
                        s.stage_results = {
                            key: dict(value) for key, value in stage_results.items()
                        }
                        quant_result = s.stage_results.get("quant") or {}
                        quant_capacity = quant_result.get("capacity_tokens")
                        if isinstance(quant_capacity, int | float):
                            s.quant_capacity_tok = float(quant_capacity)
                        s.message = (
                            f"阶段 {stage_index}/{len(stages)}：正在启动 {label} 引擎。"
                        )
                        _kv_sim["s"] = s
                        rendered = _render_kv(s, str(mode))
                    yield (*rendered, hidden_results, [], s.message, gr.Timer(active=True))

                    monitor.clear()
                    for engine_message in engine_mgr.start(features):
                        with kv_state_lock:
                            s.message = f"阶段 {stage_index}/{len(stages)} · {label}：{engine_message}"
                            s.phase = "error" if engine_message.startswith("❌") else "starting"
                            rendered = _render_kv(s, str(mode))
                        yield (*rendered, hidden_results, [], s.message, gr.Timer(active=True))
                        if kv_cancel.is_set():
                            return
                    if not engine_mgr.is_alive() or not engine_mgr.health():
                        raise RuntimeError(engine_mgr.last_start_error or f"{label} 引擎未就绪")

                    if stage == "quant":
                        _warmup_f5_engine(engine_mgr.base_url, require_c8_quality=True)
                    capacity = engine_mgr.kv_capacity_tokens()
                    with kv_state_lock:
                        s.npu_capacity_tok = float(capacity) if capacity else None
                        if stage == "quant":
                            s.quant_capacity_tok = s.npu_capacity_tok
                        s.stage_results.setdefault(stage, {})["capacity_tokens"] = capacity
                        s.message = (
                            f"{label} 引擎就绪，真实 KV 容量 {capacity or '—'} tokens；"
                            f"正在运行同一 tau-bench {domain} #{tid}。"
                        )
                        s.phase = "task"
                        rendered = _render_kv(s, str(mode))
                    yield (*rendered, hidden_results, [], s.message, gr.Timer(active=True))

                    monitor.clear()
                    last_status = ""
                    task_chat = []
                    for task_chat, task_status in tau_bench_ui.run_tau_task_streaming(
                        domain=str(domain), split="test", task_id=tid,
                        engine_url=engine_url, model=model,
                        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                        user_model=str(user_sim["model"]),
                        user_provider=str(user_sim["provider"]),
                        user_api_base=str(user_sim["api_base"]),
                        user_api_key=user_sim["api_key"],
                        max_steps=int(max_steps), middlewares=None,
                    ):
                        last_status = task_status
                        with kv_state_lock:
                            update_from_samples(
                                s, monitor.snapshot(), str(mode),
                                npu_capacity_tokens=capacity,
                            )
                            s.phase = "task"
                            s.message = f"{label} · {task_status}"
                            rendered = _render_kv(s, str(mode))
                        yield (
                            *rendered, hidden_results, task_chat, s.message,
                            gr.Timer(active=True),
                        )
                        if kv_cancel.is_set():
                            return

                    with kv_state_lock:
                        update_from_samples(
                            s, monitor.snapshot(), str(mode),
                            npu_capacity_tokens=capacity,
                        )
                        result = s.stage_results.setdefault(stage, {})
                        result["status"] = last_status or "任务结束"
                        result["done"] = True
                        stage_results = {
                            key: dict(value) for key, value in s.stage_results.items()
                        }
                        failed = any(
                            mark in last_status for mark in ("失败", "出错", "未设置")
                        )
                        s.phase = "error" if failed else (
                            "done" if stage_index == len(stages) else "stage_done"
                        )
                        s.running = stage_index < len(stages)
                        s.message = f"{label} · {last_status or '任务结束'}"
                        rendered = _render_kv(s, str(mode))
                    yield (
                        *rendered,
                        gr.update(
                            value=render_static_experiment_results_md(),
                            visible=True,
                        ) if should_reveal_static_results(
                            str(mode),
                            stage_index=stage_index,
                            stage_count=len(stages),
                            failed=failed,
                        )
                        else hidden_results,
                        task_chat, s.message,
                        gr.Timer(active=stage_index < len(stages)),
                    )
                    if failed:
                        break
            except Exception as exc:  # noqa: BLE001
                with kv_state_lock:
                    s = _kv_sim["s"]
                    s.running = False
                    s.phase = "error"
                    s.message = f"运行失败：{exc}"
                    rendered = _render_kv(s, str(mode))
                yield (*rendered, hidden_results, [], s.message, gr.Timer(active=False))
            finally:
                if tau_locked:
                    tau_run_lock.release()
                kv_run_lock.release()

        def _kv_stop(mode):
            kv_cancel.set()
            engine_mgr.stop()
            with kv_state_lock:
                _kv_sim["s"].running = False
                _kv_sim["s"].phase = "stopped"
                _kv_sim["s"].message = "任务已取消，引擎已停止。"
                return (
                    *_render_kv(_kv_sim["s"], mode),
                    gr.update(value="", visible=False),
                    _kv_sim["s"].message,
                    gr.Timer(active=False),
                )

        def _kv_reset(mode):
            kv_cancel.set()
            engine_mgr.stop()
            with kv_state_lock:
                _kv_sim["s"] = SimState.fresh()
                return (
                    *_render_kv(_kv_sim["s"], mode),
                    gr.update(value="", visible=False),
                    [], "尚未运行", gr.Timer(active=False),
                )

        def _kv_tick(mode):
            with kv_state_lock:
                s = _kv_sim["s"]
                if s.phase == "task":
                    update_from_samples(
                        s, monitor.snapshot(), str(mode),
                        npu_capacity_tokens=engine_mgr.kv_capacity_tokens(),
                    )
                rendered = render_all(s, mode)
                if rendered == _kv_render_cache["value"]:
                    return tuple(gr.skip() for _ in rendered)
                _kv_render_cache["value"] = rendered
                return rendered

        def _kv_mode(mode):
            with kv_state_lock:
                return (
                    *_render_kv(_kv_sim["s"], mode),
                    gr.update(value="", visible=False),
                )

        kv_start_btn.click(
            _kv_start, [gen_mode, kv_domain, kv_task_id, kv_max_steps],
            [
                kv_tier_html, kv_quant_html, kv_static_results,
                kv_task_chat, kv_task_status, kv_timer,
            ],
            api_name="kv_start",
            concurrency_limit=None,
        )
        kv_stop_btn.click(
            _kv_stop, [gen_mode],
            [
                kv_tier_html, kv_quant_html, kv_static_results,
                kv_task_status, kv_timer,
            ], api_name="kv_stop",
            queue=False,
        )
        kv_reset_btn.click(
            _kv_reset, [gen_mode],
            [
                kv_tier_html, kv_quant_html, kv_static_results,
                kv_task_chat, kv_task_status, kv_timer,
            ],
            api_name="kv_reset",
            queue=False,
        )
        kv_timer.tick(
            _kv_tick, [gen_mode],
            [kv_tier_html, kv_quant_html],
        )
        gen_mode.change(
            _kv_mode, [gen_mode],
            [kv_tier_html, kv_quant_html, kv_static_results],
        )

        # 事件：会话对话查看已移除（统一 benchmark 改为后台 run_study 落盘，结果见本页 + 右侧历史）

        # 事件：监控刷新（共享，与对话/任务解耦）
        timer = gr.Timer(value=3.0)
        timer.tick(refresh, None, [status_md, live_plot])
        demo.load(refresh, None, [status_md, live_plot])

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
