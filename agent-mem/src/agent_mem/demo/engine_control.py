"""引擎生命周期控制（config/preset 驱动）—— demo 前端用。

与旧版（硬编码 ``CONFIG_FLAGS`` 档位）不同：本模块**任何** ``configs/*.yaml`` preset
都经 ``load_config`` → ``start_engine``（其 ``build_serve_args`` 已把 c8 / kv_transfer /
priority / middleware 全渲染成 vLLM CLI flag）→ 轮询 ``/health``。即 F1 C8、F4 LMCache、
F5 priority、unified-tau-freq/longbench 都用同一条路径启动。

``EngineHandle`` 由 Gradio ``gr.State`` 持有（无引擎单例）；``start`` 是 generator，
yield 状态文本（Gradio 5.x 流式）。
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from agent_mem.config import load_config
from agent_mem.server.vllm_server import start_engine, stop_engine

DEFAULT_MODEL_PATH = os.environ.get("AGENT_MEM_MODEL_PATH", "models/Qwen2.5-7B-Instruct")
DEFAULT_PORT = 8000
DEFAULT_LOG_DIR = os.environ.get("AGENT_MEM_LOG_DIR", "logs-demo")
# τ-bench 工具调用解析需要 hermes（longbench 同样兼容）
TOOL_CALL_PARSER = "hermes"


@dataclass
class EngineHandle:
    """由 gr.State 持有的引擎句柄（跨 callback 保活）。"""

    proc: subprocess.Popen | None = None
    base_url: str | None = None
    preset: str | None = None  # preset stem，用于打标
    log_file: str | None = None
    pid: int | None = None


def is_alive(h: EngineHandle) -> bool:
    return h.proc is not None and h.proc.poll() is None


def stop(h: EngineHandle) -> None:
    """停引擎（killpg 整个进程组，带 EngineCore/Worker）。"""
    if h.proc is not None:
        try:
            stop_engine(h.proc)
        except Exception:  # noqa: BLE001 — 停止失败不阻断 UI
            pass
    h.proc = None
    h.base_url = None
    h.preset = None
    h.pid = None


def _health(base_url: str, *, timeout: float = 3.0) -> bool:
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        return httpx.get(f"{root}/health", timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def start(
    h: EngineHandle,
    preset_path: str,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    port: int = DEFAULT_PORT,
    timeout: float = 600.0,
) -> Iterator[str]:
    """启引擎：停旧 → load_config → start_engine → 轮询 /health。

    Generator：yield 状态文本（Gradio 流式更新）。成功时 ``h`` 就绪。
    """
    yield "⏹ 停止旧引擎（若有）…"
    stop(h)
    try:
        cfg = load_config(preset_path)
    except Exception as e:  # noqa: BLE001
        yield f"❌ 配置解析失败（{preset_path}）：{e}"
        return
    stem = Path(preset_path).stem
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
    log_file = f"{DEFAULT_LOG_DIR}/engine_{stem}.log"
    yield f"▶ 启动 [{stem}] 引擎（加载权重 ~1–2 min，日志 → {log_file}）…"
    try:
        proc, base_url = start_engine(
            cfg,
            model_path=model_path,
            port=port,
            tool_call_parser=TOOL_CALL_PARSER,
            log_file=log_file,
        )
    except Exception as e:  # noqa: BLE001
        yield f"❌ 启动失败：{e}"
        return
    h.proc, h.base_url, h.preset, h.log_file, h.pid = proc, base_url, stem, log_file, proc.pid

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:  # 进程已退
            h.proc = None
            yield f"❌ [{stem}] 引擎进程异常退出（见 {log_file}）。"
            return
        if _health(base_url):
            yield f"✅ [{stem}] 引擎就绪（PID {proc.pid}，{base_url}）。"
            return
        time.sleep(3)
    yield f"❌ [{stem}] {timeout:.0f}s 未就绪（超时，见 {log_file}）。"
    stop(h)
