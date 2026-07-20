"""vLLM OpenAI server 封装（P1 引擎层）。

把 ``configs/*.yaml`` 的 engine 段翻译成 vLLM CLI 参数，subprocess 起
``vllm.entrypoints.openai.api_server``，并轮询健康端点直到就绪。

vLLM 内置即提供：OpenAI 兼容 ``/v1/chat/completions``（含 tool calling）+
Prometheus ``/metrics``（含 ``vllm:prefix_cache_hits_total``/``_queries_total``）。
所以 P1 主要是 **config→args 翻译 + 生命周期管理**。

真机：NPU 由 vllm-ascend 插件**自动识别**（不传 ``--device``）；⚠️ NPU 默认停着，
真跑前暂停等用户启动设备。``build_serve_args`` 是纯函数，无需设备即可单测。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx

from agent_mem.config import AppConfig, load_config
from agent_mem.kv import lmcache_env, lmcache_serve_flag


def build_serve_args(
    cfg: AppConfig,
    *,
    model_path: str | Path,
    served_name: str | None = None,
    device: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    tool_call_parser: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """把 config 翻译成 vLLM ``api_server`` 的 CLI 参数（不含 ``python -m ...`` 前缀）。

    - baseline 的 ``--no-enable-prefix-caching``、optimized 的 ``--kv-cache-dtype int8``
      都来自 ``engine.extra_args``（每条用 shlex 拆分，兼容 ``"--kv-cache-dtype int8"``）。
    - ``served_name`` 默认 = ``engine.model``（客户端用这个名字调）。
    """
    served = served_name or cfg.engine.model
    args: list[str] = [
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
        "--served-model-name", served,
    ]
    for a in cfg.engine.extra_args:
        args.extend(shlex.split(a))
    # 缝C（F4）：lmcache.enabled → 结构化附加 --enable-lmcache（config_file 走环境变量）
    args += lmcache_serve_flag(cfg.engine.lmcache)
    # NPU 由 vllm-ascend 插件自动识别——**不要**传 --device（vllm api_server 不认 npu/auto）。
    # 仅当显式指定 cpu/cuda/tpu/xpu（如 CPU 冒烟）时才传 --device。
    if device and device in {"cpu", "cuda", "tpu", "xpu"}:
        args += ["--device", device]
    if tool_call_parser:
        # 启用工具调用解析（Qwen 等模型真机可能需要，如 --tool-call-parser hermes）
        args += ["--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser]
    if extra:
        args += list(extra)
    return args


def engine_env(cfg: AppConfig) -> dict[str, str]:
    """引擎子进程需要的额外环境变量（缝C：LMCACHE_CONFIG_FILE 等）。

    返回**仅额外项**；:func:`start_engine` 会与 ``os.environ`` 合并后传给子进程。
    """
    env = dict(lmcache_env(cfg))
    return env


def _root_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def start_engine(
    cfg: AppConfig,
    *,
    model_path: str | Path,
    served_name: str | None = None,
    device: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
    tool_call_parser: str | None = None,
    python_exe: str | None = None,
    log_file: str | Path | None = None,
) -> tuple[subprocess.Popen, str]:
    """启动 vLLM api_server 子进程，返回 ``(proc, base_url)``（不阻塞，不等待就绪）。"""
    args = build_serve_args(
        cfg, model_path=model_path, served_name=served_name, device=device,
        host=host, port=port, tool_call_parser=tool_call_parser,
    )
    cmd = [python_exe or sys.executable, "-m", "vllm.entrypoints.openai.api_server", *args]
    out_fh = open(log_file, "wb") if log_file else subprocess.DEVNULL  # noqa: SIM115
    # 缝C：注入 LMCache 等环境变量（与 os.environ 合并）
    proc = subprocess.Popen(cmd, stdout=out_fh, stderr=subprocess.STDOUT,
                            env={**os.environ, **engine_env(cfg)})
    base_url = f"http://127.0.0.1:{port}/v1"
    return proc, base_url


def wait_for_engine(base_url: str, *, timeout: float = 600, interval: float = 2) -> bool:
    """轮询 ``/health`` 直到就绪或超时。超时抛 :class:`TimeoutError`。"""
    health = f"{_root_url(base_url)}/health"
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(health, timeout=5).status_code == 200:
                return True
        except Exception as e:  # noqa: BLE001 — 启动中连不上属正常
            last_err = e
        time.sleep(interval)
    raise TimeoutError(f"引擎 {timeout}s 未就绪（{health}）：{last_err}")


def stop_engine(proc: subprocess.Popen, *, timeout: float = 30) -> None:
    """优雅停止引擎子进程（terminate→kill）。"""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    """CLI：``python -m agent_mem.server.vllm_server --config ... --model-path ...``。"""
    import argparse

    p = argparse.ArgumentParser(description="启动 vLLM OpenAI server（config 驱动）")
    p.add_argument("--config", required=True, help="configs/*.yaml 路径")
    p.add_argument("--model-path", required=True, help="模型权重路径（vllm --model）")
    p.add_argument("--served-name", default=None, help="--served-model-name（默认 engine.model）")
    p.add_argument("--device", default=None, help="auto|cpu|npu|cuda（默认 vllm 自选）")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--tool-call-parser", default=None, help="如 hermes（启用工具调用解析）")
    p.add_argument("--log-file", default=None, help="引擎日志输出文件")
    p.add_argument("--timeout", type=float, default=600, help="等待就绪超时秒")
    args = p.parse_args()

    cfg = load_config(args.config)
    proc, base_url = start_engine(
        cfg, model_path=args.model_path, served_name=args.served_name, device=args.device,
        port=args.port, host=args.host, tool_call_parser=args.tool_call_parser,
        log_file=args.log_file,
    )
    print(f"[vllm-server] 启动中 → {base_url} (PID {proc.pid})", flush=True)
    try:
        wait_for_engine(base_url, timeout=args.timeout)
        print(f"[vllm-server] 就绪：{base_url} （Ctrl-C 停止）", flush=True)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[vllm-server] 收到中断，停止引擎...", flush=True)
        stop_engine(proc)
    except TimeoutError as e:
        print(f"[vllm-server] {e}", file=sys.stderr)
        stop_engine(proc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
