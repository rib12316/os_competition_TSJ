"""引擎生命周期管理（demo 前端按钮驱动）：按档位起/停 vLLM 子进程 + 健康轮询。

前端按钮调 :class:`EngineManager.start(config)`（生成器，逐步 yield 状态）：
停止旧引擎 → 按 config 的 flag 起 vLLM 子进程（setsid 进程组，便于整组 kill）
→ 轮询 ``/health`` 到就绪 → 设当前 config（作 bench 自动标签）。

档位 → vLLM flag（**不需合并功能分支**即可起的档；F1 C8/F2 压缩需合并，另接）::

    baseline      --no-enable-prefix-caching
    prefix-cache  （V1 默认，无额外 flag）
    lmcache       --kv-transfer-config {"kv_connector":"LMCacheAscendConnector",...}
    all-engine    prefix-cache（默认）+ lmcache

所有档都带 ``--enable-auto-tool-choice --tool-call-parser hermes``（τ-bench 工具调用必需，
否则 400）。NPU 由本进程直接拉起（用户已授权前端管引擎）。
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Iterator

import httpx

# 档位 → 额外 vLLM CLI flag（v1：全部档可起，含 F1 C8 / F4 LMCache / F5 priority）
# ⚠️ vLLM V1 prefix cache 默认开！要隔离 LMCache 必须 --no-enable-prefix-caching，否则
#    "lmcache" 实际 = prefix+lmcache（KV命中率是 prefix 的，非 LMCache），且与 all-engine 同义。
CONFIG_FLAGS: dict[str, list[str]] = {
    "baseline": ["--no-enable-prefix-caching"],
    "prefix-cache": [],  # prefix 默认开
    "lmcache": [  # 隔离 LMCache：prefix 关 + LMCache（其收益看 HBM/external hit，非 prefix 命中率）
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}',
    ],
    "c8": [  # F1：C8 int8 KV（⚠️ 需模型先 annotate + post-RoPE 校准；FULL decode 死锁故 FULL_DECODE_ONLY）
        "--quantization", "ascend",
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    ],
    "priority": ["--scheduling-policy", "priority"],  # F5 引擎层 flag（真增益靠应用层准入控制）
    "all-engine": [  # 全栈：prefix + LMCache + C8 + priority
        "--kv-transfer-config", '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}',
        "--quantization", "ascend",
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--scheduling-policy", "priority",
    ],
}
# v1：全部档可起（C8 需模型预 annotate + 校准，见 docs/F1-c8-injection.md）
PENDING_CONFIGS: () = ()

# 所有档共用的基础 flag（模型加载 + τ-bench 工具调用）
BASE_FLAGS = [
    "--max-model-len", "32768",
    "--gpu-memory-utilization", "0.9",
    "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
]


class EngineManager:
    """管一个 vLLM 引擎子进程：按档位起 / 停 / 健康轮询。"""

    def __init__(
        self,
        *,
        model_path: str,
        served_name: str = "Qwen2.5-7B-Instruct",
        port: int = 8000,
        host: str = "0.0.0.0",
        python_exe: str | None = None,
        log_file: str = "/data/os_competition_TSJ/logs-demo/engine.log",
    ):
        self.model_path = model_path
        self.served_name = served_name
        self.port = port
        self.host = host
        self.python_exe = python_exe or os.environ.get("AGENT_MEM_PYTHON", _detect_venv_python())
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)  # 日志目录（/data 数据盘）
        self.proc: subprocess.Popen | None = None
        self.config: str | None = None  # 当前引擎档位（作 bench 自动标签）

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def health(self, *, timeout: float = 3.0) -> bool:
        try:
            return httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=timeout).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def _cmd(self, config: str) -> list[str]:
        return [
            self.python_exe, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port), "--host", self.host,
            "--served-model-name", self.served_name,
            *BASE_FLAGS,
            *CONFIG_FLAGS.get(config, []),
        ]

    def _engine_env(self, config: str = "") -> dict:
        """引擎子进程 env：CANN python 目录补进 PYTHONPATH（修丢 acl 坑）+ C8 patch（F1）。"""
        env = os.environ.copy()
        cann = _cann_python_paths()
        if cann:
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(cann) + (os.pathsep + pp if pp else "")
        # F1 C8：Qwen2（Qwen2.5）需 sitecustomize 给 load_weights 打补丁才能加载 KV scale
        if config in ("c8", "all-engine"):
            env["QWEN2_C8_PATCH"] = "1"
            patch_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kv", "c8patch"
            )
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = patch_dir + (os.pathsep + pp if pp else "")
        return env

    def stop(self) -> None:
        """整组 kill 当前引擎（主进程 + EngineCore 子进程）。"""
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        self.proc = None
        self._free_port()

    def _free_port(self) -> None:
        """杀掉占用本端口的 vLLM 引擎（APIServer + EngineCore 子进程），释放 HBM。

        pkill ``api_server`` 会留下孤儿 ``VLLM::EngineCore`` 仍占 HBM，必须一并杀。
        """
        for pat in (f"api_server.*--port {self.port}", "EngineCore"):
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pat],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:  # noqa: BLE001
                pass
        time.sleep(4)  # 等 NPU HBM 释放，给下一档腾地方

    def start(self, config: str, *, timeout: float = 300.0) -> Iterator[str]:
        """按档位起引擎，生成器逐步 yield 状态文本（供 Gradio 流式显示）。

        停旧 + 释放端口 → 起新（setsid 进程组）→ 轮询 ``/health``。就绪设 ``self.config``。
        """
        if config in PENDING_CONFIGS:
            yield f"⏳ {config} 需合并对应功能分支才能真起（C8 补丁 / lingua venv）。"
            return
        if config not in CONFIG_FLAGS:
            yield f"❌ 未知档位 {config!r}"
            return

        yield f"⏹ 停止旧引擎（{self.config or '外部'}）…"
        self.stop()
        self._free_port()
        yield f"▶ 启动 [{config}] 引擎（加载权重 ~1-2min，日志 → {self.log_file}）…"

        cmd = self._cmd(config)
        log_fh = open(self.log_file, "ab")  # noqa: SIM115
        # 引擎 env：确保 CANN python 目录在 PYTHONPATH（acl/tbe 等模块），否则 EngineCore
        # 子进程 ModuleNotFoundError: acl（demo 若用 PYTHONPATH=src 覆盖会丢 CANN 路径）
        self.proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=self._engine_env(config),
            start_new_session=True,  # setsid → 新进程组，便于 os.killpg 整组杀
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                tail = _tail(self.log_file, 3)
                self.proc = None
                yield f"❌ 引擎进程退出。日志末尾：\n{tail}"
                return
            if self.health(timeout=3.0):
                self.config = config
                yield f"✅ [{config}] 引擎就绪（PID {self.proc.pid}，{self.base_url}）。后续 bench 自动用此档为标签。"
                return
            time.sleep(4)
        self.config = None
        yield f"❌ [{config}] {timeout:.0f}s 未就绪（超时）。"


def _detect_venv_python() -> str:
    for cand in ("/data/os_competition_TSJ/.venv/bin/python", ".venv/bin/python"):
        if os.path.exists(cand):
            return cand
    return "python"


def _cann_python_paths() -> list[str]:
    """检测 CANN 的 python 目录（acl/tbe/toolkit），补进引擎子进程 PYTHONPATH。

    匹配 shell set_env.sh 常注入的 PYTHONPATH 项；存在即返回。
    """
    patterns = [
        "/usr/local/Ascend/cann-*/python/site-packages",
        "/usr/local/Ascend/cann-*/opp/built-in/op_impl/ai_core/tbe",
        "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
        "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe",
    ]
    import glob

    out: list[str] = []
    for pat in patterns:
        for p in glob.glob(pat):
            if os.path.isdir(p) and p not in out:
                out.append(p)
    return out


def _tail(path: str, n: int = 5) -> str:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")[-2000:].strip().splitlines()[-n:]
    except Exception:  # noqa: BLE001
        return ""
