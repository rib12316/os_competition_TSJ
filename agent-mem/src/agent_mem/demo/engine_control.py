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

import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

# 功能 → vLLM flag（前端多选组合；prefix-cache 默认开，不选它 = baseline 关前缀）
FEATURE_FLAGS: dict[str, list[str]] = {
    "c8": [  # F1：C8 int8 KV（需模型先 annotate+校准；FULL decode 死锁故 FULL_DECODE_ONLY）
        "--quantization", "ascend",
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
    ],
    "lmcache": [  # F4：LMCache 分层（社区插件，三级 NPU/CPU/Disk）
        "--kv-transfer-config", '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}',
    ],
    "simple-offload": [  # F5-Phase2：SimpleCPUOffload（vLLM 自带，单级 lazy HBM↔CPU）
        "--kv-transfer-config",
        '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both",'
        '"kv_connector_extra_config":{"cpu_bytes_to_use":4294967296,"lazy_offload":true}}',
    ],
    "priority": ["--scheduling-policy", "priority"],  # F5 引擎层 flag（真增益靠应用层准入控制）
}
# 所有档共用的基础 flag（mem-util / max-model-len 由 EngineManager 注入）
BASE_FLAGS = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]

C8_RUNTIME_FILES = (
    "quant_model_description.json",
    "kv_cache_scales.safetensors",
    "model.safetensors.index.json",
)
C8_BACKUP_SUFFIX = ".c8bak"
C8_STOCK_RUNTIME_BACKUP = "model.safetensors.index.json.demo-stock.bak"


def flags_for(features: list[str]) -> list[str]:
    """多选功能 → 合并 vLLM flag。prefix-cache 默认开；不选它 = baseline（--no-enable-prefix-caching）。"""
    flags: list[str] = []
    if "prefix-cache" not in features:
        flags.append("--no-enable-prefix-caching")
    for f in ("c8", "lmcache", "simple-offload", "priority"):
        if f in features:
            flags.extend(FEATURE_FLAGS[f])
    return flags


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
        gpu_mem_util: float = 0.9,
        max_model_len: int = 32768,
    ):
        self.model_path = model_path
        self.served_name = served_name
        self.port = port
        self.host = host
        self.python_exe = python_exe or os.environ.get("AGENT_MEM_PYTHON", _detect_venv_python())
        self.log_file = log_file
        self.gpu_mem_util = gpu_mem_util  # 显存上限（F5 制压场景调小，如 0.27）
        self.max_model_len = max_model_len  # F5 制压(0.27)时须调小到 16384，否则 1 个 max-len 请求都装不下
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)  # 日志目录（/data 数据盘）
        self.proc: subprocess.Popen | None = None
        self.config: str | None = None  # 当前引擎档位（作 bench 自动标签）
        self.last_start_error: str | None = None

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

    def _cmd(self, features: list[str]) -> list[str]:
        return [
            self.python_exe, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port), "--host", self.host,
            "--served-model-name", self.served_name,
            "--gpu-memory-utilization", str(self.gpu_mem_util),
            "--max-model-len", str(self.max_model_len),
            *BASE_FLAGS,
            *flags_for(features),
        ]

    def _engine_env(self, features: list[str] | None = None) -> dict:
        """引擎子进程 env：CANN python 目录补进 PYTHONPATH（修丢 acl 坑）+ C8 patch（F1）。"""
        env = os.environ.copy()
        env.setdefault("VLLM_NO_USAGE_STATS", "1")
        cann = _cann_python_paths()
        if cann:
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(cann) + (os.pathsep + pp if pp else "")
        # F1 C8：Qwen2（Qwen2.5）需 sitecustomize 给 load_weights 打补丁才能加载 KV scale
        if features and "c8" in features:
            env["QWEN2_C8_PATCH"] = "1"
            patch_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kv", "c8patch"
            )
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = patch_dir + (os.pathsep + pp if pp else "")
        if features and "lmcache" in features:
            # vLLM and LMCache telemetry both spawn py-cpuinfo during startup; on this
            # container concurrent probes can return empty JSON and degrade the connector.
            env["LMCACHE_TRACK_USAGE"] = "false"
            env.setdefault("PYTHONHASHSEED", "0")
            env.setdefault("LMCACHE_LOCAL_CPU", "true")
            env.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", "4.0")
            disk_path = os.environ.get(
                "AGENT_MEM_LMCACHE_DISK", "/data/os_competition_TSJ/logs-demo/lmcache-disk"
            )
            os.makedirs(disk_path, exist_ok=True)
            env.setdefault("LMCACHE_LOCAL_DISK", disk_path)
            env.setdefault("LMCACHE_MAX_LOCAL_DISK_SIZE", "2.0")
            env.setdefault("LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR", "true")
            internal_port = 6999 + max(0, self.port - 8000) * 2
            env.setdefault("LMCACHE_INTERNAL_API_SERVER_ENABLED", "true")
            env.setdefault("LMCACHE_INTERNAL_API_SERVER_HOST", "127.0.0.1")
            env.setdefault("LMCACHE_INTERNAL_API_SERVER_PORT_START", str(internal_port))
        return env

    def kv_capacity_tokens(self) -> int | None:
        """Read the current engine's real KV token capacity from its startup log."""
        try:
            text = Path(self.log_file).read_text(encoding="utf-8", errors="replace")[-200_000:]
        except OSError:
            return None
        import re

        matches = re.findall(r"GPU KV cache size:\s*([0-9,]+) tokens", text)
        return int(matches[-1].replace(",", "")) if matches else None

    def validate_c8_artifacts(self) -> None:
        """Validate the calibrated C8 backups without changing the active model."""
        model_dir = Path(self.model_path).resolve()
        backups = {
            name: model_dir / f"{name}{C8_BACKUP_SUFFIX}"
            for name in C8_RUNTIME_FILES
        }
        missing = [path.name for path in backups.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"C8 预检失败，缺少校准产物：{', '.join(missing)}")
        empty = [path.name for path in backups.values() if path.stat().st_size <= 0]
        if empty:
            raise RuntimeError(f"C8 预检失败，产物为空：{', '.join(empty)}")

        try:
            description = json.loads(
                backups["quant_model_description.json"].read_text(encoding="utf-8")
            )
            c8_index = json.loads(
                backups["model.safetensors.index.json"].read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"C8 预检失败，元数据不可读：{exc}") from exc
        if description.get("kv_cache_type") != "C8":
            raise RuntimeError("C8 预检失败：quant_model_description 的 kv_cache_type 不是 C8")

        described_scales = {
            key for key, value in description.items()
            if key.endswith(".kv_cache_scale") and value == "C8"
        }
        weight_map = c8_index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise RuntimeError("C8 预检失败：C8 index 缺少 weight_map")
        indexed_scales = {
            key for key, value in weight_map.items()
            if key.endswith(".kv_cache_scale")
            and value == "kv_cache_scales.safetensors"
        }
        if not described_scales or described_scales != indexed_scales:
            raise RuntimeError(
                "C8 预检失败：quant description 与 index 中的 KV scale 映射不一致"
            )

        try:
            import numpy as np
            from safetensors import safe_open

            scale_path = backups["kv_cache_scales.safetensors"]
            global_min = float("inf")
            global_max = float("-inf")
            with safe_open(str(scale_path), framework="numpy") as scale_file:
                stored_keys = set(scale_file.keys())
                if stored_keys != described_scales:
                    raise RuntimeError(
                        "C8 预检失败：safetensors 中的 KV scale keys 与描述不一致"
                    )
                for key in sorted(stored_keys):
                    values = scale_file.get_tensor(key)
                    if values.size == 0 or not np.isfinite(values).all():
                        raise RuntimeError(f"C8 预检失败：{key} 包含空值或非有限值")
                    if (values <= 0).any():
                        raise RuntimeError(f"C8 预检失败：{key} 包含非正 scale")
                    global_min = min(global_min, float(values.min()))
                    global_max = max(global_max, float(values.max()))
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"C8 预检失败，scale 文件不可读：{exc}") from exc
        if global_max <= global_min * (1.0 + 1e-6):
            raise RuntimeError(
                "C8 预检失败：全部 scale 为常数，检测到未校准的占位产物"
            )

        active_index = model_dir / "model.safetensors.index.json"
        stock_backup = model_dir / C8_STOCK_RUNTIME_BACKUP
        if not active_index.is_file() and not stock_backup.is_file():
            raise RuntimeError("C8 预检失败：找不到 stock model.safetensors.index.json")

    @staticmethod
    def _copy_atomic(source: Path, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.agent-mem-{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _restore_c8_runtime(self) -> bool:
        """Restore stock metadata after a C8 engine run or interrupted activation."""
        model_dir = Path(self.model_path).resolve()
        stock_backup = model_dir / C8_STOCK_RUNTIME_BACKUP
        if not stock_backup.is_file():
            return False
        self._copy_atomic(stock_backup, model_dir / "model.safetensors.index.json")
        for name in ("quant_model_description.json", "kv_cache_scales.safetensors"):
            try:
                (model_dir / name).unlink()
            except FileNotFoundError:
                pass
        stock_backup.unlink()
        return True

    def _activate_c8_runtime(self) -> None:
        """Atomically expose calibrated ``.c8bak`` files for one C8 engine run."""
        self._restore_c8_runtime()
        self.validate_c8_artifacts()
        model_dir = Path(self.model_path).resolve()
        active_index = model_dir / "model.safetensors.index.json"
        stock_backup = model_dir / C8_STOCK_RUNTIME_BACKUP
        active_description = model_dir / "quant_model_description.json"
        active_scales = model_dir / "kv_cache_scales.safetensors"
        if active_description.exists() or active_scales.exists():
            raise RuntimeError(
                "C8 激活被拒绝：模型目录已有非 demo 管理的 active C8 文件，请先确认其来源"
            )

        try:
            stock_index = json.loads(active_index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"C8 激活失败，stock index 不可读：{exc}") from exc
        stock_scale_keys = [
            key for key in (stock_index.get("weight_map") or {})
            if key.endswith(".kv_cache_scale")
        ]
        if stock_scale_keys:
            raise RuntimeError("C8 激活被拒绝：当前 stock index 已包含 KV scale 映射")

        self._copy_atomic(active_index, stock_backup)
        try:
            # Description is copied last: its presence is what activates ModelSlim C8.
            for name in (
                "kv_cache_scales.safetensors",
                "model.safetensors.index.json",
                "quant_model_description.json",
            ):
                self._copy_atomic(
                    model_dir / f"{name}{C8_BACKUP_SUFFIX}", model_dir / name
                )
        except Exception:
            self._restore_c8_runtime()
            raise

    def stop(self) -> bool:
        """Stop the configured-port engine, including externally started instances."""
        stopped = False
        if self.proc is not None:
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                self.proc.wait(timeout=15)
                stopped = True
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    stopped = True
                except Exception:  # noqa: BLE001
                    pass
        self.proc = None
        self.config = None
        reclaimed = self._free_port() or stopped
        self._restore_c8_runtime()
        return reclaimed

    def _free_port(self) -> bool:
        """Stop only API servers bound to this manager's configured port and their children."""
        try:
            found = subprocess.run(
                ["pgrep", "-f", f"vllm.entrypoints.openai.api_server.*--port {self.port}"],
                check=False,
                capture_output=True,
                text=True,
            )
            roots = [int(value) for value in found.stdout.split() if value.isdigit()]
        except Exception:  # noqa: BLE001
            roots = []

        def descendants(pid: int) -> list[int]:
            try:
                children_path = f"/proc/{pid}/task/{pid}/children"
                with open(children_path) as children_file:
                    children = [int(value) for value in children_file.read().split()]
            except (OSError, ValueError):
                return []
            out: list[int] = []
            for child in children:
                out.extend(descendants(child))
                out.append(child)
            return out

        targets: list[int] = []
        for root in roots:
            targets.extend(descendants(root))
            targets.append(root)
        # An API server killed outside this manager can leave an EngineCore adopted by PID 1.
        for comm_path in Path("/proc").glob("[0-9]*/comm"):
            try:
                pid = int(comm_path.parent.name)
                comm = comm_path.read_text(encoding="utf-8").strip().lower()
                stat = (comm_path.parent / "stat").read_text(encoding="utf-8")
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == 1 and comm.startswith("vllm") and "enginecor" in comm:
                targets.append(pid)
        targets = list(dict.fromkeys(targets))
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in targets:
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            if targets:
                time.sleep(2)
        return bool(targets)

    def start(self, features: list[str], *, timeout: float = 300.0) -> Iterator[str]:
        """按多选功能起引擎，生成器逐步 yield 状态文本（供 Gradio 流式显示加载过程）。

        停旧 + 释放端口 → 起新（setsid 进程组）→ 轮询 ``/health``。就绪设 ``self.config`` 为标签。
        ``features`` ∈ {"prefix-cache","c8","lmcache","priority"}；空 = baseline(prefix关)。
        """
        label = "+".join(features) if features else "baseline（prefix关）"
        self.last_start_error = None
        yield f"⏹ 停止旧引擎（{self.config or '外部'}）…"
        self.stop()
        if "c8" in features:
            try:
                self._activate_c8_runtime()
            except Exception as exc:  # noqa: BLE001
                self.last_start_error = str(exc)
                yield f"❌ [{label}] {self.last_start_error}"
                return
        yield f"▶ 启动 [{label}] 引擎（加载权重 ~1-2min，日志 → {self.log_file}）…"

        cmd = self._cmd(features)
        try:
            with open(self.log_file, "ab") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=self._engine_env(features),
                    start_new_session=True,
                )
                self.proc = proc
        except Exception as exc:  # noqa: BLE001
            self.proc = None
            self._restore_c8_runtime()
            self.last_start_error = f"无法创建引擎进程：{exc}"
            yield f"❌ [{label}] {self.last_start_error}"
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc is not proc:
                self.last_start_error = f"[{label}] 启动已取消"
                yield f"⏹ {self.last_start_error}。"
                return
            if proc.poll() is not None:
                tail = _tail(self.log_file, 3)
                self.proc = None
                self._restore_c8_runtime()
                self.last_start_error = f"引擎进程退出。日志末尾：\n{tail}"
                yield f"❌ {self.last_start_error}"
                return
            if self.health(timeout=3.0):
                self.config = label
                yield f"✅ [{label}] 引擎就绪（PID {self.proc.pid}，{self.base_url}）。"
                return
            time.sleep(4)
        self.last_start_error = f"[{label}] {timeout:.0f}s 未就绪（超时）"
        self.stop()
        yield f"❌ {self.last_start_error}。"


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
            lines = f.read().decode("utf-8", errors="replace")[-4000:].strip().splitlines()
            return "\n".join(lines[-n:])
    except Exception:  # noqa: BLE001
        return ""
