"""缝D · F2 Prompt / 上下文压缩中间件（基于 LLMLingua / LongLLMLingua）。

挂在 :meth:`transform_messages`：发引擎**前**把 agent 的**冷历史**压短，热尾与
system 原样保留。命中赛题"降显存/降延迟"——更短的 prompt → 更少的 prefill 与 KV。

设计要点（实现者必读）
^^^^^^^^^^^^^^^^^^^^^^

- **方案 A：单一方法 + 触发门**。``method`` 选一个压缩器（默认
  ``longllmlingua``，最适合长会话 agent 的 question-aware 压缩）；``trigger_tokens``
  是**压/不压**的开关——冷历史太短就**直接放行**（短上下文没有 lost-in-the-middle
  问题，压了反而白费延迟）。不是"两种方法分场景"。
- **阈值增量压缩（不是每步都压）**：冷历史首次过 ``trigger_tokens`` 才压一次，之后
  **只有自上次压缩后新增的冷 >= ``recompress_delta_tokens`` 才重压**，其余步**复用**
  缓存的压缩结果（新增冷 verbatim 补在压缩段后，无信息丢失）。把压缩从 O(步数) 降到
  偶发。缓存按 session 存 ``ctx.scratch``。
- **事件日志**：设环境变量 ``F2_EVENT_LOG=<path>`` 后，每步写一条 JSONL（session/步号/
  上下文 token/动作 skip|compress|reuse/压缩次数/前后 token/耗时）。触发计量使用与引擎
  一致的 tokenizer；``sent_tokens`` 仍来自引擎响应 ``usage.prompt_tokens``。
- **正典不动**：只变换发给引擎的副本（详见 ``base.py``），压缩**无序可恢复**。
- **tool_call 配对安全**：绝不留下孤立的 ``role=tool`` 消息。做法——把整段冷历史
  压成**一条**文本消息（冷的 assistant ``tool_calls`` 与冷的 tool 结果**一起**进
  文本，互不残留引用），热尾原样保留且边界 snap 到完整 ``tool_call→tool`` 组。
- **压缩器进程级单例**：``llmlingua`` 的 ``PromptCompressor`` 要加载小模型（几百
  MB~2GB），必须**只加载一次**。缓存在 ``self._compressor``（实例由
  ``build_middlewares`` 构造一次、跨 session/step 复用），懒加载。**别**放进
  ``ctx.scratch``（那是每 session 一份，会反复加载）。
- **惰性 import**：``llmlingua`` 是重依赖（torch/transformers），import 写在
  ``_get_compressor`` 内部，未安装时不影响包导入与其它 F / 单测。

红线：压过头伤成功率，必须配 ablation，``task_success_rate`` 下降 ≤ 2pp。
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

from agent_mem.context_telemetry import messages_preview, text_preview
from agent_mem.middleware.base import BaseMiddleware, MiddlewareContext
from agent_mem.middleware.static_prompt import (
    compact_system_messages,
    optimize_tool_descriptions,
)
from agent_mem.token_counting import (
    count_text_chunks,
    estimate_text_tokens,
    get_tokenizer,
    resolve_tokenizer_path,
)

# 支持的压缩方法（同一套配置切，便于 ablation）
_METHODS: set[str] = {"llmlingua", "longllmlingua", "llmlingua2"}

# 随包发布的压缩 worker 脚本（在隔离 venv 里跑，见 _SubprocessCompressor）
_DEFAULT_WORKER = os.path.join(os.path.dirname(__file__), "_compress_worker.py")
# worker 的 stderr 日志（诊断用；不参与协议，避免 PIPE 死锁）
_WORKER_STDERR_LOG = "/tmp/llmlingua_worker.stderr.log"

_CRITICAL_KEY_PARTS = {
    "id", "status", "state", "amount", "price", "total", "balance",
    "quantity", "count", "time", "date", "email", "address", "payment",
    "reason", "confirm", "error", "name", "zip", "refund", "severity",
    "priority", "owner", "assignee", "organization", "tenant", "account",
    "permission", "role",
}


@dataclass(frozen=True)
class _HistorySegment:
    """工具感知历史片段：prefix 必须原样保留，body 才允许压缩。"""

    prefix: str
    body: str = ""
    rate: float = 1.0

    def original_text(self) -> str:
        return "\n".join(part for part in (self.prefix, self.body) if part)


def _is_critical_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    parts = set(normalized.split("_"))
    return bool(parts & _CRITICAL_KEY_PARTS) or normalized.endswith(
        ("_id", "_ids", "_at", "_time", "_date", "_timestamp")
    )


def _extract_critical_json(value: Any, key: str = "") -> Any:
    """抽取工具结果中的硬字段；空容器返回 ``None``，避免复制整份 JSON。"""
    if isinstance(value, dict):
        out = {}
        for child_key, child_value in value.items():
            if _is_critical_key(child_key):
                out[child_key] = child_value
                continue
            child = _extract_critical_json(child_value, child_key)
            if child is not None:
                out[child_key] = child
        return out or None
    if isinstance(value, list):
        out = [item for item in (_extract_critical_json(v, key) for v in value)
               if item is not None]
        return out or None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return value if key and _is_critical_key(key) else None


def _critical_tool_content(content: str) -> str:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return ""
    critical = _extract_critical_json(parsed)
    if critical is None:
        return ""
    return json.dumps(critical, ensure_ascii=False, separators=(",", ":"))


def _strip_critical_json(value: Any, key: str = "") -> Any:
    """返回仅含可压缩 narrative 字段的 JSON；硬字段已由 protected prefix 承载。"""
    if key and _is_critical_key(key):
        return None
    if isinstance(value, dict):
        out = {}
        for child_key, child_value in value.items():
            child = _strip_critical_json(child_value, child_key)
            if child is not None:
                out[child_key] = child
        return out or None
    if isinstance(value, list):
        out = [item for item in (_strip_critical_json(v, key) for v in value)
               if item is not None]
        return out or None
    if isinstance(value, str):
        return value
    # 数值/布尔/null 已由 critical snapshot 无条件保留，不在正文重复。
    return None


def _compressible_tool_content(content: str) -> str:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return content
    narrative = _strip_critical_json(parsed)
    if narrative is None:
        return ""
    return json.dumps(narrative, ensure_ascii=False, separators=(",", ":"))


def _tool_aware_segments(
    messages: list[dict], *, assistant_rate: float, tool_rate: float
) -> list[_HistorySegment]:
    """把完整 agent 轨迹分成受保护 metadata 与可压缩正文。"""
    call_names: dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            name = str(call.get("function", {}).get("name") or "")
            if call_id:
                call_names[call_id] = name

    segments: list[_HistorySegment] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = str(message.get("content") or "")
        if role == "user":
            # 历史用户目标、确认和约束高度敏感，完整保留。
            segments.append(_HistorySegment(f"[USER]\n{content}"))
            continue
        if role == "assistant" and message.get("tool_calls"):
            protected = ["[ASSISTANT_TOOL_CALL]"]
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function", {})
                protected.extend([
                    f"call_id={call.get('id') or ''}",
                    f"name={fn.get('name') or ''}",
                    f"arguments={fn.get('arguments') or '{}'}",
                ])
            segments.append(_HistorySegment("\n".join(protected), content, assistant_rate))
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            name = str(message.get("name") or call_names.get(call_id) or "")
            protected = ["[TOOL_RESULT]", f"call_id={call_id}", f"name={name}"]
            critical = _critical_tool_content(content)
            if critical:
                protected.append(f"critical_fields={critical}")
            segments.append(_HistorySegment(
                "\n".join(protected), _compressible_tool_content(content), tool_rate
            ))
            continue
        segments.append(_HistorySegment(f"[{role.upper()}]", content, assistant_rate))
    return segments


def _msg_to_text(m: dict) -> str:
    """把一条 message 压成给压缩器的文本（丢结构、留语义）。

    - ``tool_calls`` 是结构化的，不进压缩文本；但保留"调用了哪些工具"的梗概，
      免得冷历史里 assistant 轮次完全空白。
    - 空 content 返回空串（调用方会过滤掉）。
    """
    content = m.get("content") or ""
    if m.get("role") == "assistant" and m.get("tool_calls"):
        names = [
            tc.get("function", {}).get("name")
            for tc in m["tool_calls"]
            if isinstance(tc, dict)
        ]
        calls = ", ".join(n for n in names if n)
        if calls:
            content = (content + f" [called: {calls}]").strip()
    return content


class _SubprocessCompressor:
    """常驻子进程压缩器：用隔离 venv 的 python 跑 ``_compress_worker``。

    背景：llmlingua 0.2.2 只兼容 transformers 4.x，而主 venv 的 transformers 被
    vllm 锁在 5.x，无法同进程共存。解法——把真压缩放进一个独立 venv 的常驻子进程，
    模型只加载一次、跨多次压缩复用；主 venv 一点不动。

    duck-type 成 ``PromptCompressor``：暴露同签名的 ``compress_prompt(*args, **kw)``，
    原样转发给 worker 里的真 ``PromptCompressor``，于是 ``_compress_cold`` 无需改。
    """

    def __init__(
        self,
        *,
        venv_python: str,
        worker_script: str,
        model_name: str | None,
        use_llmlingua2: bool,
        device: str,
        num_threads: int = 0,
    ) -> None:
        self.venv_python = venv_python
        self.worker_script = worker_script
        self.model_name = model_name
        self.use_llmlingua2 = use_llmlingua2
        self.device = device
        self.num_threads = num_threads  # >0 时限制 worker 线程数（避免多 worker 时 torch 超订）
        self._proc: subprocess.Popen | None = None
        self._stderr_fh = None
        self._lock = threading.Lock()  # 并发跑多任务时串行化对单 worker 的访问

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        # 清掉 PYTHONPATH，避免子进程误用主 venv 的 transformers
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if self.num_threads and self.num_threads > 0:
            # 限制 worker 线程数：多 worker 并行时，避免 torch 默认各吃满全部核导致超订
            env["OMP_NUM_THREADS"] = str(self.num_threads)
            env["MKL_NUM_THREADS"] = str(self.num_threads)
        self._stderr_fh = open(_WORKER_STDERR_LOG, "a")
        self._proc = subprocess.Popen(
            [self.venv_python, self.worker_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            text=True,
            env=env,
        )
        config = json.dumps(
            {
                "model_name": self.model_name,
                "use_llmlingua2": self.use_llmlingua2,
                "device": self.device,
            }
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write(config + "\n")
        self._proc.stdin.flush()
        assert self._proc.stdout is not None
        ready_line = self._proc.stdout.readline()
        if not ready_line:
            raise RuntimeError(
                f"compress worker 启动无响应，见 {_WORKER_STDERR_LOG}"
            )
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"compress worker 启动失败: {ready}")

    def compress_prompt(self, *args: Any, **kw: Any) -> dict:
        """透明转发到 worker 的 PromptCompressor.compress_prompt（线程安全）。"""
        with self._lock:
            self._ensure_started()
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(json.dumps({"args": list(args), "kw": kw}) + "\n")
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline()
            if not resp_line:
                raise RuntimeError(
                    f"compress worker 无响应（可能崩溃），见 {_WORKER_STDERR_LOG}"
                )
            resp = json.loads(resp_line)
            if "error" in resp:
                raise RuntimeError(f"compress worker 报错: {resp['error']}")
            return resp["result"]

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write("EXIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._proc.kill()
        self._proc = None


class _SubprocessCompressorPool:
    """压缩 worker 池：N 个 ``_SubprocessCompressor``，并发任务各取一个并行压缩。

    解决"单 worker + 锁在并发下串行"的瓶颈（全量115 并发4 时延迟 +16% 的根因）。
    每次 ``compress_prompt`` 从空闲队列取一个 worker、用完归还——同一 worker 同一时刻
    只被一个线程用，故无需再加锁；最多 N 路压缩并行。

    duck-type 成压缩器：暴露 ``compress_prompt(*args, **kw)``，``_compress_cold`` 无需改。
    每个 worker 各加载一份小模型（gpt2 ~500MB×N，CPU 可承受；首次压缩时并行加载）。
    """

    def __init__(self, *, size: int, **worker_kw: Any) -> None:
        self.size = max(1, int(size))
        self._workers = [_SubprocessCompressor(**worker_kw) for _ in range(self.size)]
        self._free: queue.Queue = queue.Queue()
        for w in self._workers:
            self._free.put(w)

    def compress_prompt(self, *args: Any, **kw: Any) -> dict:
        w = self._free.get()  # 无空闲则阻塞等（concurrency > pool_size 时优雅降级）
        try:
            return w.compress_prompt(*args, **kw)
        finally:
            self._free.put(w)

    def close(self) -> None:
        for w in self._workers:
            w.close()


class CompressMiddleware(BaseMiddleware):
    """F2 Prompt 压缩中间件。

    配置（yaml ``middleware.options.compress``）::

        compress:
          method: longllmlingua        # llmlingua | longllmlingua | llmlingua2
          rate: 0.4                    # 保留比例（0.4=保留 40%）
          trigger_tokens: 4000         # 冷历史 < 此值不压（压/不压门）
          keep_hot: 6                  # 热尾保留条数（snap 到完整 tool 组）
          device: cpu                  # 压缩器小模型放 CPU，NPU 让给主 LLM
          model_name: null             # null=PromptCompressor 默认模型
          # —— LongLLMLingua 专属（method=longllmlingua 才用）——
          condition_in_question: after_condition
          dynamic_context_compression_ratio: 0.3
          condition_compare: true
          reorder_context: none        # agent 轨迹默认保时序；ablation 可试 sort
    """

    name = "compress"

    def __init__(
        self,
        method: str = "longllmlingua",
        rate: float = 0.5,
        trigger_tokens: int = 4000,
        keep_hot: int = 6,
        device: str = "cpu",
        model_name: str | None = None,
        backend: str = "subprocess",
        worker_venv: str = "",
        worker_script: str = "",
        worker_pool_size: int = 1,
        worker_threads: int = 0,
        condition_in_question: str = "after",
        dynamic_context_compression_ratio: float = 0.3,
        condition_compare: bool = False,
        reorder_context: str = "original",
        force_tokens: list[str] | None = None,
        history_role: str = "system",
        tool_aware: bool = False,
        assistant_rate: float = 0.75,
        tool_result_rate: float = 0.6,
        hot_tool_trigger_tokens: int = 0,
        optimize_static_prompt: bool = False,
        system_prompt_mode: str = "none",
        policy_artifact_path: str = "",
        policy_artifact_strict: bool = False,
        deduplicate_tool_descriptions: bool = True,
        tokenizer_model: str = "",
        recompress_delta_tokens: int | None = None,
        event_log: str | None = None,
        telemetry_preview_chars: int = 4000,
        telemetry_max_messages: int = 30,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method 必须是 {sorted(_METHODS)}，得到 {method!r}")
        if not (0.0 < rate <= 1.0):
            raise ValueError(f"rate 必须在 (0, 1]，得到 {rate}")
        if keep_hot < 1:
            raise ValueError("keep_hot 必须 >= 1（至少保留 1 条热尾）")
        if trigger_tokens < 0:
            raise ValueError("trigger_tokens 必须 >= 0")

        self.method = method
        self.rate = rate
        self.trigger_tokens = trigger_tokens
        self.keep_hot = keep_hot
        self.device = device
        self.model_name = model_name
        self.backend = backend
        self.worker_venv = worker_venv
        self.worker_script = worker_script or _DEFAULT_WORKER
        self.worker_pool_size = max(1, int(worker_pool_size))  # 并发压缩池大小（>= concurrency 才全并行）
        # 每 worker 线程上限：>0 用指定值；<=0 自动 = 核数 // pool_size
        # （顺序 pool=1 → 全部核；并发 pool=N → 各占 1/N，避免 torch 超订）
        wt = int(worker_threads)
        if wt > 0:
            self.worker_threads = wt
        else:
            self.worker_threads = max(1, (os.cpu_count() or 1) // self.worker_pool_size)
        if backend not in {"subprocess", "inprocess"}:
            raise ValueError(f"backend 必须是 subprocess 或 inprocess，得到 {backend!r}")
        self.condition_in_question = condition_in_question
        self.dynamic_context_compression_ratio = dynamic_context_compression_ratio
        self.condition_compare = condition_compare
        self.reorder_context = reorder_context
        self.force_tokens = force_tokens if force_tokens is not None else ["\n", "?", "."]
        self.history_role = history_role
        self.tool_aware = bool(tool_aware)
        self.assistant_rate = float(assistant_rate)
        self.tool_result_rate = float(tool_result_rate)
        self.hot_tool_trigger_tokens = int(hot_tool_trigger_tokens)
        self.optimize_static_prompt = bool(optimize_static_prompt)
        self.system_prompt_mode = system_prompt_mode
        self.policy_artifact_path = policy_artifact_path
        self.policy_artifact_strict = bool(policy_artifact_strict)
        self.deduplicate_tool_descriptions = bool(deduplicate_tool_descriptions)
        self.tokenizer_model = tokenizer_model
        self._tokenizer: Any = None
        self._tokenizer_lock = threading.Lock()
        self._token_count_cache: dict[bytes, int] = {}
        self._token_count_cache_lock = threading.Lock()
        if self.tool_aware and self.method != "llmlingua2":
            raise ValueError("tool_aware 当前要求 method=llmlingua2（结构字段由外层保护）")
        if not (0.0 < self.assistant_rate <= 1.0):
            raise ValueError("assistant_rate 必须在 (0, 1]")
        if not (0.0 < self.tool_result_rate <= 1.0):
            raise ValueError("tool_result_rate 必须在 (0, 1]")
        if self.hot_tool_trigger_tokens < 0:
            raise ValueError("hot_tool_trigger_tokens 必须 >= 0")
        if self.system_prompt_mode not in {"none", "retail_compact", "compiled"}:
            raise ValueError("system_prompt_mode 必须是 none、retail_compact 或 compiled")
        if self.system_prompt_mode == "compiled" and not self.policy_artifact_path:
            raise ValueError("system_prompt_mode=compiled 需要 policy_artifact_path")
        self.recompress_delta_tokens = (
            recompress_delta_tokens if recompress_delta_tokens is not None else trigger_tokens
        )
        if self.recompress_delta_tokens < 0:
            raise ValueError("recompress_delta_tokens 必须 >= 0")
        self._event_log_path = event_log or os.environ.get("F2_EVENT_LOG") or ""
        self.telemetry_preview_chars = max(200, int(telemetry_preview_chars))
        self.telemetry_max_messages = max(1, int(telemetry_max_messages))
        self._log_lock = threading.Lock()  # 并发跑多任务时串行化事件日志写文件
        self._compressor_lock = threading.Lock()  # 首次并发触发时只构造一个 worker pool
        self._compressor: Any = None  # 懒加载，进程级单例

    def prepare(self) -> None:
        """Load the configured engine tokenizer before benchmark timing starts."""
        if self.tokenizer_model:
            self._get_tokenizer()

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            with self._tokenizer_lock:
                if self._tokenizer is None:
                    self._tokenizer = get_tokenizer(self.tokenizer_model)
        return self._tokenizer

    @property
    def token_count_source(self) -> str:
        if self.tokenizer_model:
            return f"tokenizer:{resolve_tokenizer_path(self.tokenizer_model)}"
        return "unicode_heuristic"

    # ---- 压缩器加载（惰性、缓存）----

    def _get_compressor(self) -> Any:
        """懒加载压缩器并缓存（只加载/拉起一次）。

        - ``backend="subprocess"``（默认）：用隔离 venv 的常驻 worker，绕开主 venv 的
          transformers 5.x 与 llmlingua 4.x 的冲突。需配 ``worker_venv``。
        - ``backend="inprocess"``：直接在进程内 import llmlingua（仅当本环境 transformers
          为 4.x 时可用，例如跑在隔离 venv 内自身）。
        """
        if self._compressor is None:
            with self._compressor_lock:
                if self._compressor is None:
                    self._compressor = self._build_compressor()
        return self._compressor

    def _build_compressor(self) -> Any:
        """构造压缩器；只允许由 ``_get_compressor`` 的初始化锁调用。"""
        if self.backend == "subprocess":
            if not self.worker_venv:
                raise ValueError(
                    "backend=subprocess 需配置 worker_venv（隔离压缩 venv 的 python 路径，"
                    "如 .venv-compress/bin/python）"
                )
            kw = dict(
                venv_python=self.worker_venv,
                worker_script=self.worker_script,
                model_name=self.model_name,
                use_llmlingua2=(self.method == "llmlingua2"),
                device=self.device,
                num_threads=self.worker_threads,
            )
            if self.worker_pool_size > 1:
                return _SubprocessCompressorPool(size=self.worker_pool_size, **kw)
            return _SubprocessCompressor(**kw)
        try:
            from llmlingua import PromptCompressor
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "inprocess 后端需要 llmlingua（且 transformers 4.x 环境）"
            ) from e
        return PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=(self.method == "llmlingua2"),
            device_map=self.device,
        )

    # ---- 主钩子 ----

    @staticmethod
    def _state(ctx: MiddlewareContext) -> dict[str, Any]:
        st = ctx.scratch.setdefault("compress", {})
        st.setdefault("frozen_count", 0)
        st.setdefault("compressed", "")
        st.setdefault("count", 0)
        st.setdefault("body_cache", {})
        return st

    def transform_request(
        self, messages: list[dict], tools: list[dict], ctx: MiddlewareContext
    ) -> tuple[list[dict], list[dict]]:
        if not self.optimize_static_prompt:
            return self.transform_messages(messages, ctx), list(tools)

        system_text = "\n".join(
            str(message.get("content") or "")
            for message in messages if message.get("role") == "system"
        ).lower()
        system_covers_confirmation = (
            "explicit user confirmation" in system_text
            or self.system_prompt_mode == "retail_compact"
        )
        if self.deduplicate_tool_descriptions:
            out_tools, conventions, stats = optimize_tool_descriptions(
                tools, system_covers_confirmation=system_covers_confirmation
            )
        else:
            out_tools, conventions, stats = list(tools), [], {
                "tool_descriptions_replaced": 0,
                "tool_conventions": 0,
                "confirmation_sentences_removed": 0,
            }
        out_messages, compacted, policy_meta = compact_system_messages(
            messages,
            mode=self.system_prompt_mode,
            conventions=conventions,
            policy_artifact_path=self.policy_artifact_path,
            policy_artifact_strict=self.policy_artifact_strict,
        )
        ctx.scratch["compress:static_metrics"] = {
            **stats,
            **policy_meta,
            "system_prompt_compacted": compacted,
        }
        return self.transform_messages(out_messages, ctx), out_tools

    @staticmethod
    def _static_metrics(ctx: MiddlewareContext) -> dict[str, Any]:
        return dict(ctx.scratch.get("compress:static_metrics", {}))

    def _history_ready_event(
        self,
        ctx: MiddlewareContext,
        *,
        cold: list[dict],
        hot: list[dict],
        cold_tokens: int,
        compressible_cold_tokens: int,
        hot_tokens: int,
        action: str,
        reason: str,
        new_tokens: int = 0,
    ) -> None:
        if not ctx.telemetry_enabled:
            return
        cold_preview = messages_preview(
            cold,
            max_messages=self.telemetry_max_messages,
            content_limit=min(800, self.telemetry_preview_chars),
        )
        cold_preview.update({
            "tokens": cold_tokens,
            "compressible_tokens": compressible_cold_tokens,
        })
        hot_preview = messages_preview(
            hot,
            max_messages=self.telemetry_max_messages,
            content_limit=min(400, self.telemetry_preview_chars),
        )
        hot_preview["tokens"] = hot_tokens
        ctx.emit("f2.history_ready", {
            "phase": "ready",
            "action": action,
            "reason": reason,
            "cold_before": cold_preview,
            "hot_history": hot_preview,
            "trigger_tokens": self.trigger_tokens,
            "recompress_delta_tokens": self.recompress_delta_tokens,
            "assistant_rate": self.assistant_rate,
            "tool_result_rate": self.tool_result_rate,
            "new_tokens": new_tokens,
            **self._static_metrics(ctx),
        })

    def _cold_after_event(
        self,
        *,
        messages: list[dict],
        tokens: int,
        compressed: str,
    ) -> dict[str, Any]:
        preview = messages_preview(
            messages,
            max_messages=self.telemetry_max_messages,
            content_limit=min(800, self.telemetry_preview_chars),
        )
        preview.update({
            "tokens": tokens,
            "compressed_text": text_preview(
                compressed,
                limit=self.telemetry_preview_chars,
            ),
        })
        return preview

    def transform_messages(
        self, messages: list[dict], ctx: MiddlewareContext
    ) -> list[dict]:
        if not messages:
            return list(messages)

        sys_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        # 太短：热尾都凑不齐，全留
        if len(rest) <= self.keep_hot:
            send_rest, hot_extra = self._compress_hot_tool_results(rest, ctx)
            out = list(sys_msgs) + send_rest
            hot_tokens = (
                self._count_tokens([_msg_to_text(m) for m in rest])
                if ctx.telemetry_enabled else 0
            )
            self._history_ready_event(
                ctx,
                cold=[],
                hot=rest,
                cold_tokens=0,
                compressible_cold_tokens=0,
                hot_tokens=hot_tokens,
                action="skip",
                reason="history_shorter_than_keep_hot",
            )
            ctx.emit("f2.skipped", {
                "phase": "skipped",
                "action": "skip",
                "reason": "history_shorter_than_keep_hot",
                **hot_extra,
            })
            self._stage_event(ctx, {"action": "skip", "n_msgs": len(messages),
                                    "cold_n": 0, "hot_n": len(rest), "cold_tokens": 0,
                                    "token_count_source": self.token_count_source,
                                    **self._static_metrics(ctx),
                                    **hot_extra,
                                    "estimated_sent_tokens": self._estimate_message_tokens(out),
                                    "reason": "history_shorter_than_keep_hot"})
            return out

        # 切冷/热；热尾边界 snap 到完整 tool_call→tool 组（避免孤立 tool）
        split = len(rest) - self.keep_hot
        while split > 0 and rest[split].get("role") == "tool":
            split -= 1  # 把 tool 的 caller(assistant) 一起拉进热尾
        cold, hot = rest[:split], rest[split:]
        send_hot, hot_extra = self._compress_hot_tool_results(hot, ctx)

        chunks = self._history_texts(cold)
        cold_tokens = self._count_tokens(chunks)
        compressible_cold_tokens = self._count_tokens(
            self._compressible_history_texts(cold)
        )
        hot_tokens = self._count_tokens([_msg_to_text(m) for m in hot])

        # 每 session 的压缩缓存（frozen_count=已压进压缩段的冷条数；compressed=压缩文本；count=压缩次数）
        st = self._state(ctx)
        new_tokens = self._count_tokens(
            self._compressible_history_texts(cold[st["frozen_count"]:])
        )

        common = {
            "n_msgs": len(messages), "cold_n": len(cold), "hot_n": len(hot),
            "cold_tokens": cold_tokens,
            "compressible_cold_tokens": compressible_cold_tokens,
            "hot_tokens": hot_tokens,
            "new_tokens": new_tokens, "compress_count": st["count"],
            "frozen_count": st["frozen_count"],
            "token_count_source": self.token_count_source,
            **self._static_metrics(ctx),
            **hot_extra,
        }

        # 触发门：冷历史没过阈值 → 完全不压（短上下文无 lost-in-the-middle）
        gate_tokens = compressible_cold_tokens if self.tool_aware else cold_tokens
        if gate_tokens < self.trigger_tokens:
            self._history_ready_event(
                ctx,
                cold=cold,
                hot=hot,
                cold_tokens=cold_tokens,
                compressible_cold_tokens=compressible_cold_tokens,
                hot_tokens=hot_tokens,
                action="skip",
                reason="below_trigger",
                new_tokens=new_tokens,
            )
            st["frozen_count"] = 0
            st["compressed"] = ""
            out = list(sys_msgs) + list(cold) + send_hot
            ctx.emit("f2.skipped", {
                "phase": "skipped",
                "action": "skip",
                "reason": "below_trigger",
                "gate_tokens": gate_tokens,
                "trigger_tokens": self.trigger_tokens,
                **hot_extra,
            })
            self._stage_event(ctx, {**common, "action": "skip", "reason": "below_trigger",
                                    "estimated_sent_tokens": self._estimate_message_tokens(out)})
            return out

        # 决定 compress vs reuse：首次压缩，或自上次压缩后新增冷 >= delta
        do_compress = (not st["compressed"]) or (new_tokens >= self.recompress_delta_tokens)
        decision_reason = (
            "first_compression"
            if not st["compressed"]
            else "recompress_delta_reached"
            if do_compress
            else "cached_compression_reused"
        )
        self._history_ready_event(
            ctx,
            cold=cold,
            hot=hot,
            cold_tokens=cold_tokens,
            compressible_cold_tokens=compressible_cold_tokens,
            hot_tokens=hot_tokens,
            action="compress" if do_compress else "reuse",
            reason=decision_reason,
            new_tokens=new_tokens,
        )
        extra: dict = {}
        if do_compress:
            question = self._pick_question(rest)
            ctx.emit("f2.compress_started", {
                "phase": "compressing",
                "action": "compress",
                "reason": decision_reason,
                "cold_tokens": cold_tokens,
                "compressible_cold_tokens": compressible_cold_tokens,
            })
            t0 = time.monotonic()
            try:
                res = self._compress_cold(chunks, question, cold=cold, ctx=ctx)
            except Exception as exc:
                ctx.emit("f2.compress_failed", {
                    "phase": "failed",
                    "action": "compress",
                    "error": repr(exc),
                    "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 1),
                })
                raise
            ms = (time.monotonic() - t0) * 1000.0
            compressed = res.get("compressed_prompt", "")
            st["compressed"] = compressed
            st["frozen_count"] = len(cold)  # 当下整段冷都压进去了
            st["count"] += 1
            est_comp = self._count_tokens([compressed]) if compressed else 0
            extra = {
                "action": "compress",
                "compress_count": st["count"],
                "frozen_count": st["frozen_count"],
                "origin_tokens": res.get("origin_tokens", cold_tokens),
                "compressed_tokens": res.get("compressed_tokens", est_comp),
                "est_compressed_tokens": est_comp,
                "ratio": res.get("ratio"),
                "compress_ms": round(ms, 1),
            }
        else:
            compressed = st["compressed"]
            extra = {"action": "reuse"}

        # 重建：[sys] + [压缩段(覆盖 cold[:frozen_count])] + [新增冷 cold[frozen_count:] verbatim] + [hot]
        # tool_call 配对安全：frozen_count 总落在完整 tool_call→tool 组边界（见 _pick 切分）
        out: list[dict] = list(sys_msgs)
        sent_cold: list[dict] = []
        if compressed:
            sent_cold.append({
                "role": self.history_role,
                "content": f"[compressed history]\n{compressed}",
            })
        sent_cold.extend(cold[st["frozen_count"]:])
        out.extend(sent_cold)  # 自上次压缩后新增的冷（verbatim，不丢信息）
        out.extend(send_hot)

        if ctx.telemetry_enabled:
            cold_after_tokens = self._estimate_message_tokens(sent_cold)
            cold_after = self._cold_after_event(
                messages=sent_cold,
                tokens=cold_after_tokens,
                compressed=compressed,
            )
            if do_compress:
                ctx.emit("f2.compress_finished", {
                    "phase": "compressed",
                    **extra,
                    "saved_tokens": max(
                        0,
                        int(extra.get("origin_tokens") or cold_tokens)
                        - int(extra.get("compressed_tokens") or cold_after_tokens),
                    ),
                    "cold_after": cold_after,
                })
            else:
                ctx.emit("f2.reused", {
                    "phase": "reused",
                    "action": "reuse",
                    "reason": decision_reason,
                    "cold_after": cold_after,
                })

        self._stage_event(ctx, {
            **common, **extra,
            "estimated_sent_tokens": self._estimate_message_tokens(out),
        })
        return out

    def _compress_hot_tool_results(
        self, hot: list[dict], ctx: MiddlewareContext
    ) -> tuple[list[dict], dict[str, Any]]:
        """保持 hot tool 协议结构，仅压超过阈值的 result content。"""
        empty = {
            "hot_tool_compressed": 0,
            "hot_tool_saved_tokens": 0,
            "hot_compress_ms": 0.0,
        }
        if not self.tool_aware or self.hot_tool_trigger_tokens <= 0:
            return list(hot), empty

        candidates: list[tuple[int, str, str, str]] = []
        for idx, message in enumerate(hot):
            if message.get("role") != "tool":
                continue
            content = str(message.get("content") or "")
            if self._count_tokens([content]) >= self.hot_tool_trigger_tokens:
                candidates.append((
                    idx,
                    content,
                    _compressible_tool_content(content),
                    _critical_tool_content(content),
                ))
        if not candidates:
            return list(hot), empty

        t0 = time.monotonic()
        body_positions = [idx for idx, (_, _, body, _) in enumerate(candidates) if body]
        compressed_values = self._compress_bodies(
            [candidates[idx][2] for idx in body_positions],
            [self.tool_result_rate] * len(body_positions),
            ctx,
        )
        compressed_by_candidate = dict(zip(body_positions, compressed_values))
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        out = list(hot)
        count = 0
        saved = 0
        for candidate_idx, (idx, original, body, critical) in enumerate(candidates):
            body = compressed_by_candidate.get(candidate_idx, body)
            pieces = ["[compressed tool result]"]
            if critical:
                pieces.append(f"critical_fields={critical}")
            pieces.append(body)
            replacement = "\n".join(pieces)
            original_tokens = self._count_tokens([original])
            replacement_tokens = self._count_tokens([replacement])
            if replacement_tokens >= original_tokens:
                continue
            out[idx] = {**hot[idx], "content": replacement}
            count += 1
            saved += original_tokens - replacement_tokens
        return out, {
            "hot_tool_compressed": count,
            "hot_tool_saved_tokens": saved,
            "hot_compress_ms": round(elapsed_ms, 1),
        }

    def _history_texts(self, messages: list[dict]) -> list[str]:
        if not self.tool_aware:
            return [text for text in (_msg_to_text(m) for m in messages) if text]
        return [
            segment.original_text()
            for segment in _tool_aware_segments(
                messages,
                assistant_rate=self.assistant_rate,
                tool_rate=self.tool_result_rate,
            )
            if segment.original_text()
        ]

    def _compressible_history_texts(self, messages: list[dict]) -> list[str]:
        if not self.tool_aware:
            return self._history_texts(messages)
        return [
            segment.body
            for segment in _tool_aware_segments(
                messages,
                assistant_rate=self.assistant_rate,
                tool_rate=self.tool_result_rate,
            )
            if segment.body
        ]

    def _compress_bodies(
        self, bodies: list[str], rates: list[float], ctx: MiddlewareContext
    ) -> list[str]:
        """按 rate 批量压缩正文，并以内容 hash 跨 cold 重压复用。"""
        if not bodies:
            return []
        st = self._state(ctx)
        cache: dict[str, str] = st.setdefault("body_cache", {})
        output = [""] * len(bodies)
        pending_by_rate: dict[float, list[tuple[int, str, str]]] = {}
        for idx, (body, rate) in enumerate(zip(bodies, rates)):
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            key = f"{rate:.4f}:{digest}"
            if key in cache:
                output[idx] = cache[key]
            else:
                pending_by_rate.setdefault(rate, []).append((idx, key, body))

        compressor = self._get_compressor()
        for rate, pending in pending_by_rate.items():
            originals = [item[2] for item in pending]
            result = compressor.compress_prompt(
                originals,
                rate=rate,
                force_tokens=self.force_tokens,
                force_reserve_digit=True,
                use_context_level_filter=False,
            )
            compressed = result.get("compressed_prompt_list")
            if not isinstance(compressed, list) or len(compressed) != len(originals):
                compressed = originals
            for (idx, key, original), value in zip(pending, compressed):
                value = value if isinstance(value, str) and value.strip() else original
                # 压完更长时保留原文；结构保护不能以负收益为代价。
                if self._count_tokens([value]) >= self._count_tokens([original]):
                    value = original
                cache[key] = value
                output[idx] = value
        return output

    def _compress_tool_aware(
        self, cold: list[dict], ctx: MiddlewareContext
    ) -> dict:
        segments = _tool_aware_segments(
            cold,
            assistant_rate=self.assistant_rate,
            tool_rate=self.tool_result_rate,
        )
        body_segments = [(idx, segment) for idx, segment in enumerate(segments)
                         if segment.body]
        bodies = [segment.body for _, segment in body_segments]
        rates = [segment.rate for _, segment in body_segments]
        compressed_bodies = self._compress_bodies(bodies, rates, ctx)
        replacements = {
            idx: body for (idx, _), body in zip(body_segments, compressed_bodies)
        }
        pieces = []
        for idx, segment in enumerate(segments):
            pieces.append("\n".join(
                part for part in (segment.prefix, replacements.get(idx, "")) if part
            ))
        original = "\n\n".join(segment.original_text() for segment in segments)
        compressed = "\n\n".join(pieces)
        origin_tokens = self._count_tokens([original])
        compressed_tokens = self._count_tokens([compressed])
        ratio = origin_tokens / max(1, compressed_tokens)
        return {
            "compressed_prompt": compressed,
            "origin_tokens": origin_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": f"{ratio:.1f}x",
        }

    def _estimate_message_tokens(self, msgs: list[dict]) -> int:
        """Count serialized message text; tools/chat-template overhead remains excluded."""
        return self._count_tokens([_msg_to_text(m) for m in msgs])

    def _stage_event(self, ctx: MiddlewareContext, payload: dict) -> None:
        """暂存本步事件，等模型响应带回真实 prompt token 后再落盘。"""
        ctx.scratch[f"{self.name}:pending_event"] = payload

    def after_model_call(
        self, prompt_tokens: int | None, ctx: MiddlewareContext
    ) -> None:
        payload = ctx.scratch.pop(f"{self.name}:pending_event", None)
        if payload is None:
            return
        payload["sent_tokens"] = prompt_tokens
        payload["sent_tokens_source"] = (
            "response.usage.prompt_tokens" if prompt_tokens is not None else "unavailable"
        )
        ctx.emit("f2.request_completed", {
            "sent_prompt_tokens": prompt_tokens,
            "sent_tokens_source": payload["sent_tokens_source"],
        })
        self._log_event(ctx, payload)

    @staticmethod
    def _pick_question(rest: list[dict]) -> str:
        """LongLingua 的相关性锚点 = 最近一条带内容的 user 消息（无则退化）。"""
        q = next(
            (m.get("content") or "" for m in reversed(rest)
             if m.get("role") == "user" and m.get("content")),
            "",
        )
        if not q:
            # LongLLMLingua 必须非空 question（llmlingua 内部 assert）
            q = next(
                (m.get("content") or "" for m in reversed(rest) if m.get("content")),
                "",
            ) or "Continue the task."
        return q

    def _log_event(self, ctx: MiddlewareContext, payload: dict) -> None:
        """每步写一条 JSONL 到 F2_EVENT_LOG（未设则不记）。"""
        if not self._event_log_path:
            return
        payload.setdefault("session_id", ctx.session_id)
        payload.setdefault("step", ctx.step)
        payload["ts"] = time.time()
        try:
            with self._log_lock:
                with open(self._event_log_path, "a") as f:
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:  # noqa: BLE001 — 日志失败不影响主流程
            pass

    # ---- 压缩分发 ----

    def _compress_cold(
        self,
        chunks: list[str],
        question: str,
        *,
        cold: list[dict] | None = None,
        ctx: MiddlewareContext | None = None,
    ) -> dict:
        """调压缩器，返回 llmlingua 的完整结果 dict（含 compressed_prompt/origin_tokens/...）。"""
        if self.tool_aware:
            if cold is None or ctx is None:
                raise ValueError("tool_aware 压缩需要 cold messages 与 MiddlewareContext")
            return self._compress_tool_aware(cold, ctx)
        c = self._get_compressor()
        if self.method == "longllmlingua":
            # LongLLMLingua：按块传入，question-aware 打分 + 分段动态率
            return c.compress_prompt(
                chunks,
                question=question,
                rate=self.rate,
                rank_method="longllmlingua",
                condition_in_question=self.condition_in_question,
                dynamic_context_compression_ratio=self.dynamic_context_compression_ratio,
                condition_compare=self.condition_compare,
                reorder_context=self.reorder_context,
            )
        # llmlingua / llmlingua2：拼成一段文本（compress_prompt 首参 context），按 rate 压
        context = "\n\n".join(chunks)
        kw: dict[str, Any] = dict(rate=self.rate, force_tokens=self.force_tokens)
        if question:
            kw["question"] = question
        return c.compress_prompt(context, **kw)

    def _count_tokens(self, chunks: list[str]) -> int:
        """Count trigger tokens with the engine tokenizer when one is configured."""
        text = "\n\n".join(chunk for chunk in chunks if chunk)
        if not text:
            return 0
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        with self._token_count_cache_lock:
            cached = self._token_count_cache.get(digest)
        if cached is not None:
            return cached
        count = (
            count_text_chunks(self._get_tokenizer(), [text])
            if self.tokenizer_model
            else estimate_text_tokens([text])
        )
        with self._token_count_cache_lock:
            if len(self._token_count_cache) >= 4096:
                self._token_count_cache.pop(next(iter(self._token_count_cache)))
            self._token_count_cache[digest] = count
        return count
