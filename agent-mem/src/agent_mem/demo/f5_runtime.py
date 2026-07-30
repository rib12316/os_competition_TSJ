"""F5 judge-demo runtime: tier definitions and per-request admission control."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class F5Tier:
    """One fixed layer in the final baseline -> ours comparison."""

    key: str
    label: str
    features: tuple[str, ...]
    priority_strategy: str
    adaptive_admission: bool
    c8_enabled: bool
    offload_label: str


def build_f5_tiers(kv_mode: str | None = None) -> list[F5Tier]:
    """Return the native baseline stack and the validated F5 stack (KV-pool 准入控制).

    对齐 ``configs/f5-evict-dynamic.yaml``（真机已验证的赢家）：ours = ``--scheduling-policy priority``
    + **KV-pool 准入控制（背压）**——读 ``vllm:kv_cache_usage_perc``，>85% 暂不放新 session、降至 70% 再放，
    从源头防止 KV 池溢出 → 抢占归零、KV 命中率回升、延迟下降。**无** KV offload（offload 层在当前版本未承重）。

    注：早先的 progress（SRTF 优先级）方案经公平 conc-6 复测**无效**（优先级只改「谁被抢」不改「抢几次」），
    真正起作用的是准入控制（防溢出而非事后抢救）。

    ``kv_mode`` 保留为可选入参仅为向后兼容，已不再校验/使用。
    """
    return [
        F5Tier(
            key="baseline",
            label="baseline",
            features=("prefix-cache",),
            priority_strategy="fcfs",
            adaptive_admission=False,
            c8_enabled=False,
            offload_label="关闭",
        ),
        F5Tier(
            key="ours",
            label="ours",
            features=("prefix-cache", "priority"),           # 引擎开 priority 调度（准入控制的协同层）
            priority_strategy="idle",                        # priority-evict（idle 抬优先级）——主机制是准入控制
            adaptive_admission=True,                          # ★ KV-pool 准入控制（70-85% 背压），防溢出消除抢占
            c8_enabled=False,
            offload_label="关闭",
        ),
    ]


class UserSimResponseCache:
    """Replay one recorded user script per task turn, including observed latency."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._lock = threading.Lock()
        self._responses: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0
        self._clock = clock
        self._sleeper = sleeper

    @staticmethod
    def _key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        messages = kwargs.get("messages") or []
        system_content = next(
            (
                message.get("content", "")
                for message in messages
                if isinstance(message, dict) and message.get("role") == "system"
            ),
            "",
        )
        user_turn = sum(
            1
            for message in messages
            if isinstance(message, dict) and message.get("role") == "assistant"
        )
        payload = {
            "args": args,
            "model": kwargs.get("model"),
            "provider": kwargs.get("custom_llm_provider"),
            "task_instruction": system_content,
            "user_turn": user_turn,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def complete(self, delegate: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("temperature", 0.0)
        key = self._key(args, kwargs)
        with self._lock:
            cached = self._responses.get(key)
            if cached is not None:
                self._hits += 1
                response, latency_s = cached
            else:
                response, latency_s = None, 0.0
        if response is not None:
            self._sleeper(latency_s)
            return copy.deepcopy(response)

        started = self._clock()
        response = delegate(*args, **kwargs)
        latency_s = max(0.0, self._clock() - started)
        with self._lock:
            existing = self._responses.setdefault(
                key, (copy.deepcopy(response), latency_s)
            )
            self._misses += 1
        return copy.deepcopy(existing[0])

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._responses),
                "hits": self._hits,
                "misses": self._misses,
            }


class RequestAdmissionController:
    """Gate every model request and expose scheduler decisions for the F5 UI.

    Fixed mode is used by baseline/native so all three layers have the same client-side
    request cap. Adaptive mode changes that cap according to vLLM KV-pool utilization.
    """

    def __init__(
        self,
        *,
        max_requests: int,
        adaptive: bool,
        min_requests: int = 1,
        initial_requests: int | None = None,
        kv_pct_fn: Callable[[], float] | None = None,
        target_lo: float = 70.0,
        target_hi: float = 85.0,
        poll_interval_s: float = 0.5,
        adjustment_interval_s: float = 1.0,
        cancel_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_requests < 1:
            raise ValueError("max_requests 必须 >= 1")
        if not 1 <= min_requests <= max_requests:
            raise ValueError("min_requests 必须满足 1 <= min_requests <= max_requests")
        if initial_requests is None:
            initial_requests = max_requests
        if not min_requests <= initial_requests <= max_requests:
            raise ValueError(
                "initial_requests 必须满足 min_requests <= initial_requests <= max_requests"
            )
        if not 0 <= target_lo <= target_hi <= 100:
            raise ValueError("target_lo/target_hi 必须满足 0 <= lo <= hi <= 100")
        self.max_requests = int(max_requests)
        self.min_requests = int(min_requests)
        self.adaptive = bool(adaptive)
        self.target_lo = float(target_lo)
        self.target_hi = float(target_hi)
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.adjustment_interval_s = max(0.0, float(adjustment_interval_s))
        self._kv_pct_fn = kv_pct_fn
        self._cancel = cancel_event or threading.Event()
        self._clock = clock

        self._cond = threading.Condition(threading.RLock())
        self._kv_read_lock = threading.Lock()
        self._active: set[str] = set()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=512)
        self._limit = int(initial_requests)
        self._last_adjust_at = float("-inf")
        self._latest_kv_pct: float | None = None
        self._kv_peak_pct: float | None = None
        self._kv_read_at = float("-inf")
        self._kv_read_cache: float | None = None
        self._admitted = 0
        self._deferred = 0
        self._max_active = 0

    def _read_kv_pct(self) -> float | None:
        if self._kv_pct_fn is None:
            return None
        with self._kv_read_lock:
            now = self._clock()
            if now - self._kv_read_at < self.poll_interval_s:
                return self._kv_read_cache
            try:
                value = float(self._kv_pct_fn())
            except Exception:
                value = -1.0
            self._kv_read_at = now
            self._kv_read_cache = None if value < 0 else min(100.0, max(0.0, value))
            return self._kv_read_cache

    def _record(self, event: str, session_id: str, **data: Any) -> None:
        self._events.append({
            "t": self._clock(),
            "event": event,
            "session_id": session_id,
            **data,
        })

    def _adjust_limit(self, kv_pct: float | None) -> None:
        if not self.adaptive or kv_pct is None:
            return
        now = self._clock()
        if now - self._last_adjust_at < self.adjustment_interval_s:
            return
        previous = self._limit
        if kv_pct >= self.target_hi:
            self._limit = max(self.min_requests, self._limit - 1)
        elif kv_pct <= self.target_lo:
            self._limit = min(self.max_requests, self._limit + 1)
        if self._limit != previous:
            self._last_adjust_at = now
            self._record("limit.changed", "", before=previous, after=self._limit, kv_pct=kv_pct)

    def before_request(self, session_id: str, *, step: int, priority: int) -> None:
        """Block until this model request is admitted."""
        waiting_since = self._clock()
        deferred_recorded = False
        with self._cond:
            self._sessions[session_id] = {
                "state": "WAITING",
                "step": int(step),
                "priority": int(priority),
                "waiting_since": waiting_since,
                "wait_ms": 0.0,
            }
            self._record("request.waiting", session_id, step=step, priority=priority)

        while True:
            if self._cancel.is_set():
                raise RuntimeError("F5 运行已停止")
            kv_pct = self._read_kv_pct()
            with self._cond:
                self._latest_kv_pct = kv_pct
                if kv_pct is not None:
                    self._kv_peak_pct = max(self._kv_peak_pct or kv_pct, kv_pct)
                self._adjust_limit(kv_pct)
                if len(self._active) < self._limit:
                    self._active.add(session_id)
                    self._admitted += 1
                    self._max_active = max(self._max_active, len(self._active))
                    wait_ms = max(0.0, (self._clock() - waiting_since) * 1000.0)
                    self._sessions[session_id].update(state="RUNNING", wait_ms=wait_ms)
                    self._record(
                        "request.admitted",
                        session_id,
                        step=step,
                        priority=priority,
                        wait_ms=wait_ms,
                        limit=self._limit,
                        kv_pct=kv_pct,
                    )
                    return
                if not deferred_recorded:
                    deferred_recorded = True
                    self._deferred += 1
                    self._record(
                        "request.deferred",
                        session_id,
                        step=step,
                        priority=priority,
                        limit=self._limit,
                        kv_pct=kv_pct,
                    )
                self._sessions[session_id]["wait_ms"] = max(
                    0.0, (self._clock() - waiting_since) * 1000.0
                )
                self._cond.wait(timeout=self.poll_interval_s)

    def after_request(self, session_id: str) -> None:
        """Release one request slot and mark the session as user/tool thinking."""
        with self._cond:
            self._active.discard(session_id)
            session = self._sessions.setdefault(session_id, {})
            session["state"] = "THINKING"
            self._record("request.completed", session_id, step=session.get("step", 0))
            self._cond.notify_all()

    def mark_done(self, session_id: str, *, error: str | None = None) -> None:
        with self._cond:
            self._active.discard(session_id)
            session = self._sessions.setdefault(session_id, {})
            session["state"] = "ERROR" if error else "DONE"
            if error:
                session["error"] = error
            self._record("session.error" if error else "session.done", session_id, error=error)
            self._cond.notify_all()

    def cancel(self) -> None:
        self._cancel.set()
        with self._cond:
            self._cond.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            return {
                "adaptive": self.adaptive,
                "active_requests": len(self._active),
                "current_limit": self._limit,
                "min_requests": self.min_requests,
                "max_active_requests": self._max_active,
                "admitted_requests": self._admitted,
                "deferred_requests": self._deferred,
                "latest_kv_pct": self._latest_kv_pct,
                "kv_peak_pct": self._kv_peak_pct,
                "sessions": {sid: dict(data) for sid, data in self._sessions.items()},
                "events": list(self._events),
            }
