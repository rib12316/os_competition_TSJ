"""缝E · 动态资源回收事件计数器（F5）。

:class:`EvictionTracker` 记录一次并发 benchmark 里"资源被回收"的事件：总 eviction
次数、其中命中 idle session 的比例（"eviction-hit-idle%"，赛题关心的回收质量指标）。
由 :class:`agent_mem.scheduler.strategies.PriorityEvictionStrategy`（抬优先级=软驱逐）
和 :class:`agent_mem.scheduler.admission.AdmissionController`（idle 回队列）上报，
由 :mod:`agent_mem.metrics` 的 run summary 消费。

线程安全（并发 session 线程 + 后台 sweep 线程都会上报）。纯计数，无 NPU 依赖，可单测。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class EvictionEvent:
    """单次回收事件（观测/日志用，键值由上报方填）。"""

    t: float                 # 事件发生的单调时钟
    session_id: str
    was_idle: bool           # 被回收的 session 当时是否 idle（命中 idle）
    reason: str = ""         # 上报方自填：如 "priority-raise" / "admit-evict"


class EvictionTracker:
    """回收事件计数器。线程安全。

    用法::

        tracker = EvictionTracker()
        tracker.record_eviction("tau-3", was_idle=True, reason="priority-raise")
        tracker.snapshot()   # {"evictions": 1, "idle_hits": 1, "idle_hit_rate": 1.0}
    """

    def __init__(self, *, max_events: int = 4096):
        self._lock = threading.Lock()
        self._evictions = 0
        self._idle_hits = 0
        self._events: list[EvictionEvent] = []
        self._max_events = max_events

    def record_eviction(self, session_id: str, *, was_idle: bool, reason: str = "") -> None:
        """记录一次回收事件。``was_idle`` 标记被回收 session 当时是否 idle。"""
        ev = EvictionEvent(t=time.monotonic(), session_id=session_id,
                           was_idle=bool(was_idle), reason=reason)
        with self._lock:
            self._evictions += 1
            if ev.was_idle:
                self._idle_hits += 1
            self._events.append(ev)
            if len(self._events) > self._max_events:
                # 丢弃最旧，保留近 max_events 条（防止超长 run 无界增长）
                self._events = self._events[-self._max_events:]

    @property
    def evictions(self) -> int:
        with self._lock:
            return self._evictions

    @property
    def idle_hits(self) -> int:
        with self._lock:
            return self._idle_hits

    @property
    def idle_hit_rate(self) -> float:
        """命中 idle 的比例（0~1）；无 eviction 时为 0.0。"""
        with self._lock:
            return (self._idle_hits / self._evictions) if self._evictions else 0.0

    def events(self) -> list[EvictionEvent]:
        """返回事件列表的副本（观测/日志用）。"""
        with self._lock:
            return list(self._events)

    def snapshot(self) -> dict:
        """供 metrics run summary 消费的快照。"""
        with self._lock:
            ev, ih = self._evictions, self._idle_hits
        return {
            "evictions": ev,
            "idle_hits": ih,
            "idle_hit_rate": (ih / ev) if ev else 0.0,
        }
