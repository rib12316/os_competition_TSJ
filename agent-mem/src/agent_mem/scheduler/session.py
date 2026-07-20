"""缝E · Session 生命周期：session 作为 first-class 实体 + 管理器。

F5（idle eviction）/ F6（KV checkpoint/恢复）都把 ``session`` 当一等公民：追踪活跃
时间、状态机、以及一个不透明的 ``kv_handle``（指向被 offload / checkpoint 的 KV，
由 缝C 的 offload connector 产生）。策略（:mod:`agent_mem.scheduler.strategies`）
决定**何时**搬 KV，机制（搬的动作）以回调注入——故策略可脱离 NPU 单测。
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class SessionState(enum.StrEnum):
    """session 的 KV 所在状态机。"""

    ACTIVE = "active"          # KV 在显存，可立即服务
    OFFLOADED = "offloaded"    # KV 已搬到 CPU/盘（F5 idle 后）
    CHECKPOINTED = "checkpointed"  # KV 落盘，进程可重启（F6）
    EVICTED = "evicted"        # KV 已丢弃（需重算 / 反向加载）


@dataclass
class Session:
    """单 agent session 的运行时记录。

    - ``last_active``：最近一次 ``touch`` 的单调时钟值（idle 判定用）。
    - ``kv_handle``：offload/checkpoint 后的 KV 句柄（由机制回调填充，策略不解释）。
    """

    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)
    state: SessionState = SessionState.ACTIVE
    kv_handle: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self, now: float | None = None) -> None:
        """标记本 session 刚被访问（更新 last_active）。"""
        self.last_active = now if now is not None else time.monotonic()


class SessionManager:
    """session 注册表 + idle 追踪 + 策略驱动入口。

    ``clock`` 可注入（测试用可控时钟）；默认 :func:`time.monotonic`。
    线程模型：非线程安全——单 driver 线程 sweep（与 benchmark 串行任务驱动一致）。
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._sessions: dict[str, Session] = {}
        self._clock = clock

    # ---- 注册 / 访问 ----

    def register(self, session_id: str, **metadata: Any) -> Session:
        """注册一个新 session（已存在则刷新 metadata 并返回既有）。"""
        if session_id in self._sessions:
            s = self._sessions[session_id]
            s.metadata.update(metadata)
            s.touch(self._clock())
            return s
        s = Session(session_id=session_id, metadata=dict(metadata))
        s.created_at = s.last_active = self._clock()
        self._sessions[session_id] = s
        return s

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def all_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    # ---- idle 计算 ----

    def touch(self, session_id: str) -> Session:
        """标记 session 活跃；不存在则按需注册。"""
        s = self._sessions.get(session_id) or self.register(session_id)
        s.touch(self._clock())
        return s

    def idle_seconds(self, session_id: str) -> float:
        """距上次活跃的秒数（不存在返回 +inf）。"""
        s = self._sessions.get(session_id)
        if s is None:
            return float("inf")
        return max(0.0, self._clock() - s.last_active)

    def idle_sessions(self, idle_timeout_s: float) -> list[Session]:
        """返回 idle >= ``idle_timeout_s`` 的 session（任意状态）。"""
        return [s for s in self._sessions.values() if self.idle_seconds(s.session_id) >= idle_timeout_s]

    # ---- 策略驱动 ----

    def sweep(self, strategy: Any) -> list[Session]:
        """遍历所有 session，交由策略决定动作（offload/checkpoint/...）。

        返回本轮被策略**改过状态**的 session（供观测/日志）。策略自己负责判定条件
        与调用机制回调。
        """
        touched: list[Session] = []
        for s in list(self._sessions.values()):
            before = s.state
            strategy.on_sweep(s, self)
            if s.state != before:
                touched.append(s)
        return touched
