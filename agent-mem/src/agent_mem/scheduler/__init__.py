"""缝E · Session 生命周期调度。

- session 作为 first-class 实体：追踪活跃时间 + 状态机（:class:`SessionManager`）
- 策略决定何时搬 KV，机制以回调注入（可脱离 NPU 单测）
- F5 并发准入控制（:class:`AdmissionController`）—— HBM 驱动的动态并发 + idle 驱逐
"""

from __future__ import annotations

from agent_mem.scheduler.admission import AdmissionController
from agent_mem.scheduler.session import Session, SessionManager, SessionState
from agent_mem.scheduler.strategies import (
    BaseStrategy,
    CheckpointStrategy,
    IdleEvictionStrategy,
    NoOpStrategy,
    SessionStrategy,
)

__all__ = [
    "AdmissionController",
    "Session",
    "SessionManager",
    "SessionState",
    "SessionStrategy",
    "BaseStrategy",
    "NoOpStrategy",
    "IdleEvictionStrategy",
    "CheckpointStrategy",
]
