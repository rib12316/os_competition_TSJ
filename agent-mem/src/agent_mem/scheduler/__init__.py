"""缝E · Session 生命周期调度（F5 idle eviction / F6 KV checkpoint-恢复）。

session 作为 first-class 实体：追踪活跃时间 + 状态机（:class:`SessionManager`），
策略（:class:`SessionStrategy`）决定何时搬 KV，机制以回调注入（可脱离 NPU 单测）。

- F5 idle eviction：:class:`IdleEvictionStrategy`
- F6 KV checkpoint/恢复：:class:`CheckpointStrategy`

二者**机制共享**（缝C offload connector）、**策略各自独立**（独立 yaml / before/after）。
"""

from __future__ import annotations

from agent_mem.scheduler.session import Session, SessionManager, SessionState
from agent_mem.scheduler.strategies import (
    BaseStrategy,
    CheckpointStrategy,
    IdleEvictionStrategy,
    NoOpStrategy,
    SessionStrategy,
)

__all__ = [
    "Session",
    "SessionManager",
    "SessionState",
    "SessionStrategy",
    "BaseStrategy",
    "NoOpStrategy",
    "IdleEvictionStrategy",
    "CheckpointStrategy",
]
